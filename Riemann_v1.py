# Potreban je NumPy ≥ 2.0
# scipy


import csv
import heapq
import math
from pathlib import Path

import numpy as np
from scipy.special import logsumexp


CSV_PATH = Path(
    "/data/loto7_4698_k80.csv"
    # "/data/loto7_4698_k80_loto_2971.csv"
    # "/data/loto7_4698_k80_loto_plus_1727.csv"
)

N = 39
K = 7
TOTAL_GAPS = N - K
TOTAL = math.comb(N, K)

GRID_SIZES = (129, 257, 513)
INTEGRATION_TOL = 1e-6

ALPHA_RANGES = (
    (0.1, 1e4),
    (1.0, 1e7),
)

SHAPE_STRENGTHS = (0.2, 1.0)


def load_csv(path):
    """
    CSV bez zaglavlja, sedam brojeva po redu.
    Prvi red najstariji, poslednji najnoviji.
    """
    rows = []

    with Path(path).open(encoding="utf-8-sig", newline="") as file:
        for line, row in enumerate(csv.reader(file), 1):
            if not row or all(not value.strip() for value in row):
                continue

            try:
                draw = sorted(int(value.strip()) for value in row)
            except ValueError as exc:
                raise ValueError(
                    f"Red {line}: neispravan ceo broj."
                ) from exc

            if (
                len(draw) != K
                or len(set(draw)) != K
                or not all(1 <= value <= N for value in draw)
            ):
                raise ValueError(
                    f"Red {line}: potrebno je sedam "
                    "različitih brojeva od 1 do 39."
                )

            rows.append(draw)

    if len(rows) < 50:
        raise ValueError("Potrebno je najmanje 50 izvlačenja.")

    return np.asarray(rows, dtype=np.int64)


def gaps(draws, n=N):
    """
    Osam razmaka, uključujući oba kraja.

    Za sedmorku s:
      g0 = s1 - 1
      gi = s(i+1) - si - 1
      g7 = 39 - s7

    Razmaci su nenegativni, njihov zbir je 32.
    Svaka validna sedmorka ima jedinstveni vektor razmaka.
    """
    extended = np.column_stack((
        np.zeros(len(draws), dtype=int),
        draws,
        np.full(len(draws), n + 1),
    ))

    return np.diff(extended, axis=1) - 1


def build_model(draw_gaps, total=TOTAL_GAPS):
    """
    Dovoljne statistike za distribucije razmaka.

    Ne rangiraju se frekvencije pojedinačnih loto brojeva.
    Zajednička distribucija nastaje uslovljavanjem svih
    osam razmaka na njihov fiksni zbir.
    """
    counts = np.asarray([
        np.bincount(
            draw_gaps[:, pos],
            minlength=total + 1,
        )
        for pos in range(draw_gaps.shape[1])
    ])

    ratio = total / (total + draw_gaps.shape[1])
    base = ratio ** np.arange(total + 1)
    base /= base.sum()

    return {
        "counts": counts,
        "draws": draw_gaps,
    }, base


def gap_kernel(counts, base, alpha):
    """
    Za svaki alpha formira zajedničku distribuciju:

        P(g | alpha) ∝ product_j q_j(g_j | alpha),

    uz obavezni uslov sum(g) = 32.

    Normalizaciona konstanta računa se dinamičkim
    programiranjem preko svih validnih vektora razmaka.
    """
    length, span = counts.shape
    size = len(alpha)

    factor = (
        counts[None, :, :]
        + alpha[:, None, None] * base[None, None, :]
    ) / (
        counts[0].sum() + alpha[:, None, None]
    )

    partition = np.zeros((
        size,
        length + 1,
        span,
    ))

    partition[:, length, 0] = 1.0

    for pos in range(length - 1, -1, -1):
        for remaining in range(span):
            values = np.arange(remaining + 1)

            partition[:, pos, remaining] = (
                factor[:, pos, values]
                * partition[:, pos + 1, remaining - values]
            ).sum(axis=1)

    # Sedam slobodnih razmaka; osmi je preostali zbir.
    transition = np.zeros((
        size,
        length - 1,
        span,
        1,
        span,
    ))

    for pos in range(length - 1):
        for remaining in range(span):
            values = np.arange(remaining + 1)

            transition[
                :, pos, remaining, 0, :remaining + 1
            ] = (
                factor[:, pos, values]
                * partition[:, pos + 1, remaining - values]
            ) / partition[:, pos, remaining, None]

    return factor, partition, transition


def integrate(model, base, alpha_range, size):
    """
    Deterministička Rimanova suma preko log(alpha).

    Težine parametara određuju se hronološkim prediktivnim
    log-scoreom: svaki blok ocenjuje se modelom naučenim
    samo na ranijim redovima.

    Završne distribucije razmaka koriste sve dostupne redove.
    """
    counts = model["counts"]
    samples = model["draws"]

    low, high = map(math.log, alpha_range)
    step = (high - low) / size

    alpha = np.exp(
        low + (np.arange(size) + 0.5) * step
    )

    log_evidence = np.zeros(size)

    boundaries = np.linspace(
        len(samples) // 2,
        len(samples),
        5,
        dtype=int,
    )

    for left, right in zip(
        boundaries[:-1],
        boundaries[1:],
    ):
        training = np.asarray([
            np.bincount(
                samples[:left, pos],
                minlength=counts.shape[1],
            )
            for pos in range(samples.shape[1])
        ])

        validation = np.asarray([
            np.bincount(
                samples[left:right, pos],
                minlength=counts.shape[1],
            )
            for pos in range(samples.shape[1])
        ])

        factor, partition, _ = gap_kernel(
            training,
            base,
            alpha,
        )

        log_evidence += (
            validation[None, :, :] * np.log(factor)
        ).sum(axis=(1, 2))

        log_evidence -= (
            (right - left) * np.log(partition[:, 0, -1])
        )

    logs = log_evidence + math.log(step)
    mass = np.exp(logs - logsumexp(logs))

    _, _, transition = gap_kernel(
        counts,
        base,
        alpha,
    )

    if (
        not np.isclose(mass.sum(), 1.0)
        or not np.allclose(
            transition.sum(axis=-1),
            1.0,
        )
    ):
        raise RuntimeError(
            "Prediktivna distribucija nije normalizovana."
        )

    return mass, transition


def probabilities(draw_gaps, mass, transition):
    """Integrisane verovatnoće celih sedmorki."""
    total = transition.shape[2] - 1
    memory = transition.shape[3] > 1

    product = np.ones((
        len(mass),
        len(draw_gaps),
    ))

    remaining = np.full(
        len(draw_gaps),
        total,
        dtype=int,
    )
    previous = np.zeros(len(draw_gaps), dtype=int)

    for pos in range(draw_gaps.shape[1] - 1):
        current = draw_gaps[:, pos]

        bucket = (
            np.minimum(previous, 2)
            if memory
            else np.zeros_like(previous)
        )

        product *= transition[
            :, pos, remaining, bucket, current
        ]

        remaining -= current
        previous = current

    return mass @ product


def suffix_tables(transition):
    """
    Računa:
    - ukupne verovatnoće nastavaka;
    - najveće verovatnoće nastavaka.

    Posebno prati broj susednih parova u sedmorki.
    """
    size, k, span, buckets, _ = transition.shape
    memory = buckets > 1

    sums = np.zeros((
        size,
        k + 1,
        span,
        buckets,
        k,
    ))
    maxima = np.zeros_like(sums)

    sums[:, k, :, :, 0] = 1.0
    maxima[:, k, :, :, 0] = 1.0

    for pos in range(k - 1, -1, -1):
        for remaining in range(span):
            for bucket in range(buckets):
                for value in range(remaining + 1):
                    extra = int(pos > 0 and value == 0)

                    next_bucket = (
                        min(value, 2) if memory else 0
                    )

                    probability = transition[
                        :, pos, remaining, bucket, value, None
                    ]

                    continuation = sums[
                        :,
                        pos + 1,
                        remaining - value,
                        next_bucket,
                        :k - extra,
                    ]

                    sums[
                        :, pos, remaining, bucket, extra:
                    ] += probability * continuation

                    continuation = maxima[
                        :,
                        pos + 1,
                        remaining - value,
                        next_bucket,
                        :k - extra,
                    ]

                    maxima[
                        :, pos, remaining, bucket, extra:
                    ] = np.maximum(
                        maxima[
                            :, pos, remaining, bucket, extra:
                        ],
                        probability * continuation,
                    )

    return sums, maxima


def shape_correction(
    draw_gaps,
    mass,
    transition,
    strength,
    tables=None,
):
    """
    Usklađuje ukupnu distribuciju susednih parova
    sa distribucijom te strukture u istorijskim izvlačenjima.

    Ne zabranjuje grupisane kombinacije.
    Korekcija je normalizovana i regularizovana.
    """
    if tables is None:
        tables = suffix_tables(transition)

    sums, _ = tables

    k = transition.shape[1]
    total = transition.shape[2] - 1
    n = total + k

    base = np.asarray([
        math.comb(k - 1, adjacency)
        * math.comb(n - k + 1, k - adjacency)
        / math.comb(n, k)
        for adjacency in range(k)
    ])

    adjacency = (
        draw_gaps[:, 1:-1] == 0
    ).sum(axis=1)

    observed = np.bincount(
        adjacency,
        minlength=k,
    )

    prior_weight = strength * len(draw_gaps)

    target = (
        observed + prior_weight * base
    ) / (
        len(draw_gaps) + prior_weight
    )

    raw = mass @ sums[:, 0, total, 0, :]

    if not np.isclose(raw.sum(), 1.0):
        raise RuntimeError(
            "Distribucija strukture nije normalizovana."
        )

    correction = target / raw

    if not np.isclose(
        np.sum(raw * correction),
        1.0,
    ):
        raise RuntimeError(
            "Kalibrisana distribucija nije normalizovana."
        )

    return correction


def select_model(draw_gaps):
    split = int(0.8 * len(draw_gaps))

    training = draw_gaps[:split]
    validation = draw_gaps[split:]

    adjacency = (
        validation[:, 1:-1] == 0
    ).sum(axis=1)

    best = None
    best_score = -math.inf

    print(
        f"Hronološka obuka={split}; "
        f"provera={len(validation)}",
        flush=True,
    )
    print(
        f"Referentni log P={-math.log(TOTAL):.9f}",
        flush=True,
    )

    model, base = build_model(training)

    for interval in ALPHA_RANGES:
        mass, transition = integrate(
            model,
            base,
            interval,
            129,
        )

        tables = suffix_tables(transition)

        raw = probabilities(
            validation,
            mass,
            transition,
        )

        for strength in SHAPE_STRENGTHS:
            correction = shape_correction(
                training,
                mass,
                transition,
                strength,
                tables,
            )

            score = float(np.log(
                raw * correction[adjacency]
            ).mean())

            print(
                f"Alpha={interval}; "
                f"struktura={strength}; "
                f"log P={score:.9f}",
                flush=True,
            )

            if score > best_score + 1e-10:
                best = (interval, strength)
                best_score = score

    if best is None:
        raise RuntimeError("Nijedan model nije izabran.")

    return best


def to_combination(prefix):
    return tuple(
        int(value)
        for value in np.cumsum(np.asarray(prefix) + 1)
    )


def maximum(mass, transition, correction, tables=None):
    """
    Globalni maksimum numerički integrisane distribucije.

    Deterministička branch-and-bound pretraga sa
    matematičkom gornjom granicom za svaku granu.
    Ne koristi random uzorkovanje niti beam search.
    """
    if tables is None:
        tables = suffix_tables(transition)

    _, bounds = tables

    size, k, span, buckets, _ = transition.shape
    total = span - 1
    memory = buckets > 1

    best = None
    best_p = -1.0
    seeds = set()

    # Deterministički početni kandidati.
    for grid in range(size):
        for adjacency in range(k):
            if bounds[
                grid, 0, total, 0, adjacency
            ] <= 0:
                continue

            remaining = total
            bucket = 0
            needed = adjacency
            prefix = []

            for pos in range(k):
                scores = []

                for value in range(remaining + 1):
                    extra = int(pos > 0 and value == 0)

                    next_bucket = (
                        min(value, 2) if memory else 0
                    )

                    if needed < extra:
                        score = 0.0
                    else:
                        score = (
                            transition[
                                grid,
                                pos,
                                remaining,
                                bucket,
                                value,
                            ]
                            * bounds[
                                grid,
                                pos + 1,
                                remaining - value,
                                next_bucket,
                                needed - extra,
                            ]
                        )

                    scores.append(score)

                value = int(np.argmax(scores))

                needed -= int(pos > 0 and value == 0)
                prefix.append(value)
                remaining -= value
                bucket = min(value, 2) if memory else 0

            seeds.add(tuple(prefix))

    for prefix in sorted(seeds):
        product = np.ones(size)
        remaining = total
        bucket = 0

        for pos, value in enumerate(prefix):
            product *= transition[
                :, pos, remaining, bucket, value
            ]

            remaining -= value
            bucket = min(value, 2) if memory else 0

        adjacency = sum(
            value == 0 for value in prefix[1:]
        )

        p = float(mass @ product) * correction[adjacency]
        combination = to_combination(prefix)

        if (
            p > best_p
            or (p == best_p and combination < best)
        ):
            best = combination
            best_p = p

    def upper_bound(
        product,
        pos,
        remaining,
        bucket,
        adjacency,
    ):
        possibilities = (
            bounds[
                :, pos, remaining, bucket, :k - adjacency
            ]
            * correction[None, adjacency:]
        )

        return float(
            mass @ (
                product * possibilities.max(axis=1)
            )
        )

    queue = [(
        -upper_bound(
            np.ones(size), 0, total, 0, 0
        ),
        (),
        np.ones(size),
    )]

    visited = 0

    while queue:
        negative_bound, prefix, product = heapq.heappop(
            queue
        )

        if -negative_bound < best_p * (1.0 - 1e-12):
            continue

        visited += 1
        pos = len(prefix)
        remaining = total - sum(prefix)

        adjacency = sum(
            value == 0 for value in prefix[1:]
        )

        bucket = (
            min(prefix[-1], 2)
            if memory and prefix
            else 0
        )

        if pos == k:
            p = (
                float(mass @ product)
                * correction[adjacency]
            )

            combination = to_combination(prefix)

            if (
                p > best_p
                or (p == best_p and combination < best)
            ):
                best = combination
                best_p = p

            continue

        for value in range(remaining + 1):
            child = prefix + (value,)

            next_bucket = (
                min(value, 2) if memory else 0
            )

            child_product = (
                product
                * transition[
                    :, pos, remaining, bucket, value
                ]
            )

            next_adjacency = (
                adjacency
                + int(pos > 0 and value == 0)
            )

            upper = upper_bound(
                child_product,
                pos + 1,
                remaining - value,
                next_bucket,
                next_adjacency,
            )

            if upper >= best_p * (1.0 - 1e-12):
                heapq.heappush(
                    queue,
                    (-upper, child, child_product),
                )

    if best is None:
        raise RuntimeError(
            "Globalni maksimum nije pronađen."
        )

    return best, best_p, visited


def fit_all(draw_gaps, selected):
    interval, strength = selected
    model, base = build_model(draw_gaps)

    previous = None
    previous_best = None
    change = math.inf

    for size in GRID_SIZES:
        mass, transition = integrate(
            model,
            base,
            interval,
            size,
        )

        tables = suffix_tables(transition)

        correction = shape_correction(
            draw_gaps,
            mass,
            transition,
            strength,
            tables,
        )

        adjacency = (
            draw_gaps[:, 1:-1] == 0
        ).sum(axis=1)

        values = (
            probabilities(
                draw_gaps,
                mass,
                transition,
            )
            * correction[adjacency]
        )

        prediction, p, visited = maximum(
            mass,
            transition,
            correction,
            tables,
        )

        if previous is not None:
            change = max(
                float(np.max(np.abs(
                    values / previous - 1.0
                ))),
                abs(p / previous_best - 1.0),
            )

            print(
                f"Integracija={size}; "
                f"relativna promena={change:.9g}",
                flush=True,
            )
        else:
            print(
                f"Integracija={size}",
                flush=True,
            )

        if change <= INTEGRATION_TOL:
            break

        previous = values
        previous_best = p

    if change > INTEGRATION_TOL:
        print(
            "Napomena: integracija je dostigla "
            "maksimalnu rezoluciju.",
            flush=True,
        )

    return prediction, p, visited


def main():
    draws = load_csv(CSV_PATH)
    draw_gaps = gaps(draws)

    if not np.all(
        draw_gaps.sum(axis=1) == TOTAL_GAPS
    ):
        raise RuntimeError(
            "Neispravno predstavljanje sedmorki."
        )

    print(
        f"Loto integral v1; CSV={CSV_PATH}; "
        f"svih {len(draws)} izvlačenja.",
        flush=True,
    )
    print(
        "Prvi red najstariji; poslednji najnoviji.",
        flush=True,
    )

    selected = select_model(draw_gaps)

    prediction, p, visited = fit_all(
        draw_gaps,
        selected,
    )

    print(
        f"Globalni maksimum proveren; "
        f"čvorova={visited}.",
        flush=True,
    )
    print("\nNEXT:", " ".join(map(str, prediction)))
    print(f"Verovatnoća po modelu: {p:.12g}")


if __name__ == "__main__":
    main()



"""
Loto integral v1; CSV=/data/loto7_4698_k80.csv; svih 4698 izvlačenja.
Prvi red najstariji; poslednji najnoviji.
Hronološka obuka=3758; provera=940
Referentni log P=-16.548639443
Alpha=(0.1, 10000.0); struktura=0.2; log P=-16.550814927
Alpha=(0.1, 10000.0); struktura=1.0; log P=-16.550803975
Alpha=(1.0, 10000000.0); struktura=0.2; log P=-16.548301839
Alpha=(1.0, 10000000.0); struktura=1.0; log P=-16.548290886
Integracija=129
Integracija=257; relativna promena=2.04438222e-09
Globalni maksimum proveren; čvorova=8.

NEXT: 8 x 20 y 26 z 34
Verovatnoća po modelu: 6.9426270208e-08





Loto integral v1; CSV=/data/loto7_4698_k80_loto_2971.csv; svih 2971 izvlačenja.
Prvi red najstariji; poslednji najnoviji.
Hronološka obuka=2376; provera=595
Referentni log P=-16.548639443
Alpha=(0.1, 10000.0); struktura=0.2; log P=-16.551808749
Alpha=(0.1, 10000.0); struktura=1.0; log P=-16.550302676
Alpha=(1.0, 10000000.0); struktura=0.2; log P=-16.550622680
Alpha=(1.0, 10000000.0); struktura=1.0; log P=-16.549116608
Integracija=129
Integracija=257; relativna promena=1.92717847e-08
Globalni maksimum proveren; čvorova=13.

NEXT: 8 x 16 y 18 z 34
Verovatnoća po modelu: 7.86207902332e-08





Loto integral v1; CSV=/data/loto7_4698_k80_loto_plus_1727.csv; svih 1727 izvlačenja.
Prvi red najstariji; poslednji najnoviji.
Hronološka obuka=1381; provera=346
Referentni log P=-16.548639443
Alpha=(0.1, 10000.0); struktura=0.2; log P=-16.546663161
Alpha=(0.1, 10000.0); struktura=1.0; log P=-16.546399952
Alpha=(1.0, 10000000.0); struktura=0.2; log P=-16.548889779
Alpha=(1.0, 10000000.0); struktura=1.0; log P=-16.548626570
Integracija=129
Integracija=257; relativna promena=0.000263827493
Integracija=513; relativna promena=6.63746474e-05
Napomena: integracija je dostigla maksimalnu rezoluciju.
Globalni maksimum proveren; čvorova=8.

NEXT: 4 x 6 y 22 z 34
Verovatnoća po modelu: 8.27408691483e-08
"""



"""
Riemann sums, antiderivatives and the fundamental theorem of calculus. 

Računam distribuciju integracijom preko nepoznatog stanja.
Računam prediktivne distribucije kada model ima skriveno stanje.
Uzima u obzir sva moguća stanja, ponderisana njihovom verovatnoćom.
- Rimanove sume: omogućavaju determinističku aproksimaciju tog integrala preko mreže stanja — bez random uzorkovanja. Korisne su ako stanje ima malo dimenzija.
- Primitivne funkcije: ako ih model dopušta, integral se može izračunati analitički, preciznije i brže.
- Osnovna teorema analize: povezuje integral i primitivnu funkciju, pa omogućava računanje verovatnoće intervala kao razlike vrednosti kumulativne funkcije.

Integracija preko neizvesnog stanja, umesto oslanjanja samo na jedno procenjeno stanje. 
Za konačan broj skrivenih stanja koristi se obična ponderisana suma; za kontinuirano stanje integral.
Same sedmorke 7/39 čine diskretan prostor, pa se njihove verovatnoće sabiraju. 
Ove metode pomažu da se distribucija izračuna; prediktivni signal mora doći iz modela i podataka.


Učiće zajedničku distribuciju sedmorke kroz uslovne distribucije sortiranih pozicija, a nepoznati parametar te distribucije integrisaće determinističkim Rimanovim sumama. 
Zatim proveriti normalizaciju i izračunati jednu kombinaciju sa najvećom verovatnoćom.
Model rasporeda svih osam razmaka sedmorke, integrisati njegov nepoznati parametar i proveriti predikcije na kasnijim izvlačenjima. 
Grupisanje će ocenjivati distribucija naučena iz CSV-a, bez proizvoljne zabrane velikih brojeva.
Proveru strukture cele sedmorke: model će usklađivati ukupnu verovatnoću takvih nizova sa njihovom distribucijom u CSV-u. 
To će promeniti ocenu grupisanih kombinacija na osnovu podataka, bez zabrane određenih brojeva.
Model koristi zajedničku distribuciju svih osam razmaka sedmorke, sa proverom strukture cele kombinacije. 
Parametar integriše deterministički; izbor modela prolazi hronološku proveru. 
Provereni su normalizacija, stabilnost integracije i globalni maksimum.
"""
