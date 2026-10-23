"""Participant-cluster sensitivity analyses and group-metric summaries."""

from __future__ import annotations

import math
import random
from collections import defaultdict


def benjamini_hochberg(p_values):
    values = [float(p) for p in p_values]
    if any(not math.isfinite(p) or not 0 <= p <= 1 for p in values):
        raise ValueError("Finite p values in [0,1] required")
    n = len(values)
    order = sorted(range(n), key=values.__getitem__)
    adjusted, previous = [0.0] * n, 1.0
    for rank in range(n, 0, -1):
        index = order[rank - 1]
        previous = min(previous, values[index] * n / rank)
        adjusted[index] = previous
    return adjusted


def participant_sign_flip(participants, before, after, *, repetitions=9999, seed=42):
    """Two-sided Monte Carlo test on person-mean paired changes; equal person weight."""
    if len(participants) != len(before) or len(before) != len(after) or repetitions < 1:
        raise ValueError("Aligned paired arrays and positive repetitions required")
    grouped = defaultdict(list)
    for person, left, right in zip(participants, before, after):
        if not person or not math.isfinite(left) or not math.isfinite(right):
            raise ValueError("Known participant and finite paired measurements required")
        grouped[person].append(right - left)
    changes = [sum(grouped[p]) / len(grouped[p]) for p in sorted(grouped)]
    if len(changes) < 2:
        raise ValueError("At least two participants required")
    observed = abs(sum(changes) / len(changes))
    rng, count = random.Random(seed), 0
    for _ in range(repetitions):
        permuted = abs(sum(d * rng.choice((-1, 1)) for d in changes) / len(changes))
        count += permuted >= observed - 1e-14
    return {
        "n_participants": len(changes),
        "n_sessions": len(before),
        "participant_mean_change": sum(changes) / len(changes),
        "p_two_sided": (count + 1) / (repetitions + 1),
        "method": "participant_mean_sign_flip",
        "repetitions": repetitions,
        "seed": seed,
    }


def pairwise_summary(rates):
    """Return separately named best/worst DI; never call max(DI) the worst pair."""
    pairs = []
    names = sorted(rates)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            a, b = rates[left], rates[right]
            values = [float(r[key]) for r in (a, b) for key in ("tpr", "fpr", "ppp")]
            if any(not math.isfinite(x) or not 0 <= x <= 1 for x in values):
                raise ValueError("Finite group rates in [0,1] required")
            if a["ppp"] == b["ppp"] == 0:
                di = None  # No silently assigned parity for undefined zero/zero.
            elif min(a["ppp"], b["ppp"]) == 0:
                di = 0.0
            else:
                di = min(a["ppp"] / b["ppp"], b["ppp"] / a["ppp"])
            pairs.append(
                {
                    "groups": [left, right],
                    "di": di,
                    "aod": (abs(a["tpr"] - b["tpr"]) + abs(a["fpr"] - b["fpr"])) / 2,
                }
            )
    if not pairs:
        raise ValueError("At least two groups required")
    valid = [p["di"] for p in pairs if p["di"] is not None]
    return {
        "mean_pair_aod": sum(p["aod"] for p in pairs) / len(pairs),
        "di_worst_pair": min(valid) if valid else None,
        "di_best_pair": max(valid) if valid else None,
        "pairs": pairs,
    }
