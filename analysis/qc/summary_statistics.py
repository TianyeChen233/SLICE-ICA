"""Explicit paired-session statistics and network FC definitions.

These functions calculate new outputs from supplied arrays. They do not encode
manuscript p values, and session-level tests do not correct participant clustering.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import wilcoxon


def paired_wilcoxon(before, after):
    """Two-sided signed-rank test; remove missing values as complete pairs.

    Use the asymptotic method explicitly for reproducible handling of ties. The
    statistic is the smaller signed-rank sum. All-zero differences give p=1;
    fewer than three finite pairs give undefined statistics, never significance.
    """
    before, after = np.asarray(before, dtype=float), np.asarray(after, dtype=float)
    if before.ndim != 1 or before.shape != after.shape:
        raise ValueError("Aligned one-dimensional before/after arrays required")
    mask = np.isfinite(before) & np.isfinite(after)
    left, right = before[mask], after[mask]
    n = len(left)
    statistic, p_value = np.nan, np.nan
    delta = right - left
    if n >= 3:
        if np.all(delta == 0):
            statistic, p_value = 0.0, 1.0
        else:
            result = wilcoxon(
                delta,
                alternative="two-sided",
                zero_method="wilcox",
                correction=False,
                method="approx",
            )
            statistic, p_value = float(result.statistic), float(result.pvalue)
    return {
        "n": n,
        "n_nonzero": int(np.count_nonzero(delta)),
        "n_missing_pairs": int(len(before) - n),
        "before_mean": float(left.mean()) if n else np.nan,
        "after_mean": float(right.mean()) if n else np.nan,
        "mean_delta_after_minus_before": float(delta.mean()) if n else np.nan,
        "median_delta_after_minus_before": float(np.median(delta)) if n else np.nan,
        "wilcoxon_stat": statistic,
        "p_value": p_value,
        "test": "wilcoxon_two_sided",
        "zero_method": "wilcox",
        "p_method": "asymptotic_no_continuity_correction",
        "analysis_unit": "session",
    }


def adjust_bh(p_values):
    """BH across finite p values; preserve undefined entries as NaN."""
    values = np.asarray(p_values, dtype=float)
    if values.ndim != 1 or np.any(np.isinf(values)):
        raise ValueError("One-dimensional p values required; infinity is invalid")
    valid = np.isfinite(values)
    if np.any((values[valid] < 0) | (values[valid] > 1)):
        raise ValueError("p values must lie in [0,1]")
    output = np.full(values.shape, np.nan)
    indices = np.flatnonzero(valid)
    order = indices[np.argsort(values[indices])]
    n = len(order)
    if n:
        scaled = values[order] * n / np.arange(1, n + 1)
        output[order] = np.minimum.accumulate(scaled[::-1])[::-1].clip(0, 1)
    return output


def network_block_means(correlation, network_indices, n_networks):
    """Equal-edge means; within-network entries exclude parcel self-correlations.

    Each within-network edge appears once (strict upper triangle). Each between
    block contains all cross-network parcel pairs. No Fisher-z transform is used.
    """
    matrix = np.asarray(correlation, dtype=float)
    indices = np.asarray(network_indices)
    if (
        matrix.ndim != 2
        or matrix.shape[0] != matrix.shape[1]
        or indices.shape != (matrix.shape[0],)
        or not np.isfinite(matrix).all()
        or not np.allclose(matrix, matrix.T, atol=1e-6)
        or not np.issubdtype(indices.dtype, np.integer)
        or n_networks < 1
        or set(indices.tolist()) != set(range(n_networks))
    ):
        raise ValueError("Finite symmetric correlations and contiguous network indices required")
    result = np.empty((n_networks, n_networks), dtype=float)
    groups = [np.flatnonzero(indices == k) for k in range(n_networks)]
    if any(len(group) < 2 for group in groups):
        raise ValueError("At least two parcels per network required for within-network FC")
    for i, left in enumerate(groups):
        for j in range(i, n_networks):
            block = matrix[np.ix_(left, groups[j])]
            values = block[np.triu_indices(len(left), k=1)] if i == j else block.ravel()
            result[i, j] = result[j, i] = values.mean()
    return result


def session_network_summaries(network_matrices):
    """Equal-network/pair weighting within each session, before group averaging.

    Ratio = session within mean / session between mean. A zero denominator
    produces NaN, not a manufactured zero. Mean(session ratios) and
    ratio(group means) must be reported separately by callers.
    """
    matrices = np.asarray(network_matrices, dtype=float)
    if (
        matrices.ndim != 3
        or matrices.shape[1] != matrices.shape[2]
        or matrices.shape[1] < 2
        or not np.isfinite(matrices).all()
    ):
        raise ValueError("Finite session-by-network-by-network matrices required")
    within = np.diagonal(matrices, axis1=1, axis2=2).mean(axis=1)
    upper = np.triu_indices(matrices.shape[1], k=1)
    between = matrices[:, upper[0], upper[1]].mean(axis=1)
    ratio = np.full(within.shape, np.nan)
    np.divide(within, between, out=ratio, where=np.abs(between) > 1e-12)
    return {"within": within, "between": between, "within_between_ratio": ratio}
