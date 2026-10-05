from __future__ import annotations

from typing import Iterable

import numpy as np
from scipy import stats
from scipy.stats import bootstrap


def mean_with_ci(
    values: Iterable[float],
    confidence: float = 0.95,
    method: str = "sem",
    max_bootstrap_samples: int = 10_000,
) -> tuple[float, float, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")

    mu = float(arr.mean())
    if arr.size < 2:
        return mu, float("nan"), float("nan")

    method = method.lower()
    if method == "sem":
        se = float(stats.sem(arr, ddof=1))
        low, high = stats.norm.interval(confidence, loc=mu, scale=se)
    elif method == "bootstrap":
        sample = arr[:max_bootstrap_samples]
        bst = bootstrap(data=[sample], statistic=np.mean, confidence_level=confidence)
        low, high = bst.confidence_interval.low, bst.confidence_interval.high
    else:
        raise ValueError("Unknown method. Use 'sem' or 'bootstrap'.")

    return mu, float(low), float(high)


__all__ = ["mean_with_ci"]
