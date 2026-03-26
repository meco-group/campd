import numpy as np


def summarize_1d(x, nan_safe=True):
    x = np.asarray(x, dtype=float)

    mean = np.nanmean if nan_safe else np.mean
    std = np.nanstd if nan_safe else np.std
    min_ = np.nanmin if nan_safe else np.min
    max_ = np.nanmax if nan_safe else np.max
    med = np.nanmedian if nan_safe else np.median
    percentile = np.nanpercentile if nan_safe else np.percentile

    p95, p99 = percentile(x, [95, 99])

    return {
        "mean": float(mean(x)),
        "std": float(std(x)),
        "min": float(min_(x)),
        "max": float(max_(x)),
        "median": float(med(x)),
        "p95": float(p95),
        "p99": float(p99),
    }


def summarize_2d(a, nan_safe=True):
    a = np.asarray(a, dtype=float)
    if a.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {a.shape}")

    mean = np.nanmean if nan_safe else np.mean
    std = np.nanstd if nan_safe else np.std
    percentile = np.nanpercentile if nan_safe else np.percentile

    # Row-wise statistics
    row_means = mean(a, axis=1)
    row_stds = std(a, axis=1)
    row_p95 = percentile(a, 95, axis=1)
    row_p99 = percentile(a, 99, axis=1)

    return {
        "row_mean": summarize_1d(row_means, nan_safe),
        "row_std":  summarize_1d(row_stds,  nan_safe),
        "row_p95":  summarize_1d(row_p95,   nan_safe),
        "row_p99":  summarize_1d(row_p99,   nan_safe),
        "all":      summarize_1d(a.ravel(), nan_safe),
    }


def summarize_all_stats(all_stats: dict[str, list]) -> dict[str, dict]:
    """Summarize all statistics in a dictionary."""
    for key, values in list(all_stats.items()):
        if not values:
            continue

        v0 = values[0]

        # Scalars → 1D summary
        if isinstance(v0, (int, float, np.number)):
            all_stats[key] = summarize_1d(values, nan_safe=True)
            continue

        # List/array-like → numpy
        if isinstance(v0, (list, tuple, np.ndarray)):
            arr = np.asarray(values, dtype=float)

            if arr.ndim == 1:
                all_stats[key] = summarize_1d(arr, nan_safe=True)
            elif arr.ndim == 2:
                all_stats[key] = summarize_2d(arr, nan_safe=True)
            else:
                raise ValueError(
                    f"{key}: expected 1D or 2D, got shape {arr.shape}")

    return all_stats
