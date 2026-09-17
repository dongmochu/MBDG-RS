"""Small feature helpers shared by the paper's classical baselines."""

import numpy as np


def relative_group_zscore(values, group_width):
    """Standardize each within-trial feature group without cross-subject data.

    This removes a shared amplitude offset and scale only within the current
    trial. It therefore does not fit or use any held-out participant statistic.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] % int(group_width):
        raise ValueError("feature width must be divisible by group_width")
    groups = values.reshape(len(values), -1, int(group_width))
    center = groups.mean(axis=2, keepdims=True)
    scale = np.maximum(groups.std(axis=2, keepdims=True), 1e-8)
    return ((groups - center) / scale).reshape(values.shape)
