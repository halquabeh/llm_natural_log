"""Backward-compatible wrappers around ``natural_log.transforms``."""

from natural_log.transforms import (
    analyze_transformed_hidden_states,
    compute_avg_distances,
    compute_distance as _compute_distance,
    compute_monotonicity as _compute_monotonicity,
    compute_scaling_rate_index as compute_sublinearity,
    select_best_layer,
    summarize_metric_runs,
    transform_hidden_states,
)


def compute_monotonicity(x, transformed_x):
    """Old scalar return style for notebooks."""

    signed_rho, _ = _compute_monotonicity(x, transformed_x)
    return signed_rho

__all__ = [
    "_compute_distance",
    "analyze_transformed_hidden_states",
    "compute_avg_distances",
    "compute_monotonicity",
    "compute_sublinearity",
    "select_best_layer",
    "summarize_metric_runs",
    "transform_hidden_states",
]

