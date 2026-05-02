"""Dimensionality reduction and geometry metrics."""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from typing import Any

import numpy as np
from scipy.spatial.distance import cosine
from scipy.stats import spearmanr
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression


LOGGER = logging.getLogger(__name__)


def _as_numpy(value: Any) -> np.ndarray:
    """Convert tensors/lists/scalars into a NumPy array."""

    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def compute_distance(rep1: Any, rep2: Any, metric: str = "euclidean") -> float:
    """Compute a pairwise distance between two hidden representations."""

    arr1 = _as_numpy(rep1).reshape(-1)
    arr2 = _as_numpy(rep2).reshape(-1)
    if metric == "cosine":
        return float(cosine(arr1, arr2))
    if metric == "euclidean":
        return float(np.linalg.norm(arr1 - arr2))
    raise ValueError(f"Unknown distance metric: {metric}")


def compute_avg_distances(
    hidden_states: Mapping[int | str, list[Any]],
    metric: str = "euclidean",
) -> dict[int | str, float]:
    """Average within-group pairwise distances."""

    distances: dict[int | str, float] = {}
    for group, hidden_state_list in hidden_states.items():
        if len(hidden_state_list) < 2:
            distances[group] = math.nan
            continue
        group_distances: list[float] = []
        for i in range(len(hidden_state_list)):
            for j in range(i + 1, len(hidden_state_list)):
                group_distances.append(
                    compute_distance(hidden_state_list[i], hidden_state_list[j], metric=metric)
                )
        distances[group] = float(np.mean(group_distances))
    return distances


def _collect_layer_arrays(layer_payload: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, dict[Any, list[int]]]:
    """Stack all group representations and answers for one layer."""

    all_representations: list[np.ndarray] = []
    all_answers: list[float] = []
    group_indices: dict[Any, list[int]] = {}
    index = 0

    for group, reps in layer_payload["hidden_states"].items():
        group_indices[group] = list(range(index, index + len(reps)))
        for rep, answer in zip(reps, layer_payload["answers"][group]):
            all_representations.append(_as_numpy(rep).reshape(-1))
            all_answers.append(float(answer))
        index += len(reps)

    return np.vstack(all_representations), np.asarray(all_answers, dtype=float), group_indices


def transform_hidden_states(
    all_layers_hidden_states: Mapping[int, Mapping[str, Any]],
    method: str = "PCA",
    num_components: int = 1,
    logger: logging.Logger | None = None,
) -> dict[int, dict[str, Any]]:
    """Project hidden states with PCA or PLS and restore the group structure."""

    logger = logger or LOGGER
    transformed_hidden_states: dict[int, dict[str, Any]] = {}
    use_pls = method.lower() == "pls"

    for layer, layer_payload in all_layers_hidden_states.items():
        representations, answers, group_indices = _collect_layer_arrays(layer_payload)
        n_components = min(num_components, representations.shape[0], representations.shape[1])

        try:
            if use_pls:
                model = PLSRegression(n_components=n_components)
                model.fit(representations, answers.reshape(-1, 1))
                projections = model.transform(representations)
                score = float(model.score(representations, answers.reshape(-1, 1)))
                first_direction = model.x_weights_[:, 0]
                score_name = "R2"
            else:
                model = PCA(n_components=n_components)
                projections = model.fit_transform(representations)
                score = float(model.explained_variance_ratio_.sum())
                first_direction = model.components_[0]
                score_name = "EV"
        except ValueError as exc:
            logger.warning("Layer %s skipped during %s: %s", layer, method, exc)
            continue

        reduced_hidden_states: dict[Any, list[list[float]]] = {}
        for group, indices in group_indices.items():
            reduced_hidden_states[group] = [projections[i].tolist() for i in indices]

        transformed_hidden_states[int(layer)] = {
            "hidden_states": reduced_hidden_states,
            "answers": layer_payload["answers"],
            "Explained_variance": score,
            "score_name": score_name,
            "First_direction": first_direction.tolist(),
            "method": method.upper(),
        }

    return transformed_hidden_states


def compute_monotonicity(x: list[float] | np.ndarray, transformed_x: list[float] | np.ndarray) -> tuple[float, float]:
    """Compute signed and absolute Spearman monotonicity."""

    x_arr = np.asarray(x, dtype=float).reshape(-1)
    y_arr = np.asarray(transformed_x, dtype=float).reshape(-1)
    if x_arr.size != y_arr.size:
        raise ValueError("x and transformed_x must have the same length")
    signed, _ = spearmanr(x_arr, y_arr)
    signed = float(signed)
    return signed, abs(signed)


def _sort_group_key(group: Any) -> tuple[int, str]:
    try:
        return (0, f"{int(group):020d}")
    except (TypeError, ValueError):
        return (1, str(group))


def compute_scaling_rate_index(
    hidden_states: Mapping[str, Any],
    component_index: int = 0,
    epsilon: float = 1e-8,
) -> float:
    """Estimate beta from consecutive group-mean gaps in the projected axis."""

    group_means: list[float] = []
    sorted_groups = sorted(hidden_states["hidden_states"].keys(), key=_sort_group_key)

    for group in sorted_groups:
        values = np.asarray(hidden_states["hidden_states"][group], dtype=float)
        if values.ndim == 1:
            component_values = values
        else:
            component_values = values[:, component_index]
        group_means.append(float(np.mean(component_values)))

    group_diffs = np.abs(np.diff(group_means))
    group_diffs = np.clip(group_diffs, epsilon, None)
    if len(group_diffs) < 2:
        return math.nan

    x = np.arange(1, len(group_diffs) + 1, dtype=float).reshape(-1, 1)
    y = np.log(group_diffs).reshape(-1, 1)
    regression = LinearRegression().fit(x, y)
    return float(np.exp(regression.coef_[0][0]))


def analyze_transformed_hidden_states(
    transformed_hidden_states: Mapping[int, Mapping[str, Any]],
    component_index: int = 0,
) -> dict[int, dict[str, Any]]:
    """Add monotonicity and scaling metrics to transformed hidden states."""

    analyzed: dict[int, dict[str, Any]] = {}

    for layer, hidden_states in transformed_hidden_states.items():
        all_projections: list[float] = []
        all_answers: list[float] = []

        for group in hidden_states["hidden_states"].keys():
            projections = np.asarray(hidden_states["hidden_states"][group], dtype=float)
            if projections.ndim == 1:
                component = projections
            else:
                component = projections[:, component_index]
            all_projections.extend(component.tolist())
            all_answers.extend(float(answer) for answer in hidden_states["answers"][group])

        signed_rho, abs_rho = compute_monotonicity(all_answers, all_projections)
        beta = compute_scaling_rate_index(hidden_states, component_index=component_index)

        layer_payload = dict(hidden_states)
        layer_payload["monotonicity_signed"] = signed_rho
        layer_payload["monotonicity_metric"] = abs_rho
        layer_payload["sublinearity_metric"] = beta
        analyzed[int(layer)] = layer_payload

    return analyzed


def summarize_metric_runs(runs: list[Mapping[int, Mapping[str, Any]]]) -> dict[int, dict[str, float]]:
    """Aggregate per-layer metrics over repeated runs."""

    if not runs:
        return {}

    layers = sorted(set.intersection(*(set(run.keys()) for run in runs)))
    summary: dict[int, dict[str, float]] = {}

    metric_map = {
        "EV": "Explained_variance",
        "rho": "monotonicity_metric",
        "beta": "sublinearity_metric",
    }
    for layer in layers:
        summary[layer] = {}
        for output_name, source_name in metric_map.items():
            values = np.asarray([run[layer][source_name] for run in runs], dtype=float)
            summary[layer][output_name] = float(np.nanmean(values))
            summary[layer][f"{output_name}_std"] = float(np.nanstd(values))
    return summary


def select_best_layer(
    metrics_by_layer: Mapping[int, Mapping[str, float]],
    score_key: str = "EV",
) -> tuple[int, Mapping[str, float]]:
    """Select the layer with the highest score key."""

    if not metrics_by_layer:
        raise ValueError("metrics_by_layer is empty")
    return max(metrics_by_layer.items(), key=lambda item: item[1].get(score_key, float("-inf")))
