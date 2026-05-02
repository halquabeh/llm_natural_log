"""Publication-style plotting utilities."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt
import numpy as np


def set_paper_style() -> None:
    """Set a consistent, journal-friendly Matplotlib style."""

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.titlesize": 11,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.6,
            "savefig.dpi": 300,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.5,
            "axes.edgecolor": "#333333",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _group_sort_key(group: Any) -> tuple[int, str]:
    try:
        return (0, f"{int(group):020d}")
    except (TypeError, ValueError):
        return (1, str(group))


def magnitude_palette(
    groups: list[Any],
    cmap_name: str = "viridis",
    min_intensity: float = 0.20,
    max_intensity: float = 0.92,
) -> dict[Any, Any]:
    """Assign systematic colors where larger groups receive stronger intensity."""

    sorted_groups = sorted(groups, key=_group_sort_key)
    cmap = plt.get_cmap(cmap_name)
    if len(sorted_groups) == 1:
        values = [max_intensity]
    else:
        values = np.linspace(min_intensity, max_intensity, len(sorted_groups))
    return {group: cmap(value) for group, value in zip(sorted_groups, values)}


def save_figure(fig: plt.Figure, output_path: str | Path, formats: tuple[str, ...] = ("png", "pdf")) -> list[Path]:
    """Save a figure as one or more formats."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []

    if output_path.suffix:
        fig.savefig(output_path, bbox_inches="tight")
        saved_paths.append(output_path)
    else:
        for fmt in formats:
            path = output_path.with_suffix(f".{fmt}")
            fig.savefig(path, bbox_inches="tight")
            saved_paths.append(path)
    return saved_paths


def _extract_projection_points(layer_payload: dict[str, Any], component_index: int = 0) -> dict[str, np.ndarray]:
    answers: list[float] = []
    projections: list[float] = []
    groups: list[Any] = []

    for group in sorted(layer_payload["hidden_states"].keys(), key=_group_sort_key):
        values = np.asarray(layer_payload["hidden_states"][group], dtype=float)
        if values.ndim == 1:
            component = values
        else:
            component = values[:, component_index]
        group_answers = np.asarray(layer_payload["answers"][group], dtype=float)
        answers.extend(group_answers.tolist())
        projections.extend(component.tolist())
        groups.extend([group] * len(component))

    return {
        "answers": np.asarray(answers, dtype=float),
        "projections": np.asarray(projections, dtype=float),
        "groups": np.asarray(groups, dtype=object),
    }


def plot_pca_projections(
    transformed_results: dict[int, dict[str, Any]],
    output_path: str | Path,
    layers: list[int] | None = None,
    title: str | None = None,
    component_index: int = 0,
    cmap_name: str = "viridis",
) -> list[Path]:
    """Plot projected hidden states against log-scaled magnitudes.

    Groups are colored by a single ordered colormap, so the color intensity
    increases systematically with magnitude or symbol length.
    """

    set_paper_style()
    if layers is None:
        layers = sorted(transformed_results.keys())
    else:
        layers = [layer for layer in layers if layer in transformed_results]
    if not layers:
        raise ValueError("No requested layers are present in transformed_results")

    fig_height = max(2.4, 2.1 * len(layers))
    fig, axes = plt.subplots(
        len(layers),
        1,
        figsize=(4.8, fig_height),
        sharex=False,
        constrained_layout=True,
    )
    if len(layers) == 1:
        axes = [axes]

    all_groups = sorted(
        {
            group
            for layer in layers
            for group in transformed_results[layer]["hidden_states"].keys()
        },
        key=_group_sort_key,
    )
    palette = magnitude_palette(all_groups, cmap_name=cmap_name)

    for axis, layer in zip(axes, layers):
        payload = transformed_results[layer]
        points = _extract_projection_points(payload, component_index=component_index)
        x = np.log10(np.clip(points["answers"], 1e-12, None))
        y = points["projections"]
        groups = points["groups"]

        for group in all_groups:
            mask = groups == group
            if not np.any(mask):
                continue
            axis.scatter(
                x[mask],
                y[mask],
                s=18,
                color=palette[group],
                edgecolor="white",
                linewidth=0.25,
                alpha=0.82,
                label=f"G{group}",
            )

        group_mean_x: list[float] = []
        group_mean_y: list[float] = []
        for group in all_groups:
            mask = groups == group
            if np.any(mask):
                group_mean_x.append(float(np.mean(x[mask])))
                group_mean_y.append(float(np.mean(y[mask])))
        axis.plot(group_mean_x, group_mean_y, color="#222222", marker="o", markersize=3)

        rho = payload.get("monotonicity_metric")
        beta = payload.get("sublinearity_metric")
        subtitle = f"Layer {layer}"
        if rho is not None and beta is not None:
            subtitle = f"{subtitle} | rho={rho:.2f}, beta={beta:.2f}"
        axis.set_title(subtitle)
        axis.set_xlabel("log10(value)")
        axis.set_ylabel("Projected state")

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="center left",
            bbox_to_anchor=(1.01, 0.5),
            title="Group",
            frameon=False,
        )
    if title:
        fig.suptitle(title)

    saved_paths = save_figure(fig, output_path)
    plt.close(fig)
    return saved_paths


def plot_layer_metrics(
    metrics_by_layer: dict[int, dict[str, float]],
    output_path: str | Path,
    title: str | None = None,
) -> list[Path]:
    """Plot EV, monotonicity, and scaling rate over model depth."""

    set_paper_style()
    layers = sorted(metrics_by_layer.keys())
    metrics = {
        "Explained variance": [metrics_by_layer[layer]["EV"] for layer in layers],
        "Monotonicity |rho|": [metrics_by_layer[layer]["rho"] for layer in layers],
        "Scaling rate beta": [metrics_by_layer[layer]["beta"] for layer in layers],
    }
    colors = ["#277C8E", "#7A5195", "#D95F02"]

    fig, axes = plt.subplots(1, 3, figsize=(8.2, 2.4), constrained_layout=True)
    for axis, (label, values), color in zip(axes, metrics.items(), colors):
        axis.plot(layers, values, color=color, marker="o", markersize=2.8)
        axis.set_title(label)
        axis.set_xlabel("Layer")
    axes[0].set_ylabel("Score")
    if title:
        fig.suptitle(title)

    saved_paths = save_figure(fig, output_path)
    plt.close(fig)
    return saved_paths


def plot_context_sweep(
    rows: list[dict[str, Any]],
    output_path: str | Path,
    title: str | None = None,
) -> list[Path]:
    """Plot best-layer metrics as the number of context examples changes."""

    set_paper_style()
    by_model: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_model.setdefault(row["model"], []).append(row)

    fig, axes = plt.subplots(1, 3, figsize=(8.2, 2.4), constrained_layout=True)
    metric_specs = [
        ("EV", "Explained variance"),
        ("rho", "Monotonicity |rho|"),
        ("beta", "Scaling rate beta"),
    ]
    palette = magnitude_palette(list(by_model.keys()), cmap_name="tab10", min_intensity=0.05, max_intensity=0.95)

    for model, model_rows in by_model.items():
        model_rows = sorted(model_rows, key=lambda row: row["num_examples"])
        x = [row["num_examples"] for row in model_rows]
        for axis, (metric_key, label) in zip(axes, metric_specs):
            axis.plot(
                x,
                [row[metric_key] for row in model_rows],
                marker="o",
                markersize=3,
                label=model,
                color=palette[model],
            )
            axis.set_title(label)
            axis.set_xlabel("Context examples")
    axes[0].set_ylabel("Score")
    axes[-1].axhline(1.0, color="#555555", linewidth=0.8, linestyle="--")
    axes[-1].text(0.02, 0.93, "beta=1", transform=axes[-1].transAxes, fontsize=7)
    axes[0].legend(loc="best", frameon=False)
    if title:
        fig.suptitle(title)

    saved_paths = save_figure(fig, output_path)
    plt.close(fig)
    return saved_paths


def plot_numeric_comparison(
    rows: list[dict[str, Any]],
    output_path: str | Path,
    title: str | None = None,
) -> list[Path]:
    """Plot scale-bin accuracy for the 'which is larger?' task."""

    set_paper_style()
    demonstrations = sorted({row["num_demonstrations"] for row in rows})
    palette = magnitude_palette(demonstrations, cmap_name="Blues", min_intensity=0.45, max_intensity=0.88)

    fig, axis = plt.subplots(1, 1, figsize=(4.6, 2.8), constrained_layout=True)
    for num_dem in demonstrations:
        demo_rows = sorted(
            [row for row in rows if row["num_demonstrations"] == num_dem],
            key=lambda row: row["group"],
        )
        axis.plot(
            [row["group"] for row in demo_rows],
            [row["accuracy"] for row in demo_rows],
            marker="o",
            markersize=3.5,
            color=palette[num_dem],
            label=f"{num_dem} demos",
        )
    axis.set_xlabel("Scale bin i in 10^i +/- 20")
    axis.set_ylabel("Accuracy")
    axis.set_ylim(-0.02, 1.02)
    axis.legend(frameon=False)
    if title:
        axis.set_title(title)

    saved_paths = save_figure(fig, output_path)
    plt.close(fig)
    return saved_paths
