"""Backward-compatible wrappers around ``natural_log.plotting``."""

import matplotlib.pyplot as plt

from natural_log.plotting import (
    magnitude_palette,
    plot_context_sweep,
    plot_layer_metrics,
    plot_numeric_comparison,
    plot_pca_projections,
    save_figure,
    set_paper_style,
)


def plot_predictions_across_layers(prompts, hidden_states):
    """Quick diagnostic image plot retained for old notebooks."""

    fig, axes = plt.subplots(len(hidden_states), 1, figsize=(10, 15))
    if len(hidden_states) == 1:
        axes = [axes]
    for layer, state in enumerate(hidden_states):
        axes[layer].imshow(state)
        axes[layer].set_title(f"Layer {layer} hidden states")
    plt.show()


__all__ = [
    "magnitude_palette",
    "plot_context_sweep",
    "plot_layer_metrics",
    "plot_numeric_comparison",
    "plot_pca_projections",
    "plot_predictions_across_layers",
    "save_figure",
    "set_paper_style",
]

