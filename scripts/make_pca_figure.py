"""Generate PCA projection figures for the paper."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from natural_log.analysis import Analyzer
from natural_log.config import ExperimentConfig, search_models
from natural_log.logging_utils import setup_logging
from natural_log.plotting import plot_pca_projections
from natural_log.transforms import select_best_layer


def parse_groups(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def parse_layers(value: str | None, best_layer: int) -> list[int]:
    if value is None or value == "best":
        return [best_layer]
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Model registry key or Hugging Face id")
    parser.add_argument("--data", default="numerics")
    parser.add_argument("--groups", default="1,2,3,4")
    parser.add_argument("--k", type=int, default=30)
    parser.add_argument("--num-examples", type=int, default=3)
    parser.add_argument("--device", default="0")
    parser.add_argument("--context", default="random")
    parser.add_argument("--layers", default="best", help="'best' or comma-separated layer ids")
    parser.add_argument("--save-dir", default="ICLR_results")
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    model_name = search_models(args.model) or args.model
    config = ExperimentConfig(
        model_name=model_name,
        data=args.data,
        groups=parse_groups(args.groups),
        k=args.k,
        num_examples=args.num_examples,
        context=args.context,
        runs=1,
        device=args.device,
        save_dir=Path(args.save_dir),
        log_dir=Path(args.log_dir),
        local_files_only=args.local_files_only,
    )
    logger = setup_logging("make_pca_figure", args.log_dir)
    analyzer = Analyzer(config, logger=logger)
    analyzer.states = analyzer.collect_states()
    transformed = analyzer.analyze(transform="PCA")

    # Avoid a second model pass: choose the best layer from this transformed run.
    metrics_by_layer = {
        layer: {
            "EV": payload["Explained_variance"],
            "rho": payload["monotonicity_metric"],
            "beta": payload["sublinearity_metric"],
        }
        for layer, payload in transformed.items()
    }
    best_layer, _ = select_best_layer(metrics_by_layer, score_key="EV")
    layers = parse_layers(args.layers, best_layer)

    stem = f"{config.slug()}_pca_projection"
    analyzer.save_to_file(transformed, Path(args.save_dir) / "figures" / f"{stem}.pkl")
    saved = plot_pca_projections(
        transformed,
        Path(args.save_dir) / "figures" / stem,
        layers=layers,
        title=f"{model_name} | {args.data}",
    )
    logger.info("Saved PCA projection figure: %s", ", ".join(str(path) for path in saved))


if __name__ == "__main__":
    main()
