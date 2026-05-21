"""Run the context-example sweep used for the prompt-format figure."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from natural_log.analysis import Analyzer
from natural_log.config import ExperimentConfig, search_models
from natural_log.logging_utils import setup_logging
from natural_log.plotting import plot_context_sweep
from natural_log.transforms import select_best_layer


def parse_groups(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def parse_ints(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", required=True, help="Comma-separated registry keys or HF ids")
    parser.add_argument("--num-examples", default="0,1,2,3,4,5")
    parser.add_argument("--groups", default="1,2,3,4")
    parser.add_argument("--k", type=int, default=30)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--device", default="0")
    parser.add_argument("--context", default="random")
    parser.add_argument("--save-dir", default="f_results")
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    logger = setup_logging("run_context_sweep", args.log_dir)
    rows: list[dict[str, float | int | str]] = []
    groups = parse_groups(args.groups)

    for model_key in [part.strip() for part in args.models.split(",") if part.strip()]:
        model_name = search_models(model_key) or model_key
        for num_examples in parse_ints(args.num_examples):
            config = ExperimentConfig(
                model_name=model_name,
                data="numerics",
                groups=groups,
                k=args.k,
                num_examples=num_examples,
                context=args.context,
                runs=args.runs,
                device=args.device,
                save_dir=Path(args.save_dir),
                log_dir=Path(args.log_dir),
                local_files_only=args.local_files_only,
            )
            analyzer = Analyzer(config, logger=logger)
            summary = analyzer.run_multiple(methods=("PCA",))["PCA"]
            layer, metrics = select_best_layer(summary, score_key="EV")
            rows.append(
                {
                    "model": model_name.split("/")[-1],
                    "num_examples": num_examples,
                    "layer": layer,
                    "EV": metrics["EV"],
                    "rho": metrics["rho"],
                    "beta": metrics["beta"],
                }
            )

    output_json = Path(args.save_dir) / "figures" / "context_sweep.json"
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)
    logger.info("Saved context sweep data to %s", output_json)
    saved = plot_context_sweep(rows, Path(args.save_dir) / "figures" / "context_sweep")
    logger.info("Saved context sweep: %s", ", ".join(str(path) for path in saved))


if __name__ == "__main__":
    main()
