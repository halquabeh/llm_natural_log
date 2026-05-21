"""Run and plot the 'which is larger?' motivation experiment."""

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
from natural_log.plotting import plot_numeric_comparison


def parse_groups(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def parse_ints(value: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Model registry key or Hugging Face id")
    parser.add_argument("--groups", default="1,2,3,4,5")
    parser.add_argument("--demos", default="0,3")
    parser.add_argument("--pairs", type=int, default=100)
    parser.add_argument("--gaps", default="1,5,10")
    parser.add_argument("--device", default="0")
    parser.add_argument("--save-dir", default="f_results")
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    model_name = search_models(args.model) or args.model
    logger = setup_logging("run_numeric_comparison", args.log_dir)
    config = ExperimentConfig(
        model_name=model_name,
        data="numerics",
        groups=parse_groups(args.groups),
        device=args.device,
        save_dir=Path(args.save_dir),
        log_dir=Path(args.log_dir),
        local_files_only=args.local_files_only,
    )
    analyzer = Analyzer(config, logger=logger)
    rows = analyzer.run_numeric_comparison(
        num_demonstrations=parse_ints(args.demos),
        n_pairs=args.pairs,
        gaps=parse_ints(args.gaps),
    )
    output_stem = Path(args.save_dir) / "figures" / f"{config.model_name.replace('/', '_')}_numeric_comparison"
    analyzer.save_json(rows, output_stem.with_suffix(".json"))
    saved = plot_numeric_comparison(rows, output_stem, title=model_name.split("/")[-1])
    logger.info("Saved numeric comparison figure: %s", ", ".join(str(path) for path in saved))


if __name__ == "__main__":
    main()
