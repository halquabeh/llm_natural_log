"""Generate the main PCA table and appendix PLS table."""

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
from natural_log.transforms import select_best_layer


def parse_groups(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def format_metric(mean: float, std: float) -> str:
    return f"{mean:.2f} +/- {std:.3f}"


def table_row(model: str, data: str, layer: int, metrics: dict[str, float]) -> dict[str, str]:
    return {
        "model": model,
        "group": data,
        "layer": str(layer),
        "rho": format_metric(metrics["rho"], metrics["rho_std"]),
        "beta": format_metric(metrics["beta"], metrics["beta_std"]),
        "score": format_metric(metrics["EV"], metrics["EV_std"]),
    }


def write_tsv(path: Path, rows: list[dict[str, str]], score_label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = ["model", "group", "layer", "rho", "beta", score_label]
    with path.open("w", encoding="utf-8") as handle:
        handle.write("\t".join(header) + "\n")
        for row in rows:
            handle.write(
                "\t".join(
                    [
                        row["model"],
                        row["group"],
                        row["layer"],
                        row["rho"],
                        row["beta"],
                        row["score"],
                    ]
                )
                + "\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Model registry key or Hugging Face id")
    parser.add_argument("--data", default="numerics,letters", help="Comma-separated data groups")
    parser.add_argument("--groups", default="1,2,3,4", help="Comma-separated magnitude groups")
    parser.add_argument("--k", type=int, default=30)
    parser.add_argument("--num-examples", type=int, default=3)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--device", default="1")
    parser.add_argument("--context", default="random")
    parser.add_argument("--save-dir", default="f_results")
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    model_name = search_models(args.model) or args.model
    logger = setup_logging("make_table1", args.log_dir)
    logger.info("Resolved model: %s", model_name)

    pca_rows: list[dict[str, str]] = []
    pls_rows: list[dict[str, str]] = []
    for data_name in [part.strip() for part in args.data.split(",") if part.strip()]:
        config = ExperimentConfig(
            model_name=model_name,
            data=data_name,
            groups=parse_groups(args.groups),
            k=args.k,
            num_examples=args.num_examples,
            context=args.context,
            runs=args.runs,
            device=args.device,
            save_dir=Path(args.save_dir),
            log_dir=Path(args.log_dir),
            local_files_only=args.local_files_only,
        )
        analyzer = Analyzer(config, logger=logger)
        summary = analyzer.run_multiple(methods=("PCA", "PLS"))
        analyzer.save_to_file(summary, Path(args.save_dir) / "tables" / f"{config.slug()}_table_summary.pkl")

        pca_layer, pca_metrics = select_best_layer(summary["PCA"], score_key="EV")
        pls_layer, pls_metrics = select_best_layer(summary["PLS"], score_key="EV")
        pca_rows.append(table_row(model_name, data_name, pca_layer, dict(pca_metrics)))
        pls_rows.append(table_row(model_name, data_name, pls_layer, dict(pls_metrics)))

    write_tsv(Path(args.save_dir) / "tables" / "main_pca_table.tsv", pca_rows, "sigma2")
    write_tsv(Path(args.save_dir) / "appendix" / "pls_decodability_table.tsv", pls_rows, "R2")
    logger.info("Wrote PCA main table and PLS appendix table")


if __name__ == "__main__":
    main()
