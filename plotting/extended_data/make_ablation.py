"""Figures-only runner for gold-test age ablation analyses."""

from __future__ import annotations

import json

import pandas as pd

import ablation_studies_gold_test_age as analysis
from refactored_io import figures_dir, redirect_figure_writes, results_dir, write_numerical_results_markdown


def main() -> None:
    result_dir = results_dir("ablation_gold_test_age")
    fig_dir = figures_dir("ablation_gold_test_age")
    metrics_path = result_dir / "ablation_age_gold_test_epoch_metrics.csv"
    if not metrics_path.is_file():
        raise FileNotFoundError(f"No cached ablation metrics found: {metrics_path}")

    metrics = pd.read_csv(metrics_path).to_dict("records")
    pyppg_path = result_dir / "results_cache" / "pyppg_age_metrics.json"
    pyppg_metrics = json.loads(pyppg_path.read_text()) if pyppg_path.is_file() else None
    write_numerical_results_markdown(
        fig_dir / "ablation_gold_test_age_numerical_results.md",
        "Gold-Test Age Ablation Numerical Results",
        [
            metrics_path,
            result_dir / "ablation_age_gold_test_metric_repeats.csv",
            result_dir / "ablation_age_gold_test_pyppg_metrics.csv",
            pyppg_path,
            *sorted((result_dir / "results_cache").glob("epoch_*_age_metrics.json")),
            *sorted((result_dir / "results_cache").glob("epoch_*_age_metric_repeats.csv")),
        ],
    )
    epochs = [int(row["epoch"]) for row in metrics]
    analysis.OUTPUT_DIR = str(result_dir)
    with redirect_figure_writes(fig_dir):
        for metric in analysis.METRICS:
            analysis._plot_metric(metric, epochs, metrics, pyppg_metrics)


if __name__ == "__main__":
    main()

