"""Figures-only runner for temporal age-prediction analyses."""

from __future__ import annotations

import os

import pandas as pd

import evaluation_over_time_clean as analysis
from refactored_io import figures_dir, redirect_figure_writes, results_dir, write_numerical_results_markdown


def main() -> None:
    result_dir = results_dir("temporal_age_prediction")
    fig_dir = figures_dir("temporal_age_prediction")
    results = analysis.load_age_prediction_results(str(result_dir))
    if results is None:
        raise FileNotFoundError(f"No cached age prediction JSON found in {result_dir}")
    data, feat_cols = analysis.load_embeddings_and_labels()

    perf = analysis.load_and_aggregate_group_results(str(result_dir))
    numerical_sources = [analysis._results_path(str(result_dir))]
    numerical_sources.extend(sorted(result_dir.glob("temporal_performance_group*.csv")))
    write_numerical_results_markdown(
        fig_dir / "temporal_age_prediction_numerical_results.md",
        "Temporal Age Prediction Numerical Results",
        numerical_sources,
    )
    with redirect_figure_writes(fig_dir):
        if perf is not None and not perf.empty:
            analysis.plot_results_over_time(perf, str(result_dir), pooled_results=results)
        analysis.create_all_figures(results, data=data, feat_cols=feat_cols, output_dir=str(result_dir))


if __name__ == "__main__":
    main()

