"""Figures-only runner for within-person variability analyses."""

from __future__ import annotations

import numpy as np
import pandas as pd

import within_person_variability_analysis as analysis
from refactored_io import copy_existing_figures, figures_dir, redirect_figure_writes, results_dir, write_numerical_results_markdown


def main() -> None:
    result_dir = results_dir("within_person_variability")
    fig_dir = figures_dir("within_person_variability")
    analysis.OUT_DIR = str(result_dir)
    caches = sorted(result_dir.glob("*_distribution_values.csv"))
    write_numerical_results_markdown(
        fig_dir / "within_person_variability_numerical_results.md",
        "Within-Person Variability Numerical Results",
        caches + sorted(result_dir.glob("*.csv")),
    )
    with redirect_figure_writes(fig_dir):
        for cache in caches:
            df = pd.read_csv(cache)
            if df.empty:
                continue
            title = str(df["title"].iloc[0])
            xlabel = str(df["xlabel"].iloc[0])
            kind = str(df["plot_kind"].iloc[0])
            out_path = str(result_dir / cache.name.replace("_distribution_values.csv", ".png"))
            values = {
                name: group["value"].to_numpy(dtype=float)
                for name, group in df.groupby("distribution")
            }
            if kind == "two_distributions":
                analysis.plot_distributions(
                    values.get("Within person", np.array([])),
                    values.get("Between persons", np.array([])),
                    title,
                    out_path,
                    xlabel=xlabel,
                )
            elif kind == "prediction_three_distributions":
                x_max = df["x_max_cap"].dropna()
                analysis.plot_three_distributions_pred(
                    values.get("Within person, within stage", np.array([])),
                    values.get("Within person, between stages", np.array([])),
                    values.get("Between persons", np.array([])),
                    title,
                    out_path,
                    xlabel=xlabel,
                    x_max_cap=float(x_max.iloc[0]) if not x_max.empty else None,
                )
            else:
                analysis.plot_three_distributions(
                    values.get("Within person, within stage", np.array([])),
                    values.get("Within person, between stages", np.array([])),
                    values.get("Between persons", np.array([])),
                    title,
                    out_path,
                    xlabel=xlabel,
                )
    copied = copy_existing_figures(result_dir, fig_dir)
    print(f"Regenerated {len(caches)} distribution figures and copied {len(copied)} cached figures to {fig_dir}")


if __name__ == "__main__":
    main()

