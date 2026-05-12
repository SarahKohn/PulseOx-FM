"""Figures-only runner for day-after CGM, food, and wearable association analyses."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import predict_cgm_change_from_embeddings as analysis
from refactored_io import figures_dir, redirect_figure_writes, results_dir, write_numerical_results_markdown


RESULT_FILES = (
    "cgm_log_ratio_prediction_results.csv",
    "food_log_ratio_prediction_results.csv",
    "activity_wearables_prediction_results.csv",
)


_DEMO_COLOR = "#a8a8a8"
_DEMO_EDGE = "#707070"
_COMB_DOT_SIZE = 4.5
_CONNECT_COLOR = "#d4d4d4"
_TYPE_COLORS = {
    "CGM": "#0b3d44",
    "Food": "#196874",
    "Wearables": "#84c6cf",
}


def _make_demo_vs_embeddings_forest_figure(result_dir, fig_dir) -> None:
    """Forest plot: demographics-only vs demographics+embeddings Pearson r for next-day targets.

    Styled to match target_prediction forest plots (FDR-significant targets only, sorted by
    R_combined descending, significance stars at right).
    """
    radar_path = result_dir / "radar_demo_vs_embeddings_cache.csv"
    if not radar_path.is_file():
        print(f"Skipping demo-vs-embeddings forest: {radar_path} not found")
        return

    df = pd.read_csv(radar_path)
    needed = {"target", "R_demo", "R_combined", "p_combined"}
    if not needed.issubset(df.columns):
        print(f"Skipping demo-vs-embeddings forest: missing columns {needed - set(df.columns)}")
        return

    df["R_demo"] = pd.to_numeric(df["R_demo"], errors="coerce")
    df["R_combined"] = pd.to_numeric(df["R_combined"], errors="coerce")
    df["p_combined"] = pd.to_numeric(df["p_combined"], errors="coerce")
    df = df.dropna(subset=["R_demo", "R_combined", "p_combined"])
    if df.empty:
        print("Skipping demo-vs-embeddings forest: no valid rows")
        return

    # BH-FDR (reuse the same function used elsewhere in this pipeline)
    p_arr = df["p_combined"].to_numpy(dtype=float)
    q_arr = analysis._fdr_correct(p_arr)
    df = df[q_arr < 0.05].copy()
    q_kept = q_arr[q_arr < 0.05]
    if df.empty:
        print("No FDR-significant targets for demo-vs-embeddings forest")
        return

    df["_q"] = q_kept
    df = df.sort_values("R_combined", ascending=True)  # ascending → y=0 at bottom, highest at top

    n = len(df)
    y = np.arange(n, dtype=float)
    r_demo = df["R_demo"].to_numpy(dtype=float)
    r_comb = df["R_combined"].to_numpy(dtype=float)
    y_comb = y + 0.18
    y_demo = y - 0.18

    fig, ax = plt.subplots(figsize=(7.0, max(2.5, 0.18 * n + 0.80)))

    # PulseOx-FM dots (coloured by target type) — combined model on top
    for i, (_, row) in enumerate(df.iterrows()):
        t_color = _TYPE_COLORS.get(str(row.get("target_type", "")), "#196874")
        ax.plot(r_comb[i], y_comb[i], "o", color=t_color, markeredgecolor="none",
                markersize=_COMB_DOT_SIZE, zorder=4)
    # Demographics dots (silver) — below
    ax.plot(r_demo, y_demo, "o", color=_DEMO_COLOR, markeredgecolor="none",
            markersize=_COMB_DOT_SIZE, zorder=3)

    ax.axvline(0, color="#777777", linestyle=":", linewidth=0.9, zorder=1)
    ax.grid(True, axis="x", color="#d7d7d7", alpha=0.75, linewidth=0.5, zorder=0)

    x_lo = min(float(np.nanmin(r_demo)), float(np.nanmin(r_comb))) - 0.02
    star_x = float(np.nanmax(r_comb)) + 0.015
    x_hi = star_x + 0.06
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(-0.45, n - 1 + 0.45)

    for i, q in enumerate(df["_q"].to_numpy()):
        stars = "***" if q < 0.001 else ("**" if q < 0.01 else ("*" if q < 0.05 else ""))
        if stars:
            ax.text(star_x, y[i], stars, va="center", ha="left", fontsize=9, color="#222222", clip_on=False)

    target_labels = [
        str(t).replace("_", " ").replace("step count", "Step count")
        .replace("active energy burned", "Active energy")
        .replace("basal energy burned", "Basal energy")
        for t in df["target"]
    ]
    ax.set_yticks(y)
    ax.set_yticklabels(target_labels, fontsize=10)
    ax.tick_params(axis="x", labelsize=10)
    ax.set_xlabel("Pearson $r$", fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Legend
    legend_handles = [
        plt.Line2D([0], [0], marker="o", color="none", markersize=_COMB_DOT_SIZE,
                   markerfacecolor="#196874", markeredgecolor="none",
                   label="Age, sex, BMI + PulseOx-FM embeddings"),
        plt.Line2D([0], [0], marker="o", color="none", markersize=_COMB_DOT_SIZE,
                   markerfacecolor=_DEMO_COLOR, markeredgecolor="none",
                   label="Age, sex, BMI"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=9, frameon=False)

    plt.tight_layout()

    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(fig_dir / "next_day_demo_vs_embeddings_forest.png"),
                dpi=150, facecolor="white", bbox_inches="tight")
    fig.savefig(str(fig_dir / "next_day_demo_vs_embeddings_forest.pdf"),
                facecolor="white", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    result_dir = results_dir("day_after_associations")
    fig_dir = figures_dir("day_after_associations")
    analysis.OUT_DIR = str(result_dir)
    all_results = []
    for name in RESULT_FILES:
        path = result_dir / name
        if not path.is_file():
            print(f"Missing cached result: {path}")
            continue
        rows = pd.read_csv(path).to_dict("records")
        if name.startswith("cgm"):
            target_type = "CGM"
        elif name.startswith("food"):
            target_type = "Food"
        else:
            target_type = "Wearables"
        for row in rows:
            row["target_type"] = row.get("target_type", target_type)
        all_results.extend(rows)
    if not all_results:
        raise FileNotFoundError(f"No cached association result CSVs found in {result_dir}")
    write_numerical_results_markdown(
        fig_dir / "next_day_associations_numerical_results.md",
        "Next-Day Association Numerical Results",
        [
            result_dir / "cgm_log_ratio_prediction_results.csv",
            result_dir / "cgm_log_ratio_prediction_metric_repeats.csv",
            result_dir / "food_log_ratio_prediction_results.csv",
            result_dir / "food_log_ratio_prediction_metric_repeats.csv",
            result_dir / "activity_wearables_prediction_results.csv",
            result_dir / "activity_wearables_prediction_metric_repeats.csv",
            result_dir / "radar_demo_vs_embeddings_cache.csv",
        ],
    )

    with redirect_figure_writes(fig_dir):
        # Combined bar plots: BH-FDR within each panel, Tol muted palette, etc.—see _plot_combined_volcanos.
        analysis._plot_combined_volcanos(all_results, str(result_dir))
        radar_path = result_dir / "radar_demo_vs_embeddings_cache.csv"
        if radar_path.is_file():
            radar = pd.read_csv(radar_path).to_dict("records")
            p_vals = [r["p_combined"] for r in radar if pd.notna(r.get("p_combined"))]
            if p_vals:
                q_vals = analysis._fdr_correct(p_vals)
                sig = [r for r, q in zip(radar, q_vals) if q < analysis.RADAR_FDR_ALPHA]
                analysis._plot_forest_demo_vs_embeddings(sig, str(result_dir))

    _make_demo_vs_embeddings_forest_figure(result_dir, fig_dir)


if __name__ == "__main__":
    main()

