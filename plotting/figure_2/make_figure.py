#!/usr/bin/env python3
"""Create signal reconstruction figures from cached subset-evaluation metrics."""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache-sleep-fm")
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib

matplotlib.use("Agg", force=True)
from matplotlib import colors as mcolors
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

DOWNSTREAM_DIR = Path(__file__).resolve().parent
if str(DOWNSTREAM_DIR) not in sys.path:
    sys.path.insert(0, str(DOWNSTREAM_DIR))

from refactored_io import figures_dir, results_dir, write_numerical_results_markdown


ANALYSIS_NAME = "signal_reconstructions"
OUR_COLOR = "#196874"
BASELINE_COLOR = "#B0B0B0"
BASELINE_DARK_COLOR = "#7F7F7F"
OUR_MARKER = "o"
BASELINE_MARKER = "s"
CI_Z = 1.96
CONTEXT_UNMASKED = "#C8C8C8"
CONTEXT_SHADE_ALPHA = 0.55
TRUTH_COLOR = "#000000"
RECON_LINEWIDTH = 2.4
GRID_COLOR = "#CCCCCC"


def _apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "Liberation Sans"],
            "font.size": 9,
            "axes.titlesize": 9,
            "axes.labelsize": 9,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "lines.linewidth": 1.1,
            "legend.frameon": False,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _save_figure(fig, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight", format="pdf")
    plt.close(fig)


def _metric_dict(block: dict[str, Any], metric: str, suffix: str) -> dict[float, float]:
    values = block.get(f"{metric}_{suffix}", {})
    return {float(k): float(v) for k, v in values.items()}


def _ci_values(block: dict[str, Any], metric: str, x_values: list[float]) -> list[float]:
    std = _metric_dict(block, metric, "std")
    n = _metric_dict(block, metric, "n")
    ci = []
    for x in x_values:
        count = n.get(x, 0.0)
        spread = std.get(x, float("nan"))
        if count > 0 and math.isfinite(spread):
            ci.append(CI_Z * spread / math.sqrt(count))
        else:
            ci.append(float("nan"))
    return ci


def _series(block: dict[str, Any], metric: str, x_values: list[float]) -> tuple[list[float], list[float]]:
    mean = _metric_dict(block, metric, "mean")
    return [mean.get(x, float("nan")) for x in x_values], _ci_values(block, metric, x_values)


def _plot_metric_lines(
    task: dict[str, Any],
    x_values: list[float],
    output_path: Path,
    metric: str,
    xlabel: str,
    ylabel: str,
    log_y: bool = False,
    x_tick_labels: list[str] | None = None,
    show_legend: bool = True,
) -> None:
    fig, ax = plt.subplots(figsize=(3.45, 2.35))
    for label, color, marker, block in (
        ("PulseOx-FM", OUR_COLOR, OUR_MARKER, task["our_model"]),
        ("Baseline model", BASELINE_COLOR, BASELINE_MARKER, task["baseline"]),
    ):
        y, ci = _series(block, metric, x_values)
        y_arr = [float(v) for v in y]
        ci_arr = [float(v) for v in ci]
        lower = [v - e if math.isfinite(v) and math.isfinite(e) else float("nan") for v, e in zip(y_arr, ci_arr)]
        upper = [v + e if math.isfinite(v) and math.isfinite(e) else float("nan") for v, e in zip(y_arr, ci_arr)]
        ax.plot(
            x_values,
            y_arr,
            linestyle="-",
            marker=marker,
            color=color,
            markerfacecolor=color,
            markeredgecolor=BASELINE_DARK_COLOR if color == BASELINE_COLOR else "white",
            markeredgewidth=0.5,
            markersize=3.0,
            label=label,
            zorder=3,
        )
        ax.fill_between(x_values, lower, upper, color=color, alpha=0.28, linewidth=0, zorder=2)
        ax.plot(x_values, lower, color=color, alpha=0.35, linewidth=0.35, zorder=2)
        ax.plot(x_values, upper, color=color, alpha=0.35, linewidth=0.35, zorder=2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if x_tick_labels is not None:
        ax.set_xticks(x_values)
        ax.set_xticklabels(x_tick_labels)
    if log_y:
        ax.set_yscale("log")
        ax.grid(axis="y", which="minor", color=GRID_COLOR, linestyle=":", linewidth=0.3, alpha=0.65)
    ax.grid(axis="y", color=GRID_COLOR, linestyle="--", linewidth=0.4, alpha=0.9)
    handles = [
        Line2D([0], [0], color=OUR_COLOR, marker=OUR_MARKER, linestyle="-", markersize=3.0, label="PulseOx-FM"),
        Line2D(
            [0],
            [0],
            color=BASELINE_COLOR,
            marker=BASELINE_MARKER,
            linestyle="-",
            markersize=3.0,
            label="Baseline model",
        ),
    ]
    if show_legend:
        ax.legend(handles=handles, loc="best", handlelength=1.6)
    fig.tight_layout()
    _save_figure(fig, output_path)


def _shade_unmasked_only(ax, time_axis: np.ndarray, mask: np.ndarray) -> None:
    unmasked = mask <= 0.5
    if not unmasked.any():
        return
    in_run = False
    start_t = 0.0
    for ti, is_unmasked in zip(time_axis, unmasked):
        if is_unmasked and not in_run:
            start_t = float(ti)
            in_run = True
        elif not is_unmasked and in_run:
            ax.axvspan(start_t, float(ti), color=CONTEXT_UNMASKED, alpha=CONTEXT_SHADE_ALPHA, zorder=0)
            in_run = False
    if in_run:
        ax.axvspan(start_t, float(time_axis[-1]), color=CONTEXT_UNMASKED, alpha=CONTEXT_SHADE_ALPHA, zorder=0)


def _plot_ground_truth_by_mask(ax, time_axis: np.ndarray, true_sig: np.ndarray, mask: np.ndarray) -> None:
    unmasked = mask <= 0.5
    true_context = true_sig.astype(float).copy()
    true_context[~unmasked] = np.nan
    true_masked = true_sig.astype(float).copy()
    true_masked[unmasked] = np.nan
    ax.plot(time_axis, true_context, color=TRUTH_COLOR, linestyle="-", linewidth=0.9, zorder=3)
    ax.plot(time_axis, true_masked, color=TRUTH_COLOR, linestyle=":", linewidth=1.0, zorder=3)


def _unified_example_handles() -> list:
    handles = [
        Line2D([0], [0], color=TRUTH_COLOR, linestyle="-", linewidth=0.9, label="Ground truth (context)"),
        Line2D([0], [0], color=TRUTH_COLOR, linestyle=":", linewidth=1.0, label="Ground truth (masked)"),
        Line2D([0], [0], color=OUR_COLOR, linewidth=RECON_LINEWIDTH, label="PulseOx-FM"),
        Line2D([0], [0], color=BASELINE_COLOR, linewidth=RECON_LINEWIDTH, label="Baseline model"),
        Patch(facecolor=CONTEXT_UNMASKED, edgecolor="none", alpha=CONTEXT_SHADE_ALPHA, label="Context (unmasked)"),
    ]
    return handles


def _add_unified_example_legend(ax) -> None:
    ax.legend(
        handles=_unified_example_handles(),
        loc="upper right",
        bbox_to_anchor=(0.995, 0.995),
        borderaxespad=0.0,
        fontsize=7.5,
        frameon=True,
        framealpha=0.90,
        facecolor="white",
        edgecolor="#BBBBBB",
        handlelength=1.7,
        labelspacing=0.22,
    )


def _add_forecast_horizon_markers(ax, time_axis: np.ndarray, mask: np.ndarray) -> None:
    masked = mask > 0.5
    if not masked.any():
        return
    t0 = float(time_axis[int(np.argmax(masked))])
    y_lo, y_hi = ax.get_ylim()
    y_txt = y_hi - 0.07 * (y_hi - y_lo)
    for horizon in (0, 5, 10):
        tx = t0 + float(horizon)
        if float(time_axis[0]) <= tx <= float(time_axis[-1]):
            ax.axvline(tx, color="#555555", linestyle=":", linewidth=0.8, alpha=0.9, zorder=2)
            ax.text(
                tx,
                y_txt,
                f"{horizon} s",
                ha="center",
                va="top",
                fontsize=7.5,
                color="#222222",
                bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="#BBBBBB", linewidth=0.3, alpha=0.95),
                zorder=5,
            )


def _zoom_slice_forecast(mask: np.ndarray, sampling_rate: int, ctx_sec: float = 15.0, fc_sec: float = 15.0) -> tuple[int, int]:
    masked = mask > 0.5
    if not masked.any():
        n = len(mask)
        width = min(n, int(round((ctx_sec + fc_sec) * sampling_rate)))
        return 0, width
    start = int(np.argmax(masked))
    s0 = max(0, start - int(round(ctx_sec * sampling_rate)))
    s1 = min(len(mask), start + int(round(fc_sec * sampling_rate)))
    return s0, s1


def _zoom_slice_random_mask(mask: np.ndarray, sampling_rate: int, total_sec: float = 30.0) -> tuple[int, int]:
    masked = mask > 0.5
    n = len(mask)
    width = min(n, int(round(total_sec * sampling_rate)))
    if not masked.any():
        return 0, width
    centers = np.flatnonzero(masked)
    center = int(centers[len(centers) // 2])
    start = min(max(0, center - width // 2), max(0, n - width))
    return start, min(n, start + width)


def _full_length_reconstruction(reconstruction: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = np.full(mask.shape, np.nan, dtype=float)
    masked = mask > 0.5
    values = np.asarray(reconstruction, dtype=float).reshape(-1)
    n = min(int(masked.sum()), values.size)
    out[np.flatnonzero(masked)[:n]] = values[:n]
    return out


def _plot_forecast_examples(npz_path: Path, fig_dir: Path, n_examples: int = 3) -> None:
    if not npz_path.is_file():
        print(f"Skipping forecast examples; missing {npz_path}")
        return
    data = dict(np.load(npz_path))
    original_full = data["original_full"]
    our = data["our_model_reconstruction"]
    baseline = data["baseline_reconstruction"]
    full_time_sec = data["full_time_sec"]
    masks = data["forecast_mask"]
    sampling_rate = int(np.asarray(data.get("sampling_rate", 125)).item())
    n = min(n_examples, original_full.shape[0])
    for idx in range(n):
        mask = np.repeat(masks[idx].astype(float), sampling_rate)
        true_sig = original_full[idx, 0].astype(float)
        our_full = _full_length_reconstruction(our[idx, 0], mask)
        baseline_full = _full_length_reconstruction(baseline[idx, 0], mask)
        s0, s1 = _zoom_slice_forecast(mask, sampling_rate)
        x = full_time_sec[s0:s1]
        mask_zoom = mask[s0:s1]
        true_zoom = true_sig[s0:s1]
        our_zoom = our_full[s0:s1]
        baseline_zoom = baseline_full[s0:s1]

        fig, axes = plt.subplots(3, 1, figsize=(5.0, 4.25), sharex=True, sharey=True)
        for ax, (panel, row_title) in zip(
            axes,
            [("none", "No reconstruction"), ("mae", "PulseOx-FM"), ("baseline", "Baseline model")],
        ):
            _shade_unmasked_only(ax, x, mask_zoom)
            _plot_ground_truth_by_mask(ax, x, true_zoom, mask_zoom)
            if panel == "mae":
                ax.plot(x, our_zoom, color=OUR_COLOR, linewidth=RECON_LINEWIDTH, zorder=4)
            elif panel == "baseline":
                ax.plot(x, baseline_zoom, color=BASELINE_COLOR, linewidth=RECON_LINEWIDTH, zorder=4)
            ax.set_title(row_title, fontsize=7)
            ax.set_ylabel("")
            ax.set_xlim(float(x[0]), float(x[-1]))
            ax.grid(color=GRID_COLOR, linestyle="--", linewidth=0.35, alpha=0.85)
        for ax in axes:
            _add_forecast_horizon_markers(ax, x, mask_zoom)
        axes[-1].set_xlabel("Time (s)")
        fig.tight_layout(h_pad=0.35)
        _save_figure(fig, fig_dir / f"fig_example_forecast_{idx + 1}.png")


def _plot_random_masking_examples(npz_path: Path, fig_dir: Path, n_examples: int = 3) -> None:
    if not npz_path.is_file():
        print(f"Skipping random masking examples; missing {npz_path}")
        return
    data = dict(np.load(npz_path))
    original_full = data["original_full"]
    our = data["our_model_reconstruction"]
    baseline = data["baseline_reconstruction"]
    full_time_sec = data["full_time_sec"]
    masks = data["random_mask"]
    sampling_rate = int(np.asarray(data.get("sampling_rate", 125)).item())
    n = min(n_examples, original_full.shape[0])
    for idx in range(n):
        mask = np.repeat(masks[idx].astype(float), sampling_rate)
        true_sig = original_full[idx, 0].astype(float)
        our_sig = our[idx, 0].astype(float)
        baseline_sig = baseline[idx, 0].astype(float)
        our_plot = our_sig.copy()
        baseline_plot = baseline_sig.copy()
        our_plot[mask <= 0.5] = np.nan
        baseline_plot[mask <= 0.5] = np.nan
        s0, s1 = _zoom_slice_random_mask(mask, sampling_rate)
        x = full_time_sec[s0:s1]
        mask_zoom = mask[s0:s1]
        true_zoom = true_sig[s0:s1]
        our_zoom = our_plot[s0:s1]
        baseline_zoom = baseline_plot[s0:s1]

        fig, axes = plt.subplots(3, 1, figsize=(5.0, 4.25), sharex=True, sharey=True)
        for ax, (panel, row_title) in zip(
            axes,
            [("none", "No reconstruction"), ("mae", "PulseOx-FM"), ("baseline", "Baseline model")],
        ):
            _shade_unmasked_only(ax, x, mask_zoom)
            _plot_ground_truth_by_mask(ax, x, true_zoom, mask_zoom)
            if panel == "mae":
                ax.plot(x, our_zoom, color=OUR_COLOR, linewidth=RECON_LINEWIDTH, zorder=4)
            elif panel == "baseline":
                ax.plot(x, baseline_zoom, color=BASELINE_COLOR, linewidth=RECON_LINEWIDTH, zorder=4)
            ax.set_title(row_title, fontsize=7)
            ax.set_ylabel("")
            ax.set_xlim(float(x[0]), float(x[-1]))
            ax.grid(color=GRID_COLOR, linestyle="--", linewidth=0.35, alpha=0.85)
        _add_unified_example_legend(axes[0])
        axes[-1].set_xlabel("Time (s)")
        fig.tight_layout(h_pad=0.35)
        _save_figure(fig, fig_dir / f"fig_example_random_masking_50pct_{idx + 1}.png")


def main() -> None:
    _apply_style()
    result_dir = results_dir(ANALYSIS_NAME)
    fig_dir = figures_dir(ANALYSIS_NAME)
    metrics_path = result_dir / "metrics.json"
    if not metrics_path.is_file():
        raise FileNotFoundError(f"No cached signal reconstruction metrics found: {metrics_path}")

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    random_masking = metrics["random_masking"]
    forecasting = metrics["forecasting"]

    mask_ratios = [float(x) for x in random_masking["mask_ratios"]]
    mask_ratio_labels = [f"{int(x * 100)}%" for x in mask_ratios]
    _plot_metric_lines(
        random_masking,
        mask_ratios,
        fig_dir / "fig_random_masking_pearson.png",
        "pearson",
        "Masking ratio",
        "Pearson $r$ (masked patches)",
        x_tick_labels=mask_ratio_labels,
    )
    _plot_metric_lines(
        random_masking,
        mask_ratios,
        fig_dir / "fig_random_masking_mse.png",
        "mse",
        "Masking ratio",
        "Reconstruction MSE",
        x_tick_labels=mask_ratio_labels,
    )

    horizons = [float(x) for x in forecasting["time_horizons_sec"]]
    horizon_labels = [f"{int(x):g} s" for x in horizons]
    _plot_metric_lines(
        forecasting,
        horizons,
        fig_dir / "fig_forecast_pearson.png",
        "pearson",
        "Forecast horizon start",
        "Pearson $r$ (1 s window)",
        x_tick_labels=horizon_labels,
    )
    _plot_metric_lines(
        forecasting,
        horizons,
        fig_dir / "fig_forecast_mse.png",
        "mse",
        "Forecast horizon start",
        "Forecast MSE (1 s window)",
        x_tick_labels=horizon_labels,
        show_legend=False,
    )
    _plot_forecast_examples(result_dir / forecasting.get("signals_npz", "forecast_signals_gold_test_subset.npz"), fig_dir)
    _plot_random_masking_examples(
        result_dir / random_masking.get("signals_npz", "random_masking_signals_gold_test_subset.npz"),
        fig_dir,
    )
    write_numerical_results_markdown(
        fig_dir / "signal_reconstructions_numerical_results.md",
        "Signal Reconstructions Numerical Results",
        [metrics_path, result_dir / forecasting.get("signals_npz", "forecast_signals_gold_test_subset.npz")],
    )
    print(f"Figures written to {fig_dir}")


if __name__ == "__main__":
    main()
