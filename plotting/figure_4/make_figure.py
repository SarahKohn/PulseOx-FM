"""Figures-only runner for target prediction analyses."""

from __future__ import annotations

import glob
import io
import math
import os
import shutil
from pathlib import Path

# Must run before NumPy / SciPy import BLAS. Scheduler-provided thread counts (e.g. SLURM_CPUS_PER_TASK)
# often inflate OMP/OpenBLAS/MKL threads and can segfault on import or fit.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_MAX_THREADS"] = "1"
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import multiprocessing as _mp

try:
    _mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass

# Import before NumPy/SciPy initialize BLAS — avoids segfault on some clusters when joblib/sklearn
# loads ``multiprocessing.resource_tracker`` later (faulthandler: create_module in resource_tracker.py).
import multiprocessing.resource_tracker as _mp_resource_tracker  # noqa: F401

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Patch
from scipy.stats import rankdata, ttest_ind

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ.setdefault("JOBLIB_TEMP_FOLDER", "/tmp")

# ``analysis`` imports ``utils``, which loads torch at import time; avoid CUDA init segfaults on CPU/broken-driver nodes.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import target_prediction_evaluation_short as analysis
from refactored_io import figures_dir, redirect_figure_writes, results_dir, write_numerical_results_markdown

# Old output directory produced by the ensemble TabICL run (bone density figures live here)
_OLD_TABICL_OUTPUT = Path(
    "/net/mraid20/export/jafar/Sarah/ssl_sleep/target_predictions_ensemble_TabICL_output"
)
_PULSEOX_TEAL = "#196874"
_DEMO_GRAY = "#b0b0b0"


def _roc_curve_binary(
    y_true: np.ndarray, y_score: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Binary ROC (labels 0/1); matches ``sklearn.metrics.roc_curve`` layout close enough for plotting."""
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_score = np.asarray(y_score, dtype=np.float64).ravel()
    if y_true.shape != y_score.shape:
        raise ValueError("y_true and y_score must have the same shape")
    desc = np.argsort(y_score, kind="mergesort")[::-1]
    y_true = y_true[desc]
    y_score = y_score[desc]
    distinct = np.where(np.diff(y_score))[0]
    thresh_idx = np.r_[distinct, y_true.size - 1]
    tps = np.cumsum(y_true)[thresh_idx]
    fps = np.cumsum(1.0 - y_true)[thresh_idx]
    tps = np.r_[0.0, tps]
    fps = np.r_[0.0, fps]
    denom_f = fps[-1]
    denom_t = tps[-1]
    fpr = np.zeros_like(fps) if denom_f <= 0 else fps / denom_f
    tpr = np.zeros_like(tps) if denom_t <= 0 else tps / denom_t
    thresholds = np.r_[np.inf, y_score[thresh_idx]]
    return fpr, tpr, thresholds


def _roc_auc_score_binary(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Binary ROC AUC via ranked scores (equivalent to Mann–Whitney U; matches sklearn for non-single-class)."""
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_score = np.asarray(y_score, dtype=np.float64).ravel()
    pos = y_true == 1
    n_pos = int(pos.sum())
    n_neg = int((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(y_score)
    sum_ranks_pos = float(ranks[pos].sum())
    return float((sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


# Uniform matplotlib font size (pt) for manuscript Figure 5 panels a–d (≥ 8 pt; journal composite uses 10 pt).
_FIG5_FONT_PT = 10.0


def _records(path):
    if not os.path.isfile(path):
        print(f"Missing cached result: {path}")
        return []
    return pd.read_csv(path).to_dict("records")


def plot_grouped_incidence_or_forest(results_dir_path: Path, fig_dir: Path) -> None:
    """Forest plot: OR/HR per disease category from grouped incidence analysis."""
    csv_path = results_dir_path / "grouped_incidence_or_results.csv"
    if not csv_path.is_file():
        print(f"Skipping OR forest plot: {csv_path} not found. Run run_target_prediction_results.py first.")
        return
    df = pd.read_csv(csv_path)
    if df.empty:
        print("Skipping OR forest plot: grouped_incidence_or_results.csv is empty.")
        return

    df = df.dropna(subset=["sleep_risk_effect_ratio"]).reset_index(drop=True)
    # Sort by effect ratio descending so largest effect is at top
    df = df.sort_values("sleep_risk_effect_ratio", ascending=False).reset_index(drop=True)
    n = len(df)
    fig_h = max(3.5, 0.55 * n + 1.4)
    fig, ax = plt.subplots(figsize=(8.0, fig_h))

    y = np.arange(n)
    or_vals = df["sleep_risk_effect_ratio"].to_numpy(dtype=float)
    ci_low = df.get("sleep_risk_ci_low", pd.Series(np.nan, index=df.index)).to_numpy(dtype=float)
    ci_high = df.get("sleep_risk_ci_high", pd.Series(np.nan, index=df.index)).to_numpy(dtype=float)
    colors = [_PULSEOX_TEAL if v >= 1.0 else _DEMO_GRAY for v in or_vals]

    # Horizontal error bars as xerr requires [low_err, high_err] arrays
    xerr_low = np.where(np.isfinite(ci_low), np.maximum(0, or_vals - ci_low), 0.0)
    xerr_high = np.where(np.isfinite(ci_high), np.maximum(0, ci_high - or_vals), 0.0)

    ax.barh(y, or_vals, height=0.5, color=colors, edgecolor="#333333", linewidth=0.6,
            xerr=[xerr_low, xerr_high],
            error_kw={"ecolor": "#333333", "elinewidth": 1.0, "capsize": 3})
    ax.axvline(1.0, color="#555555", linestyle="--", linewidth=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels(df["category"].tolist(), fontsize=9)
    ax.invert_yaxis()
    ratio_label = "HR" if (df["effect_metric"] == "HR").any() else "OR"
    ax.set_xlabel(f"{ratio_label} (sleep embeddings, adjusted for age/sex/BMI)", fontsize=9)
    ax.set_title("Disease incidence within 2 years\nby disease category", fontsize=10)
    ax.grid(axis="x", linestyle=":", alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Annotate with OR value and p-value
    for i, row in df.iterrows():
        or_v = row.get("sleep_risk_effect_ratio", np.nan)
        pv = row.get("p_sleep_risk_effect", np.nan)
        n_ev = row.get("n_events", "")
        n_pt = row.get("n_participants", "")
        n_txt = f"N={int(n_ev)}/{int(n_pt)}" if pd.notna(n_ev) and pd.notna(n_pt) else ""
        p_txt = f"p={pv:.3f}" if pd.notna(pv) else ""
        label = f"{or_v:.2f} {p_txt} {n_txt}".strip() if pd.notna(or_v) else ""
        ci_h = ci_high[i] if i < len(ci_high) else np.nan
        x_pos = (ci_h if np.isfinite(ci_h) else or_v) + 0.03
        ax.text(x_pos, i, label, va="center", ha="left", fontsize=7.5, color="#222222")

    ax.set_xlim(left=0)
    plt.tight_layout()
    for ext in ("png", "pdf"):
        out = fig_dir / f"grouped_incidence_or_forest.{ext}"
        plt.savefig(out, dpi=300, bbox_inches="tight", format=ext)
        print(f"Saved {out}")
    plt.close()


def plot_grouped_incidence_metric_barh(results_df: pd.DataFrame, metric: str, fig_dir: Path) -> None:
    """Horizontal bar plot comparing demo-only vs demo+embeddings for one discrimination metric.

    metric: 'c_index' (Cox PH) or 'auc' (logistic regression).
    """
    demo_col = f"{metric}_demo"
    comb_col = f"{metric}_combined"
    demo_std_col = f"{metric}_demo_std"
    comb_std_col = f"{metric}_combined_std"
    p_col = f"p_{metric}_demo_vs_combined"

    required = [demo_col, comb_col]
    missing = [c for c in required if c not in results_df.columns]
    if missing:
        print(f"Skipping grouped incidence {metric} bar plot: columns {missing} missing.")
        return

    df = results_df.dropna(subset=required).copy()
    if df.empty:
        print(f"Skipping grouped incidence {metric} bar plot: no valid rows.")
        return

    df["_delta"] = df[comb_col] - df[demo_col]
    df = df.sort_values("_delta", ascending=True).reset_index(drop=True)

    n = len(df)
    fig_h = max(5, 0.52 * n + 1.5)
    fig, ax = plt.subplots(figsize=(12.5, fig_h))

    y = np.arange(n)
    h = 0.35

    demo_vals = df[demo_col].to_numpy(dtype=float)
    comb_vals = df[comb_col].to_numpy(dtype=float)
    demo_std = df[demo_std_col].fillna(0).to_numpy(dtype=float) if demo_std_col in df.columns else np.zeros(n)
    comb_std = df[comb_std_col].fillna(0).to_numpy(dtype=float) if comb_std_col in df.columns else np.zeros(n)

    ax.barh(
        y - h / 2, demo_vals, height=h,
        color="#ededed", edgecolor="silver", linewidth=0.8, hatch="///",
        xerr=demo_std, error_kw={"ecolor": "#000000", "elinewidth": 0.9, "capsize": 2},
        label="Age, sex, BMI",
    )
    ax.barh(
        y + h / 2, comb_vals, height=h,
        color="#196874", edgecolor="white", linewidth=0.5,
        xerr=comb_std, error_kw={"ecolor": "#000000", "elinewidth": 0.9, "capsize": 2},
        label="Age, sex, BMI + sleep embeddings",
    )

    ax.set_yticks(y)
    ax.set_yticklabels(df["category"].tolist(), fontsize=9)
    ax.invert_yaxis()

    x_label = "C-index (Cox PH)" if metric == "c_index" else "AUC (logistic regression)"
    ax.set_xlabel(x_label, fontsize=9.5)
    ax.set_title("Disease incidence within 2 years by category", fontsize=10)
    ax.axvline(0.5, color="#555555", linestyle="--", linewidth=0.7)
    ax.grid(axis="x", linestyle=":", alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for i, row in df.iterrows():
        p_val = row.get(p_col, np.nan)
        stars = analysis._significance_stars(p_val).strip() if pd.notna(p_val) else ""
        n_ev = row.get("n_events", "")
        n_pt = row.get("n_participants", "")
        n_txt = f"N={int(n_ev)}/{int(n_pt)}" if pd.notna(n_ev) and pd.notna(n_pt) else ""
        x_right = max(
            float(row[demo_col]) + float(demo_std[i]),
            float(row[comb_col]) + float(comb_std[i]),
        ) + 0.01
        annotation = " ".join(filter(None, [stars, n_txt]))
        if annotation:
            ax.text(x_right, i, annotation, va="center", ha="left", fontsize=8.5, color="#2a2a2a")

    all_vals = np.concatenate([demo_vals + demo_std, comb_vals + comb_std])
    ax.set_xlim(
        max(0.0, float(np.nanmin(np.concatenate([demo_vals, comb_vals]))) - 0.05),
        min(1.0, float(np.nanmax(all_vals)) + 0.22),
    )
    ax.legend(loc="lower right", frameon=False, fontsize=8.5)
    plt.tight_layout()

    stem = f"barh_{metric}_grouped_incidence"
    for ext in ("png", "pdf"):
        out = fig_dir / f"{stem}.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight", format=ext)
        print(f"Saved {out}")
    plt.close(fig)


def plot_hypertension_incidence_roc(cache_dir: Path, fig_dir: Path) -> None:
    """ROC curves for predicting hypertension incidence within 2 years."""
    matches = sorted(cache_dir.glob("classification_Hypertension_incidence_2yrs_mean_all_*.csv"))
    if not matches:
        print(f"Skipping ROC plot: no hypertension incidence cache CSV found in {cache_dir}")
        return
    df = pd.read_csv(matches[0])
    target_col = "Hypertension_incidence_2yrs"
    if target_col not in df.columns:
        print(f"Skipping ROC plot: column '{target_col}' missing from cache CSV.")
        return
    df = df.dropna(subset=[target_col])
    df["participant_id"] = df["Recordings"].astype(str).str.split("__").str[0]
    # Aggregate to participant level (take mean probability per participant)
    agg = df.groupby("participant_id", as_index=False).agg(
        y_true=(target_col, "max"),
        proba_demo=("proba_demo_mean", "mean"),
        proba_combined=("proba_combined_mean", "mean"),
    )
    y = agg["y_true"].to_numpy(dtype=float)
    if len(np.unique(y[np.isfinite(y)])) < 2:
        print("Skipping ROC plot: only one class in hypertension incidence labels.")
        return

    common_fpr = np.linspace(0.0, 1.0, 101)

    def _tpr_on_grid(y_arr: np.ndarray, scores: np.ndarray, grid: np.ndarray) -> np.ndarray | None:
        valid = np.isfinite(y_arr) & np.isfinite(scores)
        yv = y_arr[valid]
        sv = scores[valid]
        if len(np.unique(yv)) < 2:
            return None
        fpr, tpr, _ = _roc_curve_binary(yv, sv)
        if len(fpr) == 0:
            return None
        return np.clip(np.interp(grid, fpr, tpr, left=0.0, right=float(tpr[-1])), 0.0, 1.0)

    def _seed_columns(prefix: str) -> list[str]:
        seeds = getattr(analysis, "CV_SEEDS", []) or []
        out = []
        for s in seeds:
            c = f"{prefix}_{int(s)}"
            if c in df.columns:
                out.append(c)
        return out

    _f5 = _FIG5_FONT_PT
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    for pooled_col, seed_prefix, label, color in [
        ("proba_combined", "proba_combined_seed", "Age, sex, BMI + embeddings", _PULSEOX_TEAL),
        ("proba_demo", "proba_demo_seed", "Age, sex, BMI", _DEMO_GRAY),
    ]:
        if pooled_col not in agg.columns:
            continue
        proba = agg[pooled_col].to_numpy(dtype=float)
        valid = np.isfinite(proba) & np.isfinite(y)
        if valid.sum() == 0:
            continue
        fpr, tpr, _ = _roc_curve_binary(y[valid], proba[valid])
        auc = _roc_auc_score_binary(y[valid], proba[valid])

        seed_cols = _seed_columns(seed_prefix)
        if len(seed_cols) >= 2:
            tpr_rows = []
            auc_seeds = []
            for sc in seed_cols:
                agg_s = df.groupby("participant_id", as_index=False).agg(
                    y_true=(target_col, "max"),
                    proba=(sc, "mean"),
                )
                yt = agg_s["y_true"].to_numpy(dtype=float)
                pr = agg_s["proba"].to_numpy(dtype=float)
                ok = np.isfinite(yt) & np.isfinite(pr)
                if ok.sum() == 0 or len(np.unique(yt[ok])) < 2:
                    continue
                tgrid = _tpr_on_grid(yt, pr, common_fpr)
                if tgrid is not None:
                    tpr_rows.append(tgrid)
                    auc_seeds.append(float(_roc_auc_score_binary(yt[ok], pr[ok])))
            if len(tpr_rows) >= 2:
                mat = np.vstack(tpr_rows)
                mean_tpr = np.mean(mat, axis=0)
                sd_tpr = np.std(mat, axis=0, ddof=1)
                lo = np.clip(mean_tpr - sd_tpr, 0.0, 1.0)
                hi = np.clip(mean_tpr + sd_tpr, 0.0, 1.0)
                ax.fill_between(
                    common_fpr,
                    lo,
                    hi,
                    facecolor=color,
                    alpha=0.22,
                    linewidth=0,
                    zorder=1,
                )
                if len(auc_seeds) >= 2:
                    auc_mean = float(np.mean(auc_seeds))
                    auc_sd = float(np.std(auc_seeds, ddof=1))
                    auc_line = f"(AUC={auc_mean:.2f}±{auc_sd:.2f})"
                else:
                    auc_line = f"(AUC={auc:.2f})"
            else:
                auc_line = f"(AUC={auc:.2f})"
        else:
            auc_line = f"(AUC={auc:.2f})"

        ax.plot(
            fpr,
            tpr,
            color=color,
            linewidth=1.8,
            label=f"{label}\n{auc_line}",
            zorder=2,
        )

    positives = int(np.nansum(y))
    total = int(np.sum(np.isfinite(y)))
    ax.plot([0, 1], [0, 1], color="#888888", linestyle="--", linewidth=0.8)
    ax.set_xlabel("False positive rate", fontsize=_f5)
    ax.set_ylabel("True positive rate", fontsize=_f5)
    ax.set_title(
        f"Hypertension incidence within 2 years\nN={positives}/{total} positive participants",
        fontsize=_f5,
    )
    ax.tick_params(axis="both", labelsize=_f5)
    ax.legend(frameon=False, loc="lower right", fontsize=_f5, handlelength=1.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(color="#d7d7d7", linewidth=0.4, linestyle="--", zorder=0)
    plt.tight_layout()
    for ext in ("png", "pdf"):
        out = fig_dir / f"roc_hypertension_incidence_2y.{ext}"
        plt.savefig(out, dpi=300, bbox_inches="tight", format=ext)
        print(f"Saved {out}")
    plt.close()


def copy_bone_density_quantile_panels(fig_dir: Path) -> None:
    """Copy or convert existing bone density quantile PDFs to PNG in the figures dir."""
    try:
        from pdf2image import convert_from_path
        _HAS_PDF2IMAGE = True
    except ImportError:
        _HAS_PDF2IMAGE = False

    search_dirs = [
        _OLD_TABICL_OUTPUT,
        Path("/home/sarahk/JafarShortcut/SleepFM/ssl_sleep/downstream_tasks_results/target_prediction"),
    ]
    targets = {
        "bone_density_quantile_boxplots": "Total_Bone_Density_quantile_boxplots_mean_all_N*.pdf",
        "bone_density_2y_delta": "Total_Bone_Density_baseline_vs_2y_change_spearman_mean_all_N*.pdf",
    }
    for out_stem, pattern in targets.items():
        found = None
        for d in search_dirs:
            matches = sorted(d.glob(pattern))
            if matches:
                found = matches[0]
                break
            # Also try PNG directly
            png_matches = sorted(d.glob(pattern.replace(".pdf", ".png")))
            if png_matches:
                dest = fig_dir / f"{out_stem}.png"
                shutil.copy2(png_matches[0], dest)
                print(f"Copied {png_matches[0]} → {dest}")
                found = "copied"
                break
        if found is None:
            print(f"Could not find {pattern} in known output directories; skipping {out_stem}.")
            continue
        if found == "copied":
            continue
        dest_png = fig_dir / f"{out_stem}.png"
        if _HAS_PDF2IMAGE:
            try:
                pages = convert_from_path(str(found), dpi=300, first_page=1, last_page=1)
                pages[0].save(str(dest_png))
                print(f"Converted {found} → {dest_png}")
            except Exception as e:
                dest_pdf = fig_dir / f"{out_stem}.pdf"
                shutil.copy2(found, dest_pdf)
                print(
                    f"PDF→PNG conversion failed ({type(e).__name__}: {e}); copied PDF → {dest_pdf}. "
                    "Install Poppler and ensure pdfinfo is on PATH (e.g. conda install -c conda-forge poppler), "
                    "or rely on the copied PDF."
                )
        else:
            dest_pdf = fig_dir / f"{out_stem}.pdf"
            shutil.copy2(found, dest_pdf)
            print(f"Copied {found} → {dest_pdf} (pdf2image not installed; PNG conversion skipped)")


def plot_psychoanaleptic_responder_target_boxplots(out: Path, cache_dir: Path, fig_dir: Path, medication: list[dict]) -> None:
    """Boxplots for baseline targets that differ between predicted psychoanaleptic response groups."""
    psycho_target = analysis._select_psychoanaleptic_target(medication)
    if psycho_target is None:
        print("Skipping psychoanaleptic responder target boxplots: no psychoanaleptic medication target found.")
        return

    cache_path = None
    cache_pool_method = None
    for pool_method in ("mean_1to8h", "mean_all"):
        matches = sorted(cache_dir.glob(f"classification_{psycho_target}_{pool_method}_*.csv"))
        if matches:
            cache_path = matches[0]
            cache_pool_method = pool_method
            break
    if cache_path is None:
        print(f"Skipping psychoanaleptic responder target boxplots: no prediction cache found for {psycho_target}.")
        return

    labels_path = out / "10k_cov_mean_merged.csv"
    if not labels_path.is_file():
        print(f"Skipping psychoanaleptic responder target boxplots: missing {labels_path}.")
        return

    cache = pd.read_csv(cache_path)
    proba_col = "proba_combined_mean" if "proba_combined_mean" in cache.columns else "proba_combined"
    required_cols = {analysis.ID_COL, psycho_target, proba_col}
    if not required_cols.issubset(cache.columns):
        print(f"Skipping psychoanaleptic responder target boxplots: cache missing {required_cols - set(cache.columns)}.")
        return

    pred_df = cache[[analysis.ID_COL, psycho_target, proba_col]].copy()
    pred_df["participant_id"] = analysis._parse_participant_id(pred_df[analysis.ID_COL])
    pred_df["y_true"] = pd.to_numeric(pred_df[psycho_target], errors="coerce")
    pred_df["pred_proba"] = pd.to_numeric(pred_df[proba_col], errors="coerce")
    pred_df = (
        pred_df.dropna(subset=["y_true", "pred_proba"])
        .groupby("participant_id", as_index=False)
        .agg(y_true=("y_true", "max"), pred_proba=("pred_proba", "mean"))
    )
    pred_df = pred_df[(pred_df["y_true"] == 1) & (pred_df["pred_proba"] != 0.5)].copy()
    if pred_df.empty:
        print(f"Skipping psychoanaleptic responder target boxplots: no actual {psycho_target} users with non-0.5 predictions.")
        return
    pred_df["response_group"] = np.where(pred_df["pred_proba"] > 0.5, "Predicted responders", "Predicted non-responders")

    labels_df = pd.read_csv(labels_path)
    if analysis.ID_COL not in labels_df.columns and "Unnamed: 0" in labels_df.columns:
        labels_df = labels_df.rename(columns={"Unnamed: 0": analysis.ID_COL})
    labels_df[analysis.ID_COL] = labels_df[analysis.ID_COL].astype(str).str.replace(".pt", "", regex=False)
    if "participant_id" not in labels_df.columns:
        labels_df["participant_id"] = analysis._parse_participant_id(labels_df[analysis.ID_COL])
    if "research_stage" not in labels_df.columns:
        labels_df["research_stage"] = analysis._parse_stage(labels_df[analysis.ID_COL])
    participant_baseline_followup = analysis.get_participant_baseline_followup(labels_df)
    phenotypes = analysis._baseline_regression_phenotypes_by_participant(labels_df, participant_baseline_followup)
    if phenotypes.empty:
        print("Skipping psychoanaleptic responder target boxplots: no baseline target phenotypes found.")
        return

    joint = pred_df.merge(phenotypes, on="participant_id", how="inner")
    if joint.empty:
        print("Skipping psychoanaleptic responder target boxplots: no overlap between response groups and baseline targets.")
        return

    panels = []
    summary_rows = []
    for target in [c for c in analysis.TARGET_COLS if c in joint.columns]:
        vals = pd.to_numeric(joint[target], errors="coerce")
        if vals.notna().sum() < 10:
            continue
        std = vals.std(skipna=True)
        if not np.isfinite(std) or std == 0:
            continue
        z = (vals - vals.mean(skipna=True)) / std
        nonresp = z[joint["response_group"] == "Predicted non-responders"].dropna().to_numpy(dtype=float)
        resp = z[joint["response_group"] == "Predicted responders"].dropna().to_numpy(dtype=float)
        if len(nonresp) < 3 or len(resp) < 3:
            continue
        stat, p_value = ttest_ind(nonresp, resp, equal_var=False, nan_policy="omit")
        if not np.isfinite(p_value) or p_value >= 0.05:
            continue
        median_delta = float(np.median(resp) - np.median(nonresp))
        row = {
            "target": target,
            "p_ttest": float(p_value),
            "t_statistic": float(stat),
            "n_non_responders": int(len(nonresp)),
            "n_responders": int(len(resp)),
            "median_z_non_responders": float(np.median(nonresp)),
            "median_z_responders": float(np.median(resp)),
            "median_z_delta_responders_minus_non": median_delta,
        }
        panels.append((target, nonresp, resp, row))
        summary_rows.append(row)

    summary_path = fig_dir / f"psychoanaleptic_responder_nonresponder_significant_targets_{cache_pool_method}.csv"
    if summary_rows:
        pd.DataFrame(summary_rows).sort_values("p_ttest").to_csv(summary_path, index=False)
    if not panels:
        print("Skipping psychoanaleptic responder target boxplots: no baseline targets were significant by Welch t-test (p < 0.05).")
        return

    panels = sorted(panels, key=lambda item: item[3]["p_ttest"])
    n_panels = len(panels)
    n_cols = min(4, max(2, math.ceil(math.sqrt(n_panels))))
    n_rows = math.ceil(n_panels / n_cols)
    colors = ["#d3d3d3", _PULSEOX_TEAL]
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.1 * n_cols, 3.0 * n_rows), squeeze=False)
    for ax, (target, nonresp, resp, row) in zip(axes.ravel(), panels):
        bp = ax.boxplot(
            [nonresp, resp],
            positions=[0, 1],
            widths=0.55,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "#2d2d2d", "linewidth": 1.2},
        )
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_edgecolor("#333333")
            patch.set_alpha(0.85)
        ax.axhline(0, color="#777777", linestyle="--", linewidth=0.7)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Non-resp.", "Resp."], fontsize=8)
        ax.set_title(
            f"{analysis._target_display_name(target)}\np={row['p_ttest']:.3g}, Δmedian={row['median_z_delta_responders_minus_non']:.2f}",
            fontsize=8.5,
        )
        ax.grid(axis="y", alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    for ax in axes.ravel()[n_panels:]:
        ax.axis("off")

    n_resp = int((joint["response_group"] == "Predicted responders").sum())
    n_nonresp = int((joint["response_group"] == "Predicted non-responders").sum())
    fig.suptitle(
        f"{analysis._target_display_name(psycho_target)} users: baseline targets differing by predicted response group\n"
        f"Welch t-test p < 0.05; combined model threshold 0.5, {cache_pool_method}; "
        f"responders N={n_resp}, non-responders N={n_nonresp}",
        fontsize=11,
    )
    fig.supxlabel("Predicted response group", fontsize=9)
    fig.supylabel("Baseline target value (z-score)", fontsize=9)
    fig.legend(
        handles=[
            Patch(facecolor=colors[0], edgecolor="#333333", label=f"Predicted non-responders, N={n_nonresp}"),
            Patch(facecolor=colors[1], edgecolor="#333333", label=f"Predicted responders, N={n_resp}"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=2,
        frameon=False,
        fontsize=8.5,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.92))
    for ext in ("png", "pdf"):
        out_path = fig_dir / f"psychoanaleptic_responder_nonresponder_significant_target_boxplots_{cache_pool_method}.{ext}"
        fig.savefig(out_path, dpi=300, bbox_inches="tight", format=ext)
        print(f"Saved {out_path}")
    plt.close(fig)
    print(f"Saved psychoanaleptic responder/non-responder t-test summary: {summary_path}")


def _load_gold_labels_for_figures(out: Path) -> pd.DataFrame | None:
    candidate_paths = [
        Path(getattr(analysis, "GOLD_RECORDS_PATH", "")),
        out / "recordings_with_labels_gold.csv",
        Path("/home/sarahk/JafarShortcut/SleepFM/ssl_sleep/tabular_data/recordings_with_labels_gold.csv"),
    ]
    for path in candidate_paths:
        if not path.is_file():
            continue
        labels = pd.read_csv(path)
        if analysis.ID_COL not in labels.columns and "index" in labels.columns:
            labels = labels.rename(columns={"index": analysis.ID_COL})
        if analysis.ID_COL not in labels.columns:
            continue
        labels[analysis.ID_COL] = labels[analysis.ID_COL].astype(str).str.replace(".pt", "", regex=False)
        if "research_stage" not in labels.columns:
            labels["research_stage"] = analysis._parse_stage(labels[analysis.ID_COL])
        if "participant_id" not in labels.columns:
            labels["participant_id"] = analysis._parse_participant_id(labels[analysis.ID_COL])
        labels["participant_id"] = labels["participant_id"].astype(str)
        return labels
    print("Skipping BP quartile line plots: gold hypertension labels not found.")
    return None


def _baseline_prediction_cache_path(cache_dir: Path, target: str) -> Path | None:
    """Resolve baseline true/pred CSV for a regression target (same glob logic as BP plots)."""
    safe_target = analysis._safe_name_for_path(target)
    matches = sorted(cache_dir.glob(f"*__{safe_target}_mean_all_*_baseline_predictions.csv"))
    if matches:
        return matches[0]
    matches = sorted(cache_dir.glob(f"*{safe_target}*baseline_predictions.csv"))
    return matches[0] if matches else None


def plot_bp_quartile_hypertension_incidence_lines(out: Path, cache_dir: Path, fig_dir: Path) -> None:
    """Line plots of 2-year hypertension incidence across measured and predicted BP quartiles."""
    labels_df = _load_gold_labels_for_figures(out)
    if labels_df is None or labels_df.empty:
        return
    hypertension_col = analysis._find_hypertension_diagnosis_column(labels_df)
    if hypertension_col is None:
        print("Skipping BP quartile line plots: hypertension diagnosis column not found.")
        return
    participant_baseline_followup = analysis.get_participant_baseline_followup(labels_df)
    participant_baseline_followup["participant_id"] = participant_baseline_followup["participant_id"].astype(str)
    stages = participant_baseline_followup[["participant_id", "baseline_stage"]].copy()
    stages["stage_2y"] = stages["baseline_stage"].apply(lambda x: analysis._stage_after_years(x, 2))
    dx_baseline = analysis._diagnosis_by_participant_stage(
        labels_df, hypertension_col, stages, "baseline_stage"
    ).rename(columns={"baseline_stage": "hypertension_baseline"})
    dx_2y = analysis._diagnosis_by_participant_stage(
        labels_df, hypertension_col, stages, "stage_2y"
    ).rename(columns={"stage_2y": "hypertension_2y"})

    for target in ("Sitting BP diastolic", "Sitting BP systolic"):
        cache_path = _baseline_prediction_cache_path(cache_dir, target)
        if cache_path is None:
            print(f"Skipping BP quartile line plot for {target}: baseline prediction cache not found.")
            continue
        pred_df = pd.read_csv(cache_path)
        pred_col = f"{target}_pred"
        if not {analysis.ID_COL, target, pred_col}.issubset(pred_df.columns):
            print(f"Skipping BP quartile line plot for {target}: cache missing required columns.")
            continue
        if "participant_id" not in pred_df.columns:
            pred_df["participant_id"] = analysis._parse_participant_id(pred_df[analysis.ID_COL])
        pred_df["participant_id"] = pred_df["participant_id"].astype(str)
        pred_df[target] = pd.to_numeric(pred_df[target], errors="coerce")
        pred_df[pred_col] = pd.to_numeric(pred_df[pred_col], errors="coerce")
        base_bp = (
            pred_df.dropna(subset=[target, pred_col])
            .groupby("participant_id", as_index=False)
            .agg(
                measured_bp=(target, "mean"),
                predicted_bp=(pred_col, "mean"),
                n_baseline_recordings=(analysis.ID_COL, "nunique"),
            )
        )
        joint = base_bp.merge(dx_baseline, on="participant_id", how="inner")
        joint = joint[joint["hypertension_baseline"] == 0].copy()
        joint = joint.merge(dx_2y, on="participant_id", how="left").dropna(subset=["hypertension_2y"])
        if len(joint) < 20:
            print(f"Skipping BP quartile line plot for {target}: N={len(joint)} after joins.")
            continue

        rows = []
        for source, value_col, label in [
            ("measured", "measured_bp", "Measured baseline BP"),
            ("predicted", "predicted_bp", "Predicted baseline BP"),
        ]:
            try:
                joint[f"{source}_quartile"] = pd.qcut(joint[value_col], q=4, labels=["Q1", "Q2", "Q3", "Q4"])
            except ValueError:
                print(f"Skipping {source} quartiles for {target}: could not form four bins.")
                continue
            for q_label in ("Q1", "Q2", "Q3", "Q4"):
                grp = joint[joint[f"{source}_quartile"] == q_label]
                n_2y = int(grp["hypertension_2y"].notna().sum())
                cases_2y = int(grp["hypertension_2y"].sum()) if n_2y else 0
                rows.append(
                    {
                        "target": target,
                        "bp_source": source,
                        "bp_source_label": label,
                        "quartile": q_label,
                        "n_2y": n_2y,
                        "cases_2y": cases_2y,
                        "percent_2y": float(cases_2y / n_2y * 100.0) if n_2y else np.nan,
                        "n_nonhypertensive_with_2y": int(joint["participant_id"].nunique()),
                        "n_baseline_recordings_mean_pooled": int(grp["n_baseline_recordings"].sum()),
                    }
                )
        summary_df = pd.DataFrame(rows)
        if summary_df.empty:
            continue
        safe_target = analysis._safe_name_for_path(target)
        summary_path = fig_dir / f"{safe_target}_bp_hypertension_incidence_quartile_line_summary.csv"
        summary_df.to_csv(summary_path, index=False)

        fig, ax = plt.subplots(figsize=(5.8, 4.2))
        x = np.arange(1, 5)
        for source, label, color, marker in [
            ("measured", "Measured baseline BP quartiles", _DEMO_GRAY, "o"),
            ("predicted", "Predicted baseline BP quartiles", _PULSEOX_TEAL, "s"),
        ]:
            view = summary_df[summary_df["bp_source"] == source].set_index("quartile").reindex(["Q1", "Q2", "Q3", "Q4"])
            y = view["percent_2y"].to_numpy(dtype=float)
            ax.plot(x, y, marker=marker, linewidth=2.0, color=color, label=label)
            for xi, yi, cases, n in zip(x, y, view["cases_2y"], view["n_2y"]):
                if np.isfinite(yi):
                    ax.text(xi, yi + 0.8, f"{int(cases)}/{int(n)}", ha="center", va="bottom", fontsize=7.5, color=color)
        ymax = summary_df["percent_2y"].dropna().max()
        ax.set_xticks(x)
        ax.set_xticklabels(["Q1", "Q2", "Q3", "Q4"])
        ax.set_xlabel("Baseline BP quartile")
        ax.set_ylabel("Hypertension incidence within 2 years (%)")
        ax.set_ylim(0, max(5.0, float(ymax) * 1.25 if pd.notna(ymax) else 5.0))
        ax.set_title(
            f"{target}: 2-year hypertension incidence by baseline BP quartile\n"
            f"Baseline hypertension excluded, N={int(joint['participant_id'].nunique())} participants",
            fontsize=10,
        )
        ax.grid(axis="y", alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            out_path = fig_dir / f"{safe_target}_bp_hypertension_incidence_quartile_lines.{ext}"
            fig.savefig(out_path, dpi=300, bbox_inches="tight", format=ext)
            print(f"Saved {out_path}")
        plt.close(fig)
        print(f"Saved BP quartile line summary: {summary_path}")


def _resolve_age_baseline_cache(cache_dir: Path) -> tuple[str, Path] | None:
    """Pick Age vs age column name and cache path using first match with required columns."""
    for col in ("Age", "age"):
        cache_path = _baseline_prediction_cache_path(cache_dir, col)
        if cache_path is None or not cache_path.is_file():
            continue
        head = pd.read_csv(cache_path, nrows=3)
        pred_col = f"{col}_pred"
        if {analysis.ID_COL, col, pred_col}.issubset(head.columns):
            return col, cache_path
    return None


def plot_age_quartile_hypertension_incidence_lines(out: Path, cache_dir: Path, fig_dir: Path) -> None:
    """Line plots of 2-year hypertension incidence across measured vs predicted baseline age quartiles."""
    labels_df = _load_gold_labels_for_figures(out)
    if labels_df is None or labels_df.empty:
        return
    hypertension_col = analysis._find_hypertension_diagnosis_column(labels_df)
    if hypertension_col is None:
        print("Skipping age quartile line plots: hypertension diagnosis column not found.")
        return
    resolved = _resolve_age_baseline_cache(cache_dir)
    if resolved is None:
        print("Skipping age quartile line plots: baseline Age predictions cache not found.")
        return
    age_col, cache_path = resolved
    display_name = "Age"

    participant_baseline_followup = analysis.get_participant_baseline_followup(labels_df)
    participant_baseline_followup["participant_id"] = participant_baseline_followup["participant_id"].astype(str)
    stages = participant_baseline_followup[["participant_id", "baseline_stage"]].copy()
    stages["stage_2y"] = stages["baseline_stage"].apply(lambda x: analysis._stage_after_years(x, 2))
    dx_baseline = analysis._diagnosis_by_participant_stage(
        labels_df, hypertension_col, stages, "baseline_stage"
    ).rename(columns={"baseline_stage": "hypertension_baseline"})
    dx_2y = analysis._diagnosis_by_participant_stage(
        labels_df, hypertension_col, stages, "stage_2y"
    ).rename(columns={"stage_2y": "hypertension_2y"})

    pred_df = pd.read_csv(cache_path)
    pred_col = f"{age_col}_pred"
    if not {analysis.ID_COL, age_col, pred_col}.issubset(pred_df.columns):
        print(f"Skipping age quartile line plot: cache missing required columns ({cache_path}).")
        return
    if "participant_id" not in pred_df.columns:
        pred_df["participant_id"] = analysis._parse_participant_id(pred_df[analysis.ID_COL])
    pred_df["participant_id"] = pred_df["participant_id"].astype(str)
    pred_df[age_col] = pd.to_numeric(pred_df[age_col], errors="coerce")
    pred_df[pred_col] = pd.to_numeric(pred_df[pred_col], errors="coerce")
    base_age = (
        pred_df.dropna(subset=[age_col, pred_col])
        .groupby("participant_id", as_index=False)
        .agg(
            measured_age=(age_col, "mean"),
            predicted_age=(pred_col, "mean"),
            n_baseline_recordings=(analysis.ID_COL, "nunique"),
        )
    )
    joint = base_age.merge(dx_baseline, on="participant_id", how="inner")
    joint = joint[joint["hypertension_baseline"] == 0].copy()
    joint = joint.merge(dx_2y, on="participant_id", how="left").dropna(subset=["hypertension_2y"])
    if len(joint) < 20:
        print(f"Skipping age quartile line plot: N={len(joint)} after joins.")
        return

    rows = []
    for source, value_col, label in [
        ("measured", "measured_age", "Measured baseline age"),
        ("predicted", "predicted_age", "Predicted baseline age"),
    ]:
        try:
            joint[f"{source}_quartile"] = pd.qcut(joint[value_col], q=4, labels=["Q1", "Q2", "Q3", "Q4"])
        except ValueError:
            print(f"Skipping {source} age quartiles: could not form four bins.")
            continue
        for q_label in ("Q1", "Q2", "Q3", "Q4"):
            grp = joint[joint[f"{source}_quartile"] == q_label]
            n_2y = int(grp["hypertension_2y"].notna().sum())
            cases_2y = int(grp["hypertension_2y"].sum()) if n_2y else 0
            rows.append(
                {
                    "target": display_name,
                    "age_source": source,
                    "age_source_label": label,
                    "quartile": q_label,
                    "n_2y": n_2y,
                    "cases_2y": cases_2y,
                    "percent_2y": float(cases_2y / n_2y * 100.0) if n_2y else np.nan,
                    "n_nonhypertensive_with_2y": int(joint["participant_id"].nunique()),
                    "n_baseline_recordings_mean_pooled": int(grp["n_baseline_recordings"].sum()),
                }
            )
    summary_df = pd.DataFrame(rows)
    if summary_df.empty:
        return

    summary_path = fig_dir / "Age_baseline_quartile_hypertension_incidence_line_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    x = np.arange(1, 5)
    for source, label, color, marker in [
        ("measured", "Measured baseline age quartiles", _DEMO_GRAY, "o"),
        ("predicted", "Predicted baseline age quartiles", _PULSEOX_TEAL, "s"),
    ]:
        view = summary_df[summary_df["age_source"] == source].set_index("quartile").reindex(["Q1", "Q2", "Q3", "Q4"])
        y = view["percent_2y"].to_numpy(dtype=float)
        ax.plot(x, y, marker=marker, linewidth=2.0, color=color, label=label)
        for xi, yi, cases, n in zip(x, y, view["cases_2y"], view["n_2y"]):
            if np.isfinite(yi):
                ax.text(xi, yi + 0.8, f"{int(cases)}/{int(n)}", ha="center", va="bottom", fontsize=7.5, color=color)
    ymax = summary_df["percent_2y"].dropna().max()
    ax.set_xticks(x)
    ax.set_xticklabels(["Q1", "Q2", "Q3", "Q4"])
    ax.set_xlabel("Baseline age quartile")
    ax.set_ylabel("Hypertension incidence within 2 years (%)")
    ax.set_ylim(0, max(5.0, float(ymax) * 1.25 if pd.notna(ymax) else 5.0))
    ax.set_title(
        f"{display_name}: 2-year hypertension incidence by baseline age quartile\n"
        f"Baseline hypertension excluded, N={int(joint['participant_id'].nunique())} participants",
        fontsize=10,
    )
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        out_path = fig_dir / f"Age_baseline_quartile_hypertension_incidence_lines.{ext}"
        fig.savefig(out_path, dpi=300, bbox_inches="tight", format=ext)
        print(f"Saved {out_path}")
    plt.close(fig)
    print(f"Saved age quartile line summary: {summary_path}")


def _load_labels_for_regression_longitudinal_figures(out: Path) -> pd.DataFrame | None:
    """Merged covariates (preferred) or gold labels; same ID normalization as other figure loaders."""
    merged_path = getattr(analysis, "MERGED_COVARIATES_PATH", "") or ""
    if merged_path and Path(merged_path).is_file():
        labels = pd.read_csv(merged_path)
    else:
        return _load_gold_labels_for_figures(out)
    if analysis.ID_COL not in labels.columns and "index" in labels.columns:
        labels = labels.rename(columns={"index": analysis.ID_COL})
    if analysis.ID_COL not in labels.columns:
        return _load_gold_labels_for_figures(out)
    labels[analysis.ID_COL] = labels[analysis.ID_COL].astype(str).str.replace(".pt", "", regex=False)
    if "research_stage" not in labels.columns:
        labels["research_stage"] = analysis._parse_stage(labels[analysis.ID_COL])
    if "participant_id" not in labels.columns:
        labels["participant_id"] = analysis._parse_participant_id(labels[analysis.ID_COL])
    labels["participant_id"] = labels["participant_id"].astype(str)
    gold_for_regression = analysis.load_gold_labels()
    if gold_for_regression is not None and analysis.ID_COL in gold_for_regression.columns:
        gold_extra = ["Mean", "GMI", "COGI", "bt__hemoglobin"]
        gold_numeric = [
            c
            for c in gold_extra
            if c in gold_for_regression.columns and pd.api.types.is_numeric_dtype(gold_for_regression[c])
        ]
        if gold_numeric:
            labels = labels.merge(
                gold_for_regression[[analysis.ID_COL] + gold_numeric].drop_duplicates(subset=[analysis.ID_COL]),
                on=analysis.ID_COL,
                how="left",
            )
    return labels


def _regression_baseline_cache_path(cache_dir: Path, target: str) -> Path | None:
    safe_target = analysis._safe_name_for_path(target)
    for pattern in (
        f"*__{safe_target}_mean_all_*_baseline_predictions.csv",
        f"*__{safe_target}_mean_1to8h_*_baseline_predictions.csv",
        f"*{safe_target}*baseline_predictions.csv",
    ):
        matches = sorted(cache_dir.glob(pattern))
        if matches:
            return matches[0]
    return None


def _delta_summary_by_quartile(joint: pd.DataFrame, value_col: str) -> pd.DataFrame | None:
    try:
        q = pd.qcut(joint[value_col], q=4, labels=["Q1", "Q2", "Q3", "Q4"])
    except ValueError:
        return None
    j = joint.assign(_q=q).dropna(subset=["_q"])
    rows = []
    for q_label in ("Q1", "Q2", "Q3", "Q4"):
        vals = j.loc[j["_q"] == q_label, "target_delta_2y"].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        n = len(vals)
        if n == 0:
            mean_d, sd_d, lo, hi = float("nan"), float("nan"), float("nan"), float("nan")
        elif n == 1:
            mean_d = float(vals[0])
            sd_d = 0.0
            lo, hi = mean_d, mean_d
        else:
            mean_d = float(np.mean(vals))
            sd_d = float(np.std(vals, ddof=1))
            lo, hi = mean_d - sd_d, mean_d + sd_d
        rows.append(
            {
                "quartile": q_label,
                "n": n,
                "mean_delta": mean_d,
                "sd_delta": sd_d,
                "band_low": lo,
                "band_high": hi,
            }
        )
    return pd.DataFrame(rows)


def _draw_delta_quartile_panel(ax, smeas: pd.DataFrame, spred: pd.DataFrame, title: str, y_label: str, n_participants: int) -> None:
    x = np.arange(1, 5)
    for summary_df, color, marker, leg in [
        (smeas, _DEMO_GRAY, "o", "Measured baseline quartiles"),
        (spred, _PULSEOX_TEAL, "s", "Predicted baseline quartiles"),
    ]:
        view = summary_df.set_index("quartile").reindex(["Q1", "Q2", "Q3", "Q4"])
        y = view["mean_delta"].to_numpy(dtype=float)
        lo = view["band_low"].to_numpy(dtype=float)
        hi = view["band_high"].to_numpy(dtype=float)
        ax.plot(x, y, marker=marker, linewidth=2.0, color=color, label=leg)
        ax.fill_between(x, lo, hi, color=color, alpha=0.22, linewidth=0)
    ax.axhline(0.0, color="#888888", linestyle="--", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(["Q1", "Q2", "Q3", "Q4"])
    ax.set_xlabel("Baseline quartile (by value)")
    ax.set_ylabel(y_label)
    ax.set_title(f"{title}\nParticipants with baseline + follow-up, N={n_participants}", fontsize=10)
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=8)


def plot_regression_delta_2y_by_baseline_quartile_bands(out: Path, cache_dir: Path, fig_dir: Path) -> None:
    """
    For each regression target: mean 2-year Δ (follow-up − baseline) vs quartiles of baseline measured value
    and vs quartiles of baseline predicted value. Bands = mean Δ ± 1 SD of participant-level deltas within each quartile.
    Writes one PNG/PDF per target under fig_dir and one multi-page PDF under out + fig_dir for the dashboard.
    """
    labels_df = _load_labels_for_regression_longitudinal_figures(out)
    if labels_df is None or labels_df.empty:
        print("Skipping Δ vs quartile bands: labels not found.")
        return
    regression_rows = _records(out / "regression_target_prediction_summary.csv")
    if not regression_rows:
        print("Skipping Δ vs quartile bands: regression_target_prediction_summary.csv missing or empty.")
        return
    participant_baseline_followup = analysis.get_participant_baseline_followup(labels_df)
    if participant_baseline_followup is None or participant_baseline_followup.empty:
        print("Skipping Δ vs quartile bands: no baseline–follow-up pairs.")
        return

    targets = sorted({str(r.get("target")) for r in regression_rows if r.get("target")})
    summary_rows_all = []
    combined_pdf_path = out / "regression_all_targets_delta_2y_by_baseline_quartile_bands.pdf"
    min_joint_n = 30
    figures_for_pdf = []
    for target_col in targets:
        if target_col not in labels_df.columns:
            continue
        if not pd.api.types.is_numeric_dtype(labels_df[target_col]):
            continue
        cache_path = _regression_baseline_cache_path(cache_dir, target_col)
        if cache_path is None or not cache_path.is_file():
            continue
        pred_df = pd.read_csv(cache_path)
        joint = analysis.joint_baseline_followup_delta_for_target(
            labels_df, participant_baseline_followup, target_col, pred_df
        )
        if joint is None or len(joint) < min_joint_n:
            continue
        smeas = _delta_summary_by_quartile(joint, "true_target")
        spred = _delta_summary_by_quartile(joint, "pred_target")
        if smeas is None or spred is None:
            print(f"Skipping Δ vs quartile bands for {target_col}: could not form quartiles.")
            continue
        disp = analysis._target_display_name(target_col)
        safe = analysis._safe_name_for_path(target_col)
        n_part = int(joint["participant_id"].nunique())
        for src, sdf in [("measured_baseline", smeas), ("predicted_baseline", spred)]:
            for _, row in sdf.iterrows():
                summary_rows_all.append(
                    {
                        "target": target_col,
                        "target_display": disp,
                        "stratifier": src,
                        "quartile": row["quartile"],
                        "n": row["n"],
                        "mean_delta_2y": row["mean_delta"],
                        "sd_delta_2y_within_quartile": row["sd_delta"],
                        "band_mean_minus_sd": row["band_low"],
                        "band_mean_plus_sd": row["band_high"],
                        "n_participants_analysis": n_part,
                    }
                )

        fig, ax = plt.subplots(figsize=(5.8, 4.2))
        _draw_delta_quartile_panel(
            ax,
            smeas,
            spred,
            f"{disp}: 2-year change by baseline quartile",
            analysis.DELTA_2Y_LABEL,
            n_part,
        )
        fig.tight_layout()
        for ext in ("png", "pdf"):
            single_path = fig_dir / f"{safe}_delta_2y_by_baseline_quartile_bands.{ext}"
            fig.savefig(single_path, dpi=300, bbox_inches="tight", format=ext)
            print(f"Saved {single_path}")
        figures_for_pdf.append(fig)

    if summary_rows_all:
        summary_path = fig_dir / "regression_delta_2y_baseline_quartile_summary.csv"
        pd.DataFrame(summary_rows_all).to_csv(summary_path, index=False)
        print(f"Saved {summary_path}")

    if not figures_for_pdf:
        print("No per-target Δ vs quartile band figures were produced (check caches and follow-up overlap).")
        return

    with PdfPages(combined_pdf_path) as pdf:
        for fig in figures_for_pdf:
            pdf.savefig(fig)
            plt.close(fig)

    n_written = len(figures_for_pdf)
    fig_copy = fig_dir / "regression_all_targets_delta_2y_by_baseline_quartile_bands.pdf"
    try:
        shutil.copy2(combined_pdf_path, fig_copy)
        print(f"Saved {combined_pdf_path} ({n_written} targets), copied to {fig_copy}")
    except OSError as e:
        print(f"Saved {combined_pdf_path} ({n_written} targets); copy to figures dir failed: {e}")


def plot_vitaldb_preop_figures_from_results(out: Path, fig_dir: Path) -> None:
    """
    Rebuild VitalDB preoperative forest/scatter/bar figures from vitaldb_preop_predictions.csv
    (written by target_prediction_evaluation_short._run_vitaldb_preop_prediction_section).
    Saves PDFs under fig_dir so make_target_prediction_figures stays self-contained.
    """
    csv_path = out / "vitaldb_preop_predictions.csv"
    if not csv_path.is_file():
        print(
            f"Skipping VitalDB preop figures: {csv_path} not found. "
            "Run target_prediction_evaluation_short with RUN_MODE=full or vitaldb_preop_only (same results folder)."
        )
        return
    df = pd.read_csv(csv_path)
    if df.empty:
        print("Skipping VitalDB preop figures: vitaldb_preop_predictions.csv is empty.")
        return
    for old in fig_dir.glob("*vitaldb_preop_darkred*.pdf"):
        try:
            old.unlink()
        except OSError:
            pass
    rows = df.to_dict("records")
    cls_rows = [r for r in rows if str(r.get("task_type", "")).strip().lower() == "classification"]
    reg_rows = [r for r in rows if str(r.get("task_type", "")).strip().lower() == "regression"]
    color = getattr(analysis, "COLOR_VITALDB_DARK_RED", "#c60c30")
    combined_label = "Age, sex, BMI + VitalDB embeddings"

    if cls_rows:
        analysis.plot_forest_auc_demo_vs_embeddings(
            cls_rows,
            out_path=str(fig_dir / "forest_AUC_vitaldb_preop_darkred.pdf"),
            combined_color=color,
            combined_label=combined_label,
        )
        analysis.plot_scatter_auc_demo_vs_embeddings(
            cls_rows,
            out_path=str(fig_dir / "scatter_AUC_vitaldb_preop_darkred.pdf"),
            combined_color=color,
        )
        analysis.plot_bar_auc_demo_vs_embeddings(
            cls_rows,
            out_path=str(fig_dir / "bar_AUC_vitaldb_preop_darkred.pdf"),
            combined_color=color,
        )
    else:
        print("VitalDB preop: no classification rows in CSV; skipping AUC figures.")

    if reg_rows:
        analysis.plot_forest_r2_demo_vs_embeddings(
            reg_rows,
            out_path=str(fig_dir / "forest_R2_vitaldb_preop_darkred.pdf"),
            combined_color=color,
            combined_label=combined_label,
        )
        analysis.plot_scatter_r2_demo_vs_embeddings(
            reg_rows,
            out_path=str(fig_dir / "scatter_R2_vitaldb_preop_darkred.pdf"),
            combined_color=color,
        )
        analysis.plot_bar_r_demo_vs_embeddings(
            reg_rows,
            out_path=str(fig_dir / "bar_R2_vitaldb_preop_darkred.pdf"),
            combined_color=color,
        )
    for pdf in sorted(fig_dir.glob("*vitaldb_preop_darkred*.pdf")):
        try:
            shutil.copy2(pdf, out / pdf.name)
        except OSError as e:
            print(f"VitalDB preop: could not copy {pdf.name} to {out}: {e}")


# ─── Residual analysis helpers ────────────────────────────────────────────────

def _compute_residual_per_participant(pred_df: pd.DataFrame, target_col: str) -> pd.DataFrame | None:
    """Per-participant embedding residual = combined_pred_mean − demo_pred_mean."""
    combined_col = f"{target_col}_pred_age_sex_bmi_embeddings_mean"
    demo_col = f"{target_col}_pred_age_sex_bmi_mean"
    if combined_col not in pred_df.columns or demo_col not in pred_df.columns:
        return None
    df = pred_df.copy()
    if "participant_id" not in df.columns:
        df["participant_id"] = analysis._parse_participant_id(df[analysis.ID_COL])
    df["participant_id"] = df["participant_id"].astype(str)
    df["_combined"] = pd.to_numeric(df[combined_col], errors="coerce")
    df["_demo"] = pd.to_numeric(df[demo_col], errors="coerce")
    df["_res"] = df["_combined"] - df["_demo"]
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    return df.dropna(subset=["_res"]).groupby("participant_id", as_index=False).agg(
        residual=("_res", "mean"),
        true_target=(target_col, "mean"),
    )


def _list_composite_regression_pairs(cache_dir: Path) -> list[tuple[Path, str]]:
    """
    (cache_path, target_col) for each regression baseline file whose outcome contributes to
    the pooled composite (same filters as _compute_composite_residual).
    """
    pairs: list[tuple[Path, str]] = []
    skip_cols = {"Recordings", "participant_id", "R2_age_sex_bmi", "R2_age_sex_bmi_embeddings",
                 "R_demo", "R_combined", "p_demo", "p_combined_vs_demo"}
    for cache_path in sorted(cache_dir.glob("*baseline_predictions.csv")):
        try:
            pred_df = pd.read_csv(cache_path)
        except Exception:
            continue
        target_col = next(
            (c for c in pred_df.columns if c not in skip_cols and "_pred" not in c and c != "Recordings"),
            None,
        )
        if target_col is None:
            continue
        combined_col = f"{target_col}_pred_age_sex_bmi_embeddings_mean"
        demo_col = f"{target_col}_pred_age_sex_bmi_mean"
        if combined_col not in pred_df.columns or demo_col not in pred_df.columns:
            continue
        if "participant_id" not in pred_df.columns:
            pred_df["participant_id"] = analysis._parse_participant_id(pred_df[analysis.ID_COL])
        pred_df["participant_id"] = pred_df["participant_id"].astype(str)
        pred_df["_res"] = (
            pd.to_numeric(pred_df[combined_col], errors="coerce")
            - pd.to_numeric(pred_df[demo_col], errors="coerce")
        )
        agg = pred_df.dropna(subset=["_res"]).groupby("participant_id", as_index=False)["_res"].mean()
        std = agg["_res"].std()
        if not (pd.notna(std) and std > 1e-9):
            continue
        pairs.append((cache_path, target_col))
    return pairs


def _compute_composite_residual(cache_dir: Path) -> pd.DataFrame:
    """
    Composite embedding residual per participant: mean of z-scored (combined−demo) residuals
    across all available regression baseline prediction caches.
    """
    all_residuals: dict[str, list[float]] = {}
    skip_cols = {"Recordings", "participant_id", "R2_age_sex_bmi", "R2_age_sex_bmi_embeddings",
                 "R_demo", "R_combined", "p_demo", "p_combined_vs_demo"}
    for cache_path in sorted(cache_dir.glob("*baseline_predictions.csv")):
        try:
            pred_df = pd.read_csv(cache_path)
        except Exception:
            continue
        target_col = next(
            (c for c in pred_df.columns if c not in skip_cols and "_pred" not in c and c != "Recordings"),
            None,
        )
        if target_col is None:
            continue
        combined_col = f"{target_col}_pred_age_sex_bmi_embeddings_mean"
        demo_col = f"{target_col}_pred_age_sex_bmi_mean"
        if combined_col not in pred_df.columns or demo_col not in pred_df.columns:
            continue
        if "participant_id" not in pred_df.columns:
            pred_df["participant_id"] = analysis._parse_participant_id(pred_df[analysis.ID_COL])
        pred_df["participant_id"] = pred_df["participant_id"].astype(str)
        pred_df["_res"] = (
            pd.to_numeric(pred_df[combined_col], errors="coerce")
            - pd.to_numeric(pred_df[demo_col], errors="coerce")
        )
        agg = pred_df.dropna(subset=["_res"]).groupby("participant_id", as_index=False)["_res"].mean()
        std = agg["_res"].std()
        if not (pd.notna(std) and std > 1e-9):
            continue
        agg["_res_z"] = (agg["_res"] - agg["_res"].mean()) / std
        for _, row in agg.iterrows():
            all_residuals.setdefault(str(row["participant_id"]), []).append(float(row["_res_z"]))
    if not all_residuals:
        return pd.DataFrame(columns=["participant_id", "composite_residual"])
    return pd.DataFrame([
        {"participant_id": pid, "composite_residual": float(np.mean(vals))}
        for pid, vals in all_residuals.items()
    ])


def _load_classification_residual(cache_dir: Path, target_col: str) -> pd.DataFrame | None:
    """
    Load a classification prediction cache and compute per-participant
    clf_residual = proba_combined_mean − proba_demo_mean.
    """
    safe = analysis._safe_name_for_path(target_col)
    matches: list[Path] = []
    for pool in ("mean_all", "mean_1to8h"):
        m = sorted(cache_dir.glob(f"classification_{safe}_{pool}_*.csv"))
        if m:
            matches = m
            break
    if not matches:
        return None
    try:
        df = pd.read_csv(matches[0])
    except Exception:
        return None
    if "proba_combined_mean" not in df.columns or "proba_demo_mean" not in df.columns:
        return None
    if target_col not in df.columns:
        return None
    if "participant_id" not in df.columns:
        df["participant_id"] = analysis._parse_participant_id(df[analysis.ID_COL])
    df["participant_id"] = df["participant_id"].astype(str)
    df["clf_residual"] = (
        pd.to_numeric(df["proba_combined_mean"], errors="coerce")
        - pd.to_numeric(df["proba_demo_mean"], errors="coerce")
    )
    return df.dropna(subset=["clf_residual"]).groupby("participant_id", as_index=False).agg(
        clf_residual=("clf_residual", "mean"),
        y_true=(target_col, "max"),
    )


def _classification_cache_csv_path(cache_dir: Path, target_col: str) -> Path | None:
    safe = analysis._safe_name_for_path(target_col)
    for pool in ("mean_all", "mean_1to8h"):
        matches = sorted(cache_dir.glob(f"classification_{safe}_{pool}_*.csv"))
        if matches:
            return matches[0]
    return None


def _cv_seeds_with_classification_prob_columns(df: pd.DataFrame) -> list[int]:
    seeds: list[int] = []
    for s in getattr(analysis, "CV_SEEDS", []) or []:
        c_demo = f"proba_demo_seed_{int(s)}"
        c_emb = f"proba_combined_seed_{int(s)}"
        if c_demo in df.columns and c_emb in df.columns:
            seeds.append(int(s))
    return seeds


def _classification_residual_by_seed_df(
    df: pd.DataFrame, target_col: str, seed: int, out_col: str
) -> pd.DataFrame | None:
    c_demo, c_emb = f"proba_demo_seed_{int(seed)}", f"proba_combined_seed_{int(seed)}"
    if c_demo not in df.columns or c_emb not in df.columns:
        return None
    if target_col not in df.columns:
        return None
    d = df.copy()
    if "participant_id" not in d.columns:
        d["participant_id"] = analysis._parse_participant_id(d[analysis.ID_COL])
    d["participant_id"] = d["participant_id"].astype(str)
    d[out_col] = pd.to_numeric(d[c_emb], errors="coerce") - pd.to_numeric(d[c_demo], errors="coerce")
    return d.dropna(subset=[out_col]).groupby("participant_id", as_index=False).agg(
        **{out_col: (out_col, "mean")},
    )


def _cv_seeds_with_full_composite_columns(pairs: list[tuple[Path, str]]) -> list[int]:
    """Seeds where every pooled-composite target has per-seed embedding and demo prediction columns."""
    if not pairs:
        return []
    valid: list[int] = []
    for s in getattr(analysis, "CV_SEEDS", []) or []:
        s_int = int(s)
        ok = True
        for path, tc in pairs:
            try:
                hdr = pd.read_csv(path, nrows=0)
            except Exception:
                ok = False
                break
            ce = f"{tc}_pred_age_sex_bmi_embeddings_seed_{s_int}"
            dm = f"{tc}_pred_age_sex_bmi_seed_{s_int}"
            if ce not in hdr.columns or dm not in hdr.columns:
                ok = False
                break
        if ok:
            valid.append(s_int)
    return valid


def _compute_composite_residual_for_seed(
    seed: int, pairs: list[tuple[Path, str]]
) -> pd.DataFrame:
    """
    Per-seed composite residual: same z-scored mean as _compute_composite_residual but using
    *_seed_{seed} columns. Caller must pass `pairs` from _list_composite_regression_pairs and
    ensure each (path, target) has those columns for this seed.
    """
    if not pairs:
        return pd.DataFrame(columns=["participant_id", "composite_residual"])
    all_residuals: dict[str, list[float]] = {}
    s_int = int(seed)
    for cache_path, target_col in pairs:
        try:
            pred_df = pd.read_csv(cache_path)
        except Exception:
            return pd.DataFrame(columns=["participant_id", "composite_residual"])
        combined_col = f"{target_col}_pred_age_sex_bmi_embeddings_seed_{s_int}"
        demo_col = f"{target_col}_pred_age_sex_bmi_seed_{s_int}"
        if combined_col not in pred_df.columns or demo_col not in pred_df.columns:
            return pd.DataFrame(columns=["participant_id", "composite_residual"])
        if "participant_id" not in pred_df.columns:
            pred_df["participant_id"] = analysis._parse_participant_id(pred_df[analysis.ID_COL])
        pred_df["participant_id"] = pred_df["participant_id"].astype(str)
        pred_df["_res"] = (
            pd.to_numeric(pred_df[combined_col], errors="coerce")
            - pd.to_numeric(pred_df[demo_col], errors="coerce")
        )
        agg = pred_df.dropna(subset=["_res"]).groupby("participant_id", as_index=False)["_res"].mean()
        std = agg["_res"].std()
        if not (pd.notna(std) and std > 1e-9):
            continue
        agg["_res_z"] = (agg["_res"] - agg["_res"].mean()) / std
        for _, row in agg.iterrows():
            all_residuals.setdefault(str(row["participant_id"]), []).append(float(row["_res_z"]))
    if not all_residuals:
        return pd.DataFrame(columns=["participant_id", "composite_residual"])
    return pd.DataFrame([
        {"participant_id": pid, "composite_residual": float(np.mean(vals))}
        for pid, vals in all_residuals.items()
    ])


def _aggregate_seed_incidence_summaries(
    summaries: list[pd.DataFrame],
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray] | None:
    """
    Align Q1–Q4 across seeds; return median summary df plus 25th/75th pct of incidence (%) per quartile.
    """
    if len(summaries) < 2:
        return None
    q_labels = ["Q1", "Q2", "Q3", "Q4"]
    pct_rows, case_rows, n_rows = [], [], []
    for sdf in summaries:
        view = sdf.set_index("quartile").reindex(q_labels)
        pct_rows.append(view["pct"].to_numpy(dtype=float))
        case_rows.append(view["cases"].to_numpy(dtype=float))
        n_rows.append(view["n"].to_numpy(dtype=float))
    pct_m = np.vstack(pct_rows)
    case_m = np.vstack(case_rows)
    n_mm = np.vstack(n_rows)
    median_pct = np.nanmedian(pct_m, axis=0)
    low = np.nanpercentile(pct_m, 25, axis=0)
    high = np.nanpercentile(pct_m, 75, axis=0)
    med_cases = np.rint(np.nanmedian(case_m, axis=0)).astype(int)
    med_n = np.rint(np.nanmedian(n_mm, axis=0)).astype(int)
    out = pd.DataFrame({
        "quartile": q_labels,
        "pct": median_pct,
        "cases": med_cases,
        "n": med_n,
    })
    return out, low, high


def _incidence_summary_by_quartile(
    joint: pd.DataFrame, stratifier_col: str, event_col: str, q: int = 4
) -> pd.DataFrame | None:
    """Incidence rate (%) and counts per quartile of stratifier_col for binary event_col."""
    q_labels = [f"Q{i + 1}" for i in range(q)]
    try:
        joint = joint.copy()
        joint["_q"] = pd.qcut(joint[stratifier_col], q=q, labels=q_labels, duplicates="drop")
    except ValueError:
        return None
    rows = []
    for q_label in q_labels:
        grp = joint[joint["_q"] == q_label]
        n = int(grp[event_col].notna().sum())
        cases = int(grp[event_col].sum()) if n > 0 else 0
        rows.append({"quartile": q_label, "n": n, "cases": cases,
                     "pct": float(cases / n * 100) if n > 0 else np.nan})
    return pd.DataFrame(rows)


def _draw_incidence_quartile_panel(
    ax, summary_df: pd.DataFrame, title: str, color: str, n_total: int, n_excluded: int,
    ylabel: str = "Incidence within 2 years (%)",
    font_pt: float = _FIG5_FONT_PT,
    band_pct_low: np.ndarray | None = None,
    band_pct_high: np.ndarray | None = None,
) -> None:
    """Single-panel incidence line plot across quartiles; optional CV-seed IQR band behind the line."""
    n_q = len(summary_df)
    x = np.arange(1, n_q + 1, dtype=float)
    y = summary_df["pct"].to_numpy(dtype=float)
    if (
        band_pct_low is not None
        and band_pct_high is not None
        and len(band_pct_low) == n_q
        and len(band_pct_high) == n_q
    ):
        ax.fill_between(
            x,
            band_pct_low,
            band_pct_high,
            color=color,
            alpha=0.2,
            linewidth=0,
            zorder=1,
        )
    ax.plot(x, y, marker="D", linewidth=2.0, color=color, zorder=2)
    for xi, yi, cases, n in zip(x, y, summary_df["cases"], summary_df["n"]):
        if np.isfinite(yi):
            ax.text(
                xi,
                yi + max(0.26, yi * 0.032),
                f"{int(cases)}/{int(n)}",
                ha="center",
                va="bottom",
                fontsize=font_pt,
                color=color,
                clip_on=False,
                zorder=3,
            )
    ax.set_xticks(x)
    ax.set_xticklabels(summary_df["quartile"].tolist(), fontsize=font_pt)
    ax.set_xlabel("Embedding residual quartile (Q1=low, Q4=high)", fontsize=font_pt)
    ax.set_ylabel(ylabel, fontsize=font_pt)
    ymax_line = float(np.nanmax(y)) if np.any(np.isfinite(y)) else 5.0
    ymax = ymax_line
    if band_pct_high is not None and np.any(np.isfinite(band_pct_high)):
        ymax = max(ymax, float(np.nanmax(band_pct_high)))
    ax.set_ylim(0, max(2.0, ymax * 1.38))
    ax.set_xlim(0.82, float(n_q) + 0.18)
    ax.set_title(f"{title}\nN={n_total} (excl. {n_excluded} baseline cases)", fontsize=font_pt)
    ax.tick_params(axis="both", labelsize=font_pt)
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ─── Section 2: Residual quartiles → Δ 2y ────────────────────────────────────

def plot_residual_delta_2y_by_quartile_bands(out: Path, cache_dir: Path, fig_dir: Path) -> list[Path]:
    """
    For each regression target: mean 2-year Δ vs quartiles of the embedding residual
    (combined_pred − demo_pred). Saves per-target PNG/PDF and a multi-page PDF.
    Returns list of saved PNG paths for dashboard assembly.
    """
    labels_df = _load_labels_for_regression_longitudinal_figures(out)
    if labels_df is None or labels_df.empty:
        print("Skipping residual Δ vs quartile bands: labels not found.")
        return []
    regression_rows = _records(out / "regression_target_prediction_summary.csv")
    if not regression_rows:
        print("Skipping residual Δ vs quartile bands: regression_target_prediction_summary.csv not found.")
        return []
    participant_baseline_followup = analysis.get_participant_baseline_followup(labels_df)
    if participant_baseline_followup is None or participant_baseline_followup.empty:
        print("Skipping residual Δ vs quartile bands: no baseline–follow-up pairs.")
        return []

    targets = sorted({str(r.get("target")) for r in regression_rows if r.get("target")})
    summary_rows_all = []
    figures_for_pdf: list[plt.Figure] = []
    saved_pngs: list[Path] = []
    min_joint_n = 30

    for target_col in targets:
        if target_col not in labels_df.columns:
            continue
        if not pd.api.types.is_numeric_dtype(labels_df[target_col]):
            continue
        cache_path = _regression_baseline_cache_path(cache_dir, target_col)
        if cache_path is None or not cache_path.is_file():
            continue
        pred_df = pd.read_csv(cache_path)

        residual_df = _compute_residual_per_participant(pred_df, target_col)
        if residual_df is None or residual_df.empty:
            continue

        pred_col = f"{target_col}_pred"
        if pred_col not in pred_df.columns:
            if f"{target_col}_pred_mean" in pred_df.columns:
                pred_df[pred_col] = pred_df[f"{target_col}_pred_mean"]
            else:
                continue
        joint_base = analysis.joint_baseline_followup_delta_for_target(
            labels_df, participant_baseline_followup, target_col, pred_df
        )
        if joint_base is None or len(joint_base) < min_joint_n:
            continue

        joint = joint_base.merge(residual_df[["participant_id", "residual"]], on="participant_id", how="inner")
        if len(joint) < min_joint_n:
            continue

        # Reuse _delta_summary_by_quartile with residual as stratifier
        joint_r = joint.copy()
        joint_r["true_target"] = joint_r["residual"]
        sresid = _delta_summary_by_quartile(joint_r, "true_target")
        if sresid is None:
            continue

        disp = analysis._target_display_name(target_col)
        safe = analysis._safe_name_for_path(target_col)
        n_part = int(joint["participant_id"].nunique())

        for _, row in sresid.iterrows():
            summary_rows_all.append({
                "target": target_col,
                "target_display": disp,
                "stratifier": "embedding_residual",
                "quartile": row["quartile"],
                "n": row["n"],
                "mean_delta_2y": row["mean_delta"],
                "sd_delta_2y_within_quartile": row["sd_delta"],
                "n_participants_analysis": n_part,
            })

        fig, ax = plt.subplots(figsize=(5.8, 4.2))
        x = np.arange(1, 5)
        view = sresid.set_index("quartile").reindex(["Q1", "Q2", "Q3", "Q4"])
        y_v = view["mean_delta"].to_numpy(dtype=float)
        lo = view["band_low"].to_numpy(dtype=float)
        hi = view["band_high"].to_numpy(dtype=float)
        ax.plot(x, y_v, marker="D", linewidth=2.0, color=_PULSEOX_TEAL,
                label="Embedding residual quartiles")
        ax.fill_between(x, lo, hi, color=_PULSEOX_TEAL, alpha=0.22, linewidth=0)
        ax.axhline(0.0, color="#888888", linestyle="--", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(["Q1", "Q2", "Q3", "Q4"])
        ax.set_xlabel("Baseline embedding residual quartile (Q1=low, Q4=high)")
        ax.set_ylabel(analysis.DELTA_2Y_LABEL)
        ax.set_title(f"{disp}: 2-year change by embedding residual quartile\nN={n_part}", fontsize=10)
        ax.grid(axis="y", alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()

        for ext in ("png", "pdf"):
            p = fig_dir / f"{safe}_residual_delta_2y_quartile_bands.{ext}"
            fig.savefig(p, dpi=300, bbox_inches="tight", format=ext)
            if ext == "png":
                saved_pngs.append(p)
            print(f"Saved {p}")
        figures_for_pdf.append(fig)

    if summary_rows_all:
        summary_path = fig_dir / "residual_delta_2y_quartile_summary.csv"
        pd.DataFrame(summary_rows_all).to_csv(summary_path, index=False)
        print(f"Saved {summary_path}")

    if figures_for_pdf:
        combined_pdf = fig_dir / "residual_all_targets_delta_2y_quartile_bands.pdf"
        with PdfPages(combined_pdf) as pdf:
            for fig in figures_for_pdf:
                pdf.savefig(fig)
                plt.close(fig)
        print(f"Saved {combined_pdf} ({len(figures_for_pdf)} targets)")
    else:
        for fig in figures_for_pdf:
            plt.close(fig)
        print("No residual Δ vs quartile band figures produced (check caches).")

    return saved_pngs


# ─── Section 3: Residual quartiles → disease incidence ────────────────────────

def _load_gold_incidence_data(out: Path) -> pd.DataFrame | None:
    """Load gold records with incidence columns and participant/stage IDs."""
    gold_paths = [
        Path(getattr(analysis, "GOLD_RECORDS_PATH", "")),
        out / "recordings_with_labels_gold.csv",
        Path("/home/sarahk/JafarShortcut/SleepFM/ssl_sleep/tabular_data/recordings_with_labels_gold.csv"),
    ]
    for p in gold_paths:
        if not p.is_file():
            continue
        df = pd.read_csv(p)
        if analysis.ID_COL not in df.columns and "index" in df.columns:
            df = df.rename(columns={"index": analysis.ID_COL})
        if analysis.ID_COL not in df.columns:
            continue
        df[analysis.ID_COL] = df[analysis.ID_COL].astype(str).str.replace(".pt", "", regex=False)
        if "participant_id" not in df.columns:
            df["participant_id"] = analysis._parse_participant_id(df[analysis.ID_COL])
        if "research_stage" not in df.columns:
            df["research_stage"] = analysis._parse_stage(df[analysis.ID_COL])
        df["participant_id"] = df["participant_id"].astype(str)
        return df
    return None


def _baseline_rows(gold_df: pd.DataFrame, participant_baseline_followup: pd.DataFrame) -> pd.DataFrame:
    """Return one row per participant at baseline stage, aggregating labels by max."""
    lab = gold_df.merge(participant_baseline_followup, on="participant_id", how="inner")
    lab_b = lab[lab["research_stage"] == lab["baseline_stage"]].copy()
    binary_cols = [c for c in lab_b.columns if set(lab_b[c].dropna().unique()).issubset({0, 1, 0.0, 1.0})]
    agg_dict = {c: "max" for c in binary_cols}
    return lab_b.groupby("participant_id", as_index=False).agg(agg_dict)


def _incidence_to_baseline_condition_col(incidence_col: str) -> str:
    """Map '<condition>_incidence_2yrs' back to '<condition>' when possible."""
    name = str(incidence_col)
    suffix = "_incidence_2yrs"
    if name.endswith(suffix):
        return name[: -len(suffix)]
    return name


def _build_stage_resolved_disease_status(
    gold_df: pd.DataFrame,
    participant_baseline_followup: pd.DataFrame,
    condition_cols: list[str],
) -> pd.DataFrame:
    """
    Build participant-level baseline/+2y disease status from stage-resolved diagnosis labels.
    Output columns:
      - participant_id
      - <condition>__baseline
      - <condition>__2y
    """
    if gold_df is None or gold_df.empty or participant_baseline_followup is None or participant_baseline_followup.empty:
        return pd.DataFrame(columns=["participant_id"])
    stages = participant_baseline_followup[["participant_id", "baseline_stage"]].copy()
    stages["participant_id"] = stages["participant_id"].astype(str)
    stages["stage_2y"] = stages["baseline_stage"].apply(lambda x: analysis._stage_after_years(x, 2))
    out = stages[["participant_id"]].drop_duplicates().copy()
    for condition in sorted(set(str(c) for c in condition_cols if c in gold_df.columns)):
        dx_b = analysis._diagnosis_by_participant_stage(
            gold_df, condition, stages, "baseline_stage"
        ).rename(columns={"baseline_stage": f"{condition}__baseline"})
        dx_2y = analysis._diagnosis_by_participant_stage(
            gold_df, condition, stages, "stage_2y"
        ).rename(columns={"stage_2y": f"{condition}__2y"})
        out = out.merge(dx_b, on="participant_id", how="left")
        out = out.merge(dx_2y, on="participant_id", how="left")
    return out


def _panel_disease_incidence(
    ax,
    disease_status_df: pd.DataFrame,
    residual_df: pd.DataFrame,
    baseline_col: str,
    incidence_col: str,
    stratifier_col: str,
    disease_label: str,
    residual_label: str,
    color: str = _PULSEOX_TEAL,
    seed_bands: dict | None = None,
    font_pt: float = _FIG5_FONT_PT,
) -> dict | None:
    """
    Filter to non-diseased at baseline, join with residual, compute incidence by quartile, draw panel.
    seed_bands: optional {"kind": "composite", "pairs": [...]} or
      {"kind": "classification", "full_df": df, "target_col": str}
      for per-seed qcut + median line and IQR band when >= 2 CV seeds qualify.
    Returns a dict with summary stats or None if insufficient data.
    """
    if baseline_col not in disease_status_df.columns or incidence_col not in disease_status_df.columns:
        return None

    def _joint_ready(joint: pd.DataFrame) -> pd.DataFrame | None:
        j = joint.dropna(subset=[incidence_col, stratifier_col])
        if len(j) < 30 or j[incidence_col].sum() < 5:
            return None
        return j

    joint = disease_status_df[disease_status_df[baseline_col] == 0].copy()
    n_excluded = int((disease_status_df[baseline_col] == 1).sum())
    joint = joint.merge(residual_df[["participant_id", stratifier_col]], on="participant_id", how="inner")
    joint_ready = _joint_ready(joint)
    if joint_ready is None:
        return None
    summary = _incidence_summary_by_quartile(joint_ready, stratifier_col, incidence_col)
    if summary is None:
        return None

    band_low: np.ndarray | None = None
    band_high: np.ndarray | None = None
    summary_draw = summary
    used_seed_bands = False

    if seed_bands is not None:
        summaries_seed: list[pd.DataFrame] = []
        if seed_bands.get("kind") == "composite":
            pairs = seed_bands.get("pairs") or []
            seeds = _cv_seeds_with_full_composite_columns(pairs)
            if len(seeds) >= 2:
                for s in seeds:
                    cr = _compute_composite_residual_for_seed(s, pairs)
                    if cr.empty:
                        continue
                    j = disease_status_df[disease_status_df[baseline_col] == 0].copy()
                    j = j.merge(cr, on="participant_id", how="inner")
                    jr = _joint_ready(j)
                    if jr is None:
                        continue
                    sm = _incidence_summary_by_quartile(jr, "composite_residual", incidence_col)
                    if sm is not None:
                        summaries_seed.append(sm)
        elif seed_bands.get("kind") == "classification":
            df_full = seed_bands.get("full_df")
            tgt = seed_bands.get("target_col")
            if isinstance(df_full, pd.DataFrame) and isinstance(tgt, str):
                seeds = _cv_seeds_with_classification_prob_columns(df_full)
                if len(seeds) >= 2:
                    for s in seeds:
                        rs = _classification_residual_by_seed_df(df_full, tgt, s, stratifier_col)
                        if rs is None:
                            continue
                        j = disease_status_df[disease_status_df[baseline_col] == 0].copy()
                        j = j.merge(rs, on="participant_id", how="inner")
                        jr = _joint_ready(j)
                        if jr is None:
                            continue
                        sm = _incidence_summary_by_quartile(jr, stratifier_col, incidence_col)
                        if sm is not None:
                            summaries_seed.append(sm)

        agg = _aggregate_seed_incidence_summaries(summaries_seed)
        if agg is not None:
            summary_draw, band_low, band_high = agg
            used_seed_bands = True

    _draw_incidence_quartile_panel(
        ax,
        summary_draw,
        f"{disease_label} within 2 years",
        color,
        n_total=int(joint_ready["participant_id"].nunique()),
        n_excluded=n_excluded,
        font_pt=font_pt,
        band_pct_low=band_low,
        band_pct_high=band_high,
    )
    out_summary = summary_draw if used_seed_bands else summary
    return {
        "disease": disease_label,
        "n_total": len(joint_ready),
        "n_excluded": n_excluded,
        "summary": out_summary.to_dict("records"),
        "used_seed_bands": used_seed_bands,
    }


def plot_residual_quartile_disease_incidence_lines(
    out: Path, cache_dir: Path, fig_dir: Path
) -> list[Path]:
    """
    Line plots of 2-year disease incidence across embedding residual quartiles.

    Diseases:
      - Hypertension incidence: classification residual from Hypertension_incidence_2yrs cache
      - Prediabetes incidence: classification residual from Prediabetes cache, outcome from gold
      - Grouped disease categories: composite regression residual
    Baseline cases are excluded for each disease/category.
    """
    gold_df = _load_gold_incidence_data(out)
    if gold_df is None or gold_df.empty:
        print("Skipping residual→disease incidence: gold labels not found.")
        return []

    labels_df = _load_labels_for_regression_longitudinal_figures(out)
    pfup = analysis.get_participant_baseline_followup(labels_df if labels_df is not None else gold_df)
    if pfup is None or pfup.empty:
        print("Skipping residual→disease incidence: no baseline–follow-up pairs.")
        return []

    gold_baseline = _baseline_rows(gold_df, pfup)
    incidence_cols = [c for c in gold_baseline.columns if "incidence" in c.lower()]
    if not incidence_cols:
        print("Skipping residual→disease incidence: no incidence columns in gold labels.")
        return []
    condition_cols = [
        _incidence_to_baseline_condition_col(c)
        for c in incidence_cols
        if _incidence_to_baseline_condition_col(c) in gold_df.columns
    ]
    disease_status_df = _build_stage_resolved_disease_status(gold_df, pfup, condition_cols)
    if disease_status_df.empty:
        print("Skipping residual→disease incidence: failed to build stage-resolved disease labels.")
        return []

    # ── Build residual sources ──────────────────────────────────────────────
    # Classification residual: proba_combined - proba_demo for specific disease caches
    htn_residual = _load_classification_residual(cache_dir, "Hypertension_incidence_2yrs")
    prediab_residual = _load_classification_residual(cache_dir, "Prediabetes")
    if prediab_residual is not None:
        prediab_residual = prediab_residual.rename(columns={"clf_residual": "clf_residual_prediabetes"})

    # Composite regression residual for disease categories
    print("Computing composite embedding residual across all regression targets...")
    composite_residual = _compute_composite_residual(cache_dir)
    if composite_residual.empty:
        print("Composite residual is empty; disease category incidence plots will be skipped.")

    # ── Disease category mapping ────────────────────────────────────────────
    try:
        disease_cat_map = analysis._normalized_disease_category_map()
    except Exception:
        disease_cat_map = {}
    categories: dict[str, list[str]] = {}
    for cond, cat in disease_cat_map.items():
        matched = [c for c in incidence_cols if cond.lower() in c.lower()]
        for m in matched:
            categories.setdefault(cat, []).append(m)

    # ── Assemble panels ─────────────────────────────────────────────────────
    # Each entry: (ax_title, baseline_col, incidence_col, stratifier_col, residual_df, label)
    panel_specs = []

    if htn_residual is not None:
        htn_res = htn_residual.rename(columns={"clf_residual": "clf_residual_htn"})
        panel_specs.append(("Hypertension incidence",
                             "Hypertension__baseline", "Hypertension__2y",
                             "clf_residual_htn", htn_res,
                             "Classification residual\n(hypertension incidence model)"))

    if prediab_residual is not None:
        panel_specs.append(("Prediabetes incidence",
                             "Prediabetes__baseline", "Prediabetes__2y",
                             "clf_residual_prediabetes", prediab_residual,
                             "Classification residual\n(prediabetes model)"))

    if not composite_residual.empty:
        for cat, cols in sorted(categories.items()):
            cols = [c for c in cols if c in gold_baseline.columns]
            if not cols:
                continue
            cond_cols = sorted(
                set(
                    _incidence_to_baseline_condition_col(c)
                    for c in cols
                    if f"{_incidence_to_baseline_condition_col(c)}__baseline" in disease_status_df.columns
                    and f"{_incidence_to_baseline_condition_col(c)}__2y" in disease_status_df.columns
                )
            )
            if not cond_cols:
                continue
            baseline_src_cols = [f"{c}__baseline" for c in cond_cols]
            event_src_cols = [f"{c}__2y" for c in cond_cols]
            baseline_col_cat = f"_baseline_{cat}"
            incidence_col_cat = f"_incidence_{cat.replace(' ', '_')}"
            disease_status_df[baseline_col_cat] = disease_status_df[baseline_src_cols].max(axis=1, skipna=True)
            disease_status_df[incidence_col_cat] = disease_status_df[event_src_cols].max(axis=1, skipna=True)
            panel_specs.append((f"{cat} incidence",
                                 baseline_col_cat, incidence_col_cat,
                                 "composite_residual", composite_residual,
                                 "Composite embedding residual\n(all regression targets)"))

    if not panel_specs:
        print("Skipping residual→disease incidence: no valid panel specifications.")
        return []

    n_panels = len(panel_specs)
    n_cols = min(3, n_panels)
    n_rows = math.ceil(n_panels / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.5 * n_cols, 4.5 * n_rows), squeeze=False)

    summary_rows_all = []
    saved_pngs: list[Path] = []

    for idx, (disease_label, baseline_col, incidence_col, stratifier_col, res_df, res_label) in enumerate(panel_specs):
        ax = axes[idx // n_cols][idx % n_cols]
        result = _panel_disease_incidence(
            ax, disease_status_df, res_df, baseline_col, incidence_col,
            stratifier_col, disease_label, res_label,
        )
        if result is None:
            ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=10, color="gray")
            ax.set_title(f"{disease_label} within 2 years", fontsize=9)
            ax.axis("off")
        else:
            for row in result["summary"]:
                summary_rows_all.append({"disease": disease_label, "residual_type": res_label.split("\n")[0], **row})

    for ax in axes.ravel()[n_panels:]:
        ax.axis("off")

    fig.suptitle(
        "Disease incidence within 2 years by embedding residual quartile\n"
        "(residual = embedding contribution beyond age, sex, BMI)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    for ext in ("png", "pdf"):
        p = fig_dir / f"residual_quartile_disease_incidence_lines.{ext}"
        fig.savefig(p, dpi=300, bbox_inches="tight", format=ext)
        if ext == "png":
            saved_pngs.append(p)
        print(f"Saved {p}")
    plt.close(fig)

    if summary_rows_all:
        csv_path = fig_dir / "residual_quartile_disease_incidence_summary.csv"
        pd.DataFrame(summary_rows_all).to_csv(csv_path, index=False)
        print(f"Saved {csv_path}")

    return saved_pngs


def plot_residual_quartile_per_disease_panels(
    out: Path, cache_dir: Path, fig_dir: Path
) -> list[Path]:
    """
    Save individual single-panel PNGs for cardiovascular (composite residual),
    prediabetes and hypertension residual quartile incidence, for manuscript Figure 5.
    Outputs:
      residual_quartile_cardiovascular_incidence_panel.png
      residual_quartile_prediabetes_incidence_panel.png
      residual_quartile_hypertension_incidence_panel.png
    """
    gold_df = _load_gold_incidence_data(out)
    if gold_df is None or gold_df.empty:
        print("Skipping per-disease panels: gold labels not found.")
        return []

    labels_df = _load_labels_for_regression_longitudinal_figures(out)
    pfup = analysis.get_participant_baseline_followup(labels_df if labels_df is not None else gold_df)
    if pfup is None or pfup.empty:
        print("Skipping per-disease panels: no baseline–follow-up pairs.")
        return []

    gold_baseline = _baseline_rows(gold_df, pfup)
    incidence_cols = [c for c in gold_baseline.columns if "incidence" in c.lower()]
    condition_cols = [
        _incidence_to_baseline_condition_col(c)
        for c in incidence_cols
        if _incidence_to_baseline_condition_col(c) in gold_df.columns
    ]
    disease_status_df = _build_stage_resolved_disease_status(gold_df, pfup, condition_cols)
    if disease_status_df.empty:
        print("Skipping per-disease panels: failed to build stage-resolved disease labels.")
        return []

    htn_residual = _load_classification_residual(cache_dir, "Hypertension_incidence_2yrs")
    prediab_residual = _load_classification_residual(cache_dir, "Prediabetes")
    if prediab_residual is not None:
        prediab_residual = prediab_residual.rename(columns={"clf_residual": "clf_residual_prediabetes"})

    composite_pairs = _list_composite_regression_pairs(cache_dir)
    seed_bands_composite = (
        {"kind": "composite", "pairs": composite_pairs}
        if len(_cv_seeds_with_full_composite_columns(composite_pairs)) >= 2
        else None
    )

    prediab_clf_path = _classification_cache_csv_path(cache_dir, "Prediabetes")
    prediab_full_df = (
        pd.read_csv(prediab_clf_path) if prediab_clf_path and prediab_clf_path.is_file() else None
    )
    seed_bands_prediabetes = None
    if prediab_full_df is not None and len(_cv_seeds_with_classification_prob_columns(prediab_full_df)) >= 2:
        seed_bands_prediabetes = {
            "kind": "classification",
            "full_df": prediab_full_df,
            "target_col": "Prediabetes",
        }

    htn_clf_path = _classification_cache_csv_path(cache_dir, "Hypertension_incidence_2yrs")
    htn_full_df = pd.read_csv(htn_clf_path) if htn_clf_path and htn_clf_path.is_file() else None
    seed_bands_htn = None
    if htn_full_df is not None and len(_cv_seeds_with_classification_prob_columns(htn_full_df)) >= 2:
        seed_bands_htn = {
            "kind": "classification",
            "full_df": htn_full_df,
            "target_col": "Hypertension_incidence_2yrs",
        }

    specs: list[tuple] = []

    cardio_cat = "Cardiovascular disorders"
    composite_residual = _compute_composite_residual(cache_dir)
    if not composite_residual.empty and incidence_cols:
        try:
            disease_cat_map = analysis._normalized_disease_category_map()
        except Exception:
            disease_cat_map = {}
        categories: dict[str, list[str]] = {}
        for cond, cat in disease_cat_map.items():
            matched = [c for c in incidence_cols if cond.lower() in c.lower()]
            for m in matched:
                categories.setdefault(cat, []).append(m)
        cols = [c for c in categories.get(cardio_cat, []) if c in gold_baseline.columns]
        if cols:
            cond_cols = sorted(
                set(
                    _incidence_to_baseline_condition_col(c)
                    for c in cols
                    if f"{_incidence_to_baseline_condition_col(c)}__baseline" in disease_status_df.columns
                    and f"{_incidence_to_baseline_condition_col(c)}__2y" in disease_status_df.columns
                )
            )
            if not cond_cols:
                cond_cols = []
            incidence_col_cat = f"_incidence_{cardio_cat.replace(' ', '_')}"
            baseline_col_cat = f"_baseline_{cardio_cat}"
            baseline_src_cols = [f"{c}__baseline" for c in cond_cols]
            event_src_cols = [f"{c}__2y" for c in cond_cols]
            if baseline_src_cols and event_src_cols:
                disease_status_df[incidence_col_cat] = disease_status_df[event_src_cols].max(axis=1, skipna=True)
                disease_status_df[baseline_col_cat] = disease_status_df[baseline_src_cols].max(axis=1, skipna=True)
            else:
                disease_status_df[incidence_col_cat] = np.nan
                disease_status_df[baseline_col_cat] = np.nan
            specs.append((
                "residual_quartile_cardiovascular_incidence_panel",
                f"{cardio_cat} incidence",
                baseline_col_cat,
                incidence_col_cat,
                "composite_residual",
                composite_residual,
                "Composite embedding residual\n(all regression targets)",
                seed_bands_composite,
            ))

    if prediab_residual is not None:
        specs.append((
            "residual_quartile_prediabetes_incidence_panel",
            "Prediabetes incidence",
            "Prediabetes__baseline", "Prediabetes__2y",
            "clf_residual_prediabetes", prediab_residual,
            "Classification residual\n(prediabetes model)",
            seed_bands_prediabetes,
        ))
    if htn_residual is not None:
        htn_res = htn_residual.rename(columns={"clf_residual": "clf_residual_htn"})
        specs.append((
            "residual_quartile_hypertension_incidence_panel",
            "Hypertension incidence",
            "Hypertension__baseline", "Hypertension__2y",
            "clf_residual_htn", htn_res,
            "Classification residual\n(hypertension incidence model)",
            seed_bands_htn,
        ))

    saved: list[Path] = []
    for (stem, disease_label, baseline_col, incidence_col, stratifier_col, res_df, res_label, seed_bands) in specs:
        fig, ax = plt.subplots(1, 1, figsize=(5.5, 4.5))
        result = _panel_disease_incidence(
            ax, disease_status_df, res_df, baseline_col, incidence_col,
            stratifier_col, disease_label, res_label, seed_bands=seed_bands,
            font_pt=_FIG5_FONT_PT,
        )
        if result is None:
            ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=_FIG5_FONT_PT, color="gray")
            ax.set_title(f"{disease_label} within 2 years", fontsize=_FIG5_FONT_PT)
            ax.axis("off")
        fig.subplots_adjust(left=0.16, right=0.96, bottom=0.16, top=0.92)
        p = fig_dir / f"{stem}.png"
        fig.savefig(p, dpi=300, bbox_inches="tight", pad_inches=0.28, format="png")
        fig.savefig(p.with_suffix(".pdf"), format="pdf", bbox_inches="tight", pad_inches=0.28)
        plt.close(fig)
        saved.append(p)
        print(f"Saved {p} and matching PDF.")

    return saved


# ─── Dashboard: quartile longitudinal analyses ────────────────────────────────

def make_quartile_longitudinal_dashboard(out: Path, fig_dir: Path) -> None:
    """
    Build 'quartile_longitudinal_analyses.html' combining all quartile-based
    longitudinal figures into four sections.
    """
    from refactored_io import render_dashboard

    def _fig_entries(pattern: str, caption_fn=None) -> list[dict]:
        paths = sorted(fig_dir.glob(pattern))
        entries = []
        for p in paths:
            if p.suffix.lower() != ".png":
                continue
            cap = caption_fn(p.stem) if caption_fn else p.stem.replace("_", " ")
            entries.append({"path": str(p), "caption": cap})
        return entries

    def _csv_entries(pattern: str) -> list[dict]:
        return [{"path": str(p), "caption": p.name} for p in sorted(fig_dir.glob(pattern))]

    # Section 1: true/predicted quartiles → delta 2y
    sec1_figs = _fig_entries(
        "*_delta_2y_by_baseline_quartile_bands.png",
        lambda s: s.replace("_delta_2y_by_baseline_quartile_bands", "").replace("_", " "),
    )
    sec1_tables = _csv_entries("regression_delta_2y_baseline_quartile_summary.csv")

    # Section 2: residual quartiles → delta 2y
    sec2_figs = _fig_entries(
        "*_residual_delta_2y_quartile_bands.png",
        lambda s: s.replace("_residual_delta_2y_quartile_bands", "").replace("_", " "),
    )
    sec2_tables = _csv_entries("residual_delta_2y_quartile_summary.csv")

    # Section 3: residual quartiles → disease incidence
    sec3_figs = _fig_entries("residual_quartile_disease_incidence_lines.png")
    sec3_tables = _csv_entries("residual_quartile_disease_incidence_summary.csv")

    # Section 4: BP true/predicted quartiles → hypertension incidence
    sec4_figs = _fig_entries(
        "*_bp_hypertension_incidence_quartile_lines.png",
        lambda s: s.replace("_bp_hypertension_incidence_quartile_lines", "").replace("_", " "),
    )
    # Also include age quartile plot
    sec4_figs += _fig_entries(
        "*_baseline_quartile_hypertension_incidence_lines.png",
        lambda s: s.replace("_baseline_quartile_hypertension_incidence_lines", "").replace("_", " "),
    )
    sec4_tables = (
        _csv_entries("*_bp_hypertension_incidence_quartile_line_summary.csv")
        + _csv_entries("Age_baseline_quartile_hypertension_incidence_line_summary.csv")
    )

    sections = [
        {
            "title": "True vs predicted measures quartiles at baseline — correlations with Δ after 2 years",
            "summary": (
                "For each regression target, participants are stratified by quartile of their "
                "measured (grey) or embedding-predicted (teal) baseline value. Lines show mean "
                "2-year change ± 1 SD within each quartile."
            ),
            "figures": sec1_figs,
            "tables": sec1_tables,
        },
        {
            "title": "Embedding residual quartiles at baseline — correlations with Δ after 2 years",
            "summary": (
                "Embedding residual = combined model prediction − demographic model prediction "
                "(i.e. the incremental signal from sleep embeddings beyond age/sex/BMI). "
                "Participants are stratified by quartile of this residual; lines show mean 2-year "
                "change ± 1 SD."
            ),
            "figures": sec2_figs,
            "tables": sec2_tables,
        },
        {
            "title": "Embedding residual quartiles at baseline — correlations with disease incidence after 2 years",
            "summary": (
                "For each disease/category, participants with a baseline diagnosis are excluded. "
                "Remaining participants are stratified by embedding residual quartile. "
                "Lines show 2-year incidence (%) per quartile. "
                "Hypertension/Prediabetes use their respective classification model residuals; "
                "disease categories use a composite residual across all regression targets."
            ),
            "figures": sec3_figs,
            "tables": sec3_tables,
        },
        {
            "title": "True vs predicted BP quartiles at baseline — hypertension incidence after 2 years",
            "summary": (
                "Participants without hypertension at baseline are stratified by quartile of "
                "measured vs. predicted systolic/diastolic blood pressure. Lines show 2-year "
                "hypertension incidence (%) per quartile. Annotations show new cases / at-risk N."
            ),
            "figures": sec4_figs,
            "tables": sec4_tables,
        },
    ]

    dashboard_path = fig_dir / "quartile_longitudinal_analyses.html"
    render_dashboard("Quartile Longitudinal Analyses — Target Prediction", dashboard_path, sections)
    print(f"Saved dashboard: {dashboard_path}")


# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    out = results_dir("target_prediction")
    fig_dir = figures_dir("target_prediction")
    cache_dir = out / "cache_dir"
    analysis.OUTPUT_DIR = str(out)
    analysis.CACHE_DIR = str(cache_dir)

    # Copy key summary CSVs from results dir to figures dir so everything is co-located
    for csv_name in (
        "regression_target_prediction_summary.csv",
        "classification_binary_targets_AUC.csv",
        "classification_binary_medications_AUC.csv",
        "grouped_incidence_or_results.csv",
        "vitaldb_preop_predictions.csv",
        "regression_delta_2y_baseline_quartile_summary.csv",
        "regression_all_targets_delta_2y_by_baseline_quartile_bands.pdf",
    ):
        src = out / csv_name
        if src.is_file():
            dest = fig_dir / csv_name
            if not dest.exists() or src.stat().st_mtime > dest.stat().st_mtime:
                shutil.copy2(src, dest)

    # Copy BP incidence CSVs
    for bp_csv in out.glob("*_bp_hypertension_incidence_*.csv"):
        dest = fig_dir / bp_csv.name
        if not dest.exists() or bp_csv.stat().st_mtime > dest.stat().st_mtime:
            shutil.copy2(bp_csv, dest)

    regression = _records(out / "regression_target_prediction_summary.csv")
    disease = _records(out / "classification_binary_targets_AUC.csv")
    medication = _records(out / "classification_binary_medications_AUC.csv")
    bp_incidence_tables = sorted(out.glob("*_bp_hypertension_incidence_*.csv"))
    write_numerical_results_markdown(
        fig_dir / "target_prediction_numerical_results.md",
        "Target Prediction Numerical Results",
        [
            out / "regression_target_prediction_summary.csv",
            out / "classification_binary_targets_AUC.csv",
            out / "classification_binary_medications_AUC.csv",
            *(
                [out / "vitaldb_preop_predictions.csv"]
                if (out / "vitaldb_preop_predictions.csv").is_file()
                else []
            ),
            *bp_incidence_tables,
        ],
    )

    with redirect_figure_writes(fig_dir):
        regression_no_bmi = [r for r in regression if str(r.get("target", "")).lower() != "bmi"]
        regression_plot = regression_no_bmi or regression
        if regression_plot:
            analysis.plot_radar_r2(regression_plot, include_pyppg=analysis.ADD_PYPPG_TO_RADAR)
            analysis.plot_forest_r_demo_vs_embeddings(regression_plot)
            analysis.plot_scatter_r_demo_vs_embeddings(regression_plot)
            analysis.plot_bar_r_pearson_demo_vs_embeddings(regression_plot)
            analysis.plot_forest_r2_demo_vs_embeddings(regression_plot)
            analysis.plot_scatter_r2_demo_vs_embeddings(regression_plot)
            analysis.plot_bar_r_demo_vs_embeddings(regression_plot)
            analysis._plot_regression_horizontal_with_connectors(regression_plot, metric="R")
            analysis._plot_regression_horizontal_with_connectors(regression_plot, metric="R2")
        if disease:
            analysis.plot_radar_auc(disease)
            analysis.plot_forest_auc_demo_vs_embeddings(disease)
            analysis.plot_scatter_auc_demo_vs_embeddings(disease)
            analysis.plot_radar_f1(disease)
            analysis.plot_forest_f1_demo_vs_embeddings(disease)
            analysis.plot_scatter_f1_demo_vs_embeddings(disease)
        if medication:
            analysis.plot_radar_auc(medication, out_path=os.path.join(str(fig_dir), "radar_AUC_binary_medications.pdf"))
            analysis.plot_forest_auc_demo_vs_embeddings(medication, out_path=os.path.join(str(fig_dir), "forest_AUC_medications.pdf"))
            analysis.plot_scatter_auc_demo_vs_embeddings(medication, out_path=os.path.join(str(fig_dir), "scatter_AUC_medications.pdf"))
            analysis.plot_radar_f1(medication, out_path=os.path.join(str(fig_dir), "radar_F1_binary_medications.pdf"))
            analysis.plot_forest_f1_demo_vs_embeddings(medication, out_path=os.path.join(str(fig_dir), "forest_F1_medications.pdf"))
            analysis.plot_scatter_f1_demo_vs_embeddings(medication, out_path=os.path.join(str(fig_dir), "scatter_F1_medications.pdf"))
        if disease or medication:
            analysis.plot_unified_classification_auc_horizontal(disease, medication)
        if bp_incidence_tables:
            analysis.plot_cached_bp_hypertension_incidence_figures(out)

    # Direct saves (outside redirect_figure_writes — each function manages its own paths)
    _grouped_inc_csv = out / "grouped_incidence_or_results.csv"
    if _grouped_inc_csv.is_file():
        _grouped_df = pd.read_csv(_grouped_inc_csv)
        if not _grouped_df.empty:
            plot_grouped_incidence_metric_barh(_grouped_df, "auc", fig_dir)
            if "c_index_demo" in _grouped_df.columns and _grouped_df["c_index_demo"].notna().any():
                plot_grouped_incidence_metric_barh(_grouped_df, "c_index", fig_dir)
    plot_hypertension_incidence_roc(cache_dir, fig_dir)
    plot_psychoanaleptic_responder_target_boxplots(out, cache_dir, fig_dir, medication)
    plot_bp_quartile_hypertension_incidence_lines(out, cache_dir, fig_dir)
    plot_age_quartile_hypertension_incidence_lines(out, cache_dir, fig_dir)
    plot_regression_delta_2y_by_baseline_quartile_bands(out, cache_dir, fig_dir)
    plot_residual_delta_2y_by_quartile_bands(out, cache_dir, fig_dir)
    plot_residual_quartile_disease_incidence_lines(out, cache_dir, fig_dir)
    plot_residual_quartile_per_disease_panels(out, cache_dir, fig_dir)
    plot_vitaldb_preop_figures_from_results(out, fig_dir)
    copy_bone_density_quantile_panels(fig_dir)
    make_quartile_longitudinal_dashboard(out, fig_dir)


if __name__ == "__main__":
    main()

