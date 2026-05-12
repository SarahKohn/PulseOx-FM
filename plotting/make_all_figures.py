#!/usr/bin/env python3
"""Build manuscript-only composite figures and legends."""

from __future__ import annotations

import ast
import io
import json
import math
import os
import re
import shutil
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache-sleep-fm")

import matplotlib
from matplotlib.patches import Patch, Rectangle
from matplotlib.transforms import blended_transform_factory

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FormatStrFormatter, MultipleLocator, PercentFormatter
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps
from sklearn.metrics import roc_auc_score, roc_curve


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANUSCRIPT_DIR = PROJECT_ROOT / "manuscript"
FIGURES_ROOT = PROJECT_ROOT / "figures"
ABLATION_WANDB_FIG_DIR = FIGURES_ROOT / "ablation_masking_lengths"
WANDB_RECONSTRUCTION_ABLATION_PNG = (
    ABLATION_WANDB_FIG_DIR / "reconstruction_random_masking_and_forecasting_vs_train_mask_ratio.png"
)
WANDB_AGE_LINEAR_PROBING_PNG = (
    ABLATION_WANDB_FIG_DIR / "age_linear_probing_segment_length_and_training_mask_ratio.png"
)
TARGET_PREDICTION_FIG_DIR = FIGURES_ROOT / "target_prediction"
# Nature Medicine layout: 18 cm figure width (same as journal single-column max for composites built here).
FIG_WIDTH_CM = 18.0
# Maximum figure height for all manuscript composites (width remains FIG_WIDTH_CM).
FIG_MAX_HEIGHT_CM = 21.0
FIGURE_WIDTH_MM = int(round(FIG_WIDTH_CM * 10))
RECONSTRUCTION_DIR = Path("/net/mraid20/export/jafar/Sarah/ssl_sleep/reconstruction_eval_epoch143")
TEMPORAL_RESULTS = Path(
    "/home/sarahk/JafarShortcut/SleepFM/ssl_sleep/downstream_tasks_results/temporal_age_prediction/"
    "age_prediction_results_TabICL_mean_overStage_mean_cvRepeat3.json"
)
TEMPORAL_RESULTS_CANDIDATES = (
    TEMPORAL_RESULTS,
    PROJECT_ROOT / "downstream_tasks_results" / "temporal_age_prediction" / "age_prediction_results_TabICL_mean_overStage_mean_cvRepeat3.json",
)
EXTERNAL_RESULTS_DIRS = (
    Path("/home/sarahk/JafarShortcut/SleepFM/ssl_sleep/downstream_tasks_results/external_datasets"),
    Path("/net/mraid20/export/jafar/Sarah/ssl_sleep/external_dataset_evaluation_output2"),
    Path("/net/mraid20/export/jafar/Sarah/ssl_sleep/external_dataset_evaluation_output"),
)
COHORT_DESCRIPTION_CSVS = (
    PROJECT_ROOT / "tabular_data" / "cohort_description_gold_vitaldb.csv",
    Path("/home/sarahk/JafarShortcut/SleepFM/ssl_sleep/tabular_data/cohort_description_gold_vitaldb.csv"),
    Path("/net/mraid20/export/jafar/SleepFM/ssl_sleep/tabular_data/cohort_description_gold_vitaldb.csv"),
)
_TP_BASE = (
    PROJECT_ROOT / "downstream_tasks_results" / "target_prediction",
    Path("/home/sarahk/JafarShortcut/SleepFM/ssl_sleep/downstream_tasks_results/target_prediction"),
    Path("/net/mraid20/export/jafar/SleepFM/ssl_sleep/downstream_tasks_results/target_prediction"),
)
SCATTER_R_AGE_SEX_VS_EMB_N53 = "scatter_R_age_sex_vs_embeddings_Ntargets53"
CLASSIFICATION_DISEASE_AUC_CSVS = tuple(d / "classification_binary_targets_AUC.csv" for d in _TP_BASE)
CLASSIFICATION_MEDICATION_AUC_CSVS = tuple(d / "classification_binary_medications_AUC.csv" for d in _TP_BASE)
VITALDB_PREOP_PREDICTIONS_CSVS = tuple(d / "vitaldb_preop_predictions.csv" for d in _TP_BASE)
REGRESSION_TARGET_SUMMARY_CSVS = tuple(d / "regression_target_prediction_summary.csv" for d in _TP_BASE)
ABLATION_GOLD_TEST_AGE_DIRS = (
    PROJECT_ROOT / "downstream_tasks_results" / "ablation_gold_test_age",
    Path("/home/sarahk/JafarShortcut/Sarah/ssl_sleep/ablation_studies_gold_test_age"),
    Path("/home/sarahk/JafarShortcut/SleepFM/ssl_sleep/downstream_tasks_results/ablation_gold_test_age"),
    Path("/net/mraid20/export/jafar/SleepFM/ssl_sleep/downstream_tasks_results/ablation_gold_test_age"),
)
SARAH_SSL_SLEEP = Path("/net/mraid20/export/jafar/Sarah/ssl_sleep")
WANDB_ABLATION_EXPORT_SEGMENT_EXP = "segment_lengths4"
WANDB_ABLATION_EXPORT_MASK_EXP = "segment_mask_ratios4"
WANDB_ABLATION_EXPORT_DIRS = (
    Path("/net/mraid20/export/jafar/SleepFM/ssl_sleep/ablation_experiments/"),
    SARAH_SSL_SLEEP / "ablation_experiments",
    PROJECT_ROOT / "downstream_tasks_results" / "wandb_ablation_exports",
)
NEXT_DAY_BARH_ROOTS = (
    FIGURES_ROOT / "day_after_associations",
    SARAH_SSL_SLEEP / "associations_with_day_after_targets_mean_all",
    SARAH_SSL_SLEEP / "associations_with_day_after_targets_mean_1to10h",
    SARAH_SSL_SLEEP / "associations_with_day_after_targets_mean_1to8h",
    SARAH_SSL_SLEEP / "associations_with_day_after_targets_mean_1to6h",
    SARAH_SSL_SLEEP / "associations_with_day_after_targets_mean_all_clusters",
    PROJECT_ROOT / "figures" / "associations_with_day_after_targets",
)
NEXT_DAY_SIGNIFICANT_SUMMARY_CSVS = (
    SARAH_SSL_SLEEP / "associations_with_day_after_targets_mean_all" / "significant_nextday_targets_summary.csv",
    PROJECT_ROOT / "downstream_tasks_results" / "day_after_associations" / "significant_nextday_targets_summary.csv",
)
NEXT_DAY_RADAR_DEMO_CSVS = (
    Path("/home/sarahk/JafarShortcut/SleepFM/ssl_sleep/downstream_tasks_results/day_after_associations/radar_demo_vs_embeddings_cache.csv"),
    SARAH_SSL_SLEEP / "associations_with_day_after_targets_mean_all" / "radar_demo_vs_embeddings_cache.csv",
    PROJECT_ROOT / "downstream_tasks_results" / "day_after_associations" / "radar_demo_vs_embeddings_cache.csv",
)
NEXT_DAY_RESULT_CSV_GROUPS = (
    (
        Path("/home/sarahk/JafarShortcut/SleepFM/ssl_sleep/downstream_tasks_results/day_after_associations/cgm_log_ratio_prediction_results.csv"),
        SARAH_SSL_SLEEP / "associations_with_day_after_targets_mean_all" / "cgm_log_ratio_prediction_results.csv",
    ),
    (
        Path("/home/sarahk/JafarShortcut/SleepFM/ssl_sleep/downstream_tasks_results/day_after_associations/food_log_ratio_prediction_results.csv"),
        SARAH_SSL_SLEEP / "associations_with_day_after_targets_mean_all" / "food_log_ratio_prediction_results.csv",
    ),
    (
        Path("/home/sarahk/JafarShortcut/SleepFM/ssl_sleep/downstream_tasks_results/day_after_associations/activity_wearables_prediction_results.csv"),
        SARAH_SSL_SLEEP / "associations_with_day_after_targets_mean_all" / "activity_wearables_prediction_results.csv",
    ),
)
# Prefer vector PDF over PNG for the same stem (see within_person_variability_analysis).
_BASAL_PRED_VAR_PNG_BASES = (
    SARAH_SSL_SLEEP / "within_person_variability_output" / "prediction_within_vs_between_basal_energy_burned.png",
    FIGURES_ROOT / "within_person_variability" / "prediction_within_vs_between_basal_energy_burned.png",
    Path("/home/sarahk/JafarShortcut/SleepFM/ssl_sleep/downstream_tasks_results/within_person_variability")
    / "prediction_within_vs_between_basal_energy_burned.png",
)
BASAL_ENERGY_PRED_VARIABILITY_ASSETS = tuple(
    p for png_path in _BASAL_PRED_VAR_PNG_BASES for p in (png_path.with_suffix(".pdf"), png_path)
)
BASAL_ENERGY_PREDICTIONS_CSV_CANDIDATES = (
    Path("/home/sarahk/JafarShortcut/SleepFM/ssl_sleep/downstream_tasks_results")
    / "day_after_associations" / "predictions_raw_basal_energy_burned.csv",
    SARAH_SSL_SLEEP / "associations_with_day_after_targets_mean_all" / "predictions_raw_basal_energy_burned.csv",
)
GOLD_RECORDS_PATH = Path("/net/mraid20/export/jafar/SleepFM/ssl_sleep/tabular_data/recordings_with_labels_gold.csv")
MEDICATION_INTAKE_CSV = Path("/net/mraid20/export/jafar/SleepFM/ssl_sleep/tabular_data/medication_intake_per_stage.csv")
STAGE_ORDER = ("00_00_visit", "02_00_visit", "04_00_visit", "06_00_visit")
BASELINE_TO_FOLLOWUP = {"00_00_visit": "02_00_visit", "02_00_visit": "04_00_visit", "04_00_visit": "06_00_visit"}
MIN_FRACTION_OF_MINORITY_CLASS = 0.01
# Match `target_prediction_evaluation_short._filter_classification_results_for_plots`.
CLASSIFICATION_FDR_ALPHA = 0.10
SUPP_TABLE4_HPP_MEASURES_FDR_ALPHA = 0.01
ROC_BOOTSTRAP_REPEATS = 200
DISEASE_CATEGORY_PRIORITY = (
    "Cardiovascular disorders",
    "Metabolic, endocrine and reproductive disorders",
    "Sleep disorders and mental health",
    "Gastrointestinal disorders",
    "Hematologic and nutritional disorders",
    "Neurological and pain conditions",
    "Immune and allergic diseases",
    "Other",
)

DPI = 300
FIGURE_WIDTH_PX = int(round(FIGURE_WIDTH_MM / 25.4 * DPI))
FIGURE_MAX_HEIGHT_PX = int(round(FIG_MAX_HEIGHT_CM * DPI / 2.54))
# PulseOx-FM / embedding accent (HPP and embedding+AUC panels); VitalDB palette unchanged.
PULSEOX_FM_PRIMARY_HEX = "#196874"
PULSEOX_FM_EDGE_HEX = "#0f4a52"
PULSEOX_FM_LIGHT_HEX = "#c8dde2"
PULSEOX_FM_BAND_BG_HEX = "#e6f1f3"
PULSEOX_FM_BAND_TITLE_HEX = "#143d44"
AXIS_LABEL_FONT_SIZE_PT = 10
FONT_SIZE_PX = int(round(9 / 72 * DPI))
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)


def _column_content_height_for_max_figure(img: Image.Image, *, margin: int) -> int:
    """Height after uniform scaling so the image fits column width and global max figure height."""
    col_w = FIGURE_WIDTH_PX - 2 * margin
    max_row_h = FIGURE_MAX_HEIGHT_PX - 2 * margin
    scale = min(col_w / max(1, img.width), max_row_h / max(1, img.height))
    return min(int(round(img.height * scale)), max_row_h)


@dataclass(frozen=True)
class Panel:
    label: str
    image: Image.Image


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/liberation/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/urw-base35/NimbusSans-Bold.otf" if bold else "/usr/share/fonts/urw-base35/NimbusSans-Regular.otf",
    ]
    for candidate in candidates:
        path = Path(candidate)
        if path.is_file():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def _patterns_pdf_before_png(patterns: Iterable[str]) -> list[str]:
    """For each *.png glob, search the corresponding *.pdf first (vector masters for manuscript compositing)."""
    ordered: list[str] = []
    seen: set[str] = set()
    for raw in patterns:
        p = str(raw).replace("\\", "/")
        if p.lower().endswith(".png"):
            pdf_pat = p[:-4] + ".pdf"
            if pdf_pat not in seen:
                seen.add(pdf_pat)
                ordered.append(pdf_pat)
        if p not in seen:
            seen.add(p)
            ordered.append(p)
    return ordered


def _find_one(patterns: Iterable[str], roots: Iterable[Path]) -> Path:
    matches: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for pattern in _patterns_pdf_before_png(patterns):
            matches.extend(p for p in root.rglob(pattern) if p.is_file())
    if not matches:
        raise FileNotFoundError(f"Could not find any source panel matching: {', '.join(patterns)}")
    def _prio(suf: str) -> int:
        s = suf.lower()
        if s == ".pdf":
            return 0
        if s == ".png":
            return 1
        return 2
    matches.sort(key=lambda p: (_prio(p.suffix), -p.stat().st_mtime_ns))
    return matches[0]


def _prefer_pdf_asset(path: Path) -> Path:
    """Use sibling .pdf when present (wandb/export figures shipped as PNG+PDF pairs)."""
    path = Path(path)
    pdf = path.with_suffix(".pdf")
    return pdf if pdf.is_file() else path


def _rasterize_pdf_first_page(path: Path, dpi: int = DPI) -> Image.Image:
    """Rasterize first PDF page so PIL composites match manuscript DPI (vectors preserved in source PDF)."""
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:
        raise ImportError(
            "Could not load PyMuPDF (import fitz). Install with "
            "`pip install pymupdf` or `conda install -c conda-forge pymupdf`. "
            "If the package is already installed, the native library may not load on "
            "this OS (e.g. GLIBC too old for the pip wheel); try conda-forge, an "
            "older pymupdf pin, or remove sibling .pdf panels so PNG assets are used.\n"
            f"Underlying error: {exc}"
        ) from exc
    doc = fitz.open(str(path))
    try:
        page = doc.load_page(0)
        zoom = dpi / 72.0
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
        return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    finally:
        doc.close()


def _open_rgb(path: Path | str) -> Image.Image:
    path = Path(path)
    suf = path.suffix.lower()
    if suf == ".pdf":
        return _rasterize_pdf_first_page(path)
    img = Image.open(path)
    if img.mode == "RGBA":
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[3])
        return bg
    return img.convert("RGB")


def _crop_barh_figure(img: Image.Image) -> Image.Image:
    """Remove the figure title at the top and the annotation footnote at the bottom.

    Both regions are detected by scanning for gaps (runs of nearly-white rows) that
    separate them from the main chart area, so the function is robust to different
    image resolutions.
    """
    arr = np.asarray(img.convert("L"))
    h = arr.shape[0]
    row_min = arr.min(axis=1)
    has_content = row_min < 210  # True when a row contains non-white pixels

    # ── top: find and skip the figure title ──────────────────────────────
    top_limit = int(h * 0.20)
    crop_top = 0
    in_title = False
    title_end = 0
    for r in range(top_limit):
        if has_content[r]:
            in_title = True
            title_end = r
        elif in_title and r - title_end >= 5:
            # gap of ≥5 white rows found after the title; walk forward to
            # the first content row of the chart proper
            chart_start = r
            while chart_start < top_limit and not has_content[chart_start]:
                chart_start += 1
            crop_top = chart_start
            break

    # ── bottom: find and skip the annotation footnote ────────────────────
    bottom_start = int(h * 0.75)
    crop_bottom = h
    # walk up from the bottom through trailing whitespace
    i = h - 1
    while i >= bottom_start and not has_content[i]:
        i -= 1
    if i >= bottom_start:
        # walk through the footnote text
        while i >= bottom_start and has_content[i]:
            i -= 1
        # walk through the gap separating footnote from chart
        gap_end = i
        while i >= bottom_start and not has_content[i]:
            i -= 1
        gap_size = gap_end - i
        if gap_size >= 15:
            crop_bottom = i + 2   # end of chart content + tiny margin

    if crop_top >= crop_bottom:
        return img
    return img.crop((0, crop_top, img.width, crop_bottom))


def _crop_right_legend(image: Image.Image, keep_fraction: float = 0.78) -> Image.Image:
    """Drop right-side legends from source panels that duplicate adjacent panels."""
    keep_w = int(round(image.width * keep_fraction))
    return image.crop((0, 0, keep_w, image.height))


def _crop_right_colorbar(image: Image.Image, keep_fraction: float = 0.78) -> Image.Image:
    """Drop right-side colour bars when the adjacent panel keeps the same scale."""
    keep_w = int(round(image.width * keep_fraction))
    return image.crop((0, 0, keep_w, image.height))


def _recolor_umap_age_to_blues(image: Image.Image) -> Image.Image:
    """Map non-neutral age UMAP colours to a blue sequential palette while preserving text and axes."""
    rgb = image.convert("RGB")
    arr = np.asarray(rgb, dtype=np.float32) / 255.0
    max_c = arr.max(axis=2)
    min_c = arr.min(axis=2)
    chroma = max_c - min_c
    luminance = 0.2126 * arr[:, :, 0] + 0.7152 * arr[:, :, 1] + 0.0722 * arr[:, :, 2]
    colored = (chroma > 0.045) & (luminance > 0.08) & (luminance < 0.98)
    if not np.any(colored):
        return rgb

    values = 1.0 - luminance[colored]
    lo, hi = np.nanpercentile(values, [2, 98])
    if hi <= lo:
        hi = lo + 1e-6
    normed = np.clip((values - lo) / (hi - lo), 0.0, 1.0)
    cmap = plt.get_cmap("Blues")
    blue_rgb = np.asarray(cmap(0.22 + 0.76 * normed)[:, :3], dtype=np.float32)

    out = arr.copy()
    out[colored] = blue_rgb
    return Image.fromarray(np.clip(out * 255.0, 0, 255).astype(np.uint8), mode="RGB")


def _detect_y_axis_x(image: Image.Image) -> int:
    gray = np.asarray(image.convert("L"))
    if gray.size == 0:
        return 0
    y0 = int(round(gray.shape[0] * 0.12))
    y1 = int(round(gray.shape[0] * 0.94))
    x1 = max(1, int(round(gray.shape[1] * 0.35)))
    dark = gray[y0:y1, :x1] < 90
    counts = dark.sum(axis=0)
    return int(np.argmax(counts)) if counts.size else 0


def _paste_panel(
    canvas: Image.Image,
    panel: Panel,
    box: tuple[int, int, int, int],
    *,
    fit: str = "contain",
    h_align: str = "center",
    axis_x: int | None = None,
    panel_label_pt: float | None = None,
) -> None:
    x0, y0, x1, y1 = box
    target = (x1 - x0, y1 - y0)
    image = panel.image
    if fit == "cover":
        image = ImageOps.fit(image, target, method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))
    else:
        image = ImageOps.contain(image, target, method=Image.Resampling.LANCZOS)
    if axis_x is not None:
        px = int(round(axis_x - _detect_y_axis_x(image)))
    elif h_align == "left":
        px = x0
    elif h_align == "right":
        px = x1 - image.width
    else:
        px = x0 + (target[0] - image.width) // 2
    py = y0 + (target[1] - image.height) // 2
    canvas.paste(image, (px, py))
    draw = ImageDraw.Draw(canvas)
    lbl_pt = float(AXIS_LABEL_FONT_SIZE_PT if panel_label_pt is None else panel_label_pt)
    label_px = int(round(lbl_pt / 72 * DPI))
    draw.text((x0 + 6, y0 + 6), panel.label, fill=BLACK, font=_font(label_px, bold=True))


def _save_canvas(canvas: Image.Image, stem: str) -> None:
    MANUSCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    if canvas.width != FIGURE_WIDTH_PX:
        raise ValueError(f"{stem}: width {canvas.width}px != {FIGURE_WIDTH_PX}px (18 cm at {DPI} dpi)")
    if canvas.height > FIGURE_MAX_HEIGHT_PX:
        raise ValueError(
            f"{stem}: height {canvas.height}px exceeds max {FIGURE_MAX_HEIGHT_PX}px ({FIG_MAX_HEIGHT_CM} cm at {DPI} dpi)"
        )
    png_path = MANUSCRIPT_DIR / f"{stem}.png"
    pdf_path = MANUSCRIPT_DIR / f"{stem}.pdf"
    canvas.save(png_path, dpi=(DPI, DPI))
    canvas.save(pdf_path, "PDF", resolution=DPI)
    print(f"Saved {png_path}")
    print(f"Saved {pdf_path}")
    width_cm = canvas.width / DPI * 2.54
    height_cm = canvas.height / DPI * 2.54
    width_in = canvas.width / DPI
    height_in = canvas.height / DPI
    print(
        f"{stem} size: {width_cm:.2f} x {height_cm:.2f} cm "
        f"({width_in:.2f} x {height_in:.2f} in; {canvas.width} x {canvas.height} px at {DPI} dpi)"
    )


def _apply_matplotlib_rcparams() -> None:
    """Apply Nature Medicine-consistent rcParams for all generated matplotlib panels.

    Sets baseline font family, sizes, line widths, and spine visibility so that
    panels composited into the final PIL canvases share a consistent typographic style.
    Individual plotting functions may override specific sizes as needed (e.g. for panels
    that are later scaled during compositing).
    """
    matplotlib.rcParams.update(
        {
            # --- typography --------------------------------------------------
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica", "Liberation Sans"],
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 7,
            "legend.title_fontsize": 8,
            # --- line / patch weights ----------------------------------------
            "axes.linewidth": 0.6,
            "grid.linewidth": 0.4,
            "lines.linewidth": 0.8,
            "patch.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.minor.width": 0.4,
            "ytick.minor.width": 0.4,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            # --- spines (top/right removed globally) -------------------------
            "axes.spines.top": False,
            "axes.spines.right": False,
            # --- output quality ----------------------------------------------
            "figure.dpi": DPI,
            "savefig.dpi": DPI,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,   # embed Type 42 (TrueType) in PDFs – editable in Illustrator
            "ps.fonttype": 42,
        }
    )


def _style_axis(ax: plt.Axes, *, tick_labelsize: int = 10) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=tick_labelsize, width=0.6)
    ax.grid(axis="y", color="#d7d7d7", linewidth=0.4, linestyle="--", zorder=0)


def _load_temporal_results() -> dict:
    with TEMPORAL_RESULTS.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_temporal_results_optional() -> dict | None:
    for path in TEMPORAL_RESULTS_CANDIDATES:
        if path.is_file():
            try:
                with path.open("r", encoding="utf-8") as handle:
                    return json.load(handle)
            except OSError:
                continue
    return None


def _n_umap_from_filename(path: Path) -> int | None:
    m = re.search(r"_N(\d+)", path.name)
    return int(m.group(1)) if m else None


def _unique_participants_from_recording_keys(keys: object) -> int | None:
    if not keys:
        return None
    pids: set[str] = set()
    for k in keys:
        s = str(k).strip()
        if not s:
            continue
        pids.add(s.split("__", 1)[0])
    return len(pids) if pids else None


def _as_int_loose(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _draw_centered_column_title(
    canvas: Image.Image,
    text: str,
    x0: int,
    x1: int,
    y: int,
    *,
    font_pt: float = 10,
    bold: bool = False,
) -> None:
    draw = ImageDraw.Draw(canvas)
    font_px = int(round(font_pt / 72 * DPI))
    font = _font(font_px, bold=bold)
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    draw.text((x0 + max(0, (x1 - x0 - tw) // 2), y), text, fill=BLACK, font=font)


def _supplementary_fig1_sample_caption_clause(pyppg_path: Path, watchpat_path: Path) -> str:
    temporal = _load_temporal_results_optional()
    n_p = None
    n_w = None
    u_p = u_w = None
    if temporal:
        n_p = _as_int_loose(temporal.get("n_pyppg"))
        n_w = _as_int_loose(temporal.get("n_watchpat"))
        u_p = _unique_participants_from_recording_keys(temporal.get("stage_keys_pyppg"))
        u_w = _unique_participants_from_recording_keys(temporal.get("stage_keys_watchpat"))
    if n_p is None:
        n_p = _n_umap_from_filename(pyppg_path)
    if n_w is None:
        n_w = _n_umap_from_filename(watchpat_path)
    phrases: list[str] = []
    if n_p is not None:
        if u_p is not None:
            phrases.append(
                f"panels a and c (PyPPG) use N = {n_p:,} pooled participant–visit recordings from {u_p:,} unique participants"
            )
        else:
            phrases.append(f"panels a and c (PyPPG) use N = {n_p:,} pooled recordings in the UMAP")
    if n_w is not None:
        if u_w is not None:
            phrases.append(
                f"panels b and d (WatchPAT) use N = {n_w:,} pooled participant–visit recordings from {u_w:,} unique participants"
            )
        else:
            phrases.append(f"panels b and d (WatchPAT) use N = {n_w:,} pooled recordings in the UMAP")
    if not phrases:
        return (
            "Sample sizes are the numbers of pooled recordings in the PyPPG and WatchPAT UMAP outputs "
            "(see temporal age-prediction cache)."
        )
    body = "; ".join(phrases) + "."
    return body[0].upper() + body[1:]


def _read_external_case_predictions_csv(path: Path) -> pd.DataFrame:
    """Load detailed external export; skip malformed physical rows if present."""
    if not path.is_file():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.ParserError:
        return pd.read_csv(path, engine="python", on_bad_lines="skip")


def _external_detailed_csv() -> Path:
    candidates: list[Path] = []
    for root in EXTERNAL_RESULTS_DIRS:
        if root.exists():
            candidates.extend(root.rglob("external_dataset_case_predictions_with_embeddings.csv"))
    for path in sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            datasets = set(
                _read_external_case_predictions_csv(path)["dataset"].dropna().astype(str).unique()
            )
        except Exception:
            continue
        if {"Gold_test", "VitalDB"}.issubset(datasets):
            return path
    if candidates:
        return sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)[0]
    raise FileNotFoundError("Could not find external_dataset_case_predictions_with_embeddings.csv")


def _external_named_csv(name: str) -> Path | None:
    try:
        return _find_one([name], EXTERNAL_RESULTS_DIRS)
    except FileNotFoundError:
        return None


def _external_prediction_frame(dataset: str) -> pd.DataFrame:
    if dataset == "Gold_test":
        path = _external_named_csv("gold_test_age_crossval_predictions_cross_validation.csv")
    elif dataset == "VitalDB":
        path = _external_named_csv("vitaldb_age_crossval_predictions_cross_validation.csv")
    else:
        path = None
    if path is not None:
        frame = pd.read_csv(path, usecols=["age_true", "age_pred_cv"]).rename(columns={"age_pred_cv": "age_pred"})
        frame["dataset"] = dataset
        return frame

    detailed = _read_external_case_predictions_csv(_external_detailed_csv())
    frame = detailed[["dataset", "age_true", "age_pred_mae_pretrained_cross_validation"]][detailed["dataset"].astype(str) == dataset].copy()
    return frame.rename(columns={"age_pred_mae_pretrained_cross_validation": "age_pred"})


def _external_metric_r2(dataset: str) -> float | None:
    name = "gold_test_age_crossval_metrics_cross_validation.csv" if dataset == "Gold_test" else "vitaldb_age_crossval_metrics_cross_validation.csv"
    path = _external_named_csv(name)
    if path is not None:
        frame = pd.read_csv(path)
        if not frame.empty and "R2" in frame:
            return float(frame.iloc[0]["R2"])
    frame = _external_prediction_frame(dataset).dropna()
    if frame.empty:
        return None
    _, r2_value, _ = _scatter_stats(frame["age_true"].to_numpy(float), frame["age_pred"].to_numpy(float))
    return r2_value


def _external_age_cv_metrics(dataset: str) -> dict[str, tuple[float, float, int | None, float | None, float | None]]:
    name = "gold_test_age_crossval_metrics_cross_validation.csv" if dataset == "Gold_test" else "vitaldb_age_crossval_metrics_cross_validation.csv"
    path = _external_named_csv(name)
    if path is None:
        return {}
    frame = pd.read_csv(path)
    if "feature_kind" not in frame.columns:
        return {}
    out: dict[str, tuple[float, float, int | None, float | None, float | None]] = {}
    for _, row in frame.iterrows():
        feature_kind = str(row.get("feature_kind", "")).strip()
        if not feature_kind:
            continue
        r_col = "r" if "r" in frame.columns else "pearson_r"
        if r_col not in frame.columns or pd.isna(row.get(r_col)):
            continue
        n = int(row["N"]) if "N" in frame.columns and pd.notna(row.get("N")) else None
        r2_sd = None
        for col in ("R2_repeat_sd", "R2_sd"):
            if col in frame.columns and pd.notna(row.get(col)):
                r2_sd = float(row[col])
                break
        r_sd = None
        for col in ("r_repeat_sd", "r_sd", "pearson_r_sd"):
            if col in frame.columns and pd.notna(row.get(col)):
                r_sd = float(row[col])
                break
        out[feature_kind] = (float(row["R2"]), float(row[r_col]), n, r2_sd, r_sd)
    return out


def _external_age_metric_row(dataset: str, feature_kinds: Iterable[str]) -> dict[str, float | int | None]:
    name = "gold_test_age_crossval_metrics_cross_validation.csv" if dataset == "Gold_test" else "vitaldb_age_crossval_metrics_cross_validation.csv"
    path = _external_named_csv(name)
    if path is None:
        return {"Pearson_r": None, "R2": None, "MAE": None, "N_Participants": None}
    frame = pd.read_csv(path)
    if "feature_kind" not in frame.columns:
        return {"Pearson_r": None, "R2": None, "MAE": None, "N_Participants": None}
    for feature_kind in feature_kinds:
        rows = frame[frame["feature_kind"].astype(str).str.strip() == feature_kind]
        if rows.empty:
            continue
        row = rows.iloc[0]
        r_col = "r" if "r" in frame.columns else "pearson_r"
        n_col = "cv_n_groups" if "cv_n_groups" in frame.columns else "N"
        return {
            "Pearson_r": float(row[r_col]) if r_col in frame.columns and pd.notna(row.get(r_col)) else None,
            "R2": float(row["R2"]) if "R2" in frame.columns and pd.notna(row.get("R2")) else None,
            "MAE": float(row["MAE"]) if "MAE" in frame.columns and pd.notna(row.get("MAE")) else None,
            "N_Participants": int(row[n_col]) if n_col in frame.columns and pd.notna(row.get(n_col)) else None,
        }
    return {"Pearson_r": None, "R2": None, "MAE": None, "N_Participants": None}


def _cohort_participant_n(column_name: str) -> int | None:
    table = pd.read_csv(_first_existing(COHORT_DESCRIPTION_CSVS, "cohort description CSV"), keep_default_na=False)
    row = table[table["Characteristic"].astype(str) == "Participants, n"]
    if row.empty or column_name not in table.columns:
        return None
    value = str(row.iloc[0][column_name]).strip()
    try:
        return int(float(value))
    except ValueError:
        return None


def _scatter_stats(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float | None]:
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[valid]
    y_pred = y_pred[valid]
    if len(y_true) < 3:
        return float("nan"), float("nan"), None
    r_value = float(np.corrcoef(y_true, y_pred)[0, 1])
    r2_value = r_value * r_value
    try:
        from scipy import stats

        p_value = float(stats.pearsonr(y_true, y_pred).pvalue)
    except Exception:
        p_value = None
    return r_value, r2_value, p_value


def _first_existing(paths: Iterable[Path], description: str) -> Path:
    for path in paths:
        if path.is_file():
            return path
    raise FileNotFoundError(f"Could not find {description}: {', '.join(str(p) for p in paths)}")


def _find_wandb_age_ablation_csvs() -> tuple[Path, Path] | None:
    for root in WANDB_ABLATION_EXPORT_DIRS:
        seg = root / f"wandb_export_{WANDB_ABLATION_EXPORT_SEGMENT_EXP}.csv"
        mask = root / f"wandb_export_{WANDB_ABLATION_EXPORT_MASK_EXP}.csv"
        if seg.is_file() and mask.is_file():
            return seg, mask
    return None


def _scatter_r_age_sex_embeddings_n53_path() -> Path:
    """Pre-rendered regression Pearson-R scatter from target-prediction evaluation (Figure 4)."""
    for root in (TARGET_PREDICTION_FIG_DIR, *_TP_BASE):
        for ext in (".pdf", ".PDF", ".png", ".PNG"):
            cand = root / f"{SCATTER_R_AGE_SEX_VS_EMB_N53}{ext}"
            if cand.is_file():
                return cand
    raise FileNotFoundError(
        f"Could not find {SCATTER_R_AGE_SEX_VS_EMB_N53}.png (or .pdf) under "
        f"{TARGET_PREDICTION_FIG_DIR} or target_prediction result dirs"
    )


def _next_day_raw_pearson_barh_asset() -> Path:
    """Horizontal next-day performance plot: raw Pearson r (prefer vector PDF; then tabicl, then elasticnet)."""
    matches: list[Path] = []
    for root in NEXT_DAY_BARH_ROOTS:
        if root.is_dir():
            matches.extend(root.glob("next_day_performance_barh_pearson_r_raw_*.pdf"))
            matches.extend(root.glob("next_day_performance_barh_pearson_r_raw_*.PDF"))
            matches.extend(root.glob("next_day_performance_barh_pearson_r_raw_*.png"))
    if not matches:
        raise FileNotFoundError(
            "Could not find next_day_performance_barh_pearson_r_raw_*.{pdf,png} under day-after output dirs; "
            f"searched: {', '.join(str(r) for r in NEXT_DAY_BARH_ROOTS)}"
        )

    def _prio(path: Path) -> tuple[int, int, float]:
        suf = path.suffix.lower()
        pdf_first = 0 if suf == ".pdf" else 1
        tab = 0 if "tabicl" in path.name.lower() else (1 if "elasticnet" in path.name.lower() else 2)
        return pdf_first, tab, -path.stat().st_mtime_ns

    return sorted(matches, key=_prio)[0]


def _next_day_significant_summary_csv() -> Path:
    return _first_existing(NEXT_DAY_SIGNIFICANT_SUMMARY_CSVS, "significant next-day targets summary CSV")


def _next_day_raw_results_dataframe() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for candidates in NEXT_DAY_RESULT_CSV_GROUPS:
        for candidate in candidates:
            if candidate.is_file():
                frames.append(pd.read_csv(candidate))
                break
    if frames:
        df = pd.concat(frames, ignore_index=True)
        df = df[(df["analysis"].astype(str) == "raw") & (df["model"].astype(str).str.lower() == "tabicl")].copy()
        if not df.empty:
            p_values = pd.to_numeric(df["pearson_pval"], errors="coerce").to_numpy(dtype=float)
            df["q_value"] = _bh_fdr_qvalues(p_values)
            return df[df["q_value"] < 0.10].copy()

    path = _next_day_significant_summary_csv()
    df = pd.read_csv(path)
    df = df[(df["analysis"].astype(str) == "raw") & np.isfinite(pd.to_numeric(df["pearson_r"], errors="coerce"))].copy()
    preferred = df[df["model"].astype(str).str.lower() == "tabicl"]
    if not preferred.empty:
        df = preferred
    elif "elasticnet" in {m.lower() for m in df["model"].astype(str)}:
        df = df[df["model"].astype(str).str.lower() == "elasticnet"]
    if "q_value" in df:
        df["q_value"] = pd.to_numeric(df["q_value"], errors="coerce")
        df = df[df["q_value"] < 0.10]
    return df


def _next_day_stars(q_value: object) -> str:
    try:
        q = float(q_value)
    except (TypeError, ValueError):
        return ""
    if not np.isfinite(q):
        return ""
    if q < 0.001:
        return "***"
    if q < 0.01:
        return "**"
    if q < 0.10:
        return "*"
    return ""


def _format_next_day_target_label(label: object) -> str:
    raw = str(label).replace("log_ratio_", "").replace("delta_", "")
    lower = raw.lower()
    special = {
        "gmi": "GMI",
        "cv": "CV",
        "tir_70_180": "TIR 70-180",
        "tbr_below_70": "TBR below 70",
        "titr_70_140": "TITR 70-140",
        "tar_above_180": "TAR above 180",
        "ppgr_95th": "PPGR 95th",
    }
    if lower in special:
        text = special[lower]
    else:
        text = raw.replace("__", " ").replace("_", " ").lower()
        text = text[:1].upper() + text[1:]
    return textwrap.fill(text, width=34)


def _generate_next_day_raw_pearson_image(width_px: int) -> Image.Image:
    """Generate panel 6a directly so text remains 10 pt in the final composite."""
    import io

    df = _next_day_raw_results_dataframe()
    if df.empty:
        raise ValueError("No FDR-significant raw Pearson rows found for next-day panel")

    df["pearson_r"] = pd.to_numeric(df["pearson_r"], errors="coerce")
    err_col = "pearson_r_repeat_sd"
    if err_col in df:
        df[err_col] = pd.to_numeric(df[err_col], errors="coerce").fillna(0.0).clip(lower=0.0)
    else:
        df[err_col] = 0.0
    df = df.sort_values("pearson_r", ascending=False)

    type_colors = {
        "CGM": "#0b3d44",
        "Food": PULSEOX_FM_PRIMARY_HEX,
        "Wearables": "#84c6cf",
    }
    type_labels = {
        "CGM": "Continuous glucose monitoring",
        "Food": "Food logging",
        "Wearables": "Wearables",
    }
    colors = [type_colors.get(str(t), "#7f7f7f") for t in df["target_type"]]
    y_pos = np.arange(len(df))
    values = df["pearson_r"].to_numpy(dtype=float)
    errors = df[err_col].to_numpy(dtype=float)

    fig_w = width_px / DPI
    fig_h = max(4.20, 0.17 * len(df) + 0.90)
    with plt.rc_context({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
    }):
        fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=DPI)
        has_errors = bool(np.any(errors > 0))
        error_kwargs = {}
        if has_errors:
            error_kwargs = {
                "xerr": errors,
                "capsize": 3,
                "error_kw": {"ecolor": "#222222", "elinewidth": 0.9, "capthick": 0.9},
            }
        ax.barh(
            y_pos,
            values,
            color=colors,
            edgecolor="white",
            linewidth=0.6,
            zorder=2,
            **error_kwargs,
        )
        ax.set_yticks(y_pos)
        ax.set_yticklabels([_format_next_day_target_label(t) for t in df["target"]])
        ax.invert_yaxis()
        ax.axvline(0, color="#777777", linestyle=":", linewidth=0.9, zorder=1)
        ax.grid(True, axis="x", color="#d7d7d7", alpha=0.75, linewidth=0.5, zorder=0)
        ax.set_xlabel("Pearson $r$ (predicted vs observed)")
        ax.tick_params(axis="both", labelsize=10)

        min_x = float(np.nanmin(values - errors))
        max_x = float(np.nanmax(values + errors))
        span = max(max_x - min_x, 0.1)
        pad = 0.035 * span
        for i, (_, row) in enumerate(df.iterrows()):
            value = float(row["pearson_r"])
            err = float(row[err_col])
            stars = _next_day_stars(row.get("q_value"))
            text = f"{value:.2f} {stars}".rstrip()
            if value >= 0:
                x_text, ha = value + err + pad, "left"
            else:
                x_text, ha = value - err - pad, "right"
            ax.text(x_text, y_pos[i], text, va="center", ha=ha, fontsize=10, color="#222222")

        present_types = list(dict.fromkeys(str(t) for t in df["target_type"]))
        legend_handles = [
            plt.Line2D(
                [0],
                [0],
                marker="s",
                linestyle="none",
                markersize=7,
                markerfacecolor=type_colors.get(label, "#7f7f7f"),
                markeredgecolor="white",
                label=type_labels.get(label, label),
            )
            for label in present_types
        ]
        if legend_handles:
            ax.legend(handles=legend_handles, loc="lower right", frameon=False, handletextpad=0.4)
        ax.set_xlim(min(min_x - 4 * pad, -0.02), max(max_x + 8 * pad, 0.02))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout(pad=0.3)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=DPI, facecolor="white")
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf).copy().convert("RGB")


def _parse_mean_pm(cell: object) -> float:
    if cell is None:
        return float("nan")
    if isinstance(cell, (int, float, np.integer, np.floating)):
        x = float(cell)
        return float("nan") if np.isnan(x) else x
    s = str(cell).strip()
    if not s or s.lower() in {"nan", "none"}:
        return float("nan")
    if "±" in s:
        s = s.split("±")[0].strip()
    return float(s)


def _parse_p_value(cell: object) -> float | None:
    if cell is None or (isinstance(cell, float) and np.isnan(cell)):
        return None
    try:
        return float(cell)
    except (TypeError, ValueError):
        return None


def _load_disease_category_map() -> dict[str, str]:
    utils_path = PROJECT_ROOT / "utils.py"
    module = ast.parse(utils_path.read_text(encoding="utf-8"))
    for node in module.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "DISEASE_CATEGORY_MAP":
                    return ast.literal_eval(node.value)
    raise ValueError(f"Could not find DISEASE_CATEGORY_MAP in {utils_path}")


def _normalise_recording_labels(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    if "Recordings" not in frame.columns and "index" in frame.columns:
        frame = frame.rename(columns={"index": "Recordings"})
    if "Recordings" not in frame.columns:
        raise ValueError("Gold labels must contain either 'Recordings' or 'index'.")
    frame["Recordings"] = frame["Recordings"].astype(str).str.replace(".pt", "", regex=False)
    if "participant_id" not in frame.columns:
        frame["participant_id"] = frame["Recordings"].astype(str).str.split("__").str[0]
    if "research_stage" not in frame.columns:
        frame["research_stage"] = frame["Recordings"].astype(str).str.split("__").str[1]
    return frame


def _participant_baseline_followup(labels: pd.DataFrame) -> pd.DataFrame:
    stage_to_ord = {stage: idx for idx, stage in enumerate(STAGE_ORDER)}
    available = labels[["participant_id", "research_stage"]].drop_duplicates()
    available = available[available["research_stage"].isin(stage_to_ord)]
    if available.empty:
        return pd.DataFrame(columns=["participant_id", "baseline_stage", "followup_stage"])

    def first_available(stages: pd.Series) -> str | None:
        ords = [stage_to_ord[s] for s in stages if s in stage_to_ord]
        return STAGE_ORDER[min(ords)] if ords else None

    baseline = available.groupby("participant_id")["research_stage"].apply(first_available).reset_index()
    baseline.columns = ["participant_id", "baseline_stage"]
    baseline["followup_stage"] = baseline["baseline_stage"].map(BASELINE_TO_FOLLOWUP)
    baseline = baseline.dropna(subset=["followup_stage"])
    joined = labels.merge(baseline, on="participant_id", how="inner")
    has_baseline = set(joined.loc[joined["research_stage"] == joined["baseline_stage"], "participant_id"])
    has_followup = set(joined.loc[joined["research_stage"] == joined["followup_stage"], "participant_id"])
    keep = has_baseline & has_followup
    return baseline[baseline["participant_id"].isin(keep)].copy()


def _load_gold_labels(usecols: Iterable[str] | None = None) -> pd.DataFrame:
    header = pd.read_csv(GOLD_RECORDS_PATH, nrows=0).columns.tolist()
    requested = ["index", "Recordings", "participant_id", "research_stage"]
    if usecols is not None:
        requested.extend(usecols)
    cols = [c for c in dict.fromkeys(requested) if c in header]
    labels = pd.read_csv(GOLD_RECORDS_PATH, usecols=cols, low_memory=False)
    return _normalise_recording_labels(labels)


def _baseline_rows(frame: pd.DataFrame, baseline_followup: pd.DataFrame) -> pd.DataFrame:
    joined = frame.merge(baseline_followup[["participant_id", "baseline_stage"]], on="participant_id", how="inner")
    joined = joined[joined["research_stage"].astype(str) == joined["baseline_stage"].astype(str)].copy()
    return joined.drop(columns=["baseline_stage"], errors="ignore")


def _prevalence_summary(values: pd.Series) -> tuple[int, int, float] | None:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return None
    y = (numeric > 0).astype(int)
    n_total = int(y.shape[0])
    n_positive = int(y.sum())
    counts = (n_positive, n_total - n_positive)
    if min(counts) / n_total < MIN_FRACTION_OF_MINORITY_CLASS:
        return None
    return n_positive, n_total, 100.0 * n_positive / n_total


def _age_metrics(y_true: Iterable[float], y_pred: Iterable[float]) -> dict[str, float]:
    y_true_arr = np.asarray(list(y_true), dtype=float)
    y_pred_arr = np.asarray(list(y_pred), dtype=float)
    valid = np.isfinite(y_true_arr) & np.isfinite(y_pred_arr)
    y_true_arr = y_true_arr[valid]
    y_pred_arr = y_pred_arr[valid]
    r_value, _, _ = _scatter_stats(y_true_arr, y_pred_arr)
    ss_res = float(np.sum((y_true_arr - y_pred_arr) ** 2))
    ss_tot = float(np.sum((y_true_arr - np.mean(y_true_arr)) ** 2))
    r2_value = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    mae = float(np.mean(np.abs(y_true_arr - y_pred_arr))) if len(y_true_arr) else float("nan")
    return {"Pearson_r": r_value, "R2": r2_value, "MAE": mae}


def _format_metric(value: float | int | None, decimals: int = 3) -> str:
    if value is None:
        return ""
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        return ""
    if not np.isfinite(value_f):
        return ""
    return f"{value_f:.{decimals}f}"


def _format_mean_pm_sd_2dp(mean: float | None, sd: float | None) -> str:
    if mean is None:
        return ""
    try:
        m = float(mean)
    except (TypeError, ValueError):
        return ""
    if not np.isfinite(m):
        return ""
    if sd is not None:
        try:
            s = float(sd)
        except (TypeError, ValueError):
            pass
        else:
            if np.isfinite(s):
                return f"{m:.2f} ± {s:.2f}"
    return f"{m:.2f}"


def _finite_float_or_none(value: object) -> float | None:
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def _metric_sd_from_repeat_metrics(temporal: dict, metrics_key: str, field: str) -> float | None:
    metrics = temporal.get(metrics_key)
    if not metrics or not isinstance(metrics, list):
        return None
    vals: list[float] = []
    for m in metrics:
        if not isinstance(m, dict) or field not in m:
            continue
        try:
            vals.append(float(m[field]))
        except (TypeError, ValueError):
            continue
    if len(vals) < 2:
        return None
    return float(np.std(vals, ddof=0))


def _coalesce_cv_sd(temporal: dict, explicit: object, metrics_key: str, field: str) -> float | None:
    v = _finite_float_or_none(explicit)
    if v is not None:
        return v
    return _metric_sd_from_repeat_metrics(temporal, metrics_key, field)


def _mae_sd_from_emb_prediction_seeds(temporal: dict) -> float | None:
    y_true = temporal.get("y_true_emb")
    if y_true is None:
        return None
    seeds = temporal.get("cv_repeat_seeds") or []
    maes: list[float] = []
    for seed in seeds:
        key = f"y_pred_emb_seed_{int(seed)}"
        y_pred = temporal.get(key)
        if y_pred is None:
            continue
        mae = _age_metrics(y_true, y_pred)["MAE"]
        if np.isfinite(mae):
            maes.append(mae)
    if len(maes) < 2:
        return None
    return float(np.std(maes, ddof=0))


def _mae_sd_from_tabular_prediction_seeds(temporal: dict, y_true_key: str, seed_key_prefix: str) -> float | None:
    """Compute SD of MAE across repeats; keys are e.g. y_pred_pyppg_seed_<seed>."""
    y_true = temporal.get(y_true_key)
    if y_true is None:
        return None
    seeds = temporal.get("cv_repeat_seeds") or []
    maes: list[float] = []
    for seed in seeds:
        key = f"{seed_key_prefix}{int(seed)}"
        y_pred = temporal.get(key)
        if y_pred is None:
            continue
        mae = _age_metrics(y_true, y_pred)["MAE"]
        if np.isfinite(mae):
            maes.append(mae)
    if len(maes) < 2:
        return None
    return float(np.std(maes, ddof=0))


def _ensemble_dispersion_sd_from_emb_seeds(
    temporal: dict, shared_keys: list[str], emb_true: dict[str, float], pyppg_pred: dict[str, float]
) -> tuple[float | None, float | None, float | None]:
    stage_keys = temporal.get("stage_keys_emb", [])
    y_true_vec = temporal.get("y_true_emb", [])
    if not stage_keys or not y_true_vec or len(stage_keys) != len(y_true_vec):
        return None, None, None
    key_to_true = dict(zip(stage_keys, y_true_vec))
    seeds = temporal.get("cv_repeat_seeds") or []
    rs: list[float] = []
    r2s: list[float] = []
    maes: list[float] = []
    for seed in seeds:
        pk = f"y_pred_emb_seed_{int(seed)}"
        pred_vec = temporal.get(pk)
        if pred_vec is None:
            continue
        emb_seed = dict(zip(stage_keys, pred_vec))
        y_t = [float(key_to_true[k]) for k in shared_keys]
        y_p = [0.5 * (float(emb_seed[k]) + float(pyppg_pred[k])) for k in shared_keys]
        m = _age_metrics(y_t, y_p)
        if all(np.isfinite(m[k]) for k in ("Pearson_r", "R2", "MAE")):
            rs.append(m["Pearson_r"])
            r2s.append(m["R2"])
            maes.append(m["MAE"])
    if len(rs) < 2:
        return None, None, None
    return float(np.std(rs, ddof=0)), float(np.std(r2s, ddof=0)), float(np.std(maes, ddof=0))


def _pick_csv_metric_sd(row: pd.Series, col_names: tuple[str, ...]) -> float | None:
    for c in col_names:
        if c in row.index and pd.notna(row.get(c)):
            return _finite_float_or_none(row[c])
    return None


def _external_age_metric_row_full(dataset: str, feature_kinds: Iterable[str]) -> dict[str, float | int | None]:
    name = (
        "gold_test_age_crossval_metrics_cross_validation.csv"
        if dataset == "Gold_test"
        else "vitaldb_age_crossval_metrics_cross_validation.csv"
    )
    path = _external_named_csv(name)
    empty: dict[str, float | int | None] = {
        "Pearson_r": None,
        "Pearson_r_sd": None,
        "R2": None,
        "R2_sd": None,
        "MAE": None,
        "MAE_sd": None,
        "N_Participants": None,
    }
    if path is None:
        return empty
    frame = pd.read_csv(path)
    if "feature_kind" not in frame.columns:
        return empty
    for feature_kind in feature_kinds:
        rows = frame[frame["feature_kind"].astype(str).str.strip() == feature_kind]
        if rows.empty:
            continue
        row = rows.iloc[0]
        r_col = "r" if "r" in frame.columns else "pearson_r"
        n_col = "cv_n_groups" if "cv_n_groups" in frame.columns else "N"
        return {
            "Pearson_r": _finite_float_or_none(row[r_col]) if r_col in row.index else None,
            "Pearson_r_sd": _pick_csv_metric_sd(row, ("r_repeat_sd", "r_sd", "pearson_r_sd")),
            "R2": _finite_float_or_none(row["R2"]) if "R2" in row.index else None,
            "R2_sd": _pick_csv_metric_sd(row, ("R2_repeat_sd", "R2_sd")),
            "MAE": _finite_float_or_none(row["MAE"]) if "MAE" in row.index else None,
            "MAE_sd": _pick_csv_metric_sd(row, ("MAE_repeat_sd", "MAE_sd")),
            "N_Participants": int(row[n_col]) if n_col in row.index and pd.notna(row.get(n_col)) else None,
        }
    return empty


def _write_supplementary_table_3() -> None:
    temporal = _load_temporal_results()
    hpp_n = _cohort_participant_n("Gold (train+val)")

    def hpp_row_tabular(
        model: str,
        y_true_key: str,
        y_pred_key: str,
        *,
        repeat_metrics_key: str | None = None,
        r_sd_key: str | None = None,
        r2_sd_key: str | None = None,
        ypred_seed_prefix: str | None = None,
    ) -> dict[str, object]:
        yt = temporal.get(y_true_key)
        yp = temporal.get(y_pred_key)
        if yt is None or yp is None:
            return {
                "Dataset": "HPP pretraining set (train + validation)",
                "Model_or_features": model,
                "N_Participants": hpp_n,
                "Pearson_r": "",
                "R2": "",
                "MAE": "",
            }
        metrics = _age_metrics(temporal[y_true_key], temporal[y_pred_key])
        if repeat_metrics_key and r_sd_key and r2_sd_key and ypred_seed_prefix:
            r_sd = _coalesce_cv_sd(temporal, temporal.get(r_sd_key), repeat_metrics_key, "r")
            r2_sd = _coalesce_cv_sd(temporal, temporal.get(r2_sd_key), repeat_metrics_key, "r2")
            mae_sd = _mae_sd_from_tabular_prediction_seeds(temporal, y_true_key, ypred_seed_prefix)
        else:
            r_sd = r2_sd = mae_sd = None
        return {
            "Dataset": "HPP pretraining set (train + validation)",
            "Model_or_features": model,
            "N_Participants": hpp_n,
            "Pearson_r": _format_mean_pm_sd_2dp(metrics["Pearson_r"], r_sd),
            "R2": _format_mean_pm_sd_2dp(metrics["R2"], r2_sd),
            "MAE": _format_mean_pm_sd_2dp(metrics["MAE"], mae_sd),
        }

    metrics_emb = _age_metrics(temporal["y_true_emb"], temporal["y_pred_emb"])
    r_mean = metrics_emb["Pearson_r"]
    r_sd = _coalesce_cv_sd(temporal, temporal.get("r_emb_repeat_sd"), "embeddings_cv_repeat_metrics", "r")
    r2_mean = metrics_emb["R2"]
    r2_sd = _coalesce_cv_sd(temporal, temporal.get("r2_emb_repeat_sd"), "embeddings_cv_repeat_metrics", "r2")
    mae_mean = metrics_emb["MAE"]
    mae_sd = _mae_sd_from_emb_prediction_seeds(temporal)

    rows: list[dict[str, object]] = [
        {
            "Dataset": "HPP pretraining set (train + validation)",
            "Model_or_features": "PulseOx-FM",
            "N_Participants": hpp_n,
            "Pearson_r": _format_mean_pm_sd_2dp(r_mean, r_sd),
            "R2": _format_mean_pm_sd_2dp(r2_mean, r2_sd),
            "MAE": _format_mean_pm_sd_2dp(mae_mean, mae_sd),
        },
        hpp_row_tabular(
            "PyPPG",
            "y_true_pyppg",
            "y_pred_pyppg",
            repeat_metrics_key="pyppg_cv_repeat_metrics",
            r_sd_key="r_pyppg_repeat_sd",
            r2_sd_key="r2_pyppg_repeat_sd",
            ypred_seed_prefix="y_pred_pyppg_seed_",
        ),
        hpp_row_tabular(
            "WatchPAT",
            "y_true_watchpat",
            "y_pred_watchpat",
            repeat_metrics_key="watchpat_cv_repeat_metrics",
            r_sd_key="r_watchpat_repeat_sd",
            r2_sd_key="r2_watchpat_repeat_sd",
            ypred_seed_prefix="y_pred_watchpat_seed_",
        ),
    ]

    emb_pred = dict(zip(temporal.get("stage_keys_emb", []), temporal.get("y_pred_emb", [])))
    pyppg_pred = dict(zip(temporal.get("stage_keys_pyppg", []), temporal.get("y_pred_pyppg", [])))
    emb_true = dict(zip(temporal.get("stage_keys_emb", []), temporal.get("y_true_emb", [])))
    shared_keys = [key for key in temporal.get("stage_keys_emb", []) if key in pyppg_pred and key in emb_true]
    if shared_keys:
        y_true_ens = [emb_true[key] for key in shared_keys]
        y_pred_ens = [0.5 * (emb_pred[key] + pyppg_pred[key]) for key in shared_keys]
        ensemble_metrics = _age_metrics(y_true_ens, y_pred_ens)
        er_sd, e2_sd, em_sd = _ensemble_dispersion_sd_from_emb_seeds(temporal, shared_keys, emb_true, pyppg_pred)
    else:
        ensemble_metrics = {"Pearson_r": None, "R2": None, "MAE": None}
        er_sd = e2_sd = em_sd = None
    rows.append(
        {
            "Dataset": "HPP pretraining set (train + validation)",
            "Model_or_features": "Ensemble of PulseOx-FM + PyPPG",
            "N_Participants": hpp_n,
            "Pearson_r": _format_mean_pm_sd_2dp(ensemble_metrics["Pearson_r"], er_sd),
            "R2": _format_mean_pm_sd_2dp(ensemble_metrics["R2"], e2_sd),
            "MAE": _format_mean_pm_sd_2dp(ensemble_metrics["MAE"], em_sd),
        }
    )

    external_specs = [
        ("HPP test set", "PulseOx-FM", "Gold_test", ("mae_embeddings",)),
        ("HPP test set", "PyPPG", "Gold_test", ("pyppg_mean_only", "pyppg_all")),
        ("VitalDB cohort", "PulseOx-FM", "VitalDB", ("mae_embeddings",)),
        ("VitalDB cohort", "PyPPG", "VitalDB", ("pyppg_mean_only", "pyppg_all")),
    ]
    for dataset_label, model, dataset_key, feature_kinds in external_specs:
        ext = _external_age_metric_row_full(dataset_key, feature_kinds)
        rows.append(
            {
                "Dataset": dataset_label,
                "Model_or_features": model,
                "N_Participants": ext["N_Participants"],
                "Pearson_r": _format_mean_pm_sd_2dp(ext["Pearson_r"], ext["Pearson_r_sd"]),
                "R2": _format_mean_pm_sd_2dp(ext["R2"], ext["R2_sd"]),
                "MAE": _format_mean_pm_sd_2dp(ext["MAE"], ext["MAE_sd"]),
            }
        )

    for dataset_label, dataset_key in (("HPP test set", "Gold_test"), ("VitalDB cohort", "VitalDB")):
        ext = _external_ensemble_age_metric_row_full(dataset_key)
        rows.append(
            {
                "Dataset": dataset_label,
                "Model_or_features": "Ensemble of PulseOx-FM + PyPPG",
                "N_Participants": ext["N_Participants"],
                "Pearson_r": _format_mean_pm_sd_2dp(ext["Pearson_r"], ext["Pearson_r_sd"]),
                "R2": _format_mean_pm_sd_2dp(ext["R2"], ext["R2_sd"]),
                "MAE": _format_mean_pm_sd_2dp(ext["MAE"], ext["MAE_sd"]),
            }
        )

    table = pd.DataFrame(rows)
    table["N_Participants"] = pd.array(table["N_Participants"], dtype="Int64")
    path = MANUSCRIPT_DIR / "Supplementary Table 3.csv"
    table.to_csv(path, index=False)
    print(f"Saved {path}")


def _format_p_numeric_cell(p: object) -> str:
    if p is None:
        return ""
    try:
        v = float(p)
    except (TypeError, ValueError):
        return ""
    if not np.isfinite(v):
        return ""
    if abs(v) >= 1e-4:
        return f"{v:.6g}"
    return f"{v:.2e}"


def _hpp_classification_n_cell(row: pd.Series) -> str:
    return _format_clf_n_pos_total(row).replace("N=", "", 1)


def _f1_mean_sd_cells_hpp(row: pd.Series) -> tuple[str, str]:
    """Mean ± SD for F1 on HPP disease/medication CSVs when columns exist."""
    if "F1_age_sex_bmi" not in row.index:
        return "", ""
    try:
        m1 = float(row["F1_age_sex_bmi"])
    except (TypeError, ValueError):
        m1 = float("nan")
    try:
        m2 = float(row["F1_age_sex_bmi_embeddings"])
    except (TypeError, ValueError):
        m2 = float("nan")
    sd1 = _finite_float_or_none(row.get("F1_age_sex_bmi_std")) if "F1_age_sex_bmi_std" in row.index else None
    sd2 = _finite_float_or_none(row.get("F1_age_sex_bmi_embeddings_std")) if "F1_age_sex_bmi_embeddings_std" in row.index else None
    c1 = _format_mean_pm_sd_2dp(m1 if np.isfinite(m1) else None, sd1 if sd1 not in (None, 0.0) else None)
    c2 = _format_mean_pm_sd_2dp(m2 if np.isfinite(m2) else None, sd2 if sd2 not in (None, 0.0) else None)
    return c1, c2


def _supp_table4_hpp_measure_rows(reg_path: Path) -> list[dict[str, object]]:
    """Build HPP current-measures regression rows for Supplementary Table 4 (q < 0.01)."""
    df = pd.read_csv(reg_path)
    needed = {
        "target",
        "R_demo",
        "R_combined",
        "p_combined_vs_demo",
        "R_demo_std",
        "R_combined_std",
        "R2_age_sex_bmi",
        "R2_age_sex_bmi_embeddings",
        "R2_age_sex_bmi_std",
        "R2_age_sex_bmi_embeddings_std",
        "n_participants",
    }
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"regression_target_prediction_summary.csv missing columns: {missing}")

    df["R_demo"] = pd.to_numeric(df["R_demo"], errors="coerce")
    df["R_combined"] = pd.to_numeric(df["R_combined"], errors="coerce")
    df["p_combined_vs_demo"] = pd.to_numeric(df["p_combined_vs_demo"], errors="coerce")
    df["R_demo_std"] = pd.to_numeric(df["R_demo_std"], errors="coerce")
    df["R_combined_std"] = pd.to_numeric(df["R_combined_std"], errors="coerce")
    df["R2_age_sex_bmi"] = pd.to_numeric(df["R2_age_sex_bmi"], errors="coerce")
    df["R2_age_sex_bmi_embeddings"] = pd.to_numeric(df["R2_age_sex_bmi_embeddings"], errors="coerce")
    df["R2_age_sex_bmi_std"] = pd.to_numeric(df["R2_age_sex_bmi_std"], errors="coerce")
    df["R2_age_sex_bmi_embeddings_std"] = pd.to_numeric(df["R2_age_sex_bmi_embeddings_std"], errors="coerce")
    df["n_participants"] = pd.to_numeric(df["n_participants"], errors="coerce")

    df = df.dropna(subset=["R_demo", "R_combined", "p_combined_vs_demo"]).copy()
    if df.empty:
        return []

    q_arr = _bh_fdr_qvalues(df["p_combined_vs_demo"].to_numpy(dtype=float))
    df["q_bh"] = q_arr
    df = df[
        (df["q_bh"] < SUPP_TABLE4_HPP_MEASURES_FDR_ALPHA)
        & (df["R_combined"] > df["R_demo"])
    ].copy()
    if df.empty:
        return []

    df = df.sort_values(["R_combined", "target"], ascending=[False, True])

    out_rows: list[dict[str, object]] = []
    for _, row in df.iterrows():
        n_part = ""
        if np.isfinite(row["n_participants"]):
            n_part = str(int(row["n_participants"]))
        out_rows.append(
            {
                "Cohort_and_panel": "HPP sleep cohort — current measures",
                "Target": str(row["target"]),
                "Task_type": "regression",
                "Figure_primary_metric": "Pearson r",
                "ROC_AUC_or_Pearson_r_demo_mean_sd": _format_mean_pm_sd_2dp(
                    row["R_demo"], row["R_demo_std"] if np.isfinite(row["R_demo_std"]) else None
                ),
                "ROC_AUC_or_Pearson_r_embeddings_mean_sd": _format_mean_pm_sd_2dp(
                    row["R_combined"], row["R_combined_std"] if np.isfinite(row["R_combined_std"]) else None
                ),
                "R2_demo_mean_sd": _format_mean_pm_sd_2dp(
                    row["R2_age_sex_bmi"], row["R2_age_sex_bmi_std"] if np.isfinite(row["R2_age_sex_bmi_std"]) else None
                ),
                "R2_embeddings_mean_sd": _format_mean_pm_sd_2dp(
                    row["R2_age_sex_bmi_embeddings"],
                    row["R2_age_sex_bmi_embeddings_std"] if np.isfinite(row["R2_age_sex_bmi_embeddings_std"]) else None,
                ),
                "F1_demo_mean_sd": "",
                "F1_embeddings_mean_sd": "",
                "p_combined_vs_demo": _format_p_numeric_cell(row["p_combined_vs_demo"]),
                "BH_FDR_q": _format_p_numeric_cell(row["q_bh"]),
                "N_positive_over_N_participants_baseline": n_part,
                "Significance_stars_demo_vs_emb": _classification_auc_stars(row["p_combined_vs_demo"]),
            }
        )
    return out_rows


def _write_supplementary_table_4() -> None:
    """Fig. 4 panels b–c: same targets as plotted, plus N/F1/R²/P values withheld from composite bars."""
    dis_path = _first_existing(CLASSIFICATION_DISEASE_AUC_CSVS, "classification_binary_targets_AUC.csv")
    med_path = _first_existing(CLASSIFICATION_MEDICATION_AUC_CSVS, "classification_binary_medications_AUC.csv")
    vital_path = _first_existing(VITALDB_PREOP_PREDICTIONS_CSVS, "vitaldb_preop_predictions.csv")
    reg_path = _first_existing(REGRESSION_TARGET_SUMMARY_CSVS, "regression_target_prediction_summary.csv")

    hpp_plot_rows = _figure4_hpp_bar_plot_rows(dis_path, med_path)[0]
    vital_df = pd.read_csv(vital_path)
    dis_v, meas_v = _figure4_vitaldb_bar_plot_split(vital_df)

    out_rows: list[dict[str, object]] = []

    for pr in hpp_plot_rows:
        row = pr["row"]
        kind_s = pr["kind"]
        panel_group = (
            "HPP sleep cohort — current medication intake"
            if kind_s == "med"
            else "HPP sleep cohort — current diagnoses"
        )
        f1_demo, f1_emb = _f1_mean_sd_cells_hpp(row)
        out_rows.append(
            {
                "Cohort_and_panel": panel_group,
                "Target": str(row["target"]),
                "Task_type": "classification",
                "Figure_primary_metric": "ROC AUC",
                "ROC_AUC_or_Pearson_r_demo_mean_sd": _format_mean_pm_sd_2dp(pr["auc_d"], pr["std_d"]),
                "ROC_AUC_or_Pearson_r_embeddings_mean_sd": _format_mean_pm_sd_2dp(pr["auc_c"], pr["std_c"]),
                "R2_demo_mean_sd": "",
                "R2_embeddings_mean_sd": "",
                "F1_demo_mean_sd": f1_demo,
                "F1_embeddings_mean_sd": f1_emb,
                "p_combined_vs_demo": _format_p_numeric_cell(pr["p"]),
                "BH_FDR_q": _format_p_numeric_cell(pr.get("q_bh")),
                "N_positive_over_N_participants_baseline": _hpp_classification_n_cell(row),
                "Significance_stars_demo_vs_emb": _classification_auc_stars(pr["p"]),
            }
        )

    out_rows.extend(_supp_table4_hpp_measure_rows(reg_path))

    for pr in dis_v:
        row = pr["row"]
        out_rows.append(
            {
                "Cohort_and_panel": "VitalDB — preoperative diagnoses",
                "Target": str(row["target"]),
                "Task_type": "classification",
                "Figure_primary_metric": "ROC AUC",
                "ROC_AUC_or_Pearson_r_demo_mean_sd": _format_mean_pm_sd_2dp(pr["val_d"], pr["std_d"]),
                "ROC_AUC_or_Pearson_r_embeddings_mean_sd": _format_mean_pm_sd_2dp(pr["val_c"], pr["std_c"]),
                "R2_demo_mean_sd": "",
                "R2_embeddings_mean_sd": "",
                "F1_demo_mean_sd": "",
                "F1_embeddings_mean_sd": "",
                "p_combined_vs_demo": _format_p_numeric_cell(pr["p"]),
                "BH_FDR_q": _format_p_numeric_cell(pr.get("q_bh")),
                "N_positive_over_N_participants_baseline": _hpp_classification_n_cell(row),
                "Significance_stars_demo_vs_emb": _classification_auc_stars(pr["p"]),
            }
        )

    for pr in meas_v:
        row = pr["row"]
        try:
            r2d = float(row["R2_age_sex_bmi"])
            r2c = float(row["R2_age_sex_bmi_embeddings"])
        except (TypeError, ValueError):
            r2d = r2c = float("nan")
        r2_sd_d = _finite_float_or_none(row.get("R2_age_sex_bmi_std"))
        r2_sd_c = _finite_float_or_none(row.get("R2_age_sex_bmi_embeddings_std"))
        n_part = ""
        try:
            n_part = str(int(float(row["n_participants"])))
        except (TypeError, ValueError):
            if pd.notna(row.get("n_participants")):
                n_part = str(row["n_participants"])
        out_rows.append(
            {
                "Cohort_and_panel": "VitalDB — preoperative measures",
                "Target": str(row["target"]),
                "Task_type": "regression",
                "Figure_primary_metric": "Pearson r",
                "ROC_AUC_or_Pearson_r_demo_mean_sd": _format_mean_pm_sd_2dp(pr["val_d"], pr["std_d"]),
                "ROC_AUC_or_Pearson_r_embeddings_mean_sd": _format_mean_pm_sd_2dp(pr["val_c"], pr["std_c"]),
                "R2_demo_mean_sd": _format_mean_pm_sd_2dp(r2d if np.isfinite(r2d) else None, r2_sd_d),
                "R2_embeddings_mean_sd": _format_mean_pm_sd_2dp(r2c if np.isfinite(r2c) else None, r2_sd_c),
                "F1_demo_mean_sd": "",
                "F1_embeddings_mean_sd": "",
                "p_combined_vs_demo": _format_p_numeric_cell(pr["p"]),
                "BH_FDR_q": _format_p_numeric_cell(pr.get("q_bh")),
                "N_positive_over_N_participants_baseline": n_part,
                "Significance_stars_demo_vs_emb": _classification_auc_stars(pr["p"]),
            }
        )

    path = MANUSCRIPT_DIR / "Supplementary Table 4.csv"
    pd.DataFrame(out_rows).to_csv(path, index=False)
    print(f"Saved {path}")


def _write_table_1() -> None:
    source = _first_existing(COHORT_DESCRIPTION_CSVS, "cohort description CSV")
    table = pd.read_csv(source, keep_default_na=False)
    rename = {
        "Gold (train+val)": "HPP cohort (train + validation sets)",
        "Gold (test)": "HPP cohort (test set)",
        "VitalDB": "VitalDB cohort",
    }
    table = table.rename(columns=rename)
    table = table[table["Characteristic"].astype(str).ne("Unknown sex, n (%)")].reset_index(drop=True)
    country = pd.DataFrame(
        [
            {
                "Characteristic": "Country",
                "HPP cohort (train + validation sets)": "Israel",
                "HPP cohort (test set)": "Israel",
                "VitalDB cohort": "South Korea",
            }
        ]
    )
    table = pd.concat([country, table], ignore_index=True)
    path = MANUSCRIPT_DIR / "Table 1.csv"
    table.to_csv(path, index=False)
    print(f"Saved {path}")


def _write_supplementary_table_1(labels: pd.DataFrame, baseline_followup: pd.DataFrame) -> None:
    if not MEDICATION_INTAKE_CSV.is_file():
        raise FileNotFoundError(f"Could not find medication intake CSV: {MEDICATION_INTAKE_CSV}")
    meds = pd.read_csv(MEDICATION_INTAKE_CSV, low_memory=False)
    meds = meds.copy()
    meds["participant_id"] = meds["RegistrationCode"].astype(str)
    med_cols = [
        c for c in meds.columns
        if c not in {"RegistrationCode", "research_stage", "participant_id"} and "vitamin" not in c.lower()
    ]
    baseline_meds = _baseline_rows(meds, baseline_followup)
    rows: list[dict[str, object]] = []
    for medication in sorted(med_cols):
        summary = _prevalence_summary(baseline_meds[medication])
        if summary is None:
            continue
        n_positive, n_total, pct = summary
        rows.append(
            {
                "Medication_Category": medication,
                "Medication_or_ATC_Group_Used_in_Analysis": medication,
                "Prevalence_N_Total": f"{n_positive}/{n_total}",
                "Prevalence_Percent": round(pct, 1),
            }
        )
    path = MANUSCRIPT_DIR / "Supplementary Table 1.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"Saved {path}")


def _write_supplementary_table_2(labels: pd.DataFrame, baseline_followup: pd.DataFrame) -> None:
    disease_map = _load_disease_category_map()
    disease_cols = [c for c in disease_map if c in labels.columns]
    baseline_labels = _baseline_rows(labels, baseline_followup)
    rows: list[dict[str, object]] = []
    priority = {cat: idx for idx, cat in enumerate(DISEASE_CATEGORY_PRIORITY)}
    for disease in sorted(disease_cols, key=lambda d: (priority.get(disease_map[d], 999), d)):
        summary = _prevalence_summary(baseline_labels[disease])
        if summary is None:
            continue
        n_positive, n_total, pct = summary
        rows.append(
            {
                "Disease_Category": disease_map[disease],
                "Disease_Used_in_Analysis": disease,
                "Prevalence_N_Total": f"{n_positive}/{n_total}",
                "Prevalence_Percent": round(pct, 1),
            }
        )
    path = MANUSCRIPT_DIR / "Supplementary Table 2.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"Saved {path}")


def write_tables() -> None:
    disease_map = _load_disease_category_map()
    labels = _load_gold_labels(disease_map.keys())
    baseline_followup = _participant_baseline_followup(labels)
    _write_table_1()
    _write_supplementary_table_1(labels, baseline_followup)
    _write_supplementary_table_2(labels, baseline_followup)
    _write_supplementary_table_3()
    _write_supplementary_table_4()


def _add_delta_connector(
    ax: plt.Axes,
    x_baseline: float,
    x_model: float,
    y_baseline_top: float,
    y_model_top: float,
    model_value: float,
    baseline_value: float,
) -> None:
    baseline_start = y_baseline_top + 0.105
    model_start = y_model_top + 0.105
    y1 = max(baseline_start, model_start) + 0.04
    ax.plot(
        [x_baseline, x_baseline, x_model, x_model],
        [baseline_start, y1, y1, model_start],
        color="#333333",
        linewidth=1.0,
        clip_on=False,
        zorder=6,
    )
    ax.text(
        0.5 * (x_baseline + x_model),
        y1 + 0.018,
        f"+{model_value - baseline_value:.2f}",
        ha="center",
        va="bottom",
        fontsize=12,
        color="#333333",
        zorder=7,
    )


def _add_bracket_delta_connector(
    ax: plt.Axes,
    x_baseline: float,
    x_model: float,
    y_baseline_top: float,
    y_model_top: float,
    model_value: float,
    baseline_value: float,
    *,
    lift: float = 0.08,
    fontsize: float = 11.0,
) -> None:
    # Keep a small visual gap between value labels and bracket starts.
    baseline_start = y_baseline_top + 0.09
    model_start = y_model_top + 0.09
    y_bracket = max(baseline_start, model_start) + lift
    ax.plot(
        [x_baseline, x_baseline, x_model, x_model],
        [baseline_start, y_bracket, y_bracket, model_start],
        color="#333333",
        linewidth=1.0,
        clip_on=False,
        zorder=6,
    )
    xm = 0.5 * (x_baseline + x_model)
    ym = y_bracket + 0.012
    delta = model_value - baseline_value
    ax.text(
        xm,
        ym,
        f"{delta:+.2f}",
        ha="center",
        va="bottom",
        fontsize=fontsize,
        color="#333333",
        zorder=7,
    )


def _pearson_r_sd_from_repeat_metrics(temporal: dict, metrics_key: str) -> float | None:
    return _metric_sd_from_repeat_metrics(temporal, metrics_key, "r")


def _coalesce_repeat_sd(temporal: dict, explicit: object, metrics_key: str) -> float | None:
    if explicit is not None:
        try:
            v = float(explicit)
            if np.isfinite(v) and v > 0:
                return v
        except (TypeError, ValueError):
            pass
    return _pearson_r_sd_from_repeat_metrics(temporal, metrics_key)


def _external_ensemble_age_metrics(dataset: str) -> dict[str, float | int | None]:
    try:
        detailed = _read_external_case_predictions_csv(_external_detailed_csv())
    except FileNotFoundError:
        return {"Pearson_r": None, "Pearson_r_sd": None, "N": None}
    subset = detailed[detailed["dataset"].astype(str) == dataset].copy()
    if subset.empty:
        return {"Pearson_r": None, "Pearson_r_sd": None, "N": None}

    emb_base = "age_pred_mae_pretrained_cross_validation"
    py_base = "age_pred_pyppg_mean_only_cross_validation"
    if emb_base not in subset.columns or py_base not in subset.columns or "age_true" not in subset.columns:
        return {"Pearson_r": None, "Pearson_r_sd": None, "N": None}

    seed_re = re.compile(rf"^{re.escape(emb_base)}_seed_(\d+)$")
    emb_seed_cols = {int(m.group(1)): c for c in subset.columns if (m := seed_re.match(str(c)))}
    py_seed_cols = {}
    for c in subset.columns:
        m = re.match(rf"^{re.escape(py_base)}_seed_(\d+)$", str(c))
        if m:
            py_seed_cols[int(m.group(1))] = c
    shared_seeds = sorted(set(emb_seed_cols).intersection(py_seed_cols))

    rs: list[float] = []
    n_valid: int | None = None
    for seed in shared_seeds:
        y_true = subset["age_true"].to_numpy(float)
        y_pred = 0.5 * (subset[emb_seed_cols[seed]].to_numpy(float) + subset[py_seed_cols[seed]].to_numpy(float))
        valid = np.isfinite(y_true) & np.isfinite(y_pred)
        if int(np.sum(valid)) < 3:
            continue
        n_valid = int(np.sum(valid))
        rs.append(_age_metrics(y_true[valid], y_pred[valid])["Pearson_r"])

    y_true_cv = subset["age_true"].to_numpy(float)
    y_pred_cv = 0.5 * (subset[emb_base].to_numpy(float) + subset[py_base].to_numpy(float))
    valid_cv = np.isfinite(y_true_cv) & np.isfinite(y_pred_cv)
    if int(np.sum(valid_cv)) >= 3:
        point_r = _age_metrics(y_true_cv[valid_cv], y_pred_cv[valid_cv])["Pearson_r"]
        n_valid = int(np.sum(valid_cv))
    elif rs:
        point_r = float(np.mean(rs))
    else:
        point_r = None

    if rs and len(rs) >= 2:
        r_sd = float(np.std(rs, ddof=0))
    else:
        r_sd = None
    return {"Pearson_r": point_r, "Pearson_r_sd": r_sd, "N": n_valid}


def _external_ensemble_age_metric_row_full(dataset: str) -> dict[str, float | int | None]:
    """Mean metrics for 0.5*(PulseOx-FM CV + PyPPG CV) and SDs across CV-repeat seeds from detailed CSV."""
    empty: dict[str, float | int | None] = {
        "Pearson_r": None,
        "Pearson_r_sd": None,
        "R2": None,
        "R2_sd": None,
        "MAE": None,
        "MAE_sd": None,
        "N_Participants": None,
    }
    try:
        detailed = _read_external_case_predictions_csv(_external_detailed_csv())
    except FileNotFoundError:
        return empty
    subset = detailed[detailed["dataset"].astype(str) == dataset].copy()
    if subset.empty:
        return empty

    emb_base = "age_pred_mae_pretrained_cross_validation"
    py_base = "age_pred_pyppg_mean_only_cross_validation"
    if emb_base not in subset.columns or py_base not in subset.columns or "age_true" not in subset.columns:
        return empty

    seed_re = re.compile(rf"^{re.escape(emb_base)}_seed_(\d+)$")
    emb_seed_cols = {int(m.group(1)): c for c in subset.columns if (m := seed_re.match(str(c)))}
    py_seed_cols: dict[int, str] = {}
    for c in subset.columns:
        m = re.match(rf"^{re.escape(py_base)}_seed_(\d+)$", str(c))
        if m:
            py_seed_cols[int(m.group(1))] = c
    shared_seeds = sorted(set(emb_seed_cols).intersection(py_seed_cols))

    y_true_cv = subset["age_true"].to_numpy(float)
    y_pred_cv = 0.5 * (subset[emb_base].to_numpy(float) + subset[py_base].to_numpy(float))
    valid_cv = np.isfinite(y_true_cv) & np.isfinite(y_pred_cv)
    if int(np.sum(valid_cv)) < 3:
        return empty
    point = _age_metrics(y_true_cv[valid_cv], y_pred_cv[valid_cv])
    n_part = int(np.sum(valid_cv))

    rs: list[float] = []
    r2s: list[float] = []
    maes: list[float] = []
    for seed in shared_seeds:
        y_true = subset["age_true"].to_numpy(float)
        y_pred = 0.5 * (subset[emb_seed_cols[seed]].to_numpy(float) + subset[py_seed_cols[seed]].to_numpy(float))
        valid = np.isfinite(y_true) & np.isfinite(y_pred)
        if int(np.sum(valid)) < 3:
            continue
        m = _age_metrics(y_true[valid], y_pred[valid])
        if all(np.isfinite(m[k]) for k in ("Pearson_r", "R2", "MAE")):
            rs.append(m["Pearson_r"])
            r2s.append(m["R2"])
            maes.append(m["MAE"])

    def _sd(vals: list[float]) -> float | None:
        return float(np.std(vals, ddof=0)) if len(vals) >= 2 else None

    return {
        "Pearson_r": _finite_float_or_none(point["Pearson_r"]),
        "Pearson_r_sd": _sd(rs),
        "R2": _finite_float_or_none(point["R2"]),
        "R2_sd": _sd(r2s),
        "MAE": _finite_float_or_none(point["MAE"]),
        "MAE_sd": _sd(maes),
        "N_Participants": n_part,
    }


def _make_age_performance_panels(
    hpp_output: Path,
    vital_output: Path,
    *,
    row1_slot_px: int = 580,
    row1_left_slot_w: int | None = None,
    row1_right_slot_w: int | None = None,
) -> None:
    temporal = _load_temporal_results()
    vital = _external_age_cv_metrics("VitalDB")
    vital_mae = vital.get("mae_embeddings")
    vital_pyppg = vital.get("pyppg_mean_only") or vital.get("pyppg_all")
    vital_ensemble = _external_ensemble_age_metrics("VitalDB")

    hpp_color = PULSEOX_FM_PRIMARY_HEX
    hpp_light = PULSEOX_FM_LIGHT_HEX
    vital_color = "#c60c30"
    vital_light = "#f5d0d4"
    base_row1_px = 580
    fig_h_in = 3.2 * (row1_slot_px / base_row1_px)
    left_slot_w = int(row1_left_slot_w) if row1_left_slot_w is not None else row1_slot_px
    right_slot_w = int(row1_right_slot_w) if row1_right_slot_w is not None else row1_slot_px
    fig_w_left_in = max(2.2, fig_h_in * (left_slot_w / max(1, row1_slot_px)))
    fig_w_right_in = max(2.0, fig_h_in * (right_slot_w / max(1, row1_slot_px)))

    # Keep typography targets in the final composed Figure 3 after row-1 panel downscaling.
    final_target_pt = 10.0
    final_tick_pt = 8.0
    final_ylabel_pt = 8.0
    predicted_scale = row1_slot_px / max(1.0, fig_h_in * DPI)
    source_text_pt = float(np.clip(final_target_pt / max(predicted_scale, 1e-6), 9.5, 20.0))
    source_tick_pt = float(np.clip(final_tick_pt / max(predicted_scale, 1e-6), 7.5, 16.0))
    source_ylabel_pt = float(np.clip(final_ylabel_pt / max(predicted_scale, 1e-6), 7.5, 16.0))

    emb_pred = dict(zip(temporal.get("stage_keys_emb", []), temporal.get("y_pred_emb", [])))
    pyppg_pred = dict(zip(temporal.get("stage_keys_pyppg", []), temporal.get("y_pred_pyppg", [])))
    emb_true = dict(zip(temporal.get("stage_keys_emb", []), temporal.get("y_true_emb", [])))
    shared_keys = [k for k in temporal.get("stage_keys_emb", []) if k in pyppg_pred and k in emb_true]
    if shared_keys:
        y_true_ens = [emb_true[k] for k in shared_keys]
        y_pred_ens = [0.5 * (emb_pred[k] + pyppg_pred[k]) for k in shared_keys]
        hpp_ens_r = _age_metrics(y_true_ens, y_pred_ens)["Pearson_r"]
        hpp_ens_sd, _, _ = _ensemble_dispersion_sd_from_emb_seeds(temporal, shared_keys, emb_true, pyppg_pred)
    else:
        hpp_ens_r = None
        hpp_ens_sd = None

    hpp_bars = [
        {
            "label": "Ensemble model",
            "r": hpp_ens_r,
            "err": hpp_ens_sd,
            "face": hpp_color,
            "edge": "none",
            "hatch": None,
        },
        {
            "label": "PulseOx-FM\npretrained",
            "r": temporal.get("r_emb"),
            "err": _coalesce_repeat_sd(temporal, temporal.get("r_emb_repeat_sd"), "embeddings_cv_repeat_metrics"),
            "face": hpp_color,
            "edge": "none",
            "hatch": None,
        },
        {
            "label": "PyPPG\nfeatures",
            "r": temporal.get("r_pyppg"),
            "err": _coalesce_cv_sd(temporal, temporal.get("r_pyppg_repeat_sd"), "pyppg_cv_repeat_metrics", "r"),
            "face": hpp_light,
            "edge": hpp_color,
            "hatch": "///",
        },
        {
            "label": "WatchPAT\nfeatures",
            "r": temporal.get("r_watchpat"),
            "err": _coalesce_cv_sd(temporal, temporal.get("r_watchpat_repeat_sd"), "watchpat_cv_repeat_metrics", "r"),
            "face": hpp_light,
            "edge": hpp_color,
            "hatch": "///",
        },
    ]

    vital_bars = [
        {
            "label": "Ensemble model",
            "r": vital_ensemble["Pearson_r"],
            "err": vital_ensemble["Pearson_r_sd"],
            "face": vital_color,
            "edge": "none",
            "hatch": None,
        },
        {
            "label": "PyPPG\nfeatures",
            "r": vital_pyppg[1] if vital_pyppg else None,
            "err": vital_pyppg[4] if vital_pyppg and len(vital_pyppg) > 4 else None,
            "face": vital_light,
            "edge": vital_color,
            "hatch": "///",
        },
    ]

    hpp_n_raw = _cohort_participant_n("Gold (train+val)") or _finite_float_or_none(temporal.get("n_emb"))
    hpp_n = int(hpp_n_raw) if hpp_n_raw else None
    vital_n = _cohort_participant_n("VitalDB") or vital_ensemble["N"] or (vital_mae[2] if vital_mae else None)

    def _render_panel(
        bars: list[dict[str, object]],
        title: str,
        title_color: str,
        bg: str,
        outp: Path,
        connectors: list[tuple[int, int, float]],
        *,
        fig_w_in: float,
        x_positions: np.ndarray | None = None,
        xlim: tuple[float, float] | None = None,
    ) -> None:
        fig, ax = plt.subplots(figsize=(fig_w_in, fig_h_in))
        x = np.asarray(x_positions, dtype=float) if x_positions is not None else np.arange(len(bars), dtype=float)
        if len(x) != len(bars):
            raise ValueError("x_positions length must match number of bars")
        band_transform = blended_transform_factory(ax.transData, ax.transAxes)
        ax.add_patch(
            Rectangle(
                ((xlim[0] if xlim is not None else -0.55), 0),
                ((xlim[1] - xlim[0]) if xlim is not None else (len(bars) + 0.1)),
                1.16,
                transform=band_transform,
                facecolor=bg,
                edgecolor="none",
                clip_on=False,
                zorder=0,
            )
        )
        x_mid = 0.5 * ((xlim[0] + xlim[1]) if xlim is not None else float(np.min(x) + np.max(x)))
        ax.text(
            x_mid,
            1.04,
            title,
            ha="center",
            va="bottom",
            fontsize=source_text_pt,
            color=title_color,
            transform=band_transform,
        )

        heights: list[float] = []
        metric_tops: list[float] = []
        for idx, bar in enumerate(bars):
            value = bar["r"]
            if value is None or not np.isfinite(float(value)):
                heights.append(float("nan"))
                metric_tops.append(0.0)
                continue
            value = float(value)
            err_val = bar["err"]
            err = float(err_val) if err_val is not None and np.isfinite(float(err_val)) and float(err_val) > 0 else 0.0
            heights.append(value)
            metric_tops.append(value + err)
            ax.bar(
                x[idx],
                value,
                width=1.0,
                facecolor=bar["face"],
                edgecolor=bar["edge"],
                hatch=bar["hatch"],
                linewidth=1.0 if bar["hatch"] else 0.0,
                yerr=err if err > 0 else None,
                capsize=5 if err > 0 else 0,
                error_kw={"ecolor": "#111111", "elinewidth": 2.0, "capthick": 2.0},
                zorder=3,
            )
            ax.text(
                x[idx],
                value + err + 0.025,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=source_text_pt,
                zorder=5,
            )

        ax.set_ylabel("Pearson $r$ (age prediction)", fontsize=source_ylabel_pt)
        ax.set_xticks(x)
        ax.set_xticklabels([str(bar["label"]) for bar in bars], fontsize=source_tick_pt)
        if xlim is not None:
            ax.set_xlim(*xlim)
        else:
            ax.set_xlim(-0.6, len(bars) - 0.4)
        ax.set_ylim(0, 1.12)
        _style_axis(ax, tick_labelsize=int(round(source_tick_pt)))

        for i_from, i_to, lift in connectors:
            if i_from >= len(heights) or i_to >= len(heights):
                continue
            if np.isfinite(heights[i_from]) and np.isfinite(heights[i_to]):
                _add_bracket_delta_connector(
                    ax,
                    x_baseline=x[i_to],
                    x_model=x[i_from],
                    y_baseline_top=metric_tops[i_to],
                    y_model_top=metric_tops[i_from],
                    model_value=heights[i_from],
                    baseline_value=heights[i_to],
                    lift=lift,
                    fontsize=source_text_pt,
                )

        fig.tight_layout(pad=0.4)
        outp = Path(outp)
        fig.savefig(outp, dpi=DPI, bbox_inches=None, format=outp.suffix.lstrip(".").lower() or "png")
        fig.savefig(outp.with_suffix(".pdf"), format="pdf", bbox_inches=None)
        plt.close(fig)

    hpp_title = f"HPP sleep cohort, N={hpp_n:,}" if hpp_n else "HPP sleep cohort"
    vital_title = (
        f"VitalDB surgical cohort, N={int(vital_n):,}"
        if vital_n
        else "VitalDB surgical cohort"
    )
    _render_panel(
        hpp_bars,
        hpp_title,
        PULSEOX_FM_BAND_TITLE_HEX,
        PULSEOX_FM_BAND_BG_HEX,
        hpp_output,
        connectors=[(0, 2, 0.16), (1, 2, 0.09)],
        fig_w_in=fig_w_left_in,
        x_positions=np.array([0.8, 2.8, 4.8, 6.8], dtype=float),
        xlim=(0.0, 8.0),
    )
    _render_panel(
        vital_bars,
        vital_title,
        "#7d071d",
        "#fde8eb",
        vital_output,
        connectors=[(0, 1, 0.10)],
        fig_w_in=fig_w_right_in,
        x_positions=np.array([1.0, 3.0], dtype=float),
        xlim=(0.0, 4.0),
    )


def _make_external_scatter(dataset: str, title: str, color: str, output_path: Path) -> None:
    subset = _external_prediction_frame(dataset).dropna()
    if subset.empty:
        raise ValueError(f"No external prediction rows found for dataset {dataset}")
    _make_prediction_scatter(
        subset["age_true"].to_numpy(float),
        subset["age_pred"].to_numpy(float),
        title,
        color,
        output_path,
    )


def _make_temporal_hpp_scatter(output_path: Path) -> None:
    temporal = _load_temporal_results()
    y_true = np.asarray(temporal["y_true_emb"], dtype=float)
    y_pred = np.asarray(temporal["y_pred_emb"], dtype=float)
    _make_prediction_scatter(
        y_true,
        y_pred,
        "HPP cohort",
        PULSEOX_FM_PRIMARY_HEX,
        output_path,
        figsize=(3.48, 3.08),
    )


def _make_prediction_scatter(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    color: str,
    output_path: Path,
    *,
    figsize: tuple[float, float] = (3.35, 3.0),
    marker_size: float = 6.25,
) -> None:
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[valid]
    y_pred = y_pred[valid]
    r_value, r2_value, p_value = _scatter_stats(y_true, y_pred)
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(y_true, y_pred, s=marker_size, alpha=0.45, color=color, linewidths=0)
    lo = math.floor(min(y_true.min(), y_pred.min()))
    hi = math.ceil(max(y_true.max(), y_pred.max()))
    ax.plot([lo, hi], [lo, hi], color="black", linewidth=0.7)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Chronological age", fontsize=10)
    ax.set_ylabel("Predicted age", fontsize=10)
    p_text = "P not calculated" if p_value is None else ("P < 1e-300" if p_value == 0 else f"P = {p_value:.1e}")
    ax.text(
        0.05,
        0.95,
        f"Pearson $r$ = {r_value:.2f}\nR² = {r2_value:.2f}\n{p_text}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
    )
    ax.set_title(title, fontsize=10)
    _style_axis(ax)
    fig.tight_layout(pad=0.5)
    outp = Path(output_path)
    fig.savefig(outp, dpi=DPI, bbox_inches="tight", format=outp.suffix.lstrip(".").lower() or "png")
    fig.savefig(outp.with_suffix(".pdf"), format="pdf", bbox_inches="tight")
    plt.close(fig)


def build_figure_2() -> None:
    roots = (FIGURES_ROOT / "signal_reconstructions", RECONSTRUCTION_DIR, PROJECT_ROOT)
    panels = [
        Panel("a", _open_rgb(_find_one(["fig_example_forecast_2.png"], roots))),
        Panel("b", _open_rgb(_find_one(["fig_example_random_masking_50pct_3.png"], roots))),
        Panel("c", _open_rgb(_find_one(["fig_forecast_pearson.png"], roots))),
        Panel("d", _open_rgb(_find_one(["fig_random_masking_pearson.png"], roots))),
    ]
    margin, gap = 16, 10
    col_w = (FIGURE_WIDTH_PX - 2 * margin - gap) // 2
    top_h, bottom_h = 910, 650

    title_font_px = int(round(10 / 72 * DPI))
    title_font = _font(title_font_px, bold=True)
    title_h = title_font_px + 28  # vertical space reserved for section titles above panels

    canvas_h = 2 * margin + title_h + top_h + gap + bottom_h
    canvas = Image.new("RGB", (FIGURE_WIDTH_PX, canvas_h), WHITE)
    draw = ImageDraw.Draw(canvas)

    x_right = margin + col_w + gap
    right_col_w = FIGURE_WIDTH_PX - margin - x_right
    for text, col_x, col_w_px in [
        ("Forecasting evaluation", margin, col_w),
        ("Random masking evaluation", x_right, right_col_w),
    ]:
        bbox = draw.textbbox((0, 0), text, font=title_font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        tx = col_x + (col_w_px - tw) // 2
        ty = margin + (title_h - th) // 2
        draw.text((tx, ty), text, fill=BLACK, font=title_font)

    y_top = margin + title_h
    _paste_panel(canvas, panels[0], (margin, y_top, margin + col_w, y_top + top_h))
    _paste_panel(canvas, panels[1], (x_right, y_top, FIGURE_WIDTH_PX - margin, y_top + top_h))
    y = y_top + top_h + gap
    _paste_panel(canvas, panels[2], (margin, y, margin + col_w, y + bottom_h))
    _paste_panel(canvas, panels[3], (x_right, y, FIGURE_WIDTH_PX - margin, y + bottom_h))
    _save_canvas(canvas, "Figure 2")


def build_supplementary_figure_1() -> None:
    roots = (FIGURES_ROOT / "signal_reconstructions", RECONSTRUCTION_DIR, PROJECT_ROOT)
    panels = [
        Panel("a", _open_rgb(_find_one(["fig_forecast_mse.png"], roots))),
        Panel("b", _open_rgb(_find_one(["fig_random_masking_mse.png"], roots))),
    ]
    margin, gap = 34, 24
    col_w = (FIGURE_WIDTH_PX - 2 * margin - gap) // 2
    panel_h = 650
    canvas = Image.new("RGB", (FIGURE_WIDTH_PX, 2 * margin + panel_h), WHITE)
    _paste_panel(canvas, panels[0], (margin, margin, margin + col_w, margin + panel_h))
    _paste_panel(canvas, panels[1], (margin + col_w + gap, margin, FIGURE_WIDTH_PX - margin, margin + panel_h))
    _save_canvas(canvas, "Supplementary Figure 1")


def build_figure_3() -> None:
    scratch = MANUSCRIPT_DIR / "_source_panels"
    scratch.mkdir(parents=True, exist_ok=True)
    performance_hpp_path = scratch / "figure3_age_performance_hpp.png"
    performance_vital_path = scratch / "figure3_age_performance_vitaldb.png"
    hpp_scatter_path = scratch / "figure3_hpp_scatter.png"
    vital_scatter_path = scratch / "figure3_vitaldb_scatter.png"

    row2_left_h, row2_right_h = 752, 720
    row3_left_h, row3_right_h = 752, 720
    margin, gap = 50, 24
    fig3_h = FIGURE_MAX_HEIGHT_PX
    row1_h = fig3_h - (2 * margin + 2 * gap + row2_left_h + row3_left_h)
    if row1_h < 320:
        raise ValueError(f"Figure 3 row1_h={row1_h} too small; increase FIG_MAX_HEIGHT_CM or reduce lower rows.")

    row1_total_w = FIGURE_WIDTH_PX - 2 * margin - gap
    row1_left_w = int(round(0.6 * row1_total_w))
    row1_right_w = row1_total_w - row1_left_w
    _make_age_performance_panels(
        performance_hpp_path,
        performance_vital_path,
        row1_slot_px=row1_h,
        row1_left_slot_w=row1_left_w,
        row1_right_slot_w=row1_right_w,
    )
    _make_temporal_hpp_scatter(hpp_scatter_path)
    _make_external_scatter("VitalDB", "VitalDB cohort", "#D80B3A", vital_scatter_path)

    temporal_figs = FIGURES_ROOT / "temporal_age_prediction"
    age_umap = _find_one(["umap_recordings_embeddings_colored_by_age_*euclidean*.png"], [temporal_figs])
    hr_umap = _find_one(["umap_embeddings_colored_by_heart_rate_*euclidean*.png"], [temporal_figs])
    panels = [
        Panel("a", _open_rgb(_prefer_pdf_asset(performance_hpp_path))),
        Panel("b", _open_rgb(_prefer_pdf_asset(performance_vital_path))),
        Panel("c", _open_rgb(_prefer_pdf_asset(hpp_scatter_path))),
        Panel("d", _open_rgb(_prefer_pdf_asset(vital_scatter_path))),
        Panel("e", _open_rgb(age_umap)),
        Panel("f", _open_rgb(hr_umap)),
    ]
    col_w = (FIGURE_WIDTH_PX - 2 * margin - gap) // 2
    row1_x_left = margin
    row1_x_right = row1_x_left + row1_left_w + gap
    canvas = Image.new("RGB", (FIGURE_WIDTH_PX, fig3_h), WHITE)
    _paste_panel(canvas, panels[0], (row1_x_left, margin, row1_x_left + row1_left_w, margin + row1_h))
    _paste_panel(canvas, panels[1], (row1_x_right, margin, row1_x_right + row1_right_w, margin + row1_h))
    y2 = margin + row1_h + gap
    left_axis_x = margin + 88 + 36
    # Nudge right-column scatter panels (d, f) right for better visual alignment with panel b.
    right_axis_x = margin + col_w + gap + 100
    _paste_panel(canvas, panels[2], (margin, y2, margin + col_w, y2 + row2_left_h), axis_x=left_axis_x)
    row2_dy = (row2_left_h - row2_right_h) // 2
    _paste_panel(
        canvas,
        panels[3],
        (margin + col_w + gap, y2 + row2_dy, FIGURE_WIDTH_PX - margin, y2 + row2_dy + row2_right_h),
        axis_x=right_axis_x,
    )
    y3 = y2 + row2_left_h + gap
    _paste_panel(canvas, panels[4], (margin, y3, margin + col_w, y3 + row3_left_h), axis_x=left_axis_x)
    row3_dy = (row3_left_h - row3_right_h) // 2
    _paste_panel(
        canvas,
        panels[5],
        (margin + col_w + gap, y3 + row3_dy, FIGURE_WIDTH_PX - margin, y3 + row3_dy + row3_right_h),
        axis_x=right_axis_x,
    )
    _save_canvas(canvas, "Figure 3")
    shutil.rmtree(scratch, ignore_errors=True)


def _short_label(name: str, max_len: int = 42) -> str:
    name = str(name).strip()
    if len(name) <= max_len:
        return name
    return name[: max_len - 1] + "…"


def _nature_y_tick_label(raw: str) -> str:
    """Sentence-style axis tick: lowercase except leading capital and common abbreviations."""
    s = _short_label(str(raw).strip())
    if not s:
        return s
    lower = s.lower()
    out = lower[0].upper() + lower[1:] if len(lower) > 1 else lower.upper()
    for ac in ("bmi", "atc", "auc", "roc", "hdl", "ldl", "bp", "ecg", "osa", "copd", "gerd"):
        pat = re.compile(re.escape(ac), re.I)
        out = pat.sub(ac.upper(), out)
    return out


def _figure5_target_caption_label(raw: str) -> str:
    """Sentence-style label for Figure 5 y-axis; full string (no ellipsis); IBS and G6PD capitalised."""
    s = str(raw).strip()
    if not s:
        return s
    lower = s.lower()
    out = lower[0].upper() + lower[1:] if len(lower) > 1 else lower.upper()
    for ac in ("bmi", "atc", "auc", "roc", "hdl", "ldl", "bp", "ecg", "osa", "copd", "gerd"):
        pat = re.compile(re.escape(ac), re.I)
        out = pat.sub(ac.upper(), out)
    out = re.sub(r"\bibs\b", "IBS", out, flags=re.I)
    out = re.sub(r"\bg6pd\b", "G6PD", out, flags=re.I)
    return out


def _wrap_figure5_label_two_lines(text: str, max_single_line: int = 38) -> str:
    if len(text) <= max_single_line:
        return text
    mid = len(text) // 2
    best, best_d = -1, len(text) + 1
    for i, c in enumerate(text):
        if c == " ":
            d = abs(i - mid)
            if d < best_d:
                best_d, best = d, i
    if best <= 0:
        return text
    return f"{text[:best].rstrip()}\n{text[best + 1 :].lstrip()}"


def _figure4_panel_b_label(raw: str, *, max_single_line: int) -> str:
    """Panel-b label formatter with explicit manuscript-requested wraps."""
    formatted = _figure5_target_caption_label(raw)
    overrides = {
        "Heart valve disease": "Heart valve\ndisease",
        "Peptic ulcer disease": "Peptic ulcer\ndisease",
    }
    if formatted in overrides:
        return overrides[formatted]
    return _wrap_figure5_label_two_lines(formatted, max_single_line=max_single_line)


def _figure8_target_label(raw: str) -> str:
    """Strip 'Preoperative ' prefix and capitalise for Figure 8 y-axis labels."""
    label = re.sub(r"(?i)^preoperative\s+", "", raw.strip())
    return label[:1].upper() + label[1:] if label else raw


def _bh_fdr_qvalues(p_values: np.ndarray) -> np.ndarray:
    """Benjamini–Hochberg FDR q-values (same ordering as common stats implementations)."""
    p = np.asarray(p_values, dtype=float)
    n = int(p.size)
    if n == 0:
        return p
    order = np.argsort(p)
    ranked = p[order]
    q_sorted = np.empty(n, dtype=float)
    cum_min = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        q_adj = float(ranked[i] * n / rank)
        cum_min = min(cum_min, q_adj)
        q_sorted[i] = cum_min
    out = np.empty(n, dtype=float)
    out[order] = np.minimum(q_sorted, 1.0)
    return out


def _classification_auc_stars(p: object) -> str:
    """*, **, *** for raw p thresholds 0.05, 0.01, 0.001 (matches target_prediction_evaluation_short)."""
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return ""
    try:
        pv = float(p)
    except (TypeError, ValueError):
        return ""
    if not np.isfinite(pv) or pv >= 0.05:
        return ""
    if pv < 0.001:
        return "***"
    if pv < 0.01:
        return "**"
    return "*"


def _format_clf_n_pos_total(row: pd.Series) -> str:
    """N=positives/total participants (baseline), matching unified AUC bar export."""
    n_tot = row.get("n_participants")
    if n_tot is None or pd.isna(n_tot):
        n_tot = row.get("n")
    try:
        n_tot_i = int(float(n_tot))
    except (TypeError, ValueError):
        return "N=NA/NA"
    prev = row.get("prevalence")
    if prev is not None and pd.notna(prev) and n_tot_i > 0:
        try:
            n_pos = int(round(float(prev) * n_tot_i))
            n_pos = max(0, min(n_pos, n_tot_i))
            return f"N={n_pos}/{n_tot_i}"
        except (TypeError, ValueError):
            pass
    return f"N=—/{n_tot_i}"


_FIG5_MED_BG = "#FCE9FF"
_FIG5_MED_ACCENT = "#643D94"
_FIG5_DIS_BG = "#FFF3EB"
_FIG5_DIS_ACCENT = "#FF802B"
# Figure 8 – VitalDB preop colours
_FIG8_MEAS_BG = "#F0FEFF"
_FIG8_MEAS_ACCENT = "#12CBE0"
_FIG8_VITALDB_PRIMARY = "#8b0000"
_FIG8_VITALDB_EDGE = "#5c0000"
# Shared fixed bar thickness in y data-units (horizontal barh height) for Figure 4 panels b and c
_BAR_H = 0.34
# Panel b height model: inches per y-slot; panel c uses the same so bars match panel b after render
_FIG4_ROW_IN_PER_Y = 0.30
# Extend x-axes right in figure coordinates (plot + pastel background fill to figure edge)
_FIG4_BARS_AX_LEFT = 0.35
_FIG4_BARS_AX_RIGHT = 0.99
# Figure 4 bar panels: leave figure space for transAxes>1 annotation + rotated category labels.
_FIG4_BARS_SUBPLOT_RIGHT = 0.83
# Extra x-axis span (past xmax) so annotation column and category ribbons sit on tinted background
_FIG4_XLIM_RIGHT_PAD = 0.38
# Source scatter (panel a) tick/label sizes in target_prediction_evaluation_short.plot_regression_scatter_summary
_FIG4_SCATTER_AXIS_PT_SOURCE = 13.0
_FIG4_SCATTER_ANNOT_PT_SOURCE = 10.0
_FIG4_BAR_X_TICKS = (0.5, 0.7, 0.9)
# Horizontal position for significance stars on bar panels (matches 0.9 tick).
_FIG4_SIGNIFICANCE_STAR_X = 0.9
_FIG4_PEARSON_R_X_TICKS = (0.1, 0.3, 0.5, 0.7, 0.9)
# Fig. 4 panel c (VitalDB measures): sparse ticks to avoid overlap at narrow column width.
_FIG4_PEARSON_R_X_TICKS_VITALDB = (0.1, 0.5, 0.9)
_FIG4_AUC_XLIM_LO = 0.45
# Slightly past 1.0 so value labels, stars, and caps clear the axis edge.
_FIG4_BAR_AXIS_XMAX = 1.03
# Horizontal offset from error-bar tip to numeric annotation (data x-units).
_FIG4_BAR_LABEL_GAP = 0.016
# Fig. 4 panels b/c shared legend: neutral dark grey matches both bar palettes in print.
_FIG4_BC_LEGEND_EMB_HEX = "#3d3d3d"
_FIG4_BC_LEGEND_EMB_EDGE = "#2a2a2a"
# VitalDB panel c: side-by-side axes need modest extra height for x-labels and shared legend strip.
_FIG8_TWO_COL_EXTRA_H_IN = 0.45
# Embedded Fig. 4 panel c legend: larger bottom margin leaves a strip under the lower x-labels for fig.legend.
_FIG8_VITALDB_FIG_BOTTOM = 0.36
_FIG8_VITALDB_LEGEND_Y_FRAC = 0.018
# Fig. 4 panel c: shrink per-subplot axis boxes so they match panel b visual scale.
_FIG8_TWO_COL_AXIS_SCALE = 0.88
# Forest plot (Figure 4_v2 panels b/c): dot and connector colours.
_FOREST_DEMO_COLOR = "#a8a8a8"
_FOREST_DEMO_EDGE = "#707070"
_FOREST_DOT_SIZE = 4.5
_FOREST_CONNECT_COLOR = "#d4d4d4"
_FOREST_Y_OFFSET = 0.18   # vertical separation between combined (top) and demo (bottom) dots per target
_FOREST_LEGEND_VITALDB_HEX = "#505050"  # dark grey for VitalDB panel c legend marker


def _fig4_bc_legend_handles() -> tuple[Patch, Patch]:
    """Patches for demographics vs embeddings (shared by VitalDB panel c fig.legend)."""
    emb = Patch(
        facecolor=_FIG4_BC_LEGEND_EMB_HEX,
        edgecolor=_FIG4_BC_LEGEND_EMB_EDGE,
        linewidth=0.35,
        label="Age, sex, BMI + PulseOx-FM embeddings",
    )
    demo = Patch(
        facecolor="#f5f5f5",
        edgecolor="#b0b0b0",
        linewidth=0.35,
        hatch="///",
        label="Age, sex, BMI",
    )
    return emb, demo


def _fig4_save_bars(fig: plt.Figure, output_path: Path) -> None:
    """PNG for PIL compositing and sibling PDF vector master for manuscript submission."""
    outp = Path(output_path)
    face = fig.get_facecolor()
    png_kw = dict(dpi=DPI, facecolor=face, edgecolor="none", bbox_inches=None, pad_inches=0, format="png")
    pdf_kw = dict(facecolor=face, edgecolor="none", bbox_inches=None, pad_inches=0, format="pdf")
    with matplotlib.rc_context({"savefig.bbox": None}):
        fig.savefig(outp, **png_kw)
        fig.savefig(outp.with_suffix(".pdf"), **pdf_kw)


def _fig4_panel_y_span(n_targets: int) -> float:
    """Vertical extent in y data-units matching axhspan/ylim padding used in bar panels (±0.65)."""
    if n_targets <= 0:
        return 0.0
    return float(n_targets - 1) + 1.3


def _fig4_star_x_data(*, xlim_lo: float, xlim_hi: float) -> float:
    """Place significance stars at fixed x (major tick), clipped to the panel x-axis range."""
    x_star = float(_FIG4_SIGNIFICANCE_STAR_X)
    lo = float(xlim_lo) + 1e-4
    hi = float(xlim_hi) - 0.006
    return float(np.clip(x_star, lo, hi))


def _fig4_dynamic_xlim_and_star(
    val_d: np.ndarray,
    val_c: np.ndarray,
    std_d: np.ndarray,
    std_c: np.ndarray,
    *,
    xlim_lo_floor: float | None = None,
) -> tuple[float, float, float]:
    """Compute x_min, star_x, x_max using Figure 4_v2 panel-b logic."""
    all_lo = np.minimum(val_d - std_d, val_c - std_c)
    all_hi = np.maximum(val_d + std_d, val_c + std_c)
    x_min = float(np.nanmin(all_lo)) - 0.02
    if xlim_lo_floor is not None:
        x_min = max(float(xlim_lo_floor), x_min)
    star_x = float(np.nanmax(all_hi)) + 0.015
    x_max = star_x + 0.06
    return x_min, star_x, x_max


def _figure4_hpp_bar_plot_rows(disease_csv: Path, med_csv: Path) -> tuple[list[dict], int, int]:
    """Filtered + ordered classification rows shown in Fig. 4 panel b (HPP); each dict gains ``q_bh``."""
    dis = pd.read_csv(disease_csv)
    med = pd.read_csv(med_csv)
    if not {"AUC_age_sex_bmi", "AUC_age_sex_bmi_embeddings"}.issubset(dis.columns):
        raise ValueError(f"Unexpected disease AUC columns in {disease_csv}")
    if not {"AUC_age_sex_bmi", "AUC_age_sex_bmi_embeddings"}.issubset(med.columns):
        raise ValueError(f"Unexpected medication AUC columns in {med_csv}")
    if "p_combined_vs_demo" not in dis.columns or "p_combined_vs_demo" not in med.columns:
        raise ValueError("Figure 4 panel b CSVs require p_combined_vs_demo")

    dis = dis[~dis["target"].astype(str).str.contains("incidence", case=False)].copy()
    med = med[~med["target"].astype(str).str.lower().str.contains("vitamin")].copy()

    def _prep(df: pd.DataFrame, kind: str) -> list[dict]:
        out: list[dict] = []
        for _, row in df.iterrows():
            try:
                auc_d = float(_parse_mean_pm(row["AUC_age_sex_bmi"]))
                auc_c = float(_parse_mean_pm(row["AUC_age_sex_bmi_embeddings"]))
            except (TypeError, ValueError):
                continue
            if not np.isfinite(auc_d) or not np.isfinite(auc_c):
                continue
            pcd = row.get("p_combined_vs_demo")
            if pcd is None or pd.isna(pcd):
                continue
            try:
                pcd_f = float(pcd)
            except (TypeError, ValueError):
                continue
            if not np.isfinite(pcd_f):
                continue
            pdemo = row.get("p_demo")
            has_demo_p = pdemo is not None and not pd.isna(pdemo)
            try:
                std_d = float(row.get("AUC_age_sex_bmi_std", 0) or 0)
            except (TypeError, ValueError):
                std_d = 0.0
            try:
                std_c = float(row.get("AUC_age_sex_bmi_embeddings_std", 0) or 0)
            except (TypeError, ValueError):
                std_c = 0.0
            out.append(
                {
                    "kind": kind,
                    "row": row,
                    "auc_d": auc_d,
                    "auc_c": auc_c,
                    "delta": auc_c - auc_d,
                    "p": pcd_f,
                    "has_demo_p": has_demo_p,
                    "std_d": max(0.0, std_d),
                    "std_c": max(0.0, std_c),
                }
            )
        return out

    pool = _prep(dis, "dis") + _prep(med, "med")
    if not pool:
        raise ValueError("No classification rows with finite AUCs and p_combined_vs_demo for Figure 4 panel b.")

    fdr_pool = [r for r in pool if r["has_demo_p"]]
    if not fdr_pool:
        fdr_pool = list(pool)

    p_arr = np.array([r["p"] for r in fdr_pool], dtype=float)
    q_vals = _bh_fdr_qvalues(p_arr)
    q_by_key = {
        (r["kind"], str(r["row"]["target"])): float(q_vals[i]) for i, r in enumerate(fdr_pool)
    }

    kept = [
        r
        for r in pool
        if q_by_key.get((r["kind"], str(r["row"]["target"])), 1.0) < CLASSIFICATION_FDR_ALPHA
        and r["auc_c"] > r["auc_d"]
    ]
    if not kept:
        raise ValueError(
            "No targets passed Fig. 4 panel b filters (Benjamini–Hochberg q < "
            f"{CLASSIFICATION_FDR_ALPHA} on p_combined_vs_demo, and embedding AUC > demo AUC)."
        )

    med_rows = sorted([r for r in kept if r["kind"] == "med"], key=lambda x: x["delta"])
    dis_rows = sorted([r for r in kept if r["kind"] == "dis"], key=lambda x: x["delta"])
    n1, n2 = len(dis_rows), len(med_rows)
    plot_rows: list[dict] = [*med_rows, *dis_rows]

    for r in plot_rows:
        r["q_bh"] = float(q_by_key.get((r["kind"], str(r["row"]["target"])), 1.0))
    return plot_rows, n1, n2


def _figure4_vitaldb_bar_plot_split(df: pd.DataFrame) -> tuple[list[dict], list[dict]]:
    """Filtered + sorted rows shown in Fig. 4 panel c; each dict has ``val_*``, ``kind``, ``q_bh``."""

    def _prep_clf(row: pd.Series) -> dict | None:
        try:
            auc_d = float(row["AUC_age_sex_bmi"])
            auc_c = float(row["AUC_age_sex_bmi_embeddings"])
        except (TypeError, ValueError):
            return None
        if not np.isfinite(auc_d) or not np.isfinite(auc_c):
            return None
        try:
            pcd_f = float(row["p_combined_vs_demo"])
        except (TypeError, ValueError):
            return None
        if not np.isfinite(pcd_f):
            return None
        try:
            std_d = float(row.get("AUC_age_sex_bmi_std", 0) or 0)
        except (TypeError, ValueError):
            std_d = 0.0
        try:
            std_c = float(row.get("AUC_age_sex_bmi_embeddings_std", 0) or 0)
        except (TypeError, ValueError):
            std_c = 0.0
        return {
            "kind": "dis", "row": row,
            "val_d": auc_d, "val_c": auc_c, "delta": auc_c - auc_d, "p": pcd_f,
            "std_d": max(0.0, std_d), "std_c": max(0.0, std_c),
        }

    def _prep_reg(row: pd.Series) -> dict | None:
        try:
            r2_d = float(row["R2_age_sex_bmi"])
            r2_c = float(row["R2_age_sex_bmi_embeddings"])
        except (TypeError, ValueError):
            return None
        if not np.isfinite(r2_d) or not np.isfinite(r2_c):
            return None
        try:
            pcd_f = float(row["p_combined_vs_demo"])
        except (TypeError, ValueError):
            return None
        if not np.isfinite(pcd_f):
            return None
        try:
            std_r2_d = float(row.get("R2_age_sex_bmi_std", 0) or 0)
        except (TypeError, ValueError):
            std_r2_d = 0.0
        try:
            std_r2_c = float(row.get("R2_age_sex_bmi_embeddings_std", 0) or 0)
        except (TypeError, ValueError):
            std_r2_c = 0.0
        eps = 1e-9
        r_d = float(np.sqrt(max(0.0, r2_d)))
        r_c = float(np.sqrt(max(0.0, r2_c)))
        return {
            "kind": "meas", "row": row,
            "val_d": r_d, "val_c": r_c, "delta": r_c - r_d, "p": pcd_f,
            "std_d": max(0.0, std_r2_d / (2.0 * max(eps, r_d))),
            "std_c": max(0.0, std_r2_c / (2.0 * max(eps, r_c))),
        }

    pool: list[dict] = []
    for _, row in df.iterrows():
        task = str(row.get("task_type", "")).lower()
        if task == "classification":
            rec = _prep_clf(row)
        elif task == "regression":
            rec = _prep_reg(row)
        else:
            continue
        if rec is not None:
            pool.append(rec)

    if not pool:
        raise ValueError("No rows with finite metrics and p_combined_vs_demo in VitalDB CSV.")

    p_arr = np.array([r["p"] for r in pool], dtype=float)
    q_vals_full = _bh_fdr_qvalues(p_arr)

    kept: list[dict] = []
    for i, r in enumerate(pool):
        if float(q_vals_full[i]) < CLASSIFICATION_FDR_ALPHA and r["val_c"] > r["val_d"]:
            r2 = dict(r)
            r2["q_bh"] = float(q_vals_full[i])
            kept.append(r2)

    if not kept:
        raise ValueError(
            "No VitalDB targets passed Fig. 4 panel c filters "
            f"(Benjamini–Hochberg q < {CLASSIFICATION_FDR_ALPHA} and embedding > demo)."
        )

    dis_rows = sorted([r for r in kept if r["kind"] == "dis"], key=lambda x: x["delta"])
    meas_rows = sorted([r for r in kept if r["kind"] == "meas"], key=lambda x: x["delta"])
    return dis_rows, meas_rows


def _make_figure_5_unified_auc_bars(
    disease_csv: Path,
    med_csv: Path,
    output_path: Path,
    *,
    fig_width_in: float,
    ytick_wrap_max_chars: int | None = None,
    bars_axis_font_pt: float = 10.0,
    bars_value_font_pt: float = 8.5,
    bars_legend_font_pt: float = 8.5,
    subplot_right: float = _FIG4_BARS_AX_RIGHT,
    category_font_pt: float | None = None,
    ytick_labelsize: float | None = None,
) -> tuple[int, float, float, float]:
    """Horizontal AUC bars: FDR-filtered targets with higher embedding AUC; AUC labels and significance stars."""
    plot_rows, n1, n2 = _figure4_hpp_bar_plot_rows(disease_csv, med_csv)
    # Re-sort each group by auc_c ascending → highest at top of chart
    plot_rows = (
        sorted([r for r in plot_rows if r["kind"] == "med"], key=lambda x: x["auc_c"]) +
        sorted([r for r in plot_rows if r["kind"] == "dis"], key=lambda x: x["auc_c"])
    )

    wrap_n = ytick_wrap_max_chars
    if wrap_n is None:
        # Narrower figure width → shorter lines so y-tick text is not clipped.
        wrap_n = max(14, min(38, int(48.0 * float(fig_width_in) / 5.5)))

    block_gap = 0.55
    h = _BAR_H
    y_list: list[float] = []
    lab_list: list[str] = []
    y0_dis = float(n2 + block_gap) if n2 else 0.0
    for idx, r in enumerate(plot_rows):
        if idx < n2:
            y_list.append(float(idx))
        else:
            y_list.append(y0_dis + float(idx - n2))
        lab_list.append(_figure4_panel_b_label(str(r["row"]["target"]), max_single_line=wrap_n))

    y = np.asarray(y_list, dtype=float)
    auc_d_arr = np.array([r["auc_d"] for r in plot_rows], dtype=float)
    auc_c_arr = np.array([r["auc_c"] for r in plot_rows], dtype=float)
    std_d_arr = np.array([r["std_d"] for r in plot_rows], dtype=float)
    std_c_arr = np.array([r["std_c"] for r in plot_rows], dtype=float)

    y_span_lo = float(y.min()) - 0.65
    y_span_hi = float(y.max()) + 0.65
    fig_h_in = max(5.6, _FIG4_ROW_IN_PER_Y * (len(y) + block_gap) + 2.35)
    fig, ax = plt.subplots(figsize=(fig_width_in, fig_h_in))

    if n2:
        ax.axhspan(-0.65, n2 - 1 + 0.65, facecolor=_FIG5_MED_BG, edgecolor="none", zorder=0, alpha=0.75)
    if n1:
        if n2:
            d_lo, d_hi = n2 + block_gap - 0.65, n2 + block_gap + n1 - 1 + 0.65
        else:
            d_lo, d_hi = -0.65, n1 - 1 + 0.65
        ax.axhspan(d_lo, d_hi, facecolor=_FIG5_DIS_BG, edgecolor="none", zorder=0, alpha=0.75)

    err_kw = {"ecolor": "#111111", "elinewidth": 0.9, "capsize": 2}
    ax.barh(
        y + h / 2,
        auc_c_arr,
        height=h,
        color=PULSEOX_FM_PRIMARY_HEX,
        edgecolor=PULSEOX_FM_EDGE_HEX,
        linewidth=0.35,
        xerr=std_c_arr,
        error_kw=err_kw,
        zorder=3,
    )
    ax.barh(
        y - h / 2,
        auc_d_arr,
        height=h,
        color="#f5f5f5",
        edgecolor="#b0b0b0",
        linewidth=0.35,
        hatch="///",
        xerr=std_d_arr,
        error_kw=err_kw,
        zorder=2,
    )

    cat_font = category_font_pt if category_font_pt is not None else bars_axis_font_pt

    x_min, star_x, x_max = _fig4_dynamic_xlim_and_star(
        auc_d_arr,
        auc_c_arr,
        std_d_arr,
        std_c_arr,
        xlim_lo_floor=_FIG4_AUC_XLIM_LO,
    )
    if x_min < 0.5:
        ax.axvline(0.5, color="#555555", linestyle=":", linewidth=0.95, zorder=1.5)

    for i, pr in enumerate(plot_rows):
        yi = float(y[i])
        stars = _classification_auc_stars(pr["p"])
        if stars:
            ax.text(
                star_x,
                yi,
                stars,
                va="center",
                ha="left",
                fontsize=bars_axis_font_pt * 0.86,
                color="#222222",
                clip_on=False,
            )

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_span_lo, y_span_hi)
    ax.set_yticks(y, lab_list)
    for tlab, row in zip(ax.get_yticklabels(), plot_rows):
        tlab.set_multialignment("right")
        tlab.set_ha("right")
        if row["kind"] == "med":
            tlab.set_rotation(25)
            tlab.set_rotation_mode("anchor")
    ax.tick_params(axis="y", labelsize=ytick_labelsize if ytick_labelsize is not None else bars_axis_font_pt)
    ax.tick_params(axis="x", labelsize=bars_axis_font_pt)
    ax.set_xlabel("ROC AUC", fontsize=bars_axis_font_pt)
    ax.xaxis.set_major_locator(FixedLocator([0.6, 0.8]))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", color="#e8e8e8", linewidth=0.4, linestyle="--", zorder=1)

    trans = blended_transform_factory(ax.transAxes, ax.transData)
    lw_cat = 2.0
    if n1:
        if n2:
            d_lo, d_hi = n2 + block_gap - 0.65, n2 + block_gap + n1 - 1 + 0.65
        else:
            d_lo, d_hi = -0.65, n1 - 1 + 0.65
        ax.plot([0, 0], [d_lo, d_hi], color=_FIG5_DIS_ACCENT, linewidth=lw_cat,
                transform=trans, clip_on=False, zorder=5, solid_capstyle="butt")
        ax.text(
            1.04,
            0.5 * (d_lo + d_hi),
            "Current diagnoses",
            transform=trans,
            ha="left",
            va="center",
            fontsize=cat_font,
            rotation=270,
            color=_FIG5_DIS_ACCENT,
            clip_on=False,
        )
    if n2:
        ax.plot([0, 0], [-0.65, n2 - 1 + 0.65], color=_FIG5_MED_ACCENT, linewidth=lw_cat,
                transform=trans, clip_on=False, zorder=5, solid_capstyle="butt")
        ax.text(
            1.04,
            0.5 * (-0.65 + (n2 - 1 + 0.65)),
            "Current medication intake",
            transform=trans,
            ha="left",
            va="center",
            fontsize=cat_font,
            rotation=270,
            color=_FIG5_MED_ACCENT,
            clip_on=False,
        )

    fig.subplots_adjust(
        left=_FIG4_BARS_AX_LEFT,
        right=subplot_right,
        bottom=0.07,
        top=0.95,
    )
    _fig4_save_bars(fig, output_path)
    fh = float(fig.get_figheight())
    plt.close(fig)
    return len(y), fh, y_span_lo, y_span_hi


def build_figure_4() -> None:
    """Unified: a=Pearson r scatter (top-left), b=HPP AUC bars (right), c=VitalDB preop bars (bottom-left)."""
    scatter_src = _scatter_r_age_sex_embeddings_n53_path()
    dis_path = _first_existing(CLASSIFICATION_DISEASE_AUC_CSVS, "classification_binary_targets_AUC.csv")
    med_path = _first_existing(CLASSIFICATION_MEDICATION_AUC_CSVS, "classification_binary_medications_AUC.csv")
    vitaldb_path = _first_existing(VITALDB_PREOP_PREDICTIONS_CSVS, "vitaldb_preop_predictions.csv")

    scratch = MANUSCRIPT_DIR / "_source_panels_fig4"
    scratch.mkdir(parents=True, exist_ok=True)

    margin = 50
    gap = 36
    content_w = FIGURE_WIDTH_PX - 2 * margin - gap
    left_w_a = round(content_w * 0.70)
    right_w_b = content_w - left_w_a
    c_w = left_w_a

    img_scatter0 = _open_rgb(scatter_src)
    axis_pt = float(
        np.clip(
            _FIG4_SCATTER_AXIS_PT_SOURCE * left_w_a / max(1, img_scatter0.width),
            7.15,
            9.95,
        )
    )
    annot_pt = float(
        np.clip(
            _FIG4_SCATTER_ANNOT_PT_SOURCE * left_w_a / max(1, img_scatter0.width),
            6.85,
            8.9,
        )
    )
    leg_pt = float(np.clip(axis_pt - 0.85, 6.8, 8.85))

    panel_b_wrap = max(16, min(40, int(38 * (right_w_b / max(1e-6, 0.55 * content_w)))))

    pb = scratch / "fig4b_hpp_bars.png"
    pc = scratch / "fig4c_vitaldb_bars.png"
    _, fh_b, ys_lo_b, ys_hi_b = _make_figure_5_unified_auc_bars(
        dis_path,
        med_path,
        pb,
        fig_width_in=right_w_b / DPI,
        ytick_wrap_max_chars=panel_b_wrap,
        bars_axis_font_pt=axis_pt,
        bars_value_font_pt=annot_pt,
        bars_legend_font_pt=leg_pt,
        subplot_right=0.90,
        category_font_pt=axis_pt,
        ytick_labelsize=axis_pt * 0.78,
    )
    axes_frac_h = 0.95 - 0.07
    inches_per_y = (fh_b * axes_frac_h) / max(1e-6, (ys_hi_b - ys_lo_b))
    _make_figure_8_vitaldb_bars(
        vitaldb_path,
        pc,
        fig_width_in=c_w / DPI,
        inches_per_y_unit=inches_per_y,
        ytick_wrap_max_chars=None,
        bars_axis_font_pt=axis_pt,
        bars_value_font_pt=annot_pt,
        bars_legend_font_pt=leg_pt,
        subplot_right=_FIG4_BARS_SUBPLOT_RIGHT,
        category_font_pt=axis_pt,
    )

    def _fit_to_w(img: Image.Image, w: int) -> Image.Image:
        s = w / max(1, img.width)
        return img.resize((w, max(1, round(img.height * s))), Image.Resampling.LANCZOS)

    img_a = _fit_to_w(img_scatter0, left_w_a)
    img_b = _fit_to_w(_open_rgb(_prefer_pdf_asset(pb)), right_w_b)
    img_c = _fit_to_w(_open_rgb(_prefer_pdf_asset(pc)), c_w)

    h_a, h_b, h_c = img_a.height, img_b.height, img_c.height
    left_col_h = h_a + gap + h_c
    content_h = max(h_b, left_col_h)

    max_plot_h = FIGURE_MAX_HEIGHT_PX - 2 * margin
    scale_factor = 1.0
    if content_h > max_plot_h:
        scale_factor = max_plot_h / content_h
        img_a = img_a.resize((round(img_a.width * scale_factor), round(img_a.height * scale_factor)), Image.Resampling.LANCZOS)
        img_b = img_b.resize((round(img_b.width * scale_factor), round(img_b.height * scale_factor)), Image.Resampling.LANCZOS)
        img_c = img_c.resize((round(img_c.width * scale_factor), round(img_c.height * scale_factor)), Image.Resampling.LANCZOS)
        h_a, h_b, h_c = img_a.height, img_b.height, img_c.height
        left_col_h = h_a + gap + h_c
        content_h = max(h_b, left_col_h)

    canvas_h = margin + content_h + margin
    canvas = Image.new("RGB", (FIGURE_WIDTH_PX, canvas_h), WHITE)

    x_left = margin
    x_right = margin + left_w_a + gap

    _paste_panel(canvas, Panel("a", img_a), (x_left, margin, x_left + img_a.width, margin + h_a))
    y_c = margin + h_a + gap
    _paste_panel(
        canvas,
        Panel("c", img_c),
        (x_left, y_c, x_left + img_c.width, y_c + h_c),
        h_align="left",
    )
    _paste_panel(
        canvas,
        Panel("b", img_b),
        (x_right, margin, x_right + img_b.width, margin + h_b),
        h_align="left",
    )
    _save_canvas(canvas, "Figure 4")
    shutil.rmtree(scratch, ignore_errors=True)


# ---------------------------------------------------------------------------
# Figure 4_v2 – forest-plot versions of panels b and c
# ---------------------------------------------------------------------------

def _make_hpp_forest_panel(
    disease_csv: Path,
    med_csv: Path,
    output_path: Path,
    *,
    fig_width_in: float,
    axis_pt: float = 10.0,
    subplot_right: float = _FIG4_BARS_SUBPLOT_RIGHT,
    ytick_labelsize: float | None = None,
    wrap_n: int | None = None,
) -> tuple[int, float, float, float]:
    """Forest plot for HPP classification targets (Figure 4_v2 panel b).

    Returns (n_rows, fig_height_in, y_span_lo, y_span_hi).
    """
    plot_rows, n_dis, n_med = _figure4_hpp_bar_plot_rows(disease_csv, med_csv)

    if wrap_n is None:
        wrap_n = max(14, min(38, int(48.0 * float(fig_width_in) / 5.5)))

    # Re-sort each group by AUC_combined ascending → highest ends at top of its section.
    med_rows = sorted([r for r in plot_rows if r["kind"] == "med"], key=lambda x: x["auc_c"])
    dis_rows = sorted([r for r in plot_rows if r["kind"] == "dis"], key=lambda x: x["auc_c"])
    block_gap = 0.55
    y0_dis = float(n_med + block_gap) if n_med else 0.0

    rows_ordered: list[dict] = []
    y_list: list[float] = []
    for idx, r in enumerate(med_rows):
        rows_ordered.append(r)
        y_list.append(float(idx))
    for idx, r in enumerate(dis_rows):
        rows_ordered.append(r)
        y_list.append(y0_dis + float(idx))

    y = np.asarray(y_list, dtype=float)
    _PAD = 0.45
    y_span_lo = float(y.min()) - _PAD
    y_span_hi = float(y.max()) + _PAD
    n_rows = len(rows_ordered)
    fig_h_in = max(4.5, _FIG4_ROW_IN_PER_Y * (n_rows + block_gap) + 2.0)

    fig, ax = plt.subplots(figsize=(fig_width_in, fig_h_in))

    # Category backgrounds
    if n_med:
        ax.axhspan(-_PAD, n_med - 1 + _PAD, facecolor=_FIG5_MED_BG, edgecolor="none", zorder=0, alpha=0.75)
    if n_dis:
        d_lo = (y0_dis - _PAD) if n_med else -_PAD
        d_hi = (y0_dis + n_dis - 1 + _PAD) if n_med else (n_dis - 1 + _PAD)
        ax.axhspan(d_lo, d_hi, facecolor=_FIG5_DIS_BG, edgecolor="none", zorder=0, alpha=0.75)

    val_d = np.array([r["auc_d"] for r in rows_ordered])
    val_c = np.array([r["auc_c"] for r in rows_ordered])
    std_d = np.array([r["std_d"] for r in rows_ordered])
    std_c = np.array([r["std_c"] for r in rows_ordered])
    labs = [_figure4_panel_b_label(str(r["row"]["target"]), max_single_line=wrap_n) for r in rows_ordered]

    y_comb = y + _FOREST_Y_OFFSET
    y_demo = y - _FOREST_Y_OFFSET

    # PulseOx-FM dots (teal) — combined model on top
    ax.errorbar(
        val_c, y_comb, xerr=std_c, fmt="o",
        color=PULSEOX_FM_PRIMARY_HEX, markeredgecolor="none",
        markersize=_FOREST_DOT_SIZE, linestyle="none", zorder=4,
        ecolor=PULSEOX_FM_PRIMARY_HEX, elinewidth=0.8, capsize=2.5, capthick=0.8,
    )
    # Demographics dots (silver) — below
    ax.errorbar(
        val_d, y_demo, xerr=std_d, fmt="o",
        color=_FOREST_DEMO_COLOR, markeredgecolor="none",
        markersize=_FOREST_DOT_SIZE, linestyle="none", zorder=3,
        ecolor=_FOREST_DEMO_COLOR, elinewidth=0.8, capsize=2.5, capthick=0.8,
    )

    # Tight x-axis: no clip at 0.5 so silver dots below random chance are visible
    all_lo = np.minimum(val_d - std_d, val_c - std_c)
    all_hi = np.maximum(val_d + std_d, val_c + std_c)
    x_min = float(np.nanmin(all_lo)) - 0.02
    star_x = float(np.nanmax(all_hi)) + 0.015
    x_max = star_x + 0.06
    ax.set_xlim(x_min, x_max)
    if x_min < 0.5:
        ax.axvline(0.5, color="black", linestyle=":", linewidth=0.95, zorder=1.5)

    # Significance stars at centre y of each target
    for i, r in enumerate(rows_ordered):
        stars = _classification_auc_stars(r["p"])
        if stars:
            ax.text(star_x, y[i], stars, va="center", ha="left",
                    fontsize=axis_pt * 0.86, color="#222222", clip_on=False)

    ax.set_ylim(y_span_lo, y_span_hi)
    ax.set_yticks(y, labs)
    for tl, row in zip(ax.get_yticklabels(), rows_ordered):
        tl.set_multialignment("right")
        tl.set_ha("right")
        if row["kind"] == "med":
            tl.set_rotation(25)
            tl.set_rotation_mode("anchor")
    ax.tick_params(axis="y", labelsize=ytick_labelsize if ytick_labelsize is not None else axis_pt)
    ax.tick_params(axis="x", labelsize=axis_pt)
    ax.set_xlabel("ROC AUC", fontsize=axis_pt)
    ax.xaxis.set_major_locator(FixedLocator([0.6, 0.8]))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", color="#e8e8e8", linewidth=0.4, linestyle="--", zorder=1)
    for i in range(len(y) - 1):
        ax.axhline(0.5 * (y[i] + y[i + 1]), color="#cccccc", linewidth=0.4, zorder=0.5)

    # Category ribbons
    trans = blended_transform_factory(ax.transAxes, ax.transData)
    lw_cat = 2.0
    if n_dis:
        d_lo2 = (y0_dis - _PAD) if n_med else -_PAD
        d_hi2 = (y0_dis + n_dis - 1 + _PAD) if n_med else (n_dis - 1 + _PAD)
        ax.plot([0, 0], [d_lo2, d_hi2], color=_FIG5_DIS_ACCENT, linewidth=lw_cat,
                transform=trans, clip_on=False, zorder=5, solid_capstyle="butt")
        ax.text(1.04, 0.5 * (d_lo2 + d_hi2), "Current diagnoses",
                transform=trans, ha="left", va="center", fontsize=axis_pt,
                rotation=270, color=_FIG5_DIS_ACCENT, clip_on=False)
    if n_med:
        ax.plot([0, 0], [-_PAD, n_med - 1 + _PAD], color=_FIG5_MED_ACCENT, linewidth=lw_cat,
                transform=trans, clip_on=False, zorder=5, solid_capstyle="butt")
        ax.text(1.04, 0.5 * (-_PAD + (n_med - 1 + _PAD)), "Current medication intake",
                transform=trans, ha="left", va="center", fontsize=axis_pt,
                rotation=270, color=_FIG5_MED_ACCENT, clip_on=False)

    fig.subplots_adjust(left=_FIG4_BARS_AX_LEFT, right=subplot_right, bottom=0.07, top=0.95)
    _fig4_save_bars(fig, output_path)
    fh = float(fig.get_figheight())
    plt.close(fig)
    return len(y), fh, y_span_lo, y_span_hi


def _make_vitaldb_forest_panels(
    vitaldb_csv: Path,
    output_path: Path,
    *,
    fig_width_in: float,
    inches_per_y_unit: float | None = None,
    axis_pt: float = 10.0,
    subplot_right: float = _FIG4_BARS_SUBPLOT_RIGHT,
    wrap_n: int | None = None,
) -> tuple[int, float]:
    """Forest plot for VitalDB preop targets (Figure 4_v2 panel c).

    Returns (n_rows_total, fig_height_in).
    """
    df = pd.read_csv(vitaldb_csv)
    if "p_combined_vs_demo" not in df.columns:
        raise ValueError("Figure 4_v2 panel c requires p_combined_vs_demo in vitaldb_preop_predictions.csv")

    dis_rows_raw, meas_rows_raw = _figure4_vitaldb_bar_plot_split(df)
    # Re-sort each group by val_c ascending → highest at top.
    dis_rows = sorted(dis_rows_raw, key=lambda x: x["val_c"])
    meas_rows = sorted(meas_rows_raw, key=lambda x: x["val_c"])
    n_dis, n_meas = len(dis_rows), len(meas_rows)

    if wrap_n is None:
        panel_w_in = float(fig_width_in) / 2.0 if (n_dis and n_meas) else float(fig_width_in)
        wrap_n = max(14, min(38, int(48.0 * panel_w_in / 5.5)))

    _BLOCK_GAP_Y = 0.55
    TOP_OH, BOT_OH = 1.25, 0.85
    _VPAD = 0.45  # must match _PAD_V inside _draw_vitaldb_forest
    if inches_per_y_unit is not None:
        h_dis_in = max(0.55, inches_per_y_unit * max(0.1, n_dis - 1 + 2 * _VPAD)) if n_dis else 0.0
        h_meas_in = max(0.55, inches_per_y_unit * max(0.1, n_meas - 1 + 2 * _VPAD)) if n_meas else 0.0
    else:
        h_dis_in = max(1.6, n_dis * _FIG4_ROW_IN_PER_Y + TOP_OH) if n_dis else 0.0
        h_meas_in = max(1.4, n_meas * _FIG4_ROW_IN_PER_Y + BOT_OH) if n_meas else 0.0

    if n_dis and n_meas:
        ax_left_two = 0.25
        top_frac, bot_frac = 0.94, float(_FIG8_VITALDB_FIG_BOTTOM)
        band_frac = top_frac - bot_frac
        _extra_h = 0.2  # reduced to shorten panel c; legend fits in bot_frac area
        fig_h_in = (max(h_dis_in, h_meas_in) + _extra_h) / max(1e-6, band_frac)
        fig = plt.figure(figsize=(fig_width_in, fig_h_in))
        content_w_in = fig_width_in * (subplot_right - ax_left_two)
        subplot_w_in = max(1e-6, (content_w_in - ax_left_two * fig_width_in) / 2.0)
        w_col_frac = subplot_w_in / fig_width_in
        gap_frac = (ax_left_two * fig_width_in) / fig_width_in
        x0 = ax_left_two
        x1 = x0 + w_col_frac + gap_frac
        scale = float(np.clip(_FIG8_TWO_COL_AXIS_SCALE, 0.55, 1.0))
        w_scaled = w_col_frac * scale
        x_pad = 0.5 * (w_col_frac - w_scaled)
        ax_dis = fig.add_axes([x0 + x_pad, top_frac - h_dis_in / fig_h_in, w_scaled, h_dis_in / fig_h_in])
        ax_meas = fig.add_axes([x1 + x_pad, top_frac - h_meas_in / fig_h_in, w_scaled, h_meas_in / fig_h_in])
    elif n_dis:
        fig_h_in = h_dis_in
        fig, ax_dis = plt.subplots(figsize=(fig_width_in, fig_h_in))
        ax_meas = None
        fig.subplots_adjust(left=_FIG4_BARS_AX_LEFT, right=subplot_right,
                            bottom=_FIG8_VITALDB_FIG_BOTTOM, top=0.95)
    else:
        fig_h_in = h_meas_in
        fig, ax_meas = plt.subplots(figsize=(fig_width_in, fig_h_in))
        ax_dis = None
        fig.subplots_adjust(left=_FIG4_BARS_AX_LEFT, right=subplot_right,
                            bottom=_FIG8_VITALDB_FIG_BOTTOM, top=0.95)

    def _draw_vitaldb_forest(
        ax: plt.Axes,
        rows: list[dict],
        xlabel: str,
        xlim_lo: float,
        cat_lbl: str,
        cat_accent: str,
        bg: str,
        *,
        x_ticks: tuple[float, ...] | None = None,
        x_tick_labels: tuple[str, ...] | None = None,
        star_shift_left: float = 0.0,
    ) -> None:
        n = len(rows)
        y = np.arange(n, dtype=float)
        val_d = np.array([r["val_d"] for r in rows])
        val_c = np.array([r["val_c"] for r in rows])
        std_d = np.array([r["std_d"] for r in rows])
        std_c = np.array([r["std_c"] for r in rows])
        labs = [
            _wrap_figure5_label_two_lines(
                _figure8_target_label(str(r["row"]["target"])), max_single_line=wrap_n
            )
            for r in rows
        ]
        _PAD_V = 0.45
        y_lo, y_hi = -_PAD_V, n - 1 + _PAD_V
        ax.axhspan(y_lo, y_hi, facecolor=bg, edgecolor="none", zorder=0, alpha=0.75)
        y_comb = y + _FOREST_Y_OFFSET
        y_demo = y - _FOREST_Y_OFFSET
        # PulseOx-FM dots (darkred) — combined model on top
        ax.errorbar(
            val_c, y_comb, xerr=std_c, fmt="o",
            color=_FIG8_VITALDB_PRIMARY, markeredgecolor="none",
            markersize=_FOREST_DOT_SIZE, linestyle="none", zorder=4,
            ecolor=_FIG8_VITALDB_PRIMARY, elinewidth=0.8, capsize=2.5, capthick=0.8,
        )
        # Demographics dots (silver) — below
        ax.errorbar(
            val_d, y_demo, xerr=std_d, fmt="o",
            color=_FOREST_DEMO_COLOR, markeredgecolor="none",
            markersize=_FOREST_DOT_SIZE, linestyle="none", zorder=3,
            ecolor=_FOREST_DEMO_COLOR, elinewidth=0.8, capsize=2.5, capthick=0.8,
        )
        all_lo = np.minimum(val_d - std_d, val_c - std_c)
        all_hi = np.maximum(val_d + std_d, val_c + std_c)
        x_min = max(xlim_lo, float(np.nanmin(all_lo)) - 0.02)
        x_star_data = float(np.nanmax(all_hi)) + 0.015
        x_max = x_star_data + 0.06
        star_x = _fig4_star_x_data(xlim_lo=x_min, xlim_hi=x_max)
        star_x = max(x_min + 1e-4, star_x - max(0.0, float(star_shift_left)))
        xt = [0.65, 0.70] if x_ticks is None else list(x_ticks)
        if xt:
            x_min = min(x_min, min(xt) - 0.05)
            x_max = max(x_max, max(xt) + 0.05)
        ax.set_xlim(x_min, x_max)
        if xlim_lo >= 0.4 and x_min < 0.5:
            ax.axvline(0.5, color="black", linestyle=":", linewidth=0.95, zorder=1.5)
        for i, r in enumerate(rows):
            stars = _classification_auc_stars(r["p"])
            if stars:
                ax.text(star_x, y[i], stars, va="center", ha="left",
                        fontsize=axis_pt * 0.86, color="#222222", clip_on=False)
        ax.set_ylim(y_lo, y_hi)
        ax.set_yticks(y, labs)
        for tl in ax.get_yticklabels():
            tl.set_multialignment("right")
        ax.tick_params(axis="y", labelsize=axis_pt)
        ax.tick_params(axis="x", labelsize=axis_pt)
        ax.set_xlabel(xlabel, fontsize=axis_pt)
        ax.xaxis.set_major_locator(FixedLocator(xt))
        if x_tick_labels is None:
            x_tick_labels = tuple(f"{t:g}" for t in xt)
        ax.set_xticks(xt, x_tick_labels)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="x", color="#e8e8e8", linewidth=0.4, linestyle="--", zorder=1)
        for i in range(n - 1):
            ax.axhline(i + 0.5, color="#cccccc", linewidth=0.4, zorder=0.5)
        trans = blended_transform_factory(ax.transAxes, ax.transData)
        ax.plot([0, 0], [y_lo, y_hi], color=cat_accent, linewidth=2.0,
                transform=trans, clip_on=False, zorder=5, solid_capstyle="butt")
        ax.text(1.045, 0.5 * (y_lo + y_hi), cat_lbl, transform=trans,
                ha="left", va="center", fontsize=axis_pt, rotation=270,
                color=cat_accent, multialignment="center", clip_on=False)

    if ax_dis is not None:
        _draw_vitaldb_forest(
            ax_dis,
            dis_rows,
            "ROC AUC",
            0.63,
            "Preoperation\ndiagnoses",
            _FIG5_DIS_ACCENT,
            _FIG5_DIS_BG,
            x_ticks=(0.65, 0.75),
            x_tick_labels=("0.65", "0.75"),
        )
    if ax_meas is not None:
        _draw_vitaldb_forest(
            ax_meas,
            meas_rows,
            "Pearson $r$",
            0.0,
            "Preoperation\nmeasures",
            _FIG8_MEAS_ACCENT,
            _FIG8_MEAS_BG,
            x_ticks=(0.25, 0.5),
            x_tick_labels=("0.25", "0.5"),
            star_shift_left=0.02,
        )

    # Legend shown regardless of whether one or two sub-panels are present
    handles_c = [
        plt.Line2D([0], [0], marker="o", color="none", markersize=_FOREST_DOT_SIZE,
                   markerfacecolor=_FOREST_LEGEND_VITALDB_HEX, markeredgecolor="none",
                   label="Age, sex, BMI + PulseOx-FM embeddings"),
        plt.Line2D([0], [0], marker="o", color="none", markersize=_FOREST_DOT_SIZE,
                   markerfacecolor=_FOREST_DEMO_COLOR, markeredgecolor="none",
                   label="Age, sex, BMI"),
    ]
    fig.legend(handles=handles_c, loc="lower center", ncol=2, fontsize=axis_pt * 0.85,
               frameon=False, bbox_to_anchor=(0.5, 0.0))

    _fig4_save_bars(fig, output_path)
    total_h = float(fig.get_figheight())
    plt.close(fig)
    return n_dis + n_meas, total_h


def build_figure_4_v2() -> None:
    """Figure 4_v2: panel a=Pearson r scatter (unchanged), panels b/c as forest plots."""
    scatter_src = _scatter_r_age_sex_embeddings_n53_path()
    dis_path = _first_existing(CLASSIFICATION_DISEASE_AUC_CSVS, "classification_binary_targets_AUC.csv")
    med_path = _first_existing(CLASSIFICATION_MEDICATION_AUC_CSVS, "classification_binary_medications_AUC.csv")
    vitaldb_path = _first_existing(VITALDB_PREOP_PREDICTIONS_CSVS, "vitaldb_preop_predictions.csv")

    scratch = MANUSCRIPT_DIR / "_source_panels_fig4_v2"
    scratch.mkdir(parents=True, exist_ok=True)

    margin = 50
    gap = 36
    content_w = FIGURE_WIDTH_PX - 2 * margin - gap
    left_w_a = round(content_w * 0.70)
    right_w_b = content_w - left_w_a
    c_w = left_w_a

    img_scatter0 = _open_rgb(scatter_src)
    axis_pt = float(
        np.clip(
            _FIG4_SCATTER_AXIS_PT_SOURCE * left_w_a / max(1, img_scatter0.width),
            7.15,
            9.95,
        )
    )
    panel_b_wrap = max(16, min(40, int(38 * (right_w_b / max(1e-6, 0.55 * content_w)))))

    pb = scratch / "fig4v2b_hpp_forest.png"
    pc = scratch / "fig4v2c_vitaldb_forest.png"
    _, fh_b, ys_lo_b, ys_hi_b = _make_hpp_forest_panel(
        dis_path, med_path, pb,
        fig_width_in=right_w_b / DPI,
        axis_pt=axis_pt,
        subplot_right=0.90,
        ytick_labelsize=axis_pt * 0.78,
        wrap_n=panel_b_wrap,
    )
    axes_frac_h = 0.95 - 0.07  # bottom margin in _make_hpp_forest_panel
    inches_per_y = (fh_b * axes_frac_h) / max(1e-6, (ys_hi_b - ys_lo_b))
    _make_vitaldb_forest_panels(
        vitaldb_path, pc,
        fig_width_in=c_w / DPI,
        inches_per_y_unit=inches_per_y,
        axis_pt=axis_pt,
        subplot_right=_FIG4_BARS_SUBPLOT_RIGHT,
    )

    def _fit_to_w(img: Image.Image, w: int) -> Image.Image:
        s = w / max(1, img.width)
        return img.resize((w, max(1, round(img.height * s))), Image.Resampling.LANCZOS)

    img_a = _fit_to_w(img_scatter0, left_w_a)
    img_b = _fit_to_w(_open_rgb(_prefer_pdf_asset(pb)), right_w_b)
    img_c = _fit_to_w(_open_rgb(_prefer_pdf_asset(pc)), c_w)

    h_a, h_b, h_c = img_a.height, img_b.height, img_c.height
    left_col_h = h_a + gap + h_c
    content_h = max(h_b, left_col_h)

    max_plot_h = FIGURE_MAX_HEIGHT_PX - 2 * margin
    if content_h > max_plot_h:
        sf = max_plot_h / content_h
        img_a = img_a.resize((round(img_a.width * sf), round(img_a.height * sf)), Image.Resampling.LANCZOS)
        img_b = img_b.resize((round(img_b.width * sf), round(img_b.height * sf)), Image.Resampling.LANCZOS)
        img_c = img_c.resize((round(img_c.width * sf), round(img_c.height * sf)), Image.Resampling.LANCZOS)
        h_a, h_b, h_c = img_a.height, img_b.height, img_c.height
        content_h = max(h_b, h_a + gap + h_c)

    canvas_h = margin + content_h + margin
    canvas = Image.new("RGB", (FIGURE_WIDTH_PX, canvas_h), WHITE)
    x_left, x_right = margin, margin + left_w_a + gap
    _paste_panel(canvas, Panel("a", img_a), (x_left, margin, x_left + img_a.width, margin + h_a))
    y_c = margin + h_a + gap
    _paste_panel(canvas, Panel("c", img_c), (x_left, y_c, x_left + img_c.width, y_c + h_c), h_align="left")
    _paste_panel(canvas, Panel("b", img_b), (x_right, margin, x_right + img_b.width, margin + h_b), h_align="left")
    _save_canvas(canvas, "Figure 4_v2")
    shutil.rmtree(scratch, ignore_errors=True)


def _make_figure_8_vitaldb_bars(
    vitaldb_csv: Path,
    output_path: Path,
    *,
    fig_width_in: float,
    inches_per_y_unit: float | None = None,
    ytick_wrap_max_chars: int | None = None,
    bars_axis_font_pt: float = 10.0,
    bars_value_font_pt: float = 8.5,
    bars_legend_font_pt: float = 8.5,
    subplot_right: float = _FIG4_BARS_SUBPLOT_RIGHT,
    category_font_pt: float | None = None,
) -> tuple[int, float]:
    """Two-panel bar figure: ROC AUC (diagnoses) and Pearson r (measures) for VitalDB preop targets."""
    df = pd.read_csv(vitaldb_csv)
    if "p_combined_vs_demo" not in df.columns:
        raise ValueError("Figure 4 panel c requires p_combined_vs_demo in vitaldb_preop_predictions.csv")

    dis_rows, meas_rows = _figure4_vitaldb_bar_plot_split(df)
    # Re-sort by val_c ascending → highest at top of chart
    dis_rows = sorted(dis_rows, key=lambda x: x["val_c"])
    meas_rows = sorted(meas_rows, key=lambda x: x["val_c"])
    n_dis, n_meas = len(dis_rows), len(meas_rows)

    wrap_n = ytick_wrap_max_chars
    if wrap_n is None:
        panel_w_in = float(fig_width_in) / 2.0 if (n_dis and n_meas) else float(fig_width_in)
        wrap_n = max(14, min(38, int(48.0 * panel_w_in / 5.5)))

    h = _BAR_H
    # Same y-gap as panel b med/dis separation (data units → inches via inches_per_y_unit).
    _BLOCK_GAP_Y = 0.55
    TOP_OH, BOT_OH = 1.25, 0.85
    if inches_per_y_unit is not None:
        yspan_d = _fig4_panel_y_span(n_dis)
        yspan_m = _fig4_panel_y_span(n_meas)
        gap_in = inches_per_y_unit * _BLOCK_GAP_Y if (n_dis and n_meas) else 0.0
        h_dis_in = inches_per_y_unit * yspan_d if n_dis else 0.0
        h_meas_in = inches_per_y_unit * yspan_m if n_meas else 0.0
        if n_dis:
            h_dis_in = max(h_dis_in, 0.55)
        if n_meas:
            h_meas_in = max(h_meas_in, 0.55)
    else:
        gap_in = 0.55 if (n_dis and n_meas) else 0.0
        h_dis_in = max(1.6, n_dis * _FIG4_ROW_IN_PER_Y + TOP_OH) if n_dis else 0.0
        h_meas_in = max(1.4, n_meas * _FIG4_ROW_IN_PER_Y + BOT_OH) if n_meas else 0.0

    if n_dis and n_meas:
        ax_left_two = 0.25
        top_frac_two = 0.94
        bot_frac_two = float(_FIG8_VITALDB_FIG_BOTTOM)
        band_frac = top_frac_two - bot_frac_two
        h_content_max = max(h_dis_in, h_meas_in)
        _extra_h = 0.2  # reduced to shorten panel c; legend fits in bot_frac area
        fig_h_in = (h_content_max + _extra_h) / max(1e-6, band_frac)
        fig = plt.figure(figsize=(fig_width_in, fig_h_in))
        content_w_in = fig_width_in * (subplot_right - ax_left_two)
        desired_gap_in = ax_left_two * fig_width_in
        subplot_w_in = max(1e-6, (content_w_in - desired_gap_in) / 2.0)
        w_col_frac = subplot_w_in / fig_width_in
        gap_frac = desired_gap_in / fig_width_in
        x0 = ax_left_two
        x1 = x0 + w_col_frac + gap_frac
        h_dis_frac = h_dis_in / fig_h_in
        h_meas_frac = h_meas_in / fig_h_in
        scale = float(np.clip(_FIG8_TWO_COL_AXIS_SCALE, 0.55, 1.0))
        w_scaled = w_col_frac * scale
        x_pad = 0.5 * (w_col_frac - w_scaled)
        ax_dis = fig.add_axes([x0 + x_pad, top_frac_two - h_dis_frac, w_scaled, h_dis_frac])
        ax_meas = fig.add_axes([x1 + x_pad, top_frac_two - h_meas_frac, w_scaled, h_meas_frac])
    elif n_dis:
        fig_h_in = h_dis_in
        fig, ax_dis = plt.subplots(1, 1, figsize=(fig_width_in, fig_h_in))
        ax_meas = None
        fig.subplots_adjust(
            left=_FIG4_BARS_AX_LEFT,
            right=subplot_right,
            bottom=_FIG8_VITALDB_FIG_BOTTOM,
            top=0.95,
        )
    else:
        fig_h_in = h_meas_in
        fig, ax_meas = plt.subplots(1, 1, figsize=(fig_width_in, fig_h_in))
        ax_dis = None
        fig.subplots_adjust(
            left=_FIG4_BARS_AX_LEFT,
            right=subplot_right,
            bottom=_FIG8_VITALDB_FIG_BOTTOM,
            top=0.95,
        )

    err_kw = {"ecolor": "#111111", "elinewidth": 0.9, "capsize": 2}
    cat_pf = category_font_pt if category_font_pt is not None else bars_axis_font_pt

    def _draw_panel(
        ax: plt.Axes,
        rows: list[dict],
        bg: str,
        accent: str,
        xlabel: str,
        xlim_lo: float,
        xlim_hi: float | None,
        vref: float | None,
        cat_lbl: str,
        *,
        x_ticks: tuple[float, ...] | None = None,
        x_tick_labels: tuple[str, ...] | None = None,
        star_shift_left: float = 0.0,
    ) -> None:
        y = np.arange(len(rows), dtype=float)
        val_d = np.array([r["val_d"] for r in rows])
        val_c = np.array([r["val_c"] for r in rows])
        std_d = np.array([r["std_d"] for r in rows])
        std_c = np.array([r["std_c"] for r in rows])
        x_min, star_x, x_max = _fig4_dynamic_xlim_and_star(
            val_d,
            val_c,
            std_d,
            std_c,
            xlim_lo_floor=xlim_lo,
        )
        if xlim_hi is not None:
            x_max = float(xlim_hi)
        xt = list(_FIG4_BAR_X_TICKS) if x_ticks is None else list(x_ticks)
        if xt:
            x_min = min(x_min, min(xt) - 0.05)
            x_max = max(x_max, max(xt) + 0.05)
        labs = [
            _wrap_figure5_label_two_lines(
                _figure8_target_label(str(r["row"]["target"])),
                max_single_line=wrap_n,
            )
            for r in rows
        ]
        y_lo, y_hi = float(y.min()) - 0.65, float(y.max()) + 0.65

        ax.axhspan(y_lo, y_hi, facecolor=bg, edgecolor="none", zorder=0, alpha=0.75)
        if vref is not None and x_min <= vref <= x_max:
            ax.axvline(vref, color="#555555", linestyle=":", linewidth=0.95, zorder=1.5)

        ax.barh(
            y + h / 2, val_c, height=h,
            color=_FIG8_VITALDB_PRIMARY, edgecolor=_FIG8_VITALDB_EDGE, linewidth=0.35,
            xerr=std_c, error_kw=err_kw,
            zorder=3,
        )
        ax.barh(
            y - h / 2, val_d, height=h,
            color="#f5f5f5", edgecolor="#b0b0b0", linewidth=0.35, hatch="///",
            xerr=std_d, error_kw=err_kw,
            zorder=2,
        )

        star_x_visible = _fig4_star_x_data(xlim_lo=x_min, xlim_hi=x_max)
        star_x_visible = max(x_min + 1e-4, star_x_visible - max(0.0, float(star_shift_left)))
        for i, pr in enumerate(rows):
            yi = float(y[i])
            stars = _classification_auc_stars(pr["p"])
            if stars:
                ax.text(
                    star_x_visible,
                    yi,
                    stars,
                    va="center",
                    ha="left",
                    fontsize=bars_axis_font_pt * 0.86,
                    color="#222222",
                    clip_on=False,
                )

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_lo, y_hi)
        ax.set_yticks(y, labs)
        for tl in ax.get_yticklabels():
            tl.set_multialignment("right")
        ax.tick_params(axis="y", labelsize=bars_axis_font_pt)
        ax.tick_params(axis="x", labelsize=bars_axis_font_pt)
        ax.set_xlabel(xlabel, fontsize=bars_axis_font_pt)
        ax.xaxis.set_major_locator(FixedLocator(xt))
        if x_tick_labels is None:
            x_tick_labels = tuple(f"{t:g}" for t in xt)
        ax.set_xticks(xt, x_tick_labels)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="x", color="#e8e8e8", linewidth=0.4, linestyle="--", zorder=1)
        trans = blended_transform_factory(ax.transAxes, ax.transData)
        ax.plot([0, 0], [y_lo, y_hi], color=accent, linewidth=2.0,
                transform=trans, clip_on=False, zorder=5, solid_capstyle="butt")
        ax.text(
            1.045,
            0.5 * (y_lo + y_hi),
            cat_lbl,
            transform=trans,
            ha="left",
            va="center",
            fontsize=cat_pf,
            rotation=270,
            color=accent,
            multialignment="center",
            clip_on=False,
        )


    if ax_dis is not None:
        _draw_panel(
            ax_dis,
            dis_rows,
            _FIG5_DIS_BG,
            _FIG5_DIS_ACCENT,
            "ROC AUC",
            _FIG4_AUC_XLIM_LO,
            None,
            0.5,
            "Preoperation\ndiagnoses",
            x_ticks=(0.65, 0.75),
            x_tick_labels=("0.65", "0.75"),
        )
    if ax_meas is not None:
        _draw_panel(
            ax_meas,
            meas_rows,
            _FIG8_MEAS_BG,
            _FIG8_MEAS_ACCENT,
            "Pearson $r$",
            0.0,
            None,
            None,
            "Preoperation\nmeasures",
            x_ticks=(0.25, 0.5),
            x_tick_labels=("0.25", "0.5"),
            star_shift_left=0.02,
        )

    emb, demo = _fig4_bc_legend_handles()
    fig.legend(
        handles=[emb, demo],
        loc="lower right",
        bbox_to_anchor=(subplot_right, _FIG8_VITALDB_LEGEND_Y_FRAC),
        bbox_transform=fig.transFigure,
        ncol=2,
        frameon=False,
        fontsize=bars_legend_font_pt,
        columnspacing=1.35,
        handlelength=1.35,
        borderaxespad=0,
    )

    _fig4_save_bars(fig, output_path)
    fh = float(fig.get_figheight())
    plt.close(fig)
    return n_dis + n_meas, fh




def build_extended_data_figure_4() -> None:
    """Age prediction ablations: checkpoints, segment length, training mask ratio."""
    import sys

    metrics_path = _first_existing(
        tuple(d / "ablation_age_gold_test_epoch_metrics.csv" for d in ABLATION_GOLD_TEST_AGE_DIRS),
        "ablation_age_gold_test_epoch_metrics.csv",
    )

    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from downstream_tasks.make_extended_data_wandb_ablation_figures import (
        SAMPLING_HZ_DEFAULT,
        plot_extended_data_figure_2_unified,
    )

    wandb_pair = _find_wandb_age_ablation_csvs()
    if wandb_pair is not None:
        seg_csv, mr_csv = wandb_pair
        max_h_in = FIG_MAX_HEIGHT_CM / 2.54
        png_bytes = plot_extended_data_figure_2_unified(
            metrics_path,
            seg_csv,
            mr_csv,
            segment_exp_name=WANDB_ABLATION_EXPORT_SEGMENT_EXP,
            maskratio_exp_name=WANDB_ABLATION_EXPORT_MASK_EXP,
            sampling_hz=SAMPLING_HZ_DEFAULT,
            figure_width_in=FIGURE_WIDTH_MM / 25.4,
            max_height_in=max_h_in,
            dpi=DPI,
        )
        img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
        scale_w = FIGURE_WIDTH_PX / img.width
        img = img.resize(
            (FIGURE_WIDTH_PX, max(1, int(round(img.height * scale_w)))),
            Image.Resampling.LANCZOS,
        )
        if img.height > FIGURE_MAX_HEIGHT_PX:
            scale_h = FIGURE_MAX_HEIGHT_PX / img.height
            new_w = max(1, int(round(img.width * scale_h)))
            new_h = FIGURE_MAX_HEIGHT_PX
            img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            canvas = Image.new("RGB", (FIGURE_WIDTH_PX, new_h), WHITE)
            canvas.paste(img, ((FIGURE_WIDTH_PX - new_w) // 2, 0))
            img = canvas
        else:
            canvas = Image.new("RGB", (FIGURE_WIDTH_PX, img.height), WHITE)
            canvas.paste(img, (0, 0))
            img = canvas
        _save_canvas(img, "Extended Data Figure 4")
        return

    print(
        "Extended Data Figure 4: wandb export CSVs not found under "
        f"{WANDB_ABLATION_EXPORT_DIRS}; using pre-rendered age_linear_probing PNG if present."
    )

    metrics = pd.read_csv(metrics_path).sort_values("epoch")
    if metrics.empty:
        raise ValueError(f"No ablation rows found in {metrics_path}")

    fig, axes = plt.subplots(1, 3, figsize=(FIGURE_WIDTH_MM / 25.4, 2.65), sharex=True)
    plot_specs = [
        ("PearsonR", "Pearson $r$"),
        ("R2", "R²"),
        ("MAE", "MAE (years)"),
    ]
    x = metrics["epoch"].to_numpy(dtype=float)
    for ax, (metric, ylabel) in zip(axes, plot_specs):
        y_col = f"{metric}_repeat_mean" if f"{metric}_repeat_mean" in metrics.columns else metric
        y = metrics[y_col].to_numpy(dtype=float)
        ax.plot(x, y, color=PULSEOX_FM_PRIMARY_HEX, linewidth=1.8, marker="o", markersize=4.5)
        ax.set_xlabel("Pretraining epoch", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_xticks(x)
        ax.tick_params(labelsize=8.5)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.85)
        ax.grid(color="#d7d7d7", linewidth=0.4, linestyle="--", zorder=0)
        if metric == "MAE":
            ymin = max(0, float(np.nanmin(y)) - 0.18)
            ymax = float(np.nanmax(y)) + 0.18
            ax.set_ylim(ymin, ymax)
        else:
            ymin = max(0, float(np.nanmin(y)) - 0.035)
            ymax = min(1, float(np.nanmax(y)) + 0.035)
            ax.set_ylim(ymin, ymax)
    fig.tight_layout(pad=0.6, w_pad=0.9)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=DPI, facecolor="white")
    plt.close(fig)
    buf.seek(0)
    img_row1 = Image.open(buf).copy().convert("RGB")
    canvas_row1 = Image.new("RGB", (FIGURE_WIDTH_PX, img_row1.height), WHITE)
    canvas_row1.paste(img_row1, ((FIGURE_WIDTH_PX - img_row1.width) // 2, 0))

    age_asset = _prefer_pdf_asset(WANDB_AGE_LINEAR_PROBING_PNG)
    if not age_asset.is_file():
        print(f"Extended Data Figure 4: missing {WANDB_AGE_LINEAR_PROBING_PNG}; saving epoch row only.")
        _save_canvas(canvas_row1, "Extended Data Figure 4")
        return

    img_age = _open_rgb(age_asset)
    if img_age.width != FIGURE_WIDTH_PX:
        scale_w = FIGURE_WIDTH_PX / img_age.width
        img_age = img_age.resize(
            (FIGURE_WIDTH_PX, max(1, int(round(img_age.height * scale_w)))),
            Image.Resampling.LANCZOS,
        )

    gap = max(16, int(round(DPI * 0.05)))
    max_age_h = FIGURE_MAX_HEIGHT_PX - canvas_row1.height - gap
    if max_age_h <= 0:
        raise ValueError(
            "Extended Data Figure 4: top row leaves no vertical space for wandb age grid under max height."
        )
    if img_age.height > max_age_h:
        scale_h = max_age_h / img_age.height
        img_age = img_age.resize(
            (FIGURE_WIDTH_PX, max(1, int(round(img_age.height * scale_h)))),
            Image.Resampling.LANCZOS,
        )

    combined_h = canvas_row1.height + gap + img_age.height
    if combined_h > FIGURE_MAX_HEIGHT_PX:
        raise ValueError(
            "Extended Data Figure 4 stacked height exceeds journal cap; "
            f"computed {combined_h}px > {FIGURE_MAX_HEIGHT_PX}px."
        )

    stacked = Image.new("RGB", (FIGURE_WIDTH_PX, combined_h), WHITE)
    stacked.paste(canvas_row1, (0, 0))
    stacked.paste(img_age, (0, canvas_row1.height + gap))
    _save_canvas(stacked, "Extended Data Figure 4")


def build_supplementary_figure_3() -> None:
    """Wandb reconstruction ablation: random masking and forecasting vs training masking ratio."""
    recon_asset = _prefer_pdf_asset(WANDB_RECONSTRUCTION_ABLATION_PNG)
    if not recon_asset.is_file():
        raise FileNotFoundError(
            f"Missing {WANDB_RECONSTRUCTION_ABLATION_PNG} (or sibling .pdf). "
            "Run downstream_tasks/make_extended_data_wandb_ablation_figures.py first."
        )
    img = _open_rgb(recon_asset)
    scale = FIGURE_WIDTH_PX / max(1, img.width)
    tw = FIGURE_WIDTH_PX
    th = int(round(img.height * scale))
    img = img.resize((tw, th), Image.Resampling.LANCZOS)
    if th > FIGURE_MAX_HEIGHT_PX:
        shrink = FIGURE_MAX_HEIGHT_PX / th
        tw2 = max(1, int(round(img.width * shrink)))
        th2 = FIGURE_MAX_HEIGHT_PX
        img = img.resize((tw2, th2), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (FIGURE_WIDTH_PX, img.height), WHITE)
    canvas.paste(img, ((FIGURE_WIDTH_PX - img.width) // 2, 0))
    _save_canvas(canvas, "Supplementary Figure 3")


def build_extended_data_figure_5() -> None:
    """R² bar comparison from precomputed target-prediction figure (Nature extended data)."""
    stem = "bar_R2_age_sex_vs_embeddings_Ntargets53"
    src: Path | None = None
    for root in (TARGET_PREDICTION_FIG_DIR, *_TP_BASE):
        for ext in (".pdf", ".PDF", ".png", ".PNG"):
            cand = root / f"{stem}{ext}"
            if cand.is_file():
                src = cand
                break
        if src is not None:
            break
    if src is None:
        raise FileNotFoundError(
            f"Could not find {stem}.pdf (or .png) under target prediction figure dirs"
        )
    img = _open_rgb(src)
    scale = min(
        FIGURE_WIDTH_PX / max(1, img.width),
        FIGURE_MAX_HEIGHT_PX / max(1, img.height),
    )
    tw = int(round(img.width * scale))
    th = int(round(img.height * scale))
    tw = min(tw, FIGURE_WIDTH_PX)
    th = min(th, FIGURE_MAX_HEIGHT_PX)
    out_img = img.resize((tw, th), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (FIGURE_WIDTH_PX, th), WHITE)
    canvas.paste(out_img, ((FIGURE_WIDTH_PX - tw) // 2, 0))
    _save_canvas(canvas, "Extended Data Figure 5")


def _classification_cache_for_pattern(pattern: str) -> Path:
    roots: list[Path] = []
    for root in _TP_BASE:
        roots.extend([root / "cache_dir", root])
    matches: list[Path] = []
    for root in roots:
        if root.is_dir():
            matches.extend(p for p in root.glob(pattern) if p.is_file())
    if not matches:
        raise FileNotFoundError(f"Could not find cached classification predictions matching {pattern}")
    return sorted(matches, key=lambda p: p.stat().st_mtime, reverse=True)[0]


def _classification_target_column(df: pd.DataFrame) -> str:
    excluded = {"Recordings", "recordings", "record_id", "participant_id"}
    candidates = [
        c
        for c in df.columns
        if c not in excluded
        and not str(c).startswith("proba_")
        and not str(c).startswith("AUC_")
        and not str(c).startswith("F1_")
    ]
    if not candidates:
        raise ValueError("Could not identify binary target column in classification cache")
    return candidates[0]


def _seed_probability_columns(df: pd.DataFrame, prefix: str) -> list[str]:
    seed_cols = [
        c
        for c in df.columns
        if re.fullmatch(rf"{re.escape(prefix)}_seed_\d+", str(c))
    ]
    if seed_cols:
        return sorted(seed_cols, key=lambda c: int(str(c).rsplit("_", 1)[1]))
    mean_col = f"{prefix}_mean"
    return [mean_col] if mean_col in df.columns else []


def _plot_seeded_roc_band(
    ax: plt.Axes,
    y_true: np.ndarray,
    df: pd.DataFrame,
    *,
    prefix: str,
    label: str,
    color: str,
    band_alpha: float,
) -> None:
    cols = _seed_probability_columns(df, prefix)
    if not cols:
        return
    fpr_grid = np.linspace(0, 1, 201)
    tprs: list[np.ndarray] = []
    bootstrap_tprs: list[np.ndarray] = []
    aucs: list[float] = []
    rng = np.random.default_rng(20260503 + sum(ord(ch) for ch in prefix))
    for col in cols:
        score = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(y_true) & np.isfinite(score)
        yv = y_true[valid].astype(int)
        sv = score[valid]
        if len(np.unique(yv)) < 2:
            continue
        fpr, tpr, _ = roc_curve(yv, sv)
        interp_tpr = np.interp(fpr_grid, fpr, tpr)
        interp_tpr[0] = 0.0
        interp_tpr[-1] = 1.0
        tprs.append(interp_tpr)
        aucs.append(float(roc_auc_score(yv, sv)))
        pos_idx = np.flatnonzero(yv == 1)
        neg_idx = np.flatnonzero(yv == 0)
        if pos_idx.size > 0 and neg_idx.size > 0:
            for _ in range(ROC_BOOTSTRAP_REPEATS):
                boot_idx = np.concatenate(
                    [
                        rng.choice(pos_idx, size=pos_idx.size, replace=True),
                        rng.choice(neg_idx, size=neg_idx.size, replace=True),
                    ]
                )
                fpr_b, tpr_b, _ = roc_curve(yv[boot_idx], sv[boot_idx])
                interp_b = np.interp(fpr_grid, fpr_b, tpr_b)
                interp_b[0] = 0.0
                interp_b[-1] = 1.0
                bootstrap_tprs.append(interp_b)
    if not tprs:
        return
    tpr_arr = np.vstack(tprs)
    mean_tpr = np.mean(tpr_arr, axis=0)
    auc_mean = float(np.mean(aucs))
    auc_sd = float(np.std(aucs, ddof=0)) if len(aucs) > 1 else 0.0
    if bootstrap_tprs:
        boot_arr = np.vstack(bootstrap_tprs)
        lo = np.percentile(boot_arr, 2.5, axis=0)
        hi = np.percentile(boot_arr, 97.5, axis=0)
    else:
        band = 1.96 * np.std(tpr_arr, axis=0, ddof=0) / math.sqrt(max(1, len(tprs)))
        lo = mean_tpr - band
        hi = mean_tpr + band
    ax.fill_between(
        fpr_grid,
        np.clip(lo, 0, 1),
        np.clip(hi, 0, 1),
        color=color,
        alpha=band_alpha,
        linewidth=0,
    )
    ax.plot(fpr_grid, mean_tpr, color=color, linewidth=1.4, label=f"{label} (AUC {auc_mean:.2f}±{auc_sd:.2f})")


def build_supplementary_figure_2() -> None:
    """Seeded cross-validation ROC curves for selected diagnosis and medication classification tasks."""
    panels = [
        ("a", "Hypertension", "classification_Hypertension_mean_all_*.csv"),
        ("b", "Hypertension within 2 years", "classification_Hypertension_incidence_2yrs_mean_all_*.csv"),
        ("c", "Psychoanaleptics intake", "classification_PSYCHOANALEPTICS_mean_all_*.csv"),
        ("d", "Anemia", "classification_Anemia_mean_all_*.csv"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(FIGURE_WIDTH_MM / 25.4, 5.9), sharex=True, sharey=True)
    for ax, (panel_label, title, pattern) in zip(axes.ravel(), panels):
        cache_path = _classification_cache_for_pattern(pattern)
        df = pd.read_csv(cache_path)
        target_col = _classification_target_column(df)
        y_true = pd.to_numeric(df[target_col], errors="coerce").to_numpy(dtype=float)

        _plot_seeded_roc_band(
            ax,
            y_true,
            df,
            prefix="proba_combined",
            label="Age, sex, BMI + embeddings",
            color=PULSEOX_FM_PRIMARY_HEX,
            band_alpha=0.18,
        )
        _plot_seeded_roc_band(
            ax,
            y_true,
            df,
            prefix="proba_demo",
            label="Age, sex, BMI",
            color="#9c9c9c",
            band_alpha=0.14,
        )
        _plot_seeded_roc_band(
            ax,
            y_true,
            df,
            prefix="proba_emb",
            label="Embeddings only",
            color="#4a9bab",
            band_alpha=0.12,
        )

        valid_y = y_true[np.isfinite(y_true)]
        positives = int(np.nansum(valid_y))
        total = int(valid_y.size)
        ax.plot([0, 1], [0, 1], color="#333333", linestyle="--", linewidth=0.7)
        ax.set_title(f"{title}\nN={positives}/{total} positive records", fontsize=9.2)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("False positive rate", fontsize=9)
        ax.set_ylabel("True positive rate", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.text(
            -0.14,
            1.14,
            panel_label,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            fontweight="bold",
            clip_on=False,
        )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(color="#d7d7d7", linewidth=0.4, linestyle="--", zorder=0)
        ax.legend(frameon=False, loc="lower right", fontsize=6.5, handlelength=1.4, borderaxespad=0.2)
    fig.tight_layout(pad=0.7, h_pad=1.0, w_pad=0.9)

    # Save vector PDF directly from matplotlib (preserves editable text/lines).
    pdf_path = MANUSCRIPT_DIR / "Supplementary Figure 2.pdf"
    MANUSCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf_path, format="pdf", facecolor="white")
    print(f"Saved {pdf_path}")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=DPI, facecolor="white")
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf).copy().convert("RGB")
    canvas = Image.new("RGB", (FIGURE_WIDTH_PX, img.height), WHITE)
    canvas.paste(img, ((FIGURE_WIDTH_PX - img.width) // 2, 0))
    png_path = MANUSCRIPT_DIR / "Supplementary Figure 2.png"
    canvas.save(png_path, dpi=(DPI, DPI))
    print(f"Saved {png_path}")
    width_cm = canvas.width / DPI * 2.54
    height_cm = canvas.height / DPI * 2.54
    print(
        f"Supplementary Figure 2 size: {width_cm:.2f} x {height_cm:.2f} cm "
        f"({canvas.width} x {canvas.height} px at {DPI} dpi)"
    )


def _generate_basal_energy_variability_image(width_px: int | None = None) -> Image.Image:
    """Generate within/between-person prediction variability histogram for basal energy burned.

    Uses the cached predictions CSV; returns a PIL Image in Nature Medicine style.
    Falls back to a pre-rendered PNG when the CSV is unavailable.
    """
    import io
    from scipy.stats import gaussian_kde

    # Harmonized with the PulseOx-FM teal palette used in panel a.
    _C_WITHIN_WITHIN  = PULSEOX_FM_EDGE_HEX
    _C_WITHIN_BETWEEN = "#C76F5D"
    _C_BETWEEN        = "#D89C27"
    _N_BETWEEN        = 15_000
    _XMAX_CAP         = 600      # kcal
    _RNG_SEED         = 42
    _PT               = 10       # universal font size

    csv_path: Path | None = None
    for cand in BASAL_ENERGY_PREDICTIONS_CSV_CANDIDATES:
        if cand.is_file():
            csv_path = cand
            break
    if csv_path is None:
        return _open_rgb(_first_existing(BASAL_ENERGY_PRED_VARIABILITY_ASSETS, "basal energy prediction-variability figure"))

    df = pd.read_csv(csv_path)
    df["participant_id"] = df["Recordings"].astype(str).str.split("__").str[0]
    df["research_stage"] = df["Recordings"].astype(str).str.split("__").str[1]
    df["_key"]           = df["participant_id"] + "__" + df["research_stage"]
    pred_col = "pred_mean"
    rng = np.random.default_rng(_RNG_SEED)

    # Within person, within research stage → consecutive nights
    within_within: list[float] = []
    for _key, grp in df.groupby("_key", sort=False):
        if len(grp) < 2:
            continue
        idx = grp.index.tolist()
        i, j = rng.choice(len(idx), size=2, replace=False)
        within_within.append(abs(grp.loc[idx[i], pred_col] - grp.loc[idx[j], pred_col]))

    # Within person, between research stages → ~2 years apart
    within_between: list[float] = []
    for pid, pgrp in df.groupby("participant_id", sort=False):
        stages = pgrp["research_stage"].unique()
        if len(stages) < 2:
            continue
        s1, s2 = rng.choice(stages, size=2, replace=False)
        g1 = pgrp[pgrp["research_stage"] == s1]
        g2 = pgrp[pgrp["research_stage"] == s2]
        p1 = g1[pred_col].iloc[rng.integers(0, len(g1))]
        p2 = g2[pred_col].iloc[rng.integers(0, len(g2))]
        within_between.append(abs(p1 - p2))

    # Between persons
    participant_ids = df["participant_id"].unique()
    pid_to_preds = {pid: df.loc[df["participant_id"] == pid, pred_col].values for pid in participant_ids}
    between: list[float] = []
    n_target = min(_N_BETWEEN, len(participant_ids) * (len(participant_ids) - 1))
    attempts = 0
    while len(between) < n_target and attempts < n_target * 5:
        attempts += 1
        p1, p2 = rng.choice(participant_ids, size=2, replace=False)
        v1_arr, v2_arr = pid_to_preds[p1], pid_to_preds[p2]
        if len(v1_arr) == 0 or len(v2_arr) == 0:
            continue
        between.append(abs(v1_arr[rng.integers(0, len(v1_arr))] - v2_arr[rng.integers(0, len(v2_arr))]))

    ww = np.array(within_within)
    wb = np.array(within_between)
    bp = np.array(between)

    with plt.rc_context({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": _PT,
        "axes.labelsize": _PT,
        "axes.titlesize": _PT,
        "xtick.labelsize": _PT,
        "ytick.labelsize": _PT,
        "legend.fontsize": _PT,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "lines.linewidth": 1.2,
        "patch.linewidth": 0.0,
    }):
        fig_w = (width_px / DPI) if width_px is not None else 3.54
        fig, ax = plt.subplots(figsize=(fig_w, 2.70), dpi=DPI)

        all_vals = np.concatenate([ww, wb, bp])
        x_max = min(_XMAX_CAP, float(np.nanpercentile(all_vals, 99.5)) * 1.02) if len(all_vals) else _XMAX_CAP
        bins = np.linspace(0, x_max, 51)

        def _draw(values, color, label):
            vals = values[values <= x_max]
            if len(vals) == 0:
                return
            ax.hist(vals, bins=bins, alpha=0.50, density=True, color=color, edgecolor="none", label=label)
            if len(vals) > 10:
                try:
                    kde = gaussian_kde(vals, bw_method="scott")
                    x_kde = np.linspace(0, x_max, 300)
                    ax.plot(x_kde, kde(x_kde), "-", color=color, lw=1.5, alpha=0.92)
                except Exception:
                    pass

        _draw(ww, _C_WITHIN_WITHIN,  "Within person, between consecutive nights")
        _draw(wb, _C_WITHIN_BETWEEN, "Within person, 2-years apart")
        _draw(bp, _C_BETWEEN,        "Between persons")

        # Dotted mean lines with value annotations
        y_top = ax.get_ylim()[1]
        label_y_fracs = [0.97, 0.78, 0.59]  # stagger to avoid overlap
        for (arr, color), y_frac in zip(
            [(ww, _C_WITHIN_WITHIN), (wb, _C_WITHIN_BETWEEN), (bp, _C_BETWEEN)],
            label_y_fracs,
        ):
            if len(arr) > 0:
                mu = float(np.mean(arr[arr <= x_max]))
                ax.axvline(mu, color=color, linestyle=":", linewidth=1.2, alpha=0.95)
                ax.text(
                    mu + x_max * 0.015, y_top * y_frac,
                    f"{mu:.0f} kcal",
                    color=color, fontsize=_PT, va="top", ha="left",
                    rotation=90, rotation_mode="anchor",
                )

        ax.set_xlabel("|Predicted(night i) − Predicted(night j)| (kcal)", fontsize=_PT)
        ax.set_ylabel("Density (%)", fontsize=_PT)
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=1))
        ax.set_xlim(0, x_max)
        ax.legend(loc="upper right", frameon=False, fontsize=_PT, handlelength=1.2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout(pad=0.5)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=DPI, facecolor="white")
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf).copy().convert("RGB")


def build_figure_6() -> None:
    margin, gap = 50, 36
    panel_w = FIGURE_WIDTH_PX - 2 * margin

    try:
        img_a = _generate_next_day_raw_pearson_image(panel_w)
    except Exception:
        barh_asset = _next_day_raw_pearson_barh_asset()
        img_a = _crop_barh_figure(_open_rgb(barh_asset))
    img_b = _generate_basal_energy_variability_image(panel_w)

    def _scale_to_panel(img: Image.Image) -> Image.Image:
        scale = panel_w / max(1, img.width)
        return img.resize((panel_w, max(1, int(round(img.height * scale)))), Image.Resampling.LANCZOS)

    img_a_s = _scale_to_panel(img_a)
    img_b_s = _scale_to_panel(img_b)
    fig6_h = margin + img_a_s.height + gap + img_b_s.height + margin
    if fig6_h > FIGURE_MAX_HEIGHT_PX:
        available_h = FIGURE_MAX_HEIGHT_PX - 2 * margin - gap
        scale = available_h / max(1, img_a_s.height + img_b_s.height)
        img_a_s = img_a_s.resize((max(1, int(round(img_a_s.width * scale))), max(1, int(round(img_a_s.height * scale)))), Image.Resampling.LANCZOS)
        img_b_s = img_b_s.resize((max(1, int(round(img_b_s.width * scale))), max(1, int(round(img_b_s.height * scale)))), Image.Resampling.LANCZOS)
        fig6_h = margin + img_a_s.height + gap + img_b_s.height + margin
    canvas = Image.new("RGB", (FIGURE_WIDTH_PX, fig6_h), WHITE)
    _paste_panel(canvas, Panel("a", img_a_s), (margin, margin, margin + panel_w, margin + img_a_s.height))
    y_b = margin + img_a_s.height + gap
    _paste_panel(canvas, Panel("b", img_b_s), (margin, y_b, margin + panel_w, y_b + img_b_s.height))
    _save_canvas(canvas, "Figure 6")


# ---------------------------------------------------------------------------
# Figure 6_v2 – panel a replaced with demographics vs embeddings forest plot
# ---------------------------------------------------------------------------

def _generate_next_day_demo_forest_image(width_px: int) -> Image.Image:
    """Forest plot of demographics vs demographics+embeddings Pearson r for next-day targets (Figure 6_v2 panel a)."""
    demo_csv = _first_existing(NEXT_DAY_RADAR_DEMO_CSVS, "radar_demo_vs_embeddings_cache.csv")
    df = pd.read_csv(demo_csv)

    needed = {"target", "R_demo", "R_combined", "p_combined"}
    if not needed.issubset(df.columns):
        raise ValueError(f"radar_demo_vs_embeddings_cache.csv missing columns: {needed - set(df.columns)}")

    df["R_demo"] = pd.to_numeric(df["R_demo"], errors="coerce")
    df["R_combined"] = pd.to_numeric(df["R_combined"], errors="coerce")
    df["p_combined"] = pd.to_numeric(df["p_combined"], errors="coerce")
    df = df.dropna(subset=["R_demo", "R_combined", "p_combined"])

    # BH-FDR filter
    p_arr = df["p_combined"].to_numpy(dtype=float)
    q_arr = _bh_fdr_qvalues(p_arr)
    df = df[q_arr < 0.05].copy()
    q_kept = q_arr[q_arr < 0.05]
    if df.empty:
        raise ValueError("No FDR-significant targets in radar_demo_vs_embeddings_cache.csv")

    df["_q"] = q_kept
    df = df.sort_values("R_combined", ascending=True)  # ascending → highest at top (invert_yaxis or natural)

    type_colors = {
        "CGM": "#0b3d44",
        "Food": PULSEOX_FM_PRIMARY_HEX,
        "Wearables": "#84c6cf",
    }
    n = len(df)
    y = np.arange(n, dtype=float)
    r_demo = df["R_demo"].to_numpy(dtype=float)
    r_comb = df["R_combined"].to_numpy(dtype=float)
    y_comb = y + _FOREST_Y_OFFSET
    y_demo = y - _FOREST_Y_OFFSET

    fig_w = width_px / DPI
    fig_h = max(2.8, 0.18 * n + 0.80)
    with plt.rc_context({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.linewidth": 0.8,
    }):
        fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=DPI)

        # PulseOx-FM dots (coloured by target type) — combined model on top
        for i, (_, row) in enumerate(df.iterrows()):
            t_color = type_colors.get(str(row.get("target_type", "")), PULSEOX_FM_PRIMARY_HEX)
            ax.plot(r_comb[i], y_comb[i], "o", color=t_color, markeredgecolor="none",
                    markersize=_FOREST_DOT_SIZE, zorder=4)
        # Demographics dots (silver) — below
        ax.plot(r_demo, y_demo, "o", color=_FOREST_DEMO_COLOR, markeredgecolor="none",
                markersize=_FOREST_DOT_SIZE, zorder=3)

        ax.axvline(0, color="#777777", linestyle=":", linewidth=0.9, zorder=1)
        ax.grid(True, axis="x", color="#d7d7d7", alpha=0.75, linewidth=0.5, zorder=0)

        # Tight x-axis
        x_lo = min(float(np.nanmin(r_demo)), float(np.nanmin(r_comb))) - 0.02
        x_hi_raw = float(np.nanmax(r_comb))
        star_x = x_hi_raw + 0.015
        x_hi = star_x + 0.06

        # Significance stars at centre y of each target
        for i, q in enumerate(df["_q"].to_numpy()):
            if q < 0.001:
                stars = "***"
            elif q < 0.01:
                stars = "**"
            elif q < 0.05:
                stars = "*"
            else:
                stars = ""
            if stars:
                ax.text(star_x, y[i], stars, va="center", ha="left", fontsize=9, color="#222222", clip_on=False)

        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(-0.45, n - 1 + 0.45)
        ax.set_yticks(y)
        ax.set_yticklabels([_format_next_day_target_label(t) for t in df["target"]])
        ax.tick_params(axis="both", labelsize=10)
        ax.set_xlabel("Pearson $r$")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Legend
        legend_handles = [
            plt.Line2D([0], [0], marker="o", color="none", markersize=_FOREST_DOT_SIZE,
                       markerfacecolor=PULSEOX_FM_PRIMARY_HEX, markeredgecolor="none",
                       label="Age, sex, BMI + PulseOx-FM embeddings"),
            plt.Line2D([0], [0], marker="o", color="none", markersize=_FOREST_DOT_SIZE,
                       markerfacecolor=_FOREST_DEMO_COLOR, markeredgecolor="none",
                       label="Age, sex, BMI"),
        ]
        ax.legend(handles=legend_handles, loc="lower right", fontsize=9, frameon=False)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=DPI, facecolor="white", bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf).convert("RGB")


def build_figure_6_v2() -> None:
    """Figure 6_v2: panel a=demographics vs embeddings forest, panel b=basal energy variability."""
    margin, gap = 50, 36
    panel_w = FIGURE_WIDTH_PX - 2 * margin

    img_a = _generate_next_day_demo_forest_image(panel_w)
    img_b = _generate_basal_energy_variability_image(panel_w)

    def _scale_to_panel(img: Image.Image) -> Image.Image:
        scale = panel_w / max(1, img.width)
        return img.resize((panel_w, max(1, int(round(img.height * scale)))), Image.Resampling.LANCZOS)

    img_a_s = _scale_to_panel(img_a)
    img_b_s = _scale_to_panel(img_b)
    fig6_h = margin + img_a_s.height + gap + img_b_s.height + margin
    if fig6_h > FIGURE_MAX_HEIGHT_PX:
        available_h = FIGURE_MAX_HEIGHT_PX - 2 * margin - gap
        scale = available_h / max(1, img_a_s.height + img_b_s.height)
        img_a_s = img_a_s.resize(
            (max(1, int(round(img_a_s.width * scale))), max(1, int(round(img_a_s.height * scale)))),
            Image.Resampling.LANCZOS,
        )
        img_b_s = img_b_s.resize(
            (max(1, int(round(img_b_s.width * scale))), max(1, int(round(img_b_s.height * scale)))),
            Image.Resampling.LANCZOS,
        )
        fig6_h = margin + img_a_s.height + gap + img_b_s.height + margin
    canvas = Image.new("RGB", (FIGURE_WIDTH_PX, fig6_h), WHITE)
    _paste_panel(canvas, Panel("a", img_a_s), (margin, margin, margin + panel_w, margin + img_a_s.height))
    y_b = margin + img_a_s.height + gap
    _paste_panel(canvas, Panel("b", img_b_s), (margin, y_b, margin + panel_w, y_b + img_b_s.height))
    _save_canvas(canvas, "Figure 6_v2")


_FIG5_PANEL_ROOTS = (
    FIGURES_ROOT / "target_prediction",
    TARGET_PREDICTION_FIG_DIR,
)


def _load_fig5_panel(stem: str) -> Image.Image | None:
    """Load panel raster from stem in known roots; prefers PDF vector source when present."""
    for root in _FIG5_PANEL_ROOTS:
        for ext in (".pdf", ".PDF", ".png", ".PNG"):
            path = root / f"{stem}{ext}"
            if path.is_file():
                return _open_rgb(path)
    return None


def build_figure_5() -> None:
    """Residual quartiles vs incidence (a–c) and hypertension incidence ROC (d); 2×2 grid without cropping."""
    img_a = _load_fig5_panel("residual_quartile_cardiovascular_incidence_panel")
    img_b = _load_fig5_panel("residual_quartile_prediabetes_incidence_panel")
    img_c = _load_fig5_panel("residual_quartile_hypertension_incidence_panel")
    img_d = _load_fig5_panel("roc_hypertension_incidence_2y")

    missing = [
        name for name, img in [
            ("a (cardiovascular residual quartiles)", img_a),
            ("b (prediabetes residual quartiles)", img_b),
            ("c (hypertension residual quartiles)", img_c),
            ("d (hypertension incidence ROC)", img_d),
        ] if img is None
    ]
    if missing:
        raise FileNotFoundError(
            f"Figure 5: missing panel(s) {', '.join(missing)}. "
            "Run downstream_tasks/make_target_prediction_figures.py (plots ROC and quartile panels)."
        )

    margin = 50
    gap_col, gap_row = 36, 36
    inner_use_w = FIGURE_WIDTH_PX - 2 * margin
    col_w = (inner_use_w - gap_col) // 2
    grid_w = col_w * 2 + gap_col
    x_grid0 = margin + max(0, (FIGURE_WIDTH_PX - 2 * margin - grid_w) // 2)
    xa0 = x_grid0
    xb0 = x_grid0 + col_w + gap_col
    # row1_h + gap_row + row2_h ≤ figure max height minus vertical margins only.
    max_inner_h = FIGURE_MAX_HEIGHT_PX - 2 * margin

    def _scaled_row_heights() -> tuple[int, int]:
        targets = [(img_a, img_b), (img_c, img_d)]

        row_hs: list[int] = []
        for pair in targets:
            h_max = max(
                ImageOps.contain(im, (col_w, FIGURE_MAX_HEIGHT_PX)).height
                for im in pair
                if im is not None
            )
            row_hs.append(h_max)

        r1, r2 = row_hs
        rows_sum = r1 + r2
        inner_needed = rows_sum + gap_row
        if inner_needed <= max_inner_h:
            return r1, r2

        alpha = (max_inner_h - gap_row) / max(1e-6, float(rows_sum))
        return max(80, int(r1 * alpha)), max(80, int(r2 * alpha))

    row1_h, row2_h = _scaled_row_heights()

    canvas_h = margin + row1_h + gap_row + row2_h + margin
    canvas = Image.new("RGB", (FIGURE_WIDTH_PX, canvas_h), WHITE)

    y_row2 = margin + row1_h + gap_row

    _paste_panel(
        canvas,
        Panel("a", img_a),
        (xa0, margin, xa0 + col_w, margin + row1_h),
        h_align="center",
        panel_label_pt=10.0,
    )
    _paste_panel(
        canvas,
        Panel("b", img_b),
        (xb0, margin, xb0 + col_w, margin + row1_h),
        h_align="center",
        panel_label_pt=10.0,
    )
    _paste_panel(
        canvas,
        Panel("c", img_c),
        (xa0, y_row2, xa0 + col_w, y_row2 + row2_h),
        h_align="center",
        panel_label_pt=10.0,
    )
    _paste_panel(
        canvas,
        Panel("d", img_d),
        (xb0, y_row2, xb0 + col_w, y_row2 + row2_h),
        h_align="center",
        panel_label_pt=10.0,
    )
    _save_canvas(canvas, "Figure 5")


def build_extended_data_figure_1() -> None:
    """Temporal age prediction Pearson r (TabICL mean; N = 301)."""
    pdf_path = (
        FIGURES_ROOT / "temporal_age_prediction" / "age_prediction_temporal_performance_pearson_r_tabicl_mean_N301.pdf"
    )
    if not pdf_path.is_file():
        raise FileNotFoundError(f"Missing {pdf_path}")
    img = _open_rgb(pdf_path)
    scale = min(
        FIGURE_WIDTH_PX / max(1, img.width),
        FIGURE_MAX_HEIGHT_PX / max(1, img.height),
    )
    tw = int(round(img.width * scale))
    th = int(round(img.height * scale))
    tw = min(tw, FIGURE_WIDTH_PX)
    th = min(th, FIGURE_MAX_HEIGHT_PX)
    out_img = img.resize((tw, th), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (FIGURE_WIDTH_PX, th), WHITE)
    canvas.paste(out_img, ((FIGURE_WIDTH_PX - tw) // 2, 0))
    _save_canvas(canvas, "Extended Data Figure 1")


def build_extended_data_figure_2() -> str:
    temporal_figs = FIGURES_ROOT / "temporal_age_prediction"
    pyppg_age_path = _find_one(["umap_pyppg_colored_by_age_*euclidean*.png"], [temporal_figs])
    watchpat_age_path = _find_one(["umap_watchpat_colored_by_age_*euclidean*.png"], [temporal_figs])
    panels = [
        Panel(
            "a",
            _crop_right_colorbar(_recolor_umap_age_to_blues(_open_rgb(pyppg_age_path))),
        ),
        Panel("b", _recolor_umap_age_to_blues(_open_rgb(watchpat_age_path))),
        Panel(
            "c",
            _crop_right_colorbar(
                _open_rgb(_find_one(["umap_pyppg_colored_by_heart_rate_*euclidean*.png"], [temporal_figs]))
            ),
        ),
        Panel("d", _open_rgb(_find_one(["umap_watchpat_colored_by_heart_rate_*euclidean*.png"], [temporal_figs]))),
    ]
    TOP_TITLE_H = 44
    margin, gap = 34, 24
    col_w = (FIGURE_WIDTH_PX - 2 * margin - gap) // 2
    row_h = (1640 - 2 * margin - gap) // 2
    fixed_h = 2 * margin + TOP_TITLE_H + gap
    max_row_h = max(80, (FIGURE_MAX_HEIGHT_PX - fixed_h) // 2)
    row_h = min(row_h, max_row_h)
    total_h = margin + TOP_TITLE_H + row_h + gap + row_h + margin
    canvas = Image.new("RGB", (FIGURE_WIDTH_PX, total_h), WHITE)
    x_mid = margin + col_w
    x_right = FIGURE_WIDTH_PX - margin
    _draw_centered_column_title(
        canvas,
        "PyPPG features 2D projections (UMAP)",
        margin,
        x_mid,
        margin + 2,
    )
    _draw_centered_column_title(
        canvas,
        "WatchPAT features 2D projections (UMAP)",
        x_mid + gap,
        x_right,
        margin + 2,
    )
    y_panel_top = margin + TOP_TITLE_H
    _paste_panel(canvas, panels[0], (margin, y_panel_top, margin + col_w, y_panel_top + row_h))
    _paste_panel(canvas, panels[1], (margin + col_w + gap, y_panel_top, x_right, y_panel_top + row_h))
    y2 = y_panel_top + row_h + gap
    _paste_panel(canvas, panels[2], (margin, y2, margin + col_w, y2 + row_h))
    _paste_panel(canvas, panels[3], (margin + col_w + gap, y2, x_right, y2 + row_h))
    _save_canvas(canvas, "Extended Data Figure 2")
    return _supplementary_fig1_sample_caption_clause(pyppg_age_path, watchpat_age_path)


def _refresh_legend_text(captions: str) -> str:
    """Keep canonical legend wording while applying formatting normalisation."""
    # Keep SpO2 using an ASCII lowercase "2" in all generated legends.
    refreshed = re.sub(r"SpO₂|SPO2|SpO_2", "SpO2", captions)
    return refreshed


def write_legends() -> None:
    path = MANUSCRIPT_DIR / "figures_legends.md"
    if path.exists():
        captions = path.read_text(encoding="utf-8")
    else:
        captions = "# Manuscript Figure Legends\n\n"
    captions = _refresh_legend_text(captions)
    path.write_text(captions, encoding="utf-8")
    print(f"Saved {path}")


def main() -> None:
    _apply_matplotlib_rcparams()
    MANUSCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    write_tables()
    build_figure_2()
    build_supplementary_figure_1()
    build_figure_3()
    try:
        build_figure_4()
    except FileNotFoundError as exc:
        print(f"Skipping Figure 4: {exc}")
    try:
        build_figure_4_v2()
    except FileNotFoundError as exc:
        print(f"Skipping Figure 4_v2: {exc}")
    try:
        build_figure_6()
    except FileNotFoundError as exc:
        print(f"Skipping Figure 6: {exc}")
    try:
        build_figure_6_v2()
    except FileNotFoundError as exc:
        print(f"Skipping Figure 6_v2: {exc}")
    try:
        build_figure_5()
    except FileNotFoundError as exc:
        print(f"Skipping Figure 5: {exc}")
    try:
        build_extended_data_figure_1()
    except FileNotFoundError as exc:
        print(f"Skipping Extended Data Figure 1: {exc}")
    try:
        build_extended_data_figure_4()
    except FileNotFoundError as exc:
        print(f"Skipping Extended Data Figure 4: {exc}")
    try:
        build_extended_data_figure_5()
    except FileNotFoundError as exc:
        print(f"Skipping Extended Data Figure 5: {exc}")
    try:
        build_supplementary_figure_2()
    except FileNotFoundError as exc:
        print(f"Skipping Supplementary Figure 2: {exc}")
    try:
        build_supplementary_figure_3()
    except FileNotFoundError as exc:
        print(f"Skipping Supplementary Figure 3: {exc}")
    _ = build_extended_data_figure_2()
    write_legends()


if __name__ == "__main__":
    main()
