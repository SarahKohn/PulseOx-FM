"""Standalone reproduction of manuscript Figure 3 (downstream health associations).

Figure 3 summarises how PulseOx-FM embeddings improve prediction of clinical
targets over a demographics-only baseline (age, sex, BMI):

  Panel a  HPP "current measures" (regression) — Pearson r of demographics vs.
           demographics + embeddings, one point per target, identity line.
  Panel b  HPP "current diagnoses" and "current medication intake"
           (classification) — ROC AUC of demographics vs. demographics +
           embeddings, with significance stars.
  Panel c  VitalDB pre-operative diagnoses (Diabetes, ROC AUC) and measures
           (Glucose, Haemoglobin; Pearson r).

The script is fully self-contained: it reads only the aggregate, de-identified
results in ``results/Supplementary_Table_4.csv`` (no patient-level data, no
model weights). Run:

    python plotting/figure_3/make_figure.py
    python plotting/figure_3/make_figure.py --table path/to/Supplementary_Table_4.csv --outdir some/dir
"""
from __future__ import annotations

import argparse
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TEAL = "#196874"          # demographics + embeddings
GRAY = "#b0b0b0"          # demographics only
VITALDB_RED = "#8c2d2d"   # VitalDB panel accent

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_TABLE = os.path.join(REPO_ROOT, "results", "Supplementary_Table_4.csv")
DEFAULT_OUTDIR = os.path.join(REPO_ROOT, "figures")


def parse_mean_sd(value) -> tuple[float, float]:
    """Parse a ``"mean ± sd"`` cell into ``(mean, sd)``; tolerant of blanks."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return math.nan, math.nan
    s = str(value).strip()
    if not s:
        return math.nan, math.nan
    if "±" in s:
        mean_str, sd_str = s.split("±", 1)
        try:
            return float(mean_str.strip()), float(sd_str.strip())
        except ValueError:
            return math.nan, math.nan
    try:
        return float(s), 0.0
    except ValueError:
        return math.nan, math.nan


def short_label(target: str) -> str:
    """Compact a verbose target name for plotting."""
    repl = {
        " (DXA)": "", " (SM)": "", " (BT)": "", " (US)": "", " (FI)": "", " (DL)": "",
        "Median daily ": "", "caloric intake": "cal.", "cholesterol": "chol.",
    }
    out = str(target)
    for k, v in repl.items():
        out = out.replace(k, v)
    return out.strip()


def load_panels(table_path: str) -> dict[str, pd.DataFrame]:
    df = pd.read_csv(table_path)
    demo = df["ROC_AUC_or_Pearson_r_demo_mean_sd"].map(parse_mean_sd)
    emb = df["ROC_AUC_or_Pearson_r_embeddings_mean_sd"].map(parse_mean_sd)
    df["demo_mean"] = [m for m, _ in demo]
    df["demo_sd"] = [s for _, s in demo]
    df["emb_mean"] = [m for m, _ in emb]
    df["emb_sd"] = [s for _, s in emb]
    df["stars"] = df["Significance_stars_demo_vs_emb"].fillna("").astype(str).str.strip()
    return {name: g.reset_index(drop=True) for name, g in df.groupby("Cohort_and_panel")}


def _scatter_panel_a(ax, measures: pd.DataFrame, label_color: str = "#333") -> None:
    """Panel a: demographics vs. demographics + embeddings Pearson r (regression)."""
    lim = 1.0
    ax.plot([0, lim], [0, lim], ls=":", color="#444", lw=1.0, zorder=0)
    ax.scatter(measures["demo_mean"], measures["emb_mean"], s=34, color=TEAL,
               edgecolor="white", linewidth=0.4, zorder=3)
    for _, r in measures.iterrows():
        if pd.notna(r["demo_mean"]):
            ax.annotate(short_label(r["Target"]), (r["demo_mean"], r["emb_mean"]),
                        fontsize=5.5, xytext=(2, 2), textcoords="offset points", color=label_color)
    ax.set_xlim(-0.05, lim)
    ax.set_ylim(-0.05, lim)
    ax.set_xlabel("Pearson r (Age, sex, BMI)", fontsize=9)
    ax.set_ylabel("Pearson r (Age, sex, BMI + PulseOx-FM embeddings)", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _grouped_auc_barh(ax, sub: pd.DataFrame, title: str) -> None:
    sub = sub.sort_values("emb_mean", ascending=True).reset_index(drop=True)
    y = np.arange(len(sub))
    h = 0.38
    ax.barh(y - h / 2, sub["demo_mean"], height=h, xerr=sub["demo_sd"],
            color="#ededed", edgecolor="silver", hatch="///", linewidth=0.7,
            error_kw={"ecolor": "#555", "elinewidth": 0.8, "capsize": 2},
            label="Age, sex, BMI")
    ax.barh(y + h / 2, sub["emb_mean"], height=h, xerr=sub["emb_sd"],
            color=TEAL, edgecolor="white", linewidth=0.5,
            error_kw={"ecolor": "#555", "elinewidth": 0.8, "capsize": 2},
            label="Age, sex, BMI + PulseOx-FM embeddings")
    ax.axvline(0.5, color="#777", linestyle="--", linewidth=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels([short_label(t) for t in sub["Target"]], fontsize=7)
    for i, row in sub.iterrows():
        x = max(row["demo_mean"] + (row["demo_sd"] or 0), row["emb_mean"] + (row["emb_sd"] or 0))
        if row["stars"]:
            ax.text(x + 0.01, i, row["stars"], va="center", ha="left", fontsize=8, color="#222")
    ax.set_xlim(0.4, 0.92)
    ax.set_xlabel("ROC AUC", fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.tick_params(axis="x", labelsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def make_figure(table_path: str, outdir: str) -> list[str]:
    panels = load_panels(table_path)
    measures = panels.get("HPP sleep cohort — current measures", pd.DataFrame())
    diagnoses = panels.get("HPP sleep cohort — current diagnoses", pd.DataFrame())
    meds = panels.get("HPP sleep cohort — current medication intake", pd.DataFrame())
    vitaldb_dx = panels.get("VitalDB — preoperative diagnoses", pd.DataFrame())
    vitaldb_meas = panels.get("VitalDB — preoperative measures", pd.DataFrame())

    fig = plt.figure(figsize=(13, 9))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.5, 1.0, 1.0], height_ratios=[1.4, 1.0],
                          hspace=0.32, wspace=0.42)

    # ---- Panel a: regression scatter -------------------------------------
    ax_a = fig.add_subplot(gs[0, 0])
    _scatter_panel_a(ax_a, measures)
    ax_a.set_title("a  HPP current measures (regression)", fontsize=10, loc="left")

    # ---- Panel b: classification bars ------------------------------------
    ax_b1 = fig.add_subplot(gs[0, 1])
    _grouped_auc_barh(ax_b1, diagnoses, "b  HPP current diagnoses")
    ax_b2 = fig.add_subplot(gs[0, 2])
    _grouped_auc_barh(ax_b2, meds, "HPP current medication intake")
    handles, labels = ax_b1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False,
               fontsize=8.5, bbox_to_anchor=(0.5, 0.005))

    # ---- Panel c: VitalDB -------------------------------------------------
    ax_c = fig.add_subplot(gs[1, 0])
    rows, demo_v, emb_v, demo_e, emb_e, stars, kinds = [], [], [], [], [], [], []
    for _, r in pd.concat([vitaldb_dx, vitaldb_meas], ignore_index=True).iterrows():
        rows.append(r["Target"].replace("Preoperative ", "").capitalize())
        demo_v.append(r["demo_mean"]); emb_v.append(r["emb_mean"])
        demo_e.append(r["demo_sd"]); emb_e.append(r["emb_sd"])
        stars.append(r["stars"]); kinds.append(r["Figure_primary_metric"])
    y = np.arange(len(rows)); h = 0.38
    ax_c.barh(y - h / 2, demo_v, height=h, xerr=demo_e, color="#ededed",
              edgecolor="silver", hatch="///", linewidth=0.7,
              error_kw={"ecolor": "#555", "elinewidth": 0.8, "capsize": 2}, label="Age, sex, BMI")
    ax_c.barh(y + h / 2, emb_v, height=h, xerr=emb_e, color=VITALDB_RED,
              edgecolor="white", linewidth=0.5,
              error_kw={"ecolor": "#555", "elinewidth": 0.8, "capsize": 2},
              label="Age, sex, BMI + PulseOx-FM embeddings")
    ax_c.set_yticks(y)
    ax_c.set_yticklabels([f"{n}\n({k})" for n, k in zip(rows, kinds)], fontsize=7)
    for i in range(len(rows)):
        x = max((demo_v[i] or 0) + (demo_e[i] or 0), (emb_v[i] or 0) + (emb_e[i] or 0))
        if stars[i]:
            ax_c.text(x + 0.01, i, stars[i], va="center", ha="left", fontsize=8, color="#222")
    ax_c.set_xlabel("ROC AUC (diagnoses) / Pearson r (measures)", fontsize=8)
    ax_c.set_title("c  VitalDB pre-operative (out-of-distribution)", fontsize=10, loc="left")
    ax_c.tick_params(axis="x", labelsize=7)
    ax_c.spines["top"].set_visible(False)
    ax_c.spines["right"].set_visible(False)

    fig.suptitle("Figure 3 — PulseOx-FM embeddings improve clinical target prediction over demographics",
                 fontsize=12, y=0.98)

    os.makedirs(outdir, exist_ok=True)
    written = []
    for ext in ("pdf", "png"):
        out = os.path.join(outdir, f"Figure_3_reproduced.{ext}")
        fig.savefig(out, dpi=300, bbox_inches="tight")
        written.append(out)
    plt.close(fig)

    # Also emit panel a on its own (used in the README Results section).
    fig_a, ax_only = plt.subplots(figsize=(6.5, 6.5))
    _scatter_panel_a(ax_only, measures)
    ax_only.set_title("Figure 3a — HPP current measures (regression)\n"
                      "Pearson r: demographics vs. demographics + PulseOx-FM embeddings",
                      fontsize=10)
    fig_a.tight_layout()
    for ext in ("pdf", "png"):
        out = os.path.join(outdir, f"Figure_3a_reproduced.{ext}")
        fig_a.savefig(out, dpi=300, bbox_inches="tight")
        written.append(out)
    plt.close(fig_a)
    return written


def main() -> None:
    ap = argparse.ArgumentParser(description="Reproduce manuscript Figure 3 from Supplementary Table 4.")
    ap.add_argument("--table", default=DEFAULT_TABLE, help="Path to Supplementary_Table_4.csv")
    ap.add_argument("--outdir", default=DEFAULT_OUTDIR, help="Directory for output figure files")
    args = ap.parse_args()
    for f in make_figure(args.table, args.outdir):
        print(f"Saved {f}")


if __name__ == "__main__":
    main()
