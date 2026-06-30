"""Standalone reproduction of manuscript Figure 4, panel b.

Figure 4b shows next-day metabolic associations: how well next-day continuous
glucose-monitoring (CGM), food-log and wearable targets are predicted from
sleep architecture alone versus sleep architecture + PulseOx-FM embeddings,
as Pearson r (one point per target, identity line).

NOTE on panel a: manuscript Figure 4a (the histogram of within-/between-person
L2 distances in embedding space) requires per-recording embeddings, which are
not distributed with this repository. Only panel b — which depends solely on
the aggregate, de-identified results in ``results/Supplementary_Table_5.csv`` —
is reproduced here.

Run:
    python plotting/figure_4/make_figure.py
    python plotting/figure_4/make_figure.py --table path/to/Supplementary_Table_5.csv --outdir some/dir
"""
from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DOMAIN_COLORS = {
    "CGM": "#15505c",
    "Food": "#2f8093",
    "Wearables": "#7fc1cc",
}
NS_COLOR = "#b5b5b5"
Q_THRESHOLD = 0.05

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_TABLE = os.path.join(REPO_ROOT, "results", "Supplementary_Table_5.csv")
DEFAULT_OUTDIR = os.path.join(REPO_ROOT, "figures")


def make_figure(table_path: str, outdir: str) -> list[str]:
    df = pd.read_csv(table_path)
    df["significant"] = pd.to_numeric(df["q_comb_vs_arch_FDR"], errors="coerce") < Q_THRESHOLD

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    lim_hi = float(np.nanmax([df["EmbPlusArch_Pearson_r_mean"].max(),
                              df["SleepArch_Pearson_r_mean"].max()]) + 0.05)
    lim_lo = float(np.nanmin([df["EmbPlusArch_Pearson_r_mean"].min(),
                              df["SleepArch_Pearson_r_mean"].min()]) - 0.03)
    ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], ls="--", color="#999", lw=1.0, zorder=0)
    ax.axhline(0, color="#bbb", lw=0.6, zorder=0)
    ax.axvline(0, color="#bbb", lw=0.6, zorder=0)

    for domain, g in df.groupby("Domain"):
        base_color = DOMAIN_COLORS.get(domain, "#444")
        for _, r in g.iterrows():
            sig = bool(r["significant"])
            color = base_color if sig else NS_COLOR
            ax.errorbar(r["SleepArch_Pearson_r_mean"], r["EmbPlusArch_Pearson_r_mean"],
                        xerr=r["SleepArch_Pearson_r_SD"], yerr=r["EmbPlusArch_Pearson_r_SD"],
                        fmt="o", ms=7, color=color, ecolor=color, elinewidth=0.8,
                        capsize=2, markeredgecolor="white", markeredgewidth=0.5, zorder=3)
            ax.annotate(str(r["Target"]), (r["SleepArch_Pearson_r_mean"], r["EmbPlusArch_Pearson_r_mean"]),
                        fontsize=6.5, xytext=(4, 2), textcoords="offset points",
                        color="#333" if sig else "#999")

    # Legend: domains + non-significant marker
    handles = [plt.Line2D([0], [0], marker="o", ls="", ms=7, mec="white",
                          color=DOMAIN_COLORS[d], label=d) for d in ["CGM", "Food", "Wearables"]]
    handles.append(plt.Line2D([0], [0], marker="o", ls="", ms=7, mec="white",
                             color=NS_COLOR, label=f"Not significant (q ≥ {Q_THRESHOLD})"))
    ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=8.5)

    ax.set_xlim(lim_lo, lim_hi)
    ax.set_ylim(lim_lo, lim_hi)
    ax.set_xlabel("Pearson r (sleep architecture)", fontsize=10)
    ax.set_ylabel("Pearson r (sleep architecture + PulseOx-FM embeddings)", fontsize=10)
    ax.set_title("Figure 4b — Next-day metabolic associations", fontsize=11)
    ax.tick_params(labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    os.makedirs(outdir, exist_ok=True)
    written = []
    for ext in ("pdf", "png"):
        out = os.path.join(outdir, f"Figure_4b_reproduced.{ext}")
        fig.savefig(out, dpi=300, bbox_inches="tight")
        written.append(out)
    plt.close(fig)
    return written


def main() -> None:
    ap = argparse.ArgumentParser(description="Reproduce manuscript Figure 4b from Supplementary Table 5.")
    ap.add_argument("--table", default=DEFAULT_TABLE, help="Path to Supplementary_Table_5.csv")
    ap.add_argument("--outdir", default=DEFAULT_OUTDIR, help="Directory for output figure files")
    args = ap.parse_args()
    for f in make_figure(args.table, args.outdir):
        print(f"Saved {f}")
    print("Note: manuscript Figure 4a (embedding-distance histogram) requires per-recording "
          "embeddings and is not reproducible from the shipped aggregate tables.")


if __name__ == "__main__":
    main()
