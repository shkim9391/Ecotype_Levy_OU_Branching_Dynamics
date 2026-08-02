from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch


# ============================================================
# 1. CONFIG
# ============================================================
PROJECT_DIR = Path("/Multimodal_OU_Lévy_Branching_Scaffold/Figure_3")
DERIVED_DIR = PROJECT_DIR / "derived"
PANELS_DIR = PROJECT_DIR / "panels"

IN_ALL = DERIVED_DIR / "patient_timepoint_centroids_all.csv"
IN_MAIN = DERIVED_DIR / "patient_timepoint_centroids_main.csv"
MAIN_PATIENTS_TXT = DERIVED_DIR / "figure3_main_analysis_patients.txt"

OUT_PNG = PANELS_DIR / "Figure3A_patient_trajectories.png"
OUT_PDF = PANELS_DIR / "Figure3A_patient_trajectories.pdf"

TIME_ORDER = {"DX": 0, "EOI_REM": 1, "REL": 2}
TIME_COLORS = {
    "DX": "#7A8DA8",
    "EOI_REM": "#6BAF92",
    "REL": "#C97979",
}

HIGHLIGHT_PATIENT = "AML21"


# ============================================================
# 2. HELPERS
# ============================================================
def ensure_dirs() -> None:
    PANELS_DIR.mkdir(parents=True, exist_ok=True)


def load_main_patients(fp: Path) -> set[str]:
    pts = []
    with open(fp, "r") as f:
        for line in f:
            p = line.strip()
            if p:
                pts.append(p)
    return set(pts)


def style_axis(ax, panel_label: str, title: str) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

    ax.set_title("")
    ax.text(
        0.00, 1.03, panel_label,
        transform=ax.transAxes,
        fontsize=18, fontweight="bold",
        ha="left", va="bottom"
    )
    ax.text(
        0.10, 1.03, title,
        transform=ax.transAxes,
        fontsize=12, fontweight="bold",
        ha="left", va="bottom"
    )


def draw_arrow(ax, p1, p2, *, color="#666666", lw=1.0, alpha=0.6, linestyle="-", zorder=2):
    arr = FancyArrowPatch(
        p1, p2,
        arrowstyle="-|>",
        mutation_scale=10,
        linewidth=lw,
        linestyle=linestyle,
        color=color,
        alpha=alpha,
        zorder=zorder,
        shrinkA=2,
        shrinkB=2,
    )
    ax.add_patch(arr)


def add_timepoint_legend(ax):
    for tp, color in TIME_COLORS.items():
        ax.scatter([], [], s=40, c=color, label=tp)
    ax.legend(frameon=False, loc="best", fontsize=9)


# ============================================================
# 3. MAIN
# ============================================================
def main() -> None:
    ensure_dirs()

    cent_all = pd.read_csv(IN_ALL)
    cent_main = pd.read_csv(IN_MAIN)
    main_patients = load_main_patients(MAIN_PATIENTS_TXT)

    cent_all["time_order"] = cent_all["clinical_timepoint_coarse"].map(TIME_ORDER)
    cent_main["time_order"] = cent_main["clinical_timepoint_coarse"].map(TIME_ORDER)

    cent_all = cent_all.sort_values(["patient_id", "time_order", "sample_id"]).reset_index(drop=True)
    cent_main = cent_main.sort_values(["patient_id", "time_order", "sample_id"]).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(9.5, 8.5))

    # -----------------------------------------------------------------
    # Background: all retained sample centroids, faint gray
    # -----------------------------------------------------------------
    # background centroids
    ax.scatter(
        cent_all["x2d"],
        cent_all["y2d"],
        s=np.clip(np.sqrt(cent_all["n_cells"]) * 0.45, 6, 26),
        c="#D8D8D8",
        alpha=0.28,
        linewidths=0,
        zorder=1,
    )

    # -----------------------------------------------------------------
    # Main cohort trajectories: thin gray arrows, colored endpoints
    # -----------------------------------------------------------------
    for patient_id, sub in cent_main.groupby("patient_id", sort=False):
        sub = sub.sort_values("time_order")

        # Skip highlight patient here; draw separately below
        if patient_id == HIGHLIGHT_PATIENT:
            continue

        pts = sub[["x2d", "y2d"]].to_numpy()

        if len(sub) >= 2:
            for i in range(len(sub) - 1):
                t_start = sub.iloc[i]["clinical_timepoint_coarse"]
                t_end = sub.iloc[i + 1]["clinical_timepoint_coarse"]

                linestyle = "-" if t_end == "EOI_REM" else "--" if t_start == "EOI_REM" else "-"
                draw_arrow(
                    ax,
                    pts[i],
                    pts[i + 1],
                    color="#8A8A8A",
                    lw=0.8,
                    alpha=0.35,
                    linestyle=linestyle,
                    zorder=2,
                )

        for _, r in sub.iterrows():
            tp = r["clinical_timepoint_coarse"]
            ax.scatter(
                r["x2d"],
                r["y2d"],
                s=36,
                c=TIME_COLORS.get(tp, "#999999"),
                edgecolors="white",
                linewidths=0.5,
                zorder=3,
            )

    # -----------------------------------------------------------------
    # Highlight AML21 as the one three-timepoint case
    # -----------------------------------------------------------------
    aml21 = cent_main[cent_main["patient_id"] == HIGHLIGHT_PATIENT].copy()
    if not aml21.empty:
        aml21 = aml21.sort_values("time_order")
        pts = aml21[["x2d", "y2d"]].to_numpy()

        if len(aml21) >= 2:
            for i in range(len(aml21) - 1):
                t_start = aml21.iloc[i]["clinical_timepoint_coarse"]
                t_end = aml21.iloc[i + 1]["clinical_timepoint_coarse"]

                linestyle = "-" if t_end == "EOI_REM" else "--" if t_start == "EOI_REM" else "-"
                draw_arrow(
                    ax,
                    pts[i],
                    pts[i + 1],
                    color="#202020",
                    lw=2.0,
                    alpha=0.90,
                    linestyle=linestyle,
                    zorder=4,
                )

        for _, r in aml21.iterrows():
            tp = r["clinical_timepoint_coarse"]
            ax.scatter(
                r["x2d"],
                r["y2d"],
                s=70,
                c=TIME_COLORS.get(tp, "#999999"),
                edgecolors="black",
                linewidths=0.7,
                zorder=5,
            )

        # label AML21 near the REL point if present, otherwise last point
        label_row = aml21.sort_values("time_order").iloc[-1]
        ax.text(
            label_row["x2d"] + 0.14,
            label_row["y2d"] + 0.03,
            "AML21",
            fontsize=9,
            fontweight="bold",
            ha="left",
            va="center",
            color="#202020",
            zorder=6,
        )

    # -----------------------------------------------------------------
    # Styling
    # -----------------------------------------------------------------
    style_axis(ax, "A", "   Patient-level longitudinal trajectories")
    add_timepoint_legend(ax)

    all_x = cent_all["x2d"].to_numpy()
    all_y = cent_all["y2d"].to_numpy()
    pad_x = 0.15 * (all_x.max() - all_x.min())
    pad_y = 0.15 * (all_y.max() - all_y.min())
    
    ax.set_xlim(all_x.min() - pad_x, all_x.max() + pad_x)
    ax.set_ylim(all_y.min() - pad_y, all_y.max() + pad_y)

    fig.savefig(OUT_PNG, dpi=600, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    plt.close(fig)

    print(f"[DONE] Saved {OUT_PNG}")
    print(f"[DONE] Saved {OUT_PDF}")


if __name__ == "__main__":
    main()
