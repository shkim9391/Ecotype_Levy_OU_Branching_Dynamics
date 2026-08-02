from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Ellipse


# ============================================================
# 1. CONFIG
# ============================================================
PROJECT_DIR = Path("/Multimodal_OU_Lévy_Branching_Scaffold/Figure_3")
DERIVED_DIR = PROJECT_DIR / "derived"
PANELS_DIR = PROJECT_DIR / "panels"

IN_MAIN = DERIVED_DIR / "patient_timepoint_centroids_main.csv"

OUT_PNG = PANELS_DIR / "Figure3B_group_average_trajectories.png"
OUT_PDF = PANELS_DIR / "Figure3B_group_average_trajectories.pdf"

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


def draw_arrow(ax, p1, p2, *, color="#444444", lw=2.2, alpha=0.95, linestyle="-", zorder=4):
    arr = FancyArrowPatch(
        p1, p2,
        arrowstyle="-|>",
        mutation_scale=14,
        linewidth=lw,
        linestyle=linestyle,
        color=color,
        alpha=alpha,
        zorder=zorder,
        shrinkA=3,
        shrinkB=3,
    )
    ax.add_patch(arr)


def add_cov_ellipse(ax, pts: np.ndarray, color: str, alpha=0.12, zorder=1):
    if pts.shape[0] < 3:
        return

    cov = np.cov(pts.T)
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]

    vals = np.maximum(vals, 1e-8)
    angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    width, height = 2 * np.sqrt(vals)

    ell = Ellipse(
        xy=np.mean(pts, axis=0),
        width=width,
        height=height,
        angle=angle,
        facecolor=color,
        edgecolor=color,
        alpha=alpha,
        linewidth=1.0,
        zorder=zorder,
    )
    ax.add_patch(ell)


def add_timepoint_legend(ax):
    for tp, color in TIME_COLORS.items():
        ax.scatter([], [], s=45, c=color, label=tp)
    ax.legend(frameon=False, loc="best", fontsize=9)


def robust_center(df: pd.DataFrame) -> np.ndarray:
    return np.array([
        np.nanmedian(df["x2d"].to_numpy(dtype=float)),
        np.nanmedian(df["y2d"].to_numpy(dtype=float)),
    ])


# ============================================================
# 3. MAIN
# ============================================================
def main() -> None:
    ensure_dirs()

    cent = pd.read_csv(IN_MAIN)
    cent["time_order"] = cent["clinical_timepoint_coarse"].map(TIME_ORDER)
    cent = cent.sort_values(["patient_id", "time_order", "sample_id"]).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(9.5, 8.5))

    # ---------------------------------------------------------
    # Patient-level centroid clouds
    # ---------------------------------------------------------
    for tp in ["DX", "REL"]:
        sub = cent[cent["clinical_timepoint_coarse"] == tp].copy()
        if sub.empty:
            continue

        pts = sub[["x2d", "y2d"]].to_numpy(dtype=float)
        add_cov_ellipse(ax, pts, TIME_COLORS[tp], alpha=0.07, zorder=1)

        ax.scatter(
            pts[:, 0], pts[:, 1],
            s=np.clip(np.sqrt(sub["n_cells"].to_numpy()) * 0.55, 10, 34),
            c=TIME_COLORS[tp],
            alpha=0.42,
            linewidths=0,
            zorder=2,
        )

    # ---------------------------------------------------------
    # Group-average DX and REL centers
    # ---------------------------------------------------------
    dx = cent[cent["clinical_timepoint_coarse"] == "DX"].copy()
    rel = cent[cent["clinical_timepoint_coarse"] == "REL"].copy()

    if dx.empty or rel.empty:
        raise ValueError("Panel B requires both DX and REL centroids in the main cohort.")

    dx_center = robust_center(dx)
    rel_center = robust_center(rel)

    draw_arrow(
        ax,
        dx_center,
        rel_center,
        color="#333333",
        lw=2.6,
        alpha=0.95,
        linestyle="-",
        zorder=5,
    )

    ax.scatter(
        dx_center[0], dx_center[1],
        s=140, c=TIME_COLORS["DX"],
        edgecolors="black", linewidths=0.8, zorder=6
    )
    ax.scatter(
        rel_center[0], rel_center[1],
        s=140, c=TIME_COLORS["REL"],
        edgecolors="black", linewidths=0.8, zorder=6
    )

    # ---------------------------------------------------------
    # AML21 persistence overlay
    # ---------------------------------------------------------
    aml21 = cent[cent["patient_id"] == HIGHLIGHT_PATIENT].copy()
    aml21 = aml21.sort_values("time_order")

    if not aml21.empty and "EOI_REM" in set(aml21["clinical_timepoint_coarse"].astype(str)):
        pts = aml21[["x2d", "y2d"]].to_numpy(dtype=float)

        for i in range(len(aml21) - 1):
            t_start = aml21.iloc[i]["clinical_timepoint_coarse"]
            t_end = aml21.iloc[i + 1]["clinical_timepoint_coarse"]

            linestyle = "-" if t_end == "EOI_REM" else "--" if t_start == "EOI_REM" else "-"
            draw_arrow(
                ax,
                pts[i],
                pts[i + 1],
                color="#202020",
                lw=1.8,
                alpha=0.90,
                linestyle=linestyle,
                zorder=7,
            )

        for _, r in aml21.iterrows():
            tp = r["clinical_timepoint_coarse"]
            ax.scatter(
                r["x2d"], r["y2d"],
                s=70,
                c=TIME_COLORS.get(tp, "#999999"),
                edgecolors="black",
                linewidths=0.7,
                zorder=8,
            )

        rem_row = aml21[aml21["clinical_timepoint_coarse"] == "EOI_REM"]
        if not rem_row.empty:
            rr = rem_row.iloc[0]
            ax.text(
                rr["x2d"] + 0.12,
                rr["y2d"] + 0.04,
                "AML21 REM",
                fontsize=9,
                fontweight="bold",
                ha="left",
                va="center",
                color="#202020",
                zorder=9,
            )

    # ---------------------------------------------------------
    # Labels for group centers
    # ---------------------------------------------------------
    vec = rel_center - dx_center
    norm = np.linalg.norm(vec)
    u = vec / norm if norm > 0 else np.array([1.0, 0.0])
    perp = np.array([-u[1], u[0]])
    
    label_box = dict(facecolor="white", edgecolor="none", alpha=0.7, pad=1.5)
    
    ax.text(
        dx_center[0] + 0.16, dx_center[1] + 0.08,
        "DX median",
        fontsize=9, fontweight="bold",
        color="#333333",
        ha="left", va="center", zorder=7,
        bbox=label_box
    )
    
    ax.text(
        rel_center[0] + 0.16, rel_center[1] - 0.08,
        "REL median",
        fontsize=9, fontweight="bold",
        color="#333333",
        ha="left", va="center", zorder=7,
        bbox=label_box
    )

    # ---------------------------------------------------------
    # Tight limits
    # ---------------------------------------------------------
    all_x = cent["x2d"].to_numpy(dtype=float)
    all_y = cent["y2d"].to_numpy(dtype=float)
    pad_x = 0.15 * (all_x.max() - all_x.min())
    pad_y = 0.18 * (all_y.max() - all_y.min())
    ax.set_xlim(all_x.min() - pad_x, all_x.max() + pad_x)
    ax.set_ylim(all_y.min() - pad_y, all_y.max() + pad_y)

    # ---------------------------------------------------------
    # Styling
    # ---------------------------------------------------------
    style_axis(ax, "B", "Group-level contraction and relapse-associated departure")
    add_timepoint_legend(ax)

#    ax.text(
#        0.01, -0.06,
#        "Group medians summarize the 21-patient DX→REL cohort; AML21 REM is shown as the one retained persistence intermediate.",
#        transform=ax.transAxes,
#        fontsize=8.5,
#        ha="left",
#        va="top",
#        color="#555555",
#    )

    fig.savefig(OUT_PNG, dpi=600, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    plt.close(fig)

    print(f"[DONE] Saved {OUT_PNG}")
    print(f"[DONE] Saved {OUT_PDF}")


if __name__ == "__main__":
    main()
