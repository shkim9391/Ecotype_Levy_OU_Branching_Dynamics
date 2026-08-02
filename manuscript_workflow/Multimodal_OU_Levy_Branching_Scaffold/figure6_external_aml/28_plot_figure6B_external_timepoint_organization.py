from __future__ import annotations

from pathlib import Path
from scipy.stats import gaussian_kde
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch


# ============================================================
# 1. CONFIG
# ============================================================
FIG4_DERIVED = Path("/Multimodal_OU_Lévy_Branching_Scaffold/Figure_4/derived")
FIG6_DIR = Path("/Multimodal_OU_Lévy_Branching_Scaffold/Figure_6")
FIG6_DERIVED = FIG6_DIR / "derived"
FIG6_PANELS = FIG6_DIR / "panels"

IN_CENT = FIG6_DERIVED / "gse235923_sample_centroids.csv"
IN_DISC = FIG4_DERIVED / "sample_dynamic_parameters.csv"

OUT_PNG = FIG6_PANELS / "Figure6B_external_timepoint_organization.png"
OUT_PDF = FIG6_PANELS / "Figure6B_external_timepoint_organization.pdf"

TIME_ORDER = {"DX": 0, "EOI_REM": 1, "REL": 2}
TIME_COLORS = {
    "DX": "#7A8DA8",
    "EOI_REM": "#6BAF92",
    "REL": "#C97979",
}

TIME_LABELS = {
    "DX": "DX",
    "EOI_REM": "EOI/REM",
    "REL": "REL",
}

HIGHLIGHT_PATIENTS = {"Sample5", "Sample6"}


# ============================================================
# 2. HELPERS
# ============================================================
def ensure_dirs() -> None:
    FIG6_PANELS.mkdir(parents=True, exist_ok=True)


def style_axis(ax, panel_label: str, title: str,
               panel_fontsize: int = 18,
               title_fontsize: int = 12,
               title_x: float = 0.10) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

    ax.text(
        0.00, 1.03, panel_label,
        transform=ax.transAxes,
        fontsize=panel_fontsize,
        fontweight="bold",
        ha="left", va="bottom"
    )
    ax.text(
        title_x, 1.03, title,
        transform=ax.transAxes,
        fontsize=title_fontsize,
        fontweight="bold",
        ha="left", va="bottom"
    )

def draw_reference_field(
    ax,
    x,
    y,
    fill_colors=("#F3F3F3", "#ECECEC", "#E4E4E4", "#DADADA"),
    line_color="#C8C8C8",
    alpha=0.55,
    n_grid=180,
):
    """
    Draw a light density field showing the discovery reference scaffold.
    This gives the projected external cohort a visible background landscape
    without implying a hard classification boundary.
    """
    x = pd.to_numeric(pd.Series(x), errors="coerce").to_numpy()
    y = pd.to_numeric(pd.Series(y), errors="coerce").to_numpy()

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) < 8:
        return

    # Use quantile limits to avoid extreme outliers dominating the field.
    x_lo, x_hi = np.nanquantile(x, [0.02, 0.98])
    y_lo, y_hi = np.nanquantile(y, [0.02, 0.98])

    pad_x = 0.18 * (x_hi - x_lo)
    pad_y = 0.18 * (y_hi - y_lo)

    xx, yy = np.meshgrid(
        np.linspace(x_lo - pad_x, x_hi + pad_x, n_grid),
        np.linspace(y_lo - pad_y, y_hi + pad_y, n_grid),
    )

    try:
        kde = gaussian_kde(np.vstack([x, y]))
        zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
    except Exception:
        return

    positive = zz[zz > 0]
    if positive.size == 0:
        return

    levels = np.quantile(positive, [0.50, 0.68, 0.82, 0.93])
    levels = np.unique(levels)

    if len(levels) < 2:
        return

    contour_levels = list(levels) + [float(np.nanmax(zz))]

    ax.contourf(
        xx,
        yy,
        zz,
        levels=contour_levels,
        colors=fill_colors[: len(contour_levels) - 1],
        alpha=alpha,
        zorder=0,
    )

    # Outer reference boundary
    ax.contour(
        xx,
        yy,
        zz,
        levels=[levels[0]],
        colors=line_color,
        linewidths=1.0,
        alpha=0.75,
        zorder=0.5,
    )


def add_phase_legend(ax):
    for tp, c in TIME_COLORS.items():
        ax.scatter([], [], s=42, c=c, label=TIME_LABELS.get(tp, tp))

    ax.legend(
        frameon=False,
        loc="upper right",
        fontsize=9,
        handletextpad=0.4,
        borderaxespad=0.2,
    )


def draw_arrow(ax, p1, p2, *, color="#666666", lw=1.2, alpha=0.6, linestyle="-", zorder=2):
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


# ============================================================
# 3. MAIN
# ============================================================
def main() -> None:
    ensure_dirs()

    cent = pd.read_csv(IN_CENT)
    disc = pd.read_csv(IN_DISC)

    cent["clinical_timepoint_coarse"] = cent["clinical_timepoint_coarse"].astype(str)
    cent["time_order"] = cent["clinical_timepoint_coarse"].map(TIME_ORDER).fillna(999).astype(int)

    if not {"x2d", "y2d", "patient_id", "sample_id", "clinical_timepoint_coarse", "n_cells"}.issubset(cent.columns):
        raise ValueError("gse235923_sample_centroids.csv missing required centroid columns.")
    if not {"PC1", "PC2"}.issubset(disc.columns):
        raise ValueError("Discovery sample_dynamic_parameters.csv must contain PC1 and PC2.")

    fig, ax = plt.subplots(figsize=(8.8, 7.2))

    # ---------------------------------------------------------
    # Discovery reference field and anchors
    # ---------------------------------------------------------
    ax.set_facecolor("#FBFBFB")
    
    draw_reference_field(
        ax,
        disc["PC1"],
        disc["PC2"],
    )
    
    if "n_cells" in disc.columns:
        ref_sizes = np.clip(np.sqrt(disc["n_cells"]) * 0.40, 10, 34)
    else:
        ref_sizes = 16
    
    ax.scatter(
        disc["PC1"],
        disc["PC2"],
        s=ref_sizes,
        c="#CFCFCF",
        alpha=0.14,
        linewidths=0,
        zorder=1,
    )

    # ---------------------------------------------------------
    # Longitudinal patient paths
    # ---------------------------------------------------------
    for patient_id, sub in cent.groupby("patient_id", sort=False):
        sub = sub.sort_values("time_order")
        pts = sub[["x2d", "y2d"]].to_numpy(dtype=float)

        if len(sub) < 2:
            continue

        is_highlight = str(patient_id) in HIGHLIGHT_PATIENTS

        for i in range(len(sub) - 1):
            t0 = sub.iloc[i]["clinical_timepoint_coarse"]
            t1 = sub.iloc[i + 1]["clinical_timepoint_coarse"]

            # Solid for DX→EOI/REM, dashed for EOI/REM→REL when available
            linestyle = "--" if t0 == "EOI_REM" or t1 == "REL" else "-"

            draw_arrow(
                ax,
                pts[i],
                pts[i + 1],
                color="#222222" if is_highlight else "#9A9A9A",
                lw=1.8 if is_highlight else 0.9,
                alpha=0.90 if is_highlight else 0.28,
                linestyle=linestyle,
                zorder=5 if is_highlight else 2,
            )

    # ---------------------------------------------------------
    # Sample centroids
    # ---------------------------------------------------------
    for tp in ["DX", "EOI_REM", "REL"]:
        sub = cent[cent["clinical_timepoint_coarse"] == tp].copy()
        if sub.empty:
            continue

        ax.scatter(
            sub["x2d"],
            sub["y2d"],
            s=np.clip(np.sqrt(sub["n_cells"]) * 0.65, 28, 74),
            c=TIME_COLORS[tp],
            edgecolors="white",
            linewidths=0.8,
            alpha=0.95,
            zorder=4,
        )

    # ---------------------------------------------------------
    # Highlight and label Sample5 / Sample6
    # ---------------------------------------------------------
    tri = cent[cent["patient_id"].astype(str).isin(HIGHLIGHT_PATIENTS)].copy()
    for patient_id, sub in tri.groupby("patient_id", sort=False):
        sub = sub.sort_values("time_order")
        if sub.empty:
            continue
        rr = sub.iloc[-1]
        ax.text(
            rr["x2d"] + 0.04,
            rr["y2d"] + 0.03,
            str(patient_id),
            fontsize=8.5,
            fontweight="bold",
            ha="left",
            va="center",
            color="#222222",
            zorder=5,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=1.0),
        )

    # ---------------------------------------------------------
    # Limits: use percentile-based crop to reduce whitespace
    # ---------------------------------------------------------
    all_x = np.concatenate([disc["PC1"].to_numpy(dtype=float), cent["x2d"].to_numpy(dtype=float)])
    all_y = np.concatenate([disc["PC2"].to_numpy(dtype=float), cent["y2d"].to_numpy(dtype=float)])

    x_lo, x_hi = np.nanquantile(all_x, [0.01, 0.99])
    y_lo, y_hi = np.nanquantile(all_y, [0.01, 0.99])

    pad_x = 0.18 * (x_hi - x_lo)
    pad_y = 0.18 * (y_hi - y_lo)

    ax.set_xlim(x_lo - pad_x, x_hi + pad_x)
    ax.set_ylim(y_lo - pad_y, y_hi + pad_y)

    style_axis(ax, "B", "External longitudinal trajectories in projected space")
    add_phase_legend(ax)

    ax.text(
        0.98,
        0.03,
        "Highlighted cases have full DX→EOI/REM→REL trajectories",
        transform=ax.transAxes,
        fontsize=9.0,
        ha="right",
        va="bottom",
        color="#555555",
    )

    fig.savefig(OUT_PNG, dpi=600, bbox_inches="tight")
    fig.savefig(OUT_PDF, dpi=600, bbox_inches="tight")
    plt.close(fig)

    print(f"[DONE] Saved {OUT_PNG}")
    print(f"[DONE] Saved {OUT_PDF}")

    print("\n[SUMMARY]")
    print(
        cent["clinical_timepoint_coarse"]
        .value_counts(dropna=False)
        .sort_index()
        .to_string()
    )


if __name__ == "__main__":
    main()
