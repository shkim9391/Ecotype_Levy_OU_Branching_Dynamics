from __future__ import annotations

from pathlib import Path
from scipy.stats import gaussian_kde
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt


# ============================================================
# 1. CONFIG
# ============================================================
FIG4_DERIVED = Path("/Multimodal_OU_Lévy_Branching_Scaffold/Figure_4/derived")
FIG6_DIR = Path("/Multimodal_OU_Lévy_Branching_Scaffold/Figure_6")
FIG6_INPUTS = FIG6_DIR / "inputs"
FIG6_DERIVED = FIG6_DIR / "derived"
FIG6_PANELS = FIG6_DIR / "panels"

IN_H5AD = FIG6_INPUTS / "gse235923_longitudinal_malignant_projected.h5ad"
IN_CENT = FIG6_DERIVED / "gse235923_sample_centroids.csv"
IN_DISC = FIG4_DERIVED / "sample_dynamic_parameters.csv"

OUT_PNG = FIG6_PANELS / "Figure6A_external_projection.png"
OUT_PDF = FIG6_PANELS / "Figure6A_external_projection.pdf"

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
    line_color="#D2D2D2",
    alpha=0.45,
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


# ============================================================
# 3. MAIN
# ============================================================
def main() -> None:
    ensure_dirs()

    adata = sc.read_h5ad(IN_H5AD)
    cent = pd.read_csv(IN_CENT)
    disc = pd.read_csv(IN_DISC)

    if "X_fig2" not in adata.obsm:
        raise ValueError("Projected external object missing obsm['X_fig2'].")

    xy = np.asarray(adata.obsm["X_fig2"])
    obs = adata.obs.copy()
    obs["x2d"] = xy[:, 0]
    obs["y2d"] = xy[:, 1]

    # discovery reference sample centroids in the same frozen PC1/PC2 space
    if not {"PC1", "PC2", "clinical_timepoint_coarse"}.issubset(disc.columns):
        raise ValueError("Discovery sample_dynamic_parameters.csv must contain PC1, PC2, clinical_timepoint_coarse.")
    disc_ref = disc.copy()

    fig, ax = plt.subplots(figsize=(8.6, 7.2))

    # ---------------------------------------------------------
    # Discovery reference field and anchors
    # ---------------------------------------------------------
    ax.set_facecolor("#FBFBFB")
    
    draw_reference_field(
        ax,
        disc_ref["PC1"],
        disc_ref["PC2"],
    )
    
    if "n_cells" in disc_ref.columns:
        ref_sizes = np.clip(np.sqrt(disc_ref["n_cells"]) * 0.45, 10, 38)
    else:
        ref_sizes = 18
    
    ax.scatter(
        disc_ref["PC1"],
        disc_ref["PC2"],
        s=ref_sizes,
        c="#CFCFCF",
        alpha=0.16,
        linewidths=0,
        zorder=1,
    )

    # ---------------------------------------------------------
    # External projected malignant cells
    # These are sample-level projected coordinates repeated across cells,
    # giving a density-weighted projection view.
    # ---------------------------------------------------------
    for tp in ["DX", "EOI_REM", "REL"]:
        sub = obs[obs["clinical_timepoint_coarse"].astype(str) == tp].copy()
        if sub.empty:
            continue

        ax.scatter(
            sub["x2d"],
            sub["y2d"],
            s=2.0,
            c=TIME_COLORS[tp],
            alpha=0.07,
            linewidths=0,
            rasterized=True,
            zorder=2,
        )

    # ---------------------------------------------------------
    # External sample centroids overlaid
    # ---------------------------------------------------------
    cent = cent.copy()
    cent["clinical_timepoint_coarse"] = cent["clinical_timepoint_coarse"].astype(str)

    for tp in ["DX", "EOI_REM", "REL"]:
        sub = cent[cent["clinical_timepoint_coarse"] == tp].copy()
        if sub.empty:
            continue

        ax.scatter(
            sub["x2d"],
            sub["y2d"],
            s=np.clip(np.sqrt(sub["n_cells"]) * 0.55, 24, 70),
            c=TIME_COLORS[tp],
            edgecolors="white",
            linewidths=0.8,
            alpha=0.95,
            zorder=4,
        )

    # ---------------------------------------------------------
    # Highlight the two full longitudinal triads
    # ---------------------------------------------------------
    triads = {"Sample5", "Sample6"}
    tri = cent[cent["patient_id"].astype(str).isin(triads)].copy()

    for patient_id, sub in tri.groupby("patient_id", sort=False):
        sub = sub.sort_values("time_order")
        pts = sub[["x2d", "y2d"]].to_numpy(dtype=float)

        if len(sub) >= 2:
            # Background connecting line
            ax.plot(
                pts[:, 0],
                pts[:, 1],
                color="#2F2F2F",
                linewidth=1.2,
                alpha=0.65,
                zorder=5,
            )

            # Direction arrows between consecutive timepoints
            for i in range(len(pts) - 1):
                ax.annotate(
                    "",
                    xy=(pts[i + 1, 0], pts[i + 1, 1]),
                    xytext=(pts[i, 0], pts[i, 1]),
                    arrowprops=dict(
                        arrowstyle="-|>",
                        color="#2F2F2F",
                        lw=1.1,
                        alpha=0.75,
                        shrinkA=3,
                        shrinkB=3,
                    ),
                    zorder=6,
                )

        if len(sub) > 0:
            rr = sub.iloc[-1]
            ax.text(
                rr["x2d"] + 0.04,
                rr["y2d"] + 0.03,
                str(patient_id),
                fontsize=8.5,
                fontweight="bold",
                ha="left",
                va="center",
                color="#2F2F2F",
                zorder=7,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=1.0),
            )

    # ---------------------------------------------------------
    # Limits and styling
    # ---------------------------------------------------------
    all_x = np.concatenate([obs["x2d"].to_numpy(dtype=float), disc_ref["PC1"].to_numpy(dtype=float)])
    all_y = np.concatenate([obs["y2d"].to_numpy(dtype=float), disc_ref["PC2"].to_numpy(dtype=float)])

    pad_x = 0.14 * (all_x.max() - all_x.min())
    pad_y = 0.14 * (all_y.max() - all_y.min())
    ax.set_xlim(all_x.min() - pad_x, all_x.max() + pad_x)
    ax.set_ylim(all_y.min() - pad_y, all_y.max() + pad_y)

    style_axis(ax, "A", "Independent AML cohort projected into the frozen scaffold")
    add_phase_legend(ax)

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
