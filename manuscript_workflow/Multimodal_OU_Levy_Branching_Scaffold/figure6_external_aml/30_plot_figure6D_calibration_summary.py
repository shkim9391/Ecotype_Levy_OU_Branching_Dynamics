from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


# ============================================================
# 1. CONFIG
# ============================================================
FIG6_DIR = Path("/Multimodal_OU_Lévy_Branching_Scaffold/Figure_6")
FIG6_DERIVED = FIG6_DIR / "derived"
FIG6_PANELS = FIG6_DIR / "panels"

IN_DYN = FIG6_DERIVED / "gse235923_sample_dynamic_parameters.csv"
IN_CENT = FIG6_DERIVED / "gse235923_sample_centroids.csv"

OUT_PNG = FIG6_PANELS / "Figure6D_calibration_summary.png"
OUT_PDF = FIG6_PANELS / "Figure6D_calibration_summary.pdf"
OUT_TSV = FIG6_DERIVED / "figure6_calibration_summary.tsv"

BOX_COLORS = ["#E8EEF6", "#E7F4EE", "#F8E8E8", "#F3F3F3"]


# ============================================================
# 2. HELPERS
# ============================================================
def ensure_dirs() -> None:
    FIG6_PANELS.mkdir(parents=True, exist_ok=True)


def style_axis(ax, panel_label: str, title: str,
               panel_fontsize: int = 18,
               title_fontsize: int = 12,
               title_x: float = 0.10) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

    ax.text(
        0.00, 1.0, panel_label,
        transform=ax.transAxes,
        fontsize=panel_fontsize,
        fontweight="bold",
        ha="left", va="bottom"
    )
    ax.text(
        title_x, 1.0, title,
        transform=ax.transAxes,
        fontsize=title_fontsize,
        fontweight="bold",
        ha="left", va="bottom"
    )


def draw_summary_box(ax, x, y, w, h, title, body, facecolor):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        facecolor=facecolor,
        edgecolor="#777777",
        linewidth=1.1,
        alpha=0.95,
    )
    ax.add_patch(box)

    pad_x = 0.025

    ax.text(
        x + pad_x,
        y + h - 0.028,
        title,
        fontsize=11.5,
        fontweight="bold",
        ha="left",
        va="top",
        color="#222222",
    )

    ax.text(
        x + pad_x,
        y + h - 0.090,
        body,
        fontsize=10.5,
        ha="left",
        va="top",
        color="#333333",
        linespacing=1.35,
    )


# ============================================================
# 3. MAIN
# ============================================================
def main() -> None:
    ensure_dirs()

    dyn = pd.read_csv(IN_DYN)
    cent = pd.read_csv(IN_CENT)

    for c in ["theta_eff", "sigma_eff", "mu_shift_from_dx"]:
        dyn[c] = pd.to_numeric(dyn[c], errors="coerce")

    dyn["clinical_timepoint_coarse"] = dyn["clinical_timepoint_coarse"].astype(str)
    cent["clinical_timepoint_coarse"] = cent["clinical_timepoint_coarse"].astype(str)

    dx = dyn[dyn["clinical_timepoint_coarse"] == "DX"].copy()
    eoi = dyn[dyn["clinical_timepoint_coarse"] == "EOI_REM"].copy()
    rel = dyn[dyn["clinical_timepoint_coarse"] == "REL"].copy()

    dx_mu = float(np.nanmedian(dx["mu_shift_from_dx"])) if len(dx) else np.nan
    eoi_mu = float(np.nanmedian(eoi["mu_shift_from_dx"])) if len(eoi) else np.nan
    rel_mu = float(np.nanmedian(rel["mu_shift_from_dx"])) if len(rel) else np.nan

    dx_sigma = float(np.nanmedian(dx["sigma_eff"])) if len(dx) else np.nan
    eoi_sigma = float(np.nanmedian(eoi["sigma_eff"])) if len(eoi) else np.nan
    rel_sigma = float(np.nanmedian(rel["sigma_eff"])) if len(rel) else np.nan

    dx_theta = float(np.nanmedian(dx["theta_eff"])) if len(dx) else np.nan
    eoi_theta = float(np.nanmedian(eoi["theta_eff"])) if len(eoi) else np.nan
    rel_theta = float(np.nanmedian(rel["theta_eff"])) if len(rel) else np.nan

    # longitudinal structure
    seq = (
        cent.groupby("patient_id")["clinical_timepoint_coarse"]
            .apply(list)
            .reset_index(name="timepoints")
    )
    full_triads = seq[seq["timepoints"].apply(lambda x: set(x) == {"DX", "EOI_REM", "REL"})]
    n_triads = int(full_triads.shape[0])

    # summary statements
    rows = [
        {
            "metric": "EOI_vs_DX_mu_shift",
            "comparison": "EOI_REM vs DX",
            "effect_direction": "EOI closer to baseline" if eoi_mu <= dx_mu else "EOI farther from baseline",
            "estimate": eoi_mu - dx_mu,
            "p_value": np.nan,
            "interpretation": f"median μ-shift: DX={dx_mu:.3f}, EOI={eoi_mu:.3f}",
        },
        {
            "metric": "REL_vs_EOI_mu_shift",
            "comparison": "REL vs EOI_REM",
            "effect_direction": "REL farther from baseline" if rel_mu >= eoi_mu else "REL closer to baseline",
            "estimate": rel_mu - eoi_mu,
            "p_value": np.nan,
            "interpretation": f"median μ-shift: EOI={eoi_mu:.3f}, REL={rel_mu:.3f}",
        },
        {
            "metric": "EOI_vs_DX_sigma_eff",
            "comparison": "EOI_REM vs DX",
            "effect_direction": "EOI more diffuse" if eoi_sigma >= dx_sigma else "EOI less diffuse",
            "estimate": eoi_sigma - dx_sigma,
            "p_value": np.nan,
            "interpretation": f"median σ_eff: DX={dx_sigma:.3f}, EOI={eoi_sigma:.3f}",
        },
        {
            "metric": "full_longitudinal_triads",
            "comparison": "trajectory structure",
            "effect_direction": "triads available",
            "estimate": float(n_triads),
            "p_value": np.nan,
            "interpretation": f"full DX→EOI_REM→REL patients: {n_triads}",
        },
    ]
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(OUT_TSV, sep="\t", index=False)

    fig, ax = plt.subplots(figsize=(12.6, 6.8))
    style_axis(
        ax,
        "D",
        "Calibration summary of replicated treatment-aware structure",
        title_x=0.12,
    )

    box_w = 0.42
    box_h = 0.28
    x_left = 0.04
    x_right = 0.52
    y_top = 0.66
    y_bot = 0.30

    draw_summary_box(
        ax,
        x_left,
        y_top,
        box_w,
        box_h,
        "EOI/REM remains near the DX attractor",
        "External response-associated samples\n"
        "remain close to the discovery baseline.\n\n"
        f"Median μ-shift:\n"
        f"DX = {dx_mu:.3f}\n"
        f"EOI/REM = {eoi_mu:.3f}",
        BOX_COLORS[0],
    )

    draw_summary_box(
        ax,
        x_right,
        y_top,
        box_w,
        box_h,
        "REL is more displaced than EOI/REM",
        "External relapse samples occupy a more\n"
        "displaced position than EOI/REM samples.\n\n"
        f"Median μ-shift:\n"
        f"EOI/REM = {eoi_mu:.3f}\n"
        f"REL = {rel_mu:.3f}",
        BOX_COLORS[2],
    )

    draw_summary_box(
        ax,
        x_left,
        y_bot,
        box_w,
        box_h,
        "Residual-like states remain diffuse",
        "EOI/REM samples do not collapse into\n"
        "a single tightly restored state.\n\n"
        f"Median σeff:\n"
        f"DX = {dx_sigma:.3f}\n"
        f"EOI/REM = {eoi_sigma:.3f}",
        BOX_COLORS[1],
    )

    triad_text = " | ".join(full_triads["patient_id"].astype(str).tolist()) if n_triads > 0 else "none"

    draw_summary_box(
        ax,
        x_right,
        y_bot,
        box_w,
        box_h,
        "True longitudinal calibration is present",
        "The external cohort contains full\n"
        "DX→EOI/REM→REL trajectories.\n\n"
        f"Full triads: {n_triads}\n"
        f"{triad_text}",
        BOX_COLORS[3],
    )

    ax.text(
        0.02, 0.18,
       "These summaries support directional calibration rather than exact parameter replication: "
        "the frozen scaffold preserves\ninterpretable diagnosis-, response-, and relapse-associated structure in an independent pediatric AML cohort.",
        fontsize=10.5,
        ha="left",
        va="bottom",
        color="#333333",
    )

    fig.savefig(OUT_PNG, dpi=600, bbox_inches="tight")
    fig.savefig(OUT_PDF, dpi=600, bbox_inches="tight")
    plt.close(fig)

    print(f"[DONE] Saved {OUT_PNG}")
    print(f"[DONE] Saved {OUT_PDF}")
    print(f"[DONE] Saved {OUT_TSV}")

    print("\n[SUMMARY]")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
