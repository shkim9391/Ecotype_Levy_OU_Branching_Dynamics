from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu


# ============================================================
# 1. CONFIG
# ============================================================
PROJECT_DIR = Path("/Multimodal_OU_Lévy_Branching_Scaffold/Figure_4")
DERIVED_DIR = PROJECT_DIR / "derived"
PANELS_DIR = PROJECT_DIR / "panels"

IN_CSV = DERIVED_DIR / "sample_dynamic_parameters.csv"

OUT_PNG = PANELS_DIR / "Figure4A_theta_by_phase.png"
OUT_PDF = PANELS_DIR / "Figure4A_theta_by_phase.pdf"
OUT_TSV = DERIVED_DIR / "figure4A_theta_stats.tsv"

PHASE_ORDER = ["DX", "REL"]
PHASE_COLORS = {
    "DX": "#7A8DA8",
    "REL": "#C97979",
    "EOI_REM": "#6BAF92",
}

EXPLORATORY_PATIENTS = {"AML1"}
EOI_LABELS = {
    "AML21": "AML21 REM",
    "AML1": "AML1 REM\n(exploratory)",
}


# ============================================================
# 2. HELPERS
# ============================================================
def ensure_dirs() -> None:
    PANELS_DIR.mkdir(parents=True, exist_ok=True)


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    gt = 0
    lt = 0
    for xi in x:
        gt += np.sum(xi > y)
        lt += np.sum(xi < y)
    n = len(x) * len(y)
    return (gt - lt) / n if n > 0 else np.nan


def style_axis(ax, panel_label: str, title: str,
               panel_fontsize: int = 18,
               title_fontsize: int = 12,
               title_x: float = 0.10) -> None:
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)

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


# ============================================================
# 3. MAIN
# ============================================================
def main() -> None:
    ensure_dirs()

    df = pd.read_csv(IN_CSV)
    df["theta_eff"] = pd.to_numeric(df["theta_eff"], errors="coerce")

    # Main pooled comparison uses only main-analysis DX and REL samples
    main = df[df["is_main_analysis_sample"] == True].copy()
    pooled = main[main["clinical_timepoint_coarse"].isin(PHASE_ORDER)].copy()

    x_dx = pooled.loc[pooled["clinical_timepoint_coarse"] == "DX", "theta_eff"].dropna().to_numpy()
    x_rel = pooled.loc[pooled["clinical_timepoint_coarse"] == "REL", "theta_eff"].dropna().to_numpy()

    if len(x_dx) == 0 or len(x_rel) == 0:
        raise ValueError("Need both DX and REL main-analysis samples for Panel 4A.")

    # EOI points shown individually, not pooled
    eoi = df[df["clinical_timepoint_coarse"] == "EOI_REM"].copy()

    # Stats
    u, p = mannwhitneyu(x_dx, x_rel, alternative="two-sided")
    cd = cliffs_delta(x_rel, x_dx)

    stats = pd.DataFrame([
        {
            "comparison": "REL_vs_DX_main_analysis",
            "n_DX": len(x_dx),
            "n_REL": len(x_rel),
            "median_DX": float(np.nanmedian(x_dx)),
            "median_REL": float(np.nanmedian(x_rel)),
            "iqr_DX": float(np.nanquantile(x_dx, 0.75) - np.nanquantile(x_dx, 0.25)),
            "iqr_REL": float(np.nanquantile(x_rel, 0.75) - np.nanquantile(x_rel, 0.25)),
            "mannwhitney_u": float(u),
            "mannwhitney_p": float(p),
            "cliffs_delta_REL_vs_DX": float(cd),
        }
    ])
    stats.to_csv(OUT_TSV, sep="\t", index=False)

    # ---------------------------------------------------------
    # Plot
    # ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.4, 6.6))
    rng = np.random.default_rng(42)

    positions = {"DX": 1, "REL": 2}
    eoi_x = 3.05

    # Violin
    violin_data = [x_dx, x_rel]
    vp = ax.violinplot(
        violin_data,
        positions=[positions["DX"], positions["REL"]],
        widths=0.82,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )

    for body, phase in zip(vp["bodies"], PHASE_ORDER):
        body.set_facecolor(PHASE_COLORS[phase])
        body.set_edgecolor("none")
        body.set_alpha(0.20)

    # Box
    bp = ax.boxplot(
        violin_data,
        positions=[positions["DX"], positions["REL"]],
        widths=0.28,
        patch_artist=True,
        showfliers=False,
    )

    for patch, phase in zip(bp["boxes"], PHASE_ORDER):
        patch.set_facecolor(PHASE_COLORS[phase])
        patch.set_alpha(0.48)
        patch.set_edgecolor("#333333")
        patch.set_linewidth(1.0)

    for key in ["whiskers", "caps", "medians"]:
        for line in bp[key]:
            line.set_color("#333333")
            line.set_linewidth(1.0)

    # Jittered sample points
    for phase in PHASE_ORDER:
        sub = pooled[pooled["clinical_timepoint_coarse"] == phase].copy()
        x = rng.normal(loc=positions[phase], scale=0.055, size=len(sub))

        ax.scatter(
            x,
            sub["theta_eff"],
            s=34,
            c=PHASE_COLORS[phase],
            alpha=0.82,
            edgecolors="white",
            linewidths=0.35,
            zorder=3,
        )

    # EOI/REM points shown individually, not pooled into the DX/REL comparison
    if not eoi.empty:
        eoi_sorted = eoi.sort_values("theta_eff").copy()

        for i, (_, r) in enumerate(eoi_sorted.iterrows()):
            patient = str(r["patient_id"])
            y = float(r["theta_eff"])
            is_exploratory = patient in EXPLORATORY_PATIENTS

            # Slight horizontal staggering if multiple EOI/REM points are present
            x_point = eoi_x + (i - (len(eoi_sorted) - 1) / 2) * 0.035

            ax.scatter(
                [x_point],
                [y],
                s=86 if not is_exploratory else 76,
                c=PHASE_COLORS["EOI_REM"] if not is_exploratory else "white",
                edgecolors="#2F2F2F",
                linewidths=1.0,
                marker="D" if not is_exploratory else "o",
                zorder=4,
            )

            label = EOI_LABELS.get(patient, f"{patient} REM")
            ax.text(
                x_point + 0.10,
                y,
                label,
                fontsize=9.0,
                fontweight="bold" if not is_exploratory else "normal",
                ha="left",
                va="center",
                color="#2F2F2F",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.0),
                zorder=5,
            )

    # Axes / labels
    ax.set_xlim(0.55, 3.65)
    ax.set_xticks([1, 2, eoi_x])
    ax.set_xticklabels([
        f"DX\n(n={len(x_dx)})",
        f"REL\n(n={len(x_rel)})",
        f"EOI/REM\n(exploratory n={len(eoi)})",
    ])

    ax.set_ylabel(r"Effective restoring strength ($\theta_{\mathrm{eff}}$)")
    style_axis(
        ax,
        "A",
        r"Effective restoring strength by phase",
        title_x=0.12
    )

    # p-value annotation for the main DX vs REL comparison
    y_candidates = [np.nanmax(x_dx), np.nanmax(x_rel)]
    if len(eoi):
        y_candidates.append(np.nanmax(eoi["theta_eff"]))

    y_top = max(y_candidates)
    y_bar = y_top + 0.055

    ax.plot(
        [1, 1, 2, 2],
        [y_bar - 0.012, y_bar, y_bar, y_bar - 0.012],
        color="#444444",
        linewidth=1.1,
    )

    ax.text(
        1.5,
        y_bar + 0.012,
        f"DX vs REL, p = {p:.3g}",
        ha="center",
        va="bottom",
        fontsize=9.5,
        color="#444444",
    )

    # Dynamic y-limit with enough headroom for p-value bracket
    y_min = np.nanmin([np.nanmin(x_dx), np.nanmin(x_rel), np.nanmin(eoi["theta_eff"]) if len(eoi) else np.nanmin(x_dx)])
    ax.set_ylim(max(0, y_min - 0.08), y_bar + 0.08)

    # Optional subtle note inside the plotting area
    ax.text(
        0.98,
        0.03,
        "EOI/REM shown individually",
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
    print(f"[DONE] Saved {OUT_TSV}")

    print("\n[SUMMARY]")
    print("DX median:", float(np.nanmedian(x_dx)))
    print("REL median:", float(np.nanmedian(x_rel)))
    print("EOI_REM values:")
    if len(eoi):
        print(eoi[["patient_id", "sample_id", "theta_eff"]].to_string(index=False))


if __name__ == "__main__":
    main()
