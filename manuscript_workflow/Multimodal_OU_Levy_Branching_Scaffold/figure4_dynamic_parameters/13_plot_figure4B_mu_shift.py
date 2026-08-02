from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, wilcoxon


# ============================================================
# 1. CONFIG
# ============================================================
PROJECT_DIR = Path("/Multimodal_OU_Lévy_Branching_Scaffold/Figure_4")
DERIVED_DIR = PROJECT_DIR / "derived"
PANELS_DIR = PROJECT_DIR / "panels"

IN_CSV = DERIVED_DIR / "sample_dynamic_parameters.csv"

OUT_PNG = PANELS_DIR / "Figure4B_mu_shift_from_dx.png"
OUT_PDF = PANELS_DIR / "Figure4B_mu_shift_from_dx.pdf"
OUT_TSV = DERIVED_DIR / "figure4B_mu_shift_stats.tsv"

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
    df["mu_shift_from_dx"] = pd.to_numeric(df["mu_shift_from_dx"], errors="coerce")

    # Main pooled comparison uses only main-analysis DX and REL samples
    main = df[df["is_main_analysis_sample"] == True].copy()
    pooled = main[main["clinical_timepoint_coarse"].isin(PHASE_ORDER)].copy()

    x_dx = pooled.loc[pooled["clinical_timepoint_coarse"] == "DX", "mu_shift_from_dx"].dropna().to_numpy()
    x_rel = pooled.loc[pooled["clinical_timepoint_coarse"] == "REL", "mu_shift_from_dx"].dropna().to_numpy()

    if len(x_dx) == 0 or len(x_rel) == 0:
        raise ValueError("Need both DX and REL main-analysis samples for Panel 4B.")

    # EOI points shown individually, not pooled
    eoi = df[df["clinical_timepoint_coarse"] == "EOI_REM"].copy()

    # Paired test on patients with both DX and REL
    pair_df = pooled[pooled["clinical_timepoint_coarse"].isin(["DX", "REL"])].copy()
    pair_df = pair_df.pivot_table(
        index="patient_id",
        columns="clinical_timepoint_coarse",
        values="mu_shift_from_dx",
        aggfunc="first"
    ).reset_index()

    pair_df = pair_df.dropna(subset=["DX", "REL"]).copy()

    if len(pair_df) >= 2:
        w_stat, w_p = wilcoxon(pair_df["DX"], pair_df["REL"], alternative="two-sided", zero_method="wilcox")
    else:
        w_stat, w_p = np.nan, np.nan

    u_stat, u_p = mannwhitneyu(x_dx, x_rel, alternative="two-sided")
    cd = cliffs_delta(x_rel, x_dx)

    stats = pd.DataFrame([{
        "comparison": "REL_vs_DX_main_analysis",
        "n_DX": len(x_dx),
        "n_REL": len(x_rel),
        "n_pairs": int(len(pair_df)),
        "median_DX": float(np.nanmedian(x_dx)),
        "median_REL": float(np.nanmedian(x_rel)),
        "iqr_DX": float(np.nanquantile(x_dx, 0.75) - np.nanquantile(x_dx, 0.25)),
        "iqr_REL": float(np.nanquantile(x_rel, 0.75) - np.nanquantile(x_rel, 0.25)),
        "mannwhitney_u": float(u_stat),
        "mannwhitney_p": float(u_p),
        "wilcoxon_stat": float(w_stat) if pd.notna(w_stat) else np.nan,
        "wilcoxon_p": float(w_p) if pd.notna(w_p) else np.nan,
        "cliffs_delta_REL_vs_DX": float(cd),
    }])
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

    # Paired DX→REL lines for patients with both values
    for _, r in pair_df.iterrows():
        ax.plot(
            [positions["DX"], positions["REL"]],
            [r["DX"], r["REL"]],
            color="#9E9E9E",
            linewidth=0.8,
            alpha=0.22,
            zorder=1,
        )

    # Jittered sample points
    for phase in PHASE_ORDER:
        sub = pooled[pooled["clinical_timepoint_coarse"] == phase].copy()
        x = rng.normal(loc=positions[phase], scale=0.055, size=len(sub))

        ax.scatter(
            x,
            sub["mu_shift_from_dx"],
            s=34,
            c=PHASE_COLORS[phase],
            alpha=0.82,
            edgecolors="white",
            linewidths=0.35,
            zorder=3,
        )

    # EOI/REM points shown individually
    if not eoi.empty:
        eoi_sorted = eoi.sort_values("mu_shift_from_dx").copy()

        for i, (_, r) in enumerate(eoi_sorted.iterrows()):
            patient = str(r["patient_id"])
            y = float(r["mu_shift_from_dx"])
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

    ax.set_ylabel(r"Attractor displacement from DX baseline ($\mu$-shift)")

    style_axis(
        ax,
        "B",
        "Attractor displacement from diagnosis baseline",
        title_x=0.12
    )

    # Paired p-value annotation
    y_candidates = [np.nanmax(x_dx), np.nanmax(x_rel)]
    if len(eoi):
        y_candidates.append(np.nanmax(eoi["mu_shift_from_dx"]))

    y_top = max(y_candidates)
    y_bar = y_top + 0.055

    ax.plot(
        [1, 1, 2, 2],
        [y_bar - 0.012, y_bar, y_bar, y_bar - 0.012],
        color="#444444",
        linewidth=1.1,
    )

    p_label = f"paired p = {w_p:.3g}" if pd.notna(w_p) else "paired p = NA"

    ax.text(
        1.5,
        y_bar + 0.012,
        p_label,
        ha="center",
        va="bottom",
        fontsize=9.5,
        color="#444444",
    )

    # Dynamic y-limit with headroom
    y_min_candidates = [np.nanmin(x_dx), np.nanmin(x_rel)]
    if len(eoi):
        y_min_candidates.append(np.nanmin(eoi["mu_shift_from_dx"]))

    y_min = min(y_min_candidates)
    ax.set_ylim(max(0, y_min - 0.06), y_bar + 0.08)

    # Subtle note
    ax.text(
        0.98,
        0.03,
        "Paired DX→REL lines shown; EOI/REM shown individually",
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
    print("Paired Wilcoxon p:", w_p)
    print("EOI_REM values:")
    if len(eoi):
        print(eoi[["patient_id", "sample_id", "mu_shift_from_dx"]].to_string(index=False))


if __name__ == "__main__":
    main()
