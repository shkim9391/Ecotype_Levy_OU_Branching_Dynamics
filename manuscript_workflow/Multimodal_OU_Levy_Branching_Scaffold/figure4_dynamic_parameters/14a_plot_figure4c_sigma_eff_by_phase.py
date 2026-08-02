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

IN_SAMPLE = DERIVED_DIR / "sample_dynamic_parameters.csv"
IN_JUMP = Path("/Multimodal_OU_Lévy_Branching_Scaffold/Figure_3/derived/relapse_jump_candidates.csv")

OUT_PNG = PANELS_DIR / "Figure4C_sigma_and_lambda_proxy.png"
OUT_PDF = PANELS_DIR / "Figure4C_sigma_and_lambda_proxy.pdf"
OUT_TSV = DERIVED_DIR / "figure4C_sigma_lambda_stats.tsv"

PHASE_ORDER = ["DX", "REL"]
PHASE_COLORS = {
    "DX": "#7A8DA8",
    "REL": "#C97979",
    "EOI_REM": "#6BAF92",
}

JUMP_ORDER = ["Branch-continuous", "Branch-switching"]
JUMP_COLORS = {
    "Branch-continuous": "#7A8DA8",
    "Branch-switching": "#C97979",
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


def add_violin_box_jitter(ax, groups, order, colors, ycol, ylabel=None):
    rng = np.random.default_rng(42)

    arrays = [
        pd.to_numeric(groups.loc[groups["_group"] == g, ycol], errors="coerce").dropna().to_numpy()
        for g in order
    ]

    vp = ax.violinplot(
        arrays,
        positions=np.arange(1, len(order) + 1),
        widths=0.85,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )
    for body, g in zip(vp["bodies"], order):
        body.set_facecolor(colors[g])
        body.set_edgecolor("none")
        body.set_alpha(0.18)

    bp = ax.boxplot(
        arrays,
        positions=np.arange(1, len(order) + 1),
        widths=0.28,
        patch_artist=True,
        showfliers=False,
    )
    for patch, g in zip(bp["boxes"], order):
        patch.set_facecolor(colors[g])
        patch.set_alpha(0.45)
        patch.set_edgecolor("#333333")
    for key in ["whiskers", "caps", "medians"]:
        for line in bp[key]:
            line.set_color("#333333")

    for i, g in enumerate(order, start=1):
        sub = groups[groups["_group"] == g].copy()
        x = rng.normal(loc=i, scale=0.06, size=len(sub))
        ax.scatter(
            x,
            sub[ycol],
            s=34,
            c=colors[g],
            alpha=0.82,
            linewidths=0,
            zorder=3,
        )

    ax.set_xticks(np.arange(1, len(order) + 1))
    ax.set_xticklabels([
        f"{g}\n(n={int((groups['_group'] == g).sum())})" for g in order
    ])

    if ylabel is not None:
        ax.set_ylabel(ylabel)


# ============================================================
# 3. MAIN
# ============================================================
def main() -> None:
    ensure_dirs()

    # -----------------------------
    # Left side: sigma_eff by phase
    # -----------------------------
    samp = pd.read_csv(IN_SAMPLE)
    samp["sigma_eff"] = pd.to_numeric(samp["sigma_eff"], errors="coerce")

    main = samp[samp["is_main_analysis_sample"] == True].copy()
    sigma_df = main[main["clinical_timepoint_coarse"].isin(PHASE_ORDER)].copy()
    sigma_df["_group"] = sigma_df["clinical_timepoint_coarse"].astype(str)

    x_dx = sigma_df.loc[sigma_df["_group"] == "DX", "sigma_eff"].dropna().to_numpy()
    x_rel = sigma_df.loc[sigma_df["_group"] == "REL", "sigma_eff"].dropna().to_numpy()

    eoi = samp[samp["clinical_timepoint_coarse"] == "EOI_REM"].copy()

    u_sigma, p_sigma = mannwhitneyu(x_dx, x_rel, alternative="two-sided")
    cd_sigma = cliffs_delta(x_rel, x_dx)

    # -----------------------------
    # Right side: jump_score by branch class
    # -----------------------------
    jump = pd.read_csv(IN_JUMP)
    jump["jump_score"] = pd.to_numeric(jump["jump_score"], errors="coerce")
    jump["branch_switch"] = pd.to_numeric(jump["branch_switch"], errors="coerce").fillna(0).astype(int)
    jump["_group"] = np.where(
        jump["branch_switch"] == 1,
        "Branch-switching",
        "Branch-continuous"
    )

    y_cont = jump.loc[jump["_group"] == "Branch-continuous", "jump_score"].dropna().to_numpy()
    y_switch = jump.loc[jump["_group"] == "Branch-switching", "jump_score"].dropna().to_numpy()

    u_jump, p_jump = mannwhitneyu(y_cont, y_switch, alternative="two-sided")
    cd_jump = cliffs_delta(y_switch, y_cont)

    # -----------------------------
    # Stats export
    # -----------------------------
    stats = pd.DataFrame([
        {
            "panel": "sigma_eff",
            "comparison": "REL_vs_DX_main_analysis",
            "n_group1": len(x_dx),
            "n_group2": len(x_rel),
            "group1": "DX",
            "group2": "REL",
            "median_group1": float(np.nanmedian(x_dx)),
            "median_group2": float(np.nanmedian(x_rel)),
            "mannwhitney_u": float(u_sigma),
            "mannwhitney_p": float(p_sigma),
            "cliffs_delta_group2_vs_group1": float(cd_sigma),
        },
        {
            "panel": "jump_score",
            "comparison": "Branch-switching_vs_Branch-continuous",
            "n_group1": len(y_cont),
            "n_group2": len(y_switch),
            "group1": "Branch-continuous",
            "group2": "Branch-switching",
            "median_group1": float(np.nanmedian(y_cont)),
            "median_group2": float(np.nanmedian(y_switch)),
            "mannwhitney_u": float(u_jump),
            "mannwhitney_p": float(p_jump),
            "cliffs_delta_group2_vs_group1": float(cd_jump),
        },
    ])
    stats.to_csv(OUT_TSV, sep="\t", index=False)

    # -----------------------------
    # Plot
    # -----------------------------
    fig, (ax1, ax2) = plt.subplots(
    1, 2,
    figsize=(13.2, 6.6),
    gridspec_kw={"wspace": 0.22}
    )

    # Left subpanel
    add_violin_box_jitter(
        ax1,
        sigma_df,
        PHASE_ORDER,
        PHASE_COLORS,
        "sigma_eff",
        ylabel=r"Effective instability / diffusion proxy ($\sigma_{\mathrm{eff}}$)"
    )

    # EOI_REM points shown explicitly
    x_eoi = 2.88  # keep slightly inside the left panel
    
    label_offsets = {
        "AML21": (-0.14,  0.00, "right"),
        "AML1":  (-0.12, -0.02, "right"),
    }
    
    if not eoi.empty:
        for _, r in eoi.iterrows():
            patient = str(r["patient_id"])
            y = float(r["sigma_eff"])
            is_exploratory = patient in EXPLORATORY_PATIENTS
    
            ax1.scatter(
                [x_eoi],
                [y],
                s=80 if not is_exploratory else 72,
                c=PHASE_COLORS["EOI_REM"] if not is_exploratory else "white",
                edgecolors="#2F2F2F",
                linewidths=1.0,
                marker="D" if not is_exploratory else "o",
                zorder=4,
            )
    
            label = EOI_LABELS.get(patient, f"{patient} REM")
            dx, dy, ha = label_offsets.get(patient, (-0.12, 0.0, "right"))
    
            ax1.text(
                x_eoi + dx,
                y + dy,
                label,
                fontsize=8.3,
                fontweight="bold" if not is_exploratory else "normal",
                ha=ha,
                va="center",
                color="#2F2F2F",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.0),
                zorder=5,
            )

    ax1.set_xticks([1, 2, 2.88])
    ax1.set_xticklabels([
        f"DX\n(n={len(x_dx)})",
        f"REL\n(n={len(x_rel)})",
        f"EOI_REM\n(n={len(eoi)})",
    ])

    y_top1 = max(np.nanmax(x_dx), np.nanmax(x_rel), np.nanmax(eoi["sigma_eff"]) if len(eoi) else 0)
    y_bar1 = y_top1 + 0.05
    ax1.plot([1, 1, 2, 2], [y_bar1 - 0.01, y_bar1, y_bar1, y_bar1 - 0.01], color="#444444", linewidth=1.2)
    ax1.text(
        1.5, y_bar1 + 0.01,
        f"DX vs REL  p={p_sigma:.3g}",
        ha="center", va="bottom",
        fontsize=9.0, color="#444444"
    )

    style_axis(ax1, "C", r"Effective instability and jump propensity", title_x=0.12)

    # Right subpanel
    add_violin_box_jitter(
        ax2,
        jump,
        JUMP_ORDER,
        JUMP_COLORS,
        "jump_score",
        ylabel="Jump score"
    )

    y_top2 = max(np.nanmax(y_cont), np.nanmax(y_switch))
    y_bar2 = y_top2 + 0.12
    ax2.plot([1, 1, 2, 2], [y_bar2 - 0.03, y_bar2, y_bar2, y_bar2 - 0.03], color="#444444", linewidth=1.2)
    ax2.text(
        1.5, y_bar2 + 0.03,
        f"continuous vs switching  p={p_jump:.3g}",
        ha="center", va="bottom",
        fontsize=9.5, color="#444444"
    )

    # No extra panel letter here; keep right side as continuation of panel C
    for s in ["top", "right"]:
        ax2.spines[s].set_visible(False)
    ax2.set_title("")

    fig.savefig(OUT_PNG, dpi=600, bbox_inches="tight")
    fig.savefig(OUT_PDF, dpi=600, bbox_inches="tight")
    plt.close(fig)

    print(f"[DONE] Saved {OUT_PNG}")
    print(f"[DONE] Saved {OUT_PDF}")
    print(f"[DONE] Saved {OUT_TSV}")

    print("\n[SUMMARY]")
    print("sigma_eff DX median:", float(np.nanmedian(x_dx)))
    print("sigma_eff REL median:", float(np.nanmedian(x_rel)))
    print("jump_score continuous median:", float(np.nanmedian(y_cont)))
    print("jump_score switching median:", float(np.nanmedian(y_switch)))
    print("EOI_REM sigma_eff values:")
    if len(eoi):
        print(eoi[["patient_id", "sample_id", "sigma_eff"]].to_string(index=False))


if __name__ == "__main__":
    main()
