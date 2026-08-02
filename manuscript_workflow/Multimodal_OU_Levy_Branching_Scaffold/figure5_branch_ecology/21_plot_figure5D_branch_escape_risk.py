from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. CONFIG
# ============================================================
PROJECT_DIR = Path("/Multimodal_OU_Lévy_Branching_Scaffold/Figure_5")
DERIVED_DIR = PROJECT_DIR / "derived"
PANELS_DIR = PROJECT_DIR / "panels"

IN_RISK = DERIVED_DIR / "branch_escape_risk_summary.csv"
IN_TRANS = DERIVED_DIR / "branch_transition_table.csv"

OUT_PNG = PANELS_DIR / "Figure5D_branch_escape_risk.png"
OUT_PDF = PANELS_DIR / "Figure5D_branch_escape_risk.pdf"
OUT_TSV = DERIVED_DIR / "figure5D_escape_risk_stats.tsv"

BRANCH_COLORS = {
    "HSC-like basin": "#7A8DA8",
    "Progenitor-like basin": "#C97979",
    "GMP-like basin": "#D6B85A",
    "Mono/DC-like basin": "#7BB7B2",
}


# ============================================================
# 2. HELPERS
# ============================================================
def ensure_dirs() -> None:
    PANELS_DIR.mkdir(parents=True, exist_ok=True)


def assert_columns(df: pd.DataFrame, required: list[str], label: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{label} missing required columns: {missing}")


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


def clean_branch_label(x: str) -> str:
    return (
        x.replace(" basin", "")
         .replace("HSC-like", "HSC")
         .replace("Progenitor-like", "Progenitor")
         .replace("GMP-like", "GMP")
         .replace("Mono/DC-like", "Mono/DC")
    )


# ============================================================
# 3. MAIN
# ============================================================
def main() -> None:
    ensure_dirs()

    risk = pd.read_csv(IN_RISK)
    trans = pd.read_csv(IN_TRANS)

    assert_columns(
        risk,
        ["branch_id_start", "n_intervals", "switch_fraction", "mean_jump_score", "mean_displacement_hd"],
        "branch_escape_risk_summary.csv",
    )
    assert_columns(
        trans,
        ["branch_start", "jump_score", "branch_switch"],
        "branch_transition_table.csv",
    )

    risk["branch_id_start"] = risk["branch_id_start"].astype(str)
    trans["branch_start"] = trans["branch_start"].astype(str)
    trans["jump_score"] = pd.to_numeric(trans["jump_score"], errors="coerce")
    trans["branch_switch"] = pd.to_numeric(trans["branch_switch"], errors="coerce").fillna(0).astype(int)

    # order by mean jump score descending
    risk = risk.sort_values("mean_jump_score", ascending=False).reset_index(drop=True)
    branch_order = risk["branch_id_start"].tolist()

    # compact stats export
    risk.to_csv(OUT_TSV, sep="\t", index=False)

    # ---------------------------------------------------------
    # Plot
    # ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.8, 6.4))
    rng = np.random.default_rng(42)

    y_positions = np.arange(len(branch_order))[::-1]

    # x=0 reference
    ax.axvline(0, color="#C8C8C8", linewidth=1.0, zorder=0)

    # individual interval points + branch means
    for y, branch in zip(y_positions, branch_order):
        sub = trans[trans["branch_start"] == branch].copy()
        sub = sub[sub["jump_score"].notna()].copy()

        # jittered interval points
        if not sub.empty:
            yj = rng.normal(loc=y, scale=0.055, size=len(sub))
            colors = np.where(
                sub["branch_switch"].astype(int) == 1,
                "#D88A8A",   # branch-switching interval
                "#A8B6C8"    # branch-continuous interval
            )

            ax.scatter(
                sub["jump_score"],
                yj,
                s=30,
                c=colors,
                alpha=0.55,
                edgecolors="white",
                linewidths=0.25,
                zorder=2,
            )

        # branch-level mean point
        r = risk[risk["branch_id_start"] == branch].iloc[0]
        mean_jump = float(r["mean_jump_score"])
        color = BRANCH_COLORS.get(branch, "#999999")

        # horizontal stem from zero to branch mean
        ax.hlines(
            y,
            xmin=min(0, mean_jump),
            xmax=max(0, mean_jump),
            color="#B8B8B8",
            linewidth=1.7,
            zorder=1,
        )

        ax.scatter(
            [mean_jump],
            [y],
            s=130,
            c=color,
            edgecolors="#222222",
            linewidths=0.8,
            zorder=4,
        )

        # branch mean label
        ha = "left" if mean_jump >= 0 else "right"
        x_text = mean_jump + 0.08 if mean_jump >= 0 else mean_jump - 0.08

        ax.text(
            x_text,
            y,
            clean_branch_label(branch),
            fontsize=9.4,
            fontweight="bold",
            ha=ha,
            va="center",
            color="#222222",
            zorder=5,
        )

    # y labels with n
    ax.set_yticks(y_positions)
    ax.set_yticklabels([
        f"{clean_branch_label(b)}\n(n={int(risk.loc[risk['branch_id_start'] == b, 'n_intervals'].iloc[0])})"
        for b in branch_order
    ])

    # Limits after points are present
    xmin = min(-1.25, np.nanmin(trans["jump_score"]) - 0.20)
    xmax = max(
        np.nanmax(risk["mean_jump_score"]) + 0.80,
        np.nanmax(trans["jump_score"]) + 0.35
    )
    ax.set_xlim(xmin, xmax)

    # right-side switch fraction labels
    x_ann = xmax - 0.02 * (xmax - xmin)

    for y, branch in zip(y_positions, branch_order):
        r = risk[risk["branch_id_start"] == branch].iloc[0]

        ax.text(
            x_ann,
            y,
            f"switch frac. = {r['switch_fraction']:.2f}",
            fontsize=9.0,
            ha="right",
            va="center",
            color="#555555",
        )

    ax.set_xlabel("Mean jump-sensitive score by starting branch")

    style_axis(
        ax,
        "D",
        "Branch-level escape propensity",
        title_x=0.12
    )

    # Subtle legend-like annotation for interval points
    ax.scatter([], [], s=30, c="#A8B6C8", alpha=0.65, label="Branch-continuous interval")
    ax.scatter([], [], s=30, c="#D88A8A", alpha=0.65, label="Branch-switching interval")
    ax.scatter([], [], s=130, c="#999999", edgecolors="#222222", linewidths=0.8, label="Branch mean")

    ax.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.50, -0.13),
        ncol=3,
        fontsize=9.0,
        handletextpad=0.5,
        columnspacing=1.2,
        borderaxespad=0.0,
    )

    ax.set_axisbelow(True)
    ax.xaxis.grid(True, color="#E6E6E6", linewidth=0.7)
    ax.yaxis.grid(False)
    
    fig.subplots_adjust(bottom=0.22)
    
    fig.savefig(OUT_PNG, dpi=600, bbox_inches="tight")
    fig.savefig(OUT_PDF, dpi=600, bbox_inches="tight")
    plt.close(fig)

    print(f"[DONE] Saved {OUT_PNG}")
    print(f"[DONE] Saved {OUT_PDF}")
    print(f"[DONE] Saved {OUT_TSV}")

    print("\n[SUMMARY]")
    print(
        risk[["branch_id_start", "n_intervals", "switch_fraction", "mean_jump_score", "mean_displacement_hd"]]
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
