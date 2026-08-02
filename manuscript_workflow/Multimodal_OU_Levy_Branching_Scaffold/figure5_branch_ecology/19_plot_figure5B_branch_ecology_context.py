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

IN_CSV = DERIVED_DIR / "branch_ecology_summary.csv"

OUT_PNG = PANELS_DIR / "Figure5B_branch_ecology_context.png"
OUT_PDF = PANELS_DIR / "Figure5B_branch_ecology_context.pdf"
OUT_TSV = DERIVED_DIR / "figure5B_ecology_stats.tsv"

BRANCH_ORDER = [
    "HSC-like basin",
    "Progenitor-like basin",
    "GMP-like basin",
    "Mono/DC-like basin",
]

ECOTYPE_COLORS = {
    "E1_LateErythrocyte_CD8Memory": "#73C6B6",
    "E2_BCell_CD4Naive": "#F5A27A",
    "E3_NK_CD8Memory": "#6BB36B",
    "E4_CD4Naive_CD8Naive": "#E36A74",
    "Unknown": "#BDBDBD",
}

ECOTYPE_DISPLAY = {
    "E1_LateErythrocyte_CD8Memory": "E1 LateEry/CD8Mem",
    "E2_BCell_CD4Naive": "E2 B-cell/CD4Naive",
    "E3_NK_CD8Memory": "E3 NK/CD8Mem",
    "E4_CD4Naive_CD8Naive": "E4 CD4Naive/CD8Naive",
    "Unknown": "Unknown",
}

# ============================================================
# 2. HELPERS
# ============================================================
def ensure_dirs() -> None:
    PANELS_DIR.mkdir(parents=True, exist_ok=True)


def assert_columns(df: pd.DataFrame, required: list[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"branch_ecology_summary.csv missing required columns: {missing}")


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
         .replace("HSC-like", "HSC-like")
         .replace("Progenitor-like", "Progenitor-like")
         .replace("GMP-like", "GMP-like")
         .replace("Mono/DC-like", "Mono/DC-like")
    )

# ============================================================
# 3. MAIN
# ============================================================
def main() -> None:
    ensure_dirs()

    df = pd.read_csv(IN_CSV)
    assert_columns(
        df,
        [
            "branch_id_dominant",
            "ecotype_label",
            "n_samples",
            "fraction_within_branch",
            "branch_total_samples",
        ],
    )

    df["branch_id_dominant"] = df["branch_id_dominant"].astype(str)
    df["ecotype_label"] = df["ecotype_label"].astype(str)

    # Keep deterministic branch order and ecotype order by overall abundance
    branch_present = [b for b in BRANCH_ORDER if b in set(df["branch_id_dominant"])]
    eco_order = (
        df.groupby("ecotype_label")["n_samples"]
          .sum()
          .sort_values(ascending=False)
          .index
          .tolist()
    )

    pivot = (
        df.pivot_table(
            index="branch_id_dominant",
            columns="ecotype_label",
            values="fraction_within_branch",
            aggfunc="sum",
            fill_value=0.0,
        )
        .reindex(branch_present)
    )

    # stats table for manuscript reference
    top_eco = (
        df.sort_values(["branch_id_dominant", "fraction_within_branch"], ascending=[True, False])
          .groupby("branch_id_dominant")
          .head(1)
          .copy()
    )
    top_eco = top_eco.rename(columns={
        "branch_id_dominant": "branch",
        "ecotype_label": "dominant_ecotype",
        "fraction_within_branch": "dominant_fraction",
    })
    top_eco.to_csv(OUT_TSV, sep="\t", index=False)

    # ---------------------------------------------------------
    # Plot
    # ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.2, 6.6))

    x = np.arange(len(branch_present))
    bottom = np.zeros(len(branch_present), dtype=float)

    for eco in eco_order:
        vals = pivot[eco].to_numpy(dtype=float) if eco in pivot.columns else np.zeros(len(branch_present))
        color = ECOTYPE_COLORS.get(eco, "#CCCCCC")

        ax.bar(
            x,
            vals,
            bottom=bottom,
            width=0.72,
            color=color,
            edgecolor="white",
            linewidth=0.7,
            label=ECOTYPE_DISPLAY.get(eco, eco),
        )
        bottom += vals

    # Branch totals for x labels
    branch_n = (
        df[["branch_id_dominant", "branch_total_samples"]]
        .drop_duplicates()
        .set_index("branch_id_dominant")["branch_total_samples"]
        .to_dict()
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [
            f"{clean_branch_label(b)}\n(n={int(branch_n[b])})"
            for b in branch_present
        ],
        rotation=0,
        ha="center",
    )

    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Ecotype fraction within branch")

    style_axis(
        ax,
        "B",
        "Branch-specific ecological context",
        title_x=0.12,
    )

    leg = ax.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.50, -0.16),
        ncol=4,
        fontsize=8.9,
        title="Ecotype",
        title_fontsize=9.3,
        handlelength=1.2,
        handletextpad=0.4,
        columnspacing=0.9,
        borderaxespad=0.0,
    )

    # Light horizontal guides for fractions
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color="#E6E6E6", linewidth=0.7)
    ax.xaxis.grid(False)
    
    fig.subplots_adjust(bottom=0.28)

    fig.savefig(OUT_PNG, dpi=600, bbox_inches="tight")
    fig.savefig(OUT_PDF, dpi=600, bbox_inches="tight")
    plt.close(fig)

    print(f"[DONE] Saved {OUT_PNG}")
    print(f"[DONE] Saved {OUT_PDF}")
    print(f"[DONE] Saved {OUT_TSV}")

    print("\n[SUMMARY]")
    print(top_eco[["branch", "dominant_ecotype", "dominant_fraction"]].to_string(index=False))


if __name__ == "__main__":
    main()
