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

IN_CSV = DERIVED_DIR / "branch_scaffold_program_summary.csv"

OUT_PNG = PANELS_DIR / "Figure5C_branch_scaffold_programs.png"
OUT_PDF = PANELS_DIR / "Figure5C_branch_scaffold_programs.pdf"
OUT_TSV = DERIVED_DIR / "figure5C_program_stats.tsv"

BRANCH_ORDER = [
    "HSC-like basin",
    "Progenitor-like basin",
    "GMP-like basin",
    "Mono/DC-like basin",
]

PROGRAM_COLS = [
    "state_HSC_z",
    "state_Prog_z",
    "state_GMP_z",
    "state_MonoDC_z",
    "aux_EryBaso_z",
    "aux_CLP_z",
]

PROGRAM_LABELS = {
    "state_HSC_z": "HSC",
    "state_Prog_z": "Prog",
    "state_GMP_z": "GMP",
    "state_MonoDC_z": "Mono/DC",
    "aux_EryBaso_z": "Ery/Baso",
    "aux_CLP_z": "CLP",
}


# ============================================================
# 2. HELPERS
# ============================================================
def ensure_dirs() -> None:
    PANELS_DIR.mkdir(parents=True, exist_ok=True)


def assert_columns(df: pd.DataFrame, required: list[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"branch_scaffold_program_summary.csv missing required columns: {missing}")


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

    df = pd.read_csv(IN_CSV)
    assert_columns(df, ["branch_id_dominant", "n_samples"] + PROGRAM_COLS)

    df["branch_id_dominant"] = df["branch_id_dominant"].astype(str)

    # Keep only branches present, in preferred order
    present = [b for b in BRANCH_ORDER if b in set(df["branch_id_dominant"])]
    sub = (
        df.set_index("branch_id_dominant")
          .reindex(present)
          .reset_index()
    )

    # Matrix for heatmap
    mat = sub[PROGRAM_COLS].to_numpy(dtype=float)

    # export compact stats table
    stats = sub[["branch_id_dominant", "n_samples"] + PROGRAM_COLS].copy()
    stats.to_csv(OUT_TSV, sep="\t", index=False)

    # ---------------------------------------------------------
    # Plot
    # ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.6, 5.4))

    vmax = np.nanmax(np.abs(mat))
    if not np.isfinite(vmax) or vmax == 0:
        vmax = 1.0

    im = ax.imshow(
        mat,
        cmap="coolwarm",
        vmin=-vmax,
        vmax=vmax,
        aspect="auto",
        interpolation="nearest"
    )

    # Axis labels
    ax.set_xticks(np.arange(len(PROGRAM_COLS)))
    ax.set_xticklabels(
        [PROGRAM_LABELS[c] for c in PROGRAM_COLS],
        rotation=0,
        fontsize=10
    )

    ax.set_yticks(np.arange(len(present)))
    ax.set_yticklabels(
        [
            f"{clean_branch_label(b)}-like\n(n={int(sub.loc[i, 'n_samples'])})"
            if clean_branch_label(b) != "Mono/DC"
            else f"Mono/DC-like\n(n={int(sub.loc[i, 'n_samples'])})"
            for i, b in enumerate(present)
        ],
        fontsize=10
    )

    # Thin white grid lines between cells
    ax.set_xticks(np.arange(-0.5, len(PROGRAM_COLS), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(present), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)

    # Cell annotations
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            txt_color = "white" if abs(val) > 0.55 * vmax else "#222222"

            ax.text(
                j,
                i,
                f"{val:.2f}",
                ha="center",
                va="center",
                fontsize=8.8,
                color=txt_color
            )

    style_axis(
        ax,
        "C",
        "Branch-specific scaffold program composition",
        title_x=0.12
    )

    # Cleaner colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.035)
    cbar.set_label("Branch-level z-scored mean", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    fig.savefig(OUT_PNG, dpi=600, bbox_inches="tight")
    fig.savefig(OUT_PDF, dpi=600, bbox_inches="tight")
    plt.close(fig)

    print(f"[DONE] Saved {OUT_PNG}")
    print(f"[DONE] Saved {OUT_PDF}")
    print(f"[DONE] Saved {OUT_TSV}")

    print("\n[SUMMARY]")
    print(stats.to_string(index=False))


if __name__ == "__main__":
    main()
