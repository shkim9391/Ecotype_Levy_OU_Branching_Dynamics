from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# ============================================================
# 1. CONFIG
# ============================================================
PROJECT_DIR = Path("/Multimodal_OU_Lévy_Branching_Scaffold/Figure_3")
DERIVED_DIR = PROJECT_DIR / "derived"
PANELS_DIR = PROJECT_DIR / "panels"

IN_CSV = DERIVED_DIR / "relapse_jump_candidates.csv"

OUT_PNG = PANELS_DIR / "Figure3D_jump_candidates.png"
OUT_PDF = PANELS_DIR / "Figure3D_jump_candidates.pdf"

TOP_N_LABEL = 10
SWITCH_BONUS = 0.50

TOP_N_HIGHLIGHT = 10

ALPHA_LOW_LINE = 0.30
ALPHA_LOW_BONUS = 0.28
ALPHA_LOW_POINT = 0.45

ALPHA_TOP_LINE = 0.95
ALPHA_TOP_BONUS = 0.95
ALPHA_TOP_POINT = 1.00

LW_LOW_LINE = 1.6
LW_TOP_LINE = 2.2

LW_LOW_BONUS = 4.0
LW_TOP_BONUS = 5.4

S_LOW_POINT = 34
S_TOP_POINT = 78

S_LOW_DISP = 12
S_TOP_DISP = 20

COLOR_SWITCH = "#C97979"
COLOR_CONT = "#7A8DA8"
COLOR_DISP = "#BDBDBD"
COLOR_ZERO = "#D3D3D3"
COLOR_BONUS = "#E9B2B2"
COLOR_TOP_BAND = "#F8E8E8"


# ============================================================
# 2. HELPERS
# ============================================================
def ensure_dirs() -> None:
    PANELS_DIR.mkdir(parents=True, exist_ok=True)


def style_axis(ax, panel_label: str, title: str) -> None:
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)

    ax.text(
        0.00, 1.03, panel_label,
        transform=ax.transAxes,
        fontsize=18,
        fontweight="bold",
        ha="left",
        va="bottom",
    )
    ax.text(
        0.10, 1.03, title,
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def clean_branch_label(x: object) -> str:
    s = str(x)
    s = s.replace("-like basin", "")
    s = s.replace(" basin", "")
    return s.strip()


def format_transition(row: pd.Series) -> str:
    if ("branch_start" in row.index) and ("branch_end" in row.index):
        bs = row["branch_start"]
        be = row["branch_end"]
        if pd.notna(bs) and pd.notna(be):
            return f"{clean_branch_label(bs)}→{clean_branch_label(be)}"
    return ""


# ============================================================
# 3. MAIN
# ============================================================
def main() -> None:
    ensure_dirs()

    df = pd.read_csv(IN_CSV)

    required = ["patient_id", "z_displacement_hd", "branch_switch", "jump_score"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Jump candidate table missing required columns: {missing}")

    if df.empty:
        raise ValueError("Jump candidate table is empty.")

    df["z_displacement_hd"] = pd.to_numeric(df["z_displacement_hd"], errors="coerce")
    df["branch_switch"] = pd.to_numeric(df["branch_switch"], errors="coerce").fillna(0).astype(int)
    df["jump_score"] = pd.to_numeric(df["jump_score"], errors="coerce")

    df = df.sort_values(["jump_score", "z_displacement_hd"], ascending=[False, False]).reset_index(drop=True)
    if "jump_rank" not in df.columns:
        df["jump_rank"] = np.arange(1, df.shape[0] + 1)

    # Decomposition
    df["disp_component"] = df["z_displacement_hd"]
    df["switch_component"] = SWITCH_BONUS * df["branch_switch"].astype(float)

    # Sanity: jump score should equal displacement + switch bonus
    calc_score = df["disp_component"].fillna(0.0) + df["switch_component"].fillna(0.0)
    max_diff = float(np.nanmax(np.abs(calc_score - df["jump_score"])))
    if max_diff > 1e-6:
        print(f"[WARN] jump_score differs from components by max {max_diff:.6f}")

    y = np.arange(df.shape[0])[::-1]
    colors = np.where(df["branch_switch"].astype(int) == 1, COLOR_SWITCH, COLOR_CONT)

    # Top-band threshold for visual emphasis
    if df.shape[0] >= 5:
        top_band_thresh = float(df["jump_score"].nlargest(5).min())
    else:
        top_band_thresh = float(df["jump_score"].min())

    fig, ax = plt.subplots(figsize=(8.6, 7.4))

    # Background band for high escape region
    x_min_data = float(np.nanmin(np.r_[0.0, df["disp_component"].to_numpy(dtype=float), df["jump_score"].to_numpy(dtype=float)]))
    x_max_data = float(np.nanmax(np.r_[0.0, df["disp_component"].to_numpy(dtype=float), df["jump_score"].to_numpy(dtype=float)]))
    x_pad = 0.18 * (x_max_data - x_min_data if x_max_data > x_min_data else 1.0)

    ax.axvspan(top_band_thresh, x_max_data + x_pad, color=COLOR_TOP_BAND, alpha=0.35, zorder=0)
    ax.axvline(0, color=COLOR_ZERO, linewidth=1.0, zorder=0)
    
    highlight_mask = np.zeros(df.shape[0], dtype=bool)
    highlight_mask[:TOP_N_HIGHLIGHT] = True

    # 1) Displacement contribution
    for i, (yi, disp) in enumerate(zip(y, df["disp_component"])):
        is_top = highlight_mask[i]
        ax.hlines(
            yi,
            0,
            disp,
            color=COLOR_DISP,
            linewidth=LW_TOP_LINE if is_top else LW_LOW_LINE,
            alpha=ALPHA_TOP_LINE if is_top else ALPHA_LOW_LINE,
            zorder=1,
        )

    # 2) Branch-switch bonus segment (only where present)
    for i, (yi, disp, total, sw) in enumerate(zip(y, df["disp_component"], df["jump_score"], df["branch_switch"])):
        if int(sw) == 1:
            is_top = highlight_mask[i]
            ax.hlines(
                yi,
                disp,
                total,
                color=COLOR_BONUS,
                linewidth=LW_TOP_BONUS if is_top else LW_LOW_BONUS,
                alpha=ALPHA_TOP_BONUS if is_top else ALPHA_LOW_BONUS,
                zorder=2,
            )

    # 3) Total escape-score point
    # lower-ranked intervals first
    low_df = df.loc[~highlight_mask].copy()
    low_y = y[~highlight_mask]
    low_colors = colors[~highlight_mask]
    
    ax.scatter(
        low_df["jump_score"],
        low_y,
        s=S_LOW_POINT,
        c=low_colors,
        edgecolors="white",
        linewidths=0.5,
        alpha=ALPHA_LOW_POINT,
        zorder=2.8,
    )
    
    ax.scatter(
        low_df["disp_component"],
        low_y,
        s=S_LOW_DISP,
        c="#8F8F8F",
        linewidths=0,
        alpha=0.55,
        zorder=2.4,
    )
    
    # highlighted top subset on top
    top_df = df.loc[highlight_mask].copy()
    top_y = y[highlight_mask]
    top_colors = colors[highlight_mask]
    
    ax.scatter(
        top_df["jump_score"],
        top_y,
        s=S_TOP_POINT,
        c=top_colors,
        edgecolors="white",
        linewidths=0.8,
        alpha=ALPHA_TOP_POINT,
        zorder=3.2,
    )
    
    ax.scatter(
        top_df["disp_component"],
        top_y,
        s=S_TOP_DISP,
        c="#8F8F8F",
        linewidths=0,
        alpha=0.95,
        zorder=2.9,
    )

    # Labels for top candidates
    top = df.head(TOP_N_LABEL).copy()
    for _, r in top.iterrows():
        yi = y[r.name]
        transition = format_transition(r)
        label = f"{r['patient_id']}"
        if transition:
            label = f"{label}  {transition}"

        ax.text(
            float(r["jump_score"]) + 0.035,
            yi,
            label,
            fontsize=9.0,
            fontweight="bold",
            ha="left",
            va="center",
            color="#222222",
            zorder=4,
        )

    # Top-band label
    ax.text(
        top_band_thresh + 0.02,
        y.max() + 0.75,
        "highest escape-score subset",
        fontsize=9.0,
        color="#7A3B3B",
        ha="left",
        va="bottom",
    )

    ax.set_yticks([])
    ax.set_xlabel("Relapse-escape score")

    style_axis(ax, "D", "DX→REL interval escape score decomposition")

    legend_handles = [
        Line2D([0], [0], color=COLOR_DISP, lw=2.0, label="Displacement contribution"),
        Line2D([0], [0], color=COLOR_BONUS, lw=5.0, label="Branch-switch bonus"),
        Line2D([0], [0], marker="o", linestyle="None", markersize=8,
               markerfacecolor=COLOR_SWITCH, markeredgecolor="white", label="Branch-switching interval"),
        Line2D([0], [0], marker="o", linestyle="None", markersize=8,
               markerfacecolor=COLOR_CONT, markeredgecolor="white", label="Branch-continuous interval"),
    ]
    ax.legend(handles=legend_handles, frameon=False, loc="best", fontsize=9.0)

    ax.set_xlim(x_min_data - x_pad, x_max_data + x_pad)

    fig.savefig(OUT_PNG, dpi=600, bbox_inches="tight")
    fig.savefig(OUT_PDF, dpi=600, bbox_inches="tight")
    plt.close(fig)

    print(f"[DONE] Saved {OUT_PNG}")
    print(f"[DONE] Saved {OUT_PDF}")

    print("\n[TOP 10]")
    cols = [c for c in ["jump_rank", "patient_id", "branch_start", "branch_end", "disp_component", "switch_component", "jump_score"] if c in df.columns]
    print(df[cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
