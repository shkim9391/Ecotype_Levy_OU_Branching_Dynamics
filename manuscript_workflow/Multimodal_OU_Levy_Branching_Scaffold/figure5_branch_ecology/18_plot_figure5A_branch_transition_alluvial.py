from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, PathPatch
from matplotlib.path import Path as MplPath
from matplotlib.lines import Line2D


# ============================================================
# 1. CONFIG
# ============================================================
PROJECT_DIR = Path("/Multimodal_OU_Lévy_Branching_Scaffold/Figure_5")
DERIVED_DIR = PROJECT_DIR / "derived"
PANELS_DIR = PROJECT_DIR / "panels"

IN_CSV = DERIVED_DIR / "branch_transition_table.csv"

OUT_PNG = PANELS_DIR / "Figure5A_branch_transition_alluvial.png"
OUT_PDF = PANELS_DIR / "Figure5A_branch_transition_alluvial.pdf"

BRANCH_ORDER = [
    "HSC-like basin",
    "Progenitor-like basin",
    "GMP-like basin",
    "Mono/DC-like basin",
]

BRANCH_COLORS = {
    "HSC-like basin": "#7A8DA8",
    "Progenitor-like basin": "#C97979",
    "GMP-like basin": "#D6B85A",
    "Mono/DC-like basin": "#7BB7B2",
}

FLOW_COLORS = {
    "stable": "#8A9BAF",
    "switching": "#D88A8A",
}

LEFT_X = 0.17
RIGHT_X = 0.83
BAR_W = 0.08

TOP_Y = 0.88
BOT_Y = 0.12
BAR_GAP = 0.018

CURVE_PULL = 0.22


# ============================================================
# 2. HELPERS
# ============================================================
def ensure_dirs() -> None:
    PANELS_DIR.mkdir(parents=True, exist_ok=True)


def assert_columns(df: pd.DataFrame, required: list[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"branch_transition_table.csv missing required columns: {missing}")


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


def compute_bar_positions(counts: dict[str, int], branch_order: list[str]) -> dict[str, tuple[float, float]]:
    """
    Returns y-intervals (y0, y1) for each branch bar.
    """
    total_n = sum(counts.get(b, 0) for b in branch_order)
    if total_n <= 0:
        raise ValueError("No transitions available to plot.")

    n_nonzero = sum(counts.get(b, 0) > 0 for b in branch_order)
    usable_h = (TOP_Y - BOT_Y) - BAR_GAP * max(n_nonzero - 1, 0)

    pos = {}
    y = TOP_Y
    for b in branch_order:
        n = counts.get(b, 0)
        if n <= 0:
            continue
        h = usable_h * (n / total_n)
        y0 = y - h
        pos[b] = (y0, y)
        y = y0 - BAR_GAP
    return pos


def bezier_flow_patch(
    x0: float, x1: float,
    y0_top: float, y0_bot: float,
    y1_top: float, y1_bot: float,
    color: str,
    alpha: float = 0.55,
):
    """
    Create a smooth alluvial polygon between left and right stacked segments.
    """
    c0 = x0 + CURVE_PULL * (x1 - x0)
    c1 = x1 - CURVE_PULL * (x1 - x0)

    verts = [
        (x0, y0_top),
        (c0, y0_top),
        (c1, y1_top),
        (x1, y1_top),

        (x1, y1_bot),
        (c1, y1_bot),
        (c0, y0_bot),
        (x0, y0_bot),

        (x0, y0_top),
    ]
    codes = [
        MplPath.MOVETO,
        MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
        MplPath.LINETO,
        MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
        MplPath.CLOSEPOLY,
    ]
    return PathPatch(
        MplPath(verts, codes),
        facecolor=color,
        edgecolor="none",
        alpha=alpha,
        zorder=1,
    )


# ============================================================
# 3. MAIN
# ============================================================
def main() -> None:
    ensure_dirs()

    df = pd.read_csv(IN_CSV)
    assert_columns(df, ["patient_id", "branch_start", "branch_end", "branch_switch", "transition_label"])

    # Aggregate flow counts
    flows = (
        df.groupby(["branch_start", "branch_end", "branch_switch"])
          .size()
          .reset_index(name="count")
    )

    total_n = int(flows["count"].sum())

    start_counts = flows.groupby("branch_start")["count"].sum().to_dict()
    end_counts = flows.groupby("branch_end")["count"].sum().to_dict()

    left_pos = compute_bar_positions(start_counts, BRANCH_ORDER)
    right_pos = compute_bar_positions(end_counts, BRANCH_ORDER)

    # Sub-segment allocation within each bar
    flows = flows.copy()
    flows["branch_start"] = flows["branch_start"].astype(str)
    flows["branch_end"] = flows["branch_end"].astype(str)
    flows["flow_class"] = np.where(flows["branch_switch"].astype(int) == 1, "switching", "stable")

    # Sort flows for deterministic stacking
    flows["start_order"] = flows["branch_start"].map({b: i for i, b in enumerate(BRANCH_ORDER)})
    flows["end_order"] = flows["branch_end"].map({b: i for i, b in enumerate(BRANCH_ORDER)})
    flows = flows.sort_values(
        ["start_order", "end_order", "flow_class"],
        ascending=[True, True, True]
    ).reset_index(drop=True)

    left_offsets = {b: left_pos[b][1] for b in left_pos}
    right_offsets = {b: right_pos[b][1] for b in right_pos}

    # Height per single transition unit inside each bar
    left_unit = {}
    for b, (y0, y1) in left_pos.items():
        left_unit[b] = (y1 - y0) / start_counts[b]

    right_unit = {}
    for b, (y0, y1) in right_pos.items():
        right_unit[b] = (y1 - y0) / end_counts[b]

    flow_draw = []
    for _, r in flows.iterrows():
        b0 = r["branch_start"]
        b1 = r["branch_end"]
        n = int(r["count"])

        h0 = left_unit[b0] * n
        h1 = right_unit[b1] * n

        y0_top = left_offsets[b0]
        y0_bot = y0_top - h0
        left_offsets[b0] = y0_bot

        y1_top = right_offsets[b1]
        y1_bot = y1_top - h1
        right_offsets[b1] = y1_bot

        flow_draw.append({
            "branch_start": b0,
            "branch_end": b1,
            "count": n,
            "flow_class": r["flow_class"],
            "y0_top": y0_top,
            "y0_bot": y0_bot,
            "y1_top": y1_top,
            "y1_bot": y1_bot,
        })

    # ---------------------------------------------------------
    # Plot
    # ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10.5, 7.4))
    style_axis(ax, "A", "Diagnosis-to-relapse branch transitions")

    # Flows first
    for f in flow_draw:
        patch = bezier_flow_patch(
            LEFT_X + BAR_W / 2,
            RIGHT_X - BAR_W / 2,
            f["y0_top"], f["y0_bot"],
            f["y1_top"], f["y1_bot"],
            color=FLOW_COLORS[f["flow_class"]],
            alpha=0.58 if f["flow_class"] == "switching" else 0.46,
        )
        ax.add_patch(patch)

    # Bars
    for b in BRANCH_ORDER:
        if b in left_pos:
            y0, y1 = left_pos[b]
            ax.add_patch(Rectangle(
                (LEFT_X - BAR_W / 2, y0), BAR_W, y1 - y0,
                facecolor=BRANCH_COLORS[b],
                edgecolor="#444444",
                linewidth=1.0,
                alpha=0.95,
                zorder=3,
            ))
            ax.text(
                LEFT_X - BAR_W / 2 - 0.02,
                (y0 + y1) / 2,
                f"{b}\n(n={start_counts[b]})",
                ha="right", va="center",
                fontsize=9.0, color="#222222"
            )

        if b in right_pos:
            y0, y1 = right_pos[b]
            ax.add_patch(Rectangle(
                (RIGHT_X - BAR_W / 2, y0), BAR_W, y1 - y0,
                facecolor=BRANCH_COLORS[b],
                edgecolor="#444444",
                linewidth=1.0,
                alpha=0.95,
                zorder=3,
            ))
            ax.text(
                RIGHT_X + BAR_W / 2 + 0.02,
                (y0 + y1) / 2,
                f"{b}\n(n={end_counts[b]})",
                ha="left", va="center",
                fontsize=9.0, color="#222222"
            )

    # Column headings
    ax.text(LEFT_X, 0.94, "Diagnosis branch", ha="center", va="bottom",
            fontsize=10.5, fontweight="bold", color="#222222")
    ax.text(RIGHT_X, 0.94, "Relapse branch", ha="center", va="bottom",
            fontsize=10.5, fontweight="bold", color="#222222")

    # Legend
    legend_items = [
        Line2D([0], [0], color=FLOW_COLORS["stable"], lw=8, alpha=0.55, label="Branch-continuous"),
        Line2D([0], [0], color=FLOW_COLORS["switching"], lw=8, alpha=0.65, label="Branch-switching"),
    ]
    ax.legend(handles=legend_items, frameon=False, loc="upper center",
              bbox_to_anchor=(0.5, 1.01), ncol=2, fontsize=9.5)

    # Small summary annotation
    n_switch = int((df["branch_switch"].astype(int) == 1).sum())
    n_continuous = int((df["branch_switch"].astype(int) == 0).sum())
    
    ax.text(
        0.50, 0.06,
        f"{total_n} DX→REL intervals: {n_continuous} branch-continuous, {n_switch} branch-switching",
        ha="center",
        va="center",
        fontsize=11.0,
        color="#555555"
    )

    fig.savefig(OUT_PNG, dpi=600, bbox_inches="tight")
    fig.savefig(OUT_PDF, dpi=600, bbox_inches="tight")
    plt.close(fig)

    print(f"[DONE] Saved {OUT_PNG}")
    print(f"[DONE] Saved {OUT_PDF}")

    print("\n[SUMMARY]")
    print(flows[["branch_start", "branch_end", "flow_class", "count"]].to_string(index=False))


if __name__ == "__main__":
    main()
