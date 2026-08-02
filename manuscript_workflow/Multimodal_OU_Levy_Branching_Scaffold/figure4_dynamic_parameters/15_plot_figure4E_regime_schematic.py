from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch


# ============================================================
# 1. CONFIG
# ============================================================
PROJECT_DIR = Path("/Multimodal_OU_Lévy_Branching_Scaffold/Figure_4")
PANELS_DIR = PROJECT_DIR / "panels"

OUT_PNG = PANELS_DIR / "Figure4D_regime_schematic.png"
OUT_PDF = PANELS_DIR / "Figure4D_regime_schematic.pdf"


# ============================================================
# 2. HELPERS
# ============================================================
def ensure_dirs() -> None:
    PANELS_DIR.mkdir(parents=True, exist_ok=True)


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


def draw_regime_box(ax, x, y, w, h, title, subtitle, bullets, facecolor):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        facecolor=facecolor,
        edgecolor="#666666",
        linewidth=1.2,
        alpha=0.9,
    )
    ax.add_patch(box)

    ax.text(
        x + 0.003, y + h - 0.04, title,
        fontsize=11.5, fontweight="bold",
        ha="left", va="top", color="#222222"
    )
    ax.text(
        x + 0.003, y + h - 0.10, subtitle,
        fontsize=10.0,
        ha="left", va="top", color="#444444"
    )

    yy = y + h - 0.17
    for b in bullets:
        ax.text(
            x + 0.006, yy, f"• {b}",
            fontsize=8.5,
            ha="left", va="top", color="#222222"
        )
        yy -= 0.060


def draw_basin(ax, center, radius, color, label_top, label_bottom):
    circ = Circle(center, radius, facecolor=color, edgecolor="#444444", linewidth=1.2, alpha=0.95)
    ax.add_patch(circ)
    ax.text(
        center[0], center[1] + radius + 0.025, label_top,
        fontsize=10.6, fontweight="bold",
        ha="center", va="bottom", color="#222222"
    )
    ax.text(
        center[0], center[1] - radius - 0.055, label_bottom,
        fontsize=10.6,
        ha="center", va="top", color="#444444"
    )


def draw_arrow(ax, p1, p2, color="#444444", lw=1.8, linestyle="-"):
    arr = FancyArrowPatch(
        p1, p2,
        arrowstyle="-|>",
        mutation_scale=14,
        linewidth=lw,
        linestyle=linestyle,
        color=color,
        alpha=0.95,
    )
    ax.add_patch(arr)


# ============================================================
# 3. MAIN
# ============================================================
def main() -> None:
    ensure_dirs()

    fig, ax = plt.subplots(figsize=(12.5, 7.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    style_axis(ax, "D", "Schematic summary of inferred dynamical regimes")

    # ---------------------------------------------------------
    # Left to right regimes
    # ---------------------------------------------------------
    basin_y = 0.31
    basin_r = 0.075

    draw_regime_box(
        ax,
        x=0.03, y=0.60, w=0.28, h=0.30,
        title="Baseline response regime",
        subtitle="Diagnosis-anchored state",
        bullets=[
            r"higher effective restoration",
            r"lower punctuated-transition propensity",
        ],
        facecolor="#E8EEF6",
    )
    
    draw_regime_box(
        ax,
        x=0.355, y=0.60, w=0.28, h=0.30,
        title="Residual persistence regime",
        subtitle="Rare retained EOI/REM malignant state",
        bullets=[
            r"not maximally displaced from DX attractor",
            r"can remain weakly constrained",
        ],
        facecolor="#E7F4EE",
    )
    
    draw_regime_box(
        ax,
        x=0.68, y=0.60, w=0.28, h=0.30,
        title="Punctuated relapse regime",
        subtitle="Upper-tail departure subset",
        bullets=[
            r"larger DX→REL displacement",
            r"branch-switching enriched among extremes",
        ],
        facecolor="#F8E8E8",
    )

    draw_basin(
    ax,
    center=(0.17, basin_y),
    radius=basin_r,
    color="#7A8DA8",
    label_top="Constrained baseline basin",
    label_bottom=r"high $\theta_{\mathrm{eff}}$, lower $\lambda_{\mathrm{proxy}}$"
    )
    
    draw_basin(
        ax,
        center=(0.50, basin_y),
        radius=basin_r,
        color="#6BAF92",
        label_top="Residual persistent state",
        label_bottom=r"weaker constraint, high $\sigma_{\mathrm{eff}}$ in retained case"
    )
    
    draw_basin(
        ax,
        center=(0.83, basin_y),
        radius=basin_r,
        color="#C97979",
        label_top="Relapse escape regime",
        label_bottom=r"branch-switching, elevated $\lambda_{\mathrm{proxy}}$"
    )

    # ---------------------------------------------------------
    # Transitions
    # ---------------------------------------------------------
    draw_arrow(ax, (0.26, basin_y), (0.42, basin_y), color="#4A4A4A", lw=1.8, linestyle="-")
    draw_arrow(ax, (0.58, basin_y), (0.74, basin_y), color="#4A4A4A", lw=2.0, linestyle="--")

    ax.text(0.34, basin_y + 0.04, "partial contraction / persistence",
            fontsize=8.8, ha="center", va="bottom", color="#444444")
    ax.text(0.66, basin_y + 0.04, "branch-switching / escape",
            fontsize=8.8, ha="center", va="bottom", color="#444444")

    # ---------------------------------------------------------
    # Footer interpretation
    # ---------------------------------------------------------
#    ax.text(
#        0.02, 0.06,
#        "Residual disease is dynamically distinct, whereas relapse-prone trajectories are enriched for larger displacement and branch-switching escape.",
#        fontsize=9,
#        ha="left",
#        va="bottom",
#        color="#333333",
#    )

    fig.savefig(OUT_PNG, dpi=600, bbox_inches="tight")
    fig.savefig(OUT_PDF, dpi=600, bbox_inches="tight")
    plt.close(fig)

    print(f"[DONE] Saved {OUT_PNG}")
    print(f"[DONE] Saved {OUT_PDF}")


if __name__ == "__main__":
    main()
