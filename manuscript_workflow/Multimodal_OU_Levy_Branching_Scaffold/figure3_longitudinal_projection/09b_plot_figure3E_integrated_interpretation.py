from __future__ import annotations

from pathlib import Path
import textwrap

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------
BASE = Path(__file__).resolve().parent
OUTDIR = Path("/Multimodal_OU_Lévy_Branching_Scaffold/Figure_3/panels")
OUTDIR.mkdir(parents=True, exist_ok=True)

PNG_OUT = OUTDIR / "figure3E_integrated_interpretation.png"
PDF_OUT = OUTDIR / "figure3E_integrated_interpretation.pdf"
JPG_OUT = OUTDIR / "figure3E_integrated_interpretation.jpeg"


# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------
def add_wrapped_text(
    ax,
    x,
    y,
    text,
    width=26,
    fontsize=9,
    ha="center",
    va="center",
    weight="normal",
    color="black",
    zorder=5,
):
    wrapped = "\n".join(textwrap.wrap(text, width=width))
    ax.text(
        x,
        y,
        wrapped,
        fontsize=fontsize,
        ha=ha,
        va=va,
        fontweight=weight,
        color=color,
        zorder=zorder,
    )


def add_box(
    ax,
    x,
    y,
    w,
    h,
    text="",
    fc="#ffffff",
    ec="#333333",
    lw=1.2,
    fontsize=9,
    text_width=26,
    weight="normal",
    rounding=0.025,
):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.015,rounding_size={rounding}",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
        zorder=2,
    )
    ax.add_patch(patch)

    if text:
        add_wrapped_text(
            ax,
            x + w / 2,
            y + h / 2,
            text,
            width=text_width,
            fontsize=fontsize,
            weight=weight,
        )

    return patch


def add_arrow(
    ax,
    x1,
    y1,
    x2,
    y2,
    color="#555555",
    lw=1.7,
    mutation_scale=14,
    connectionstyle="arc3,rad=0.0",
    linestyle="-",
):
    arrow = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle="-|>",
        mutation_scale=mutation_scale,
        linewidth=lw,
        color=color,
        linestyle=linestyle,
        connectionstyle=connectionstyle,
        zorder=4,
    )
    ax.add_patch(arrow)
    return arrow


def add_small_metric(ax, x, y, w, h, title, value, fc="#fafafa"):
    patch = add_box(
        ax,
        x,
        y,
        w,
        h,
        "",
        fc=fc,
        ec="#b0b0b0",
        lw=1.0,
        rounding=0.018,
    )

    ax.text(
        x + w / 2,
        y + h * 0.62,
        title,
        fontsize=7.8,
        fontweight="bold",
        ha="center",
        va="center",
        color="black",
        zorder=5,
    )

    ax.text(
        x + w / 2,
        y + h * 0.30,
        value,
        fontsize=7.6,
        ha="center",
        va="center",
        color="#333333",
        zorder=5,
    )

    return patch


def add_node(ax, x, y, w, h, title, subtitle, fc, ec):
    patch = add_box(
        ax,
        x,
        y,
        w,
        h,
        "",
        fc=fc,
        ec=ec,
        lw=1.5,
        rounding=0.025,
    )

    ax.text(
        x + w / 2,
        y + h * 0.62,
        title,
        fontsize=8.9,
        fontweight="bold",
        ha="center",
        va="center",
        color="black",
        zorder=5,
    )

    ax.text(
        x + w / 2,
        y + h * 0.32,
        subtitle,
        fontsize=7.7,
        ha="center",
        va="center",
        color="#333333",
        zorder=5,
    )

    return patch


# ------------------------------------------------------------
# Main plotting function
# ------------------------------------------------------------
def main():
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, ax = plt.subplots(figsize=(14.2, 3.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Panel label and title
    ax.text(
        0.015, 0.98, "E",
        fontsize=13, fontweight="bold",
        ha="left", va="top"
    )

    ax.text(
        0.50, 0.95,
        "Integrated longitudinal interpretation",
        fontsize=12,
        fontweight="bold",
        ha="center",
        va="top"
    )

    # Background frame
    add_box(
        ax,
        0.045, 0.18, 0.91, 0.66,
        "",
        fc="#ffffff",
        ec="#d8d8d8",
        lw=1.0,
        rounding=0.020,
    )

    # Main interpretation nodes
    node_y = 0.60
    node_w = 0.17
    node_h = 0.18

    nodes = [
        (
            0.085,
            "Diagnosis scaffold",
            "pretreatment reference",
            "#e8f4fd",
            "#4E79A7",
        ),
        (
            0.310,
            "Constrained response",
            "low displacement",
            "#d9edf7",
            "#2e86c1",
        ),
        (
            0.535,
            "Residual persistence",
            "retained structure",
            "#d5f5e3",
            "#239b56",
        ),
        (
            0.760,
            "Relapse escape",
            "upper-tail departure",
            "#f5c6cb",
            "#c0392b",
        ),
    ]

    for x, title, subtitle, fc, ec in nodes:
        add_node(
            ax,
            x,
            node_y,
            node_w,
            node_h,
            title,
            subtitle,
            fc,
            ec,
        )

    # Arrows connecting nodes
    arrow_y = node_y + node_h / 2
    add_arrow(ax, 0.085 + node_w + 0.012, arrow_y, 0.310 - 0.012, arrow_y)
    add_arrow(ax, 0.310 + node_w + 0.012, arrow_y, 0.535 - 0.012, arrow_y)
    add_arrow(ax, 0.535 + node_w + 0.012, arrow_y, 0.760 - 0.012, arrow_y)

    # Small escape annotation, separated from nodes
    ax.text(
        0.845,
        0.44,
        "branch-switching\nenriched subset",
        fontsize=9.0,
        ha="center",
        va="center",
        color="#8e3b3b",
        fontweight="bold",
    )

    add_arrow(
        ax,
        0.845,
        0.48,
        0.845,
        node_y + 0.0002,
        color="#8e3b3b",
        lw=1.1,
        mutation_scale=10,
        linestyle="--",
    )

    # Metric strip
    metric_y = 0.25
    metric_w = 0.205
    metric_h = 0.12

    metrics = [
        (
            0.058,
            "Position",
            "DX → EOI/REM → REL",
        ),
        (
            0.285,
            "Displacement",
            "low/moderate → upper-tail",
        ),
        (
            0.512,
            "Branch behavior",
            "continuous → switching-enriched",
        ),
        (
            0.739,
            "Interpretation",
            "constraint → persistence → escape",
        ),
    ]

    for x, title, value in metrics:
        add_small_metric(
            ax,
            x,
            metric_y,
            metric_w,
            metric_h,
            title,
            value,
            fc="#fafafa",
        )

    # Take-home sentence
    ax.text(
        0.50,
        0.02,
        "Longitudinal projection into the frozen diagnosis scaffold separates constrained or\nresidual trajectories from an upper-tail subset of branch-switching relapse-associated escape.",
        fontsize=10.5,
        ha="center",
        va="bottom",
        color="black",
    )

    fig.savefig(PNG_OUT, dpi=600, bbox_inches="tight")
    fig.savefig(PDF_OUT, bbox_inches="tight")
    fig.savefig(JPG_OUT, dpi=600, bbox_inches="tight")
    plt.close(fig)

    print(f"[DONE] Saved:\n  {PNG_OUT}\n  {PDF_OUT}\n  {JPG_OUT}")

if __name__ == "__main__":
    main()
