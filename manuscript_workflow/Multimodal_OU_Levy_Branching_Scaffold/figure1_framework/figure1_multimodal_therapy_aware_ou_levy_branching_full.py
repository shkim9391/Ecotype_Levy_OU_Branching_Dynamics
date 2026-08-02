import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch, Circle, FancyBboxPatch, Ellipse
from matplotlib.lines import Line2D
import textwrap


# -----------------------------
# Helper functions
# -----------------------------
def add_wrapped_text(ax, x, y, text, width=40, fontsize=10, ha='center', va='center',
                     weight='normal', color='black', zorder=5):
    wrapped = "\n".join(textwrap.wrap(text, width=width))
    ax.text(
        x, y, wrapped,
        fontsize=fontsize, ha=ha, va=va,
        weight=weight, color=color, zorder=zorder
    )


def add_box(ax, x, y, w, h, text, fc="#f7f7f7", ec="black", lw=1.5,
            fontsize=10, text_width=28, style="round,pad=0.02,rounding_size=0.02",
            weight='normal'):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=style,
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=2
    )
    ax.add_patch(patch)
    add_wrapped_text(ax, x + w/2, y + h/2, text, width=text_width, fontsize=fontsize, weight=weight)
    return patch


def add_rect(ax, x, y, w, h, text="", fc="#ffffff", ec="black", lw=1.5,
             fontsize=10, text_width=22, weight='normal'):
    patch = Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec, linewidth=lw, zorder=2)
    ax.add_patch(patch)
    if text:
        add_wrapped_text(ax, x + w/2, y + h/2, text, width=text_width, fontsize=fontsize, weight=weight)
    return patch


def add_arrow(ax, x1, y1, x2, y2, color="black", lw=1.6, style='-|>', mutation_scale=14,
              connectionstyle="arc3,rad=0.0", zorder=3, linestyle='-'):
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle=style,
        mutation_scale=mutation_scale,
        linewidth=lw,
        color=color,
        connectionstyle=connectionstyle,
        linestyle=linestyle,
        zorder=zorder
    )
    ax.add_patch(arrow)
    return arrow


def add_panel_label(ax, label):
    ax.text(
        0.01, 0.98, label,
        transform=ax.transAxes,
        fontsize=16, fontweight='bold',
        va='top', ha='left'
    )

def add_regime_node(ax, x, y, r, title, subtitle, fc, ec, fontsize=9):
    circ = Circle((x, y), r, facecolor=fc, edgecolor=ec, linewidth=1.8, zorder=3)
    ax.add_patch(circ)
    add_wrapped_text(
        ax, x, y + 0.015,
        title,
        width=16,
        fontsize=fontsize,
        weight='bold'
    )
    add_wrapped_text(
        ax, x, y - 0.109,
        subtitle,
        width=22,
        fontsize=fontsize - 1,
        color="#333333"
    )
    return circ

def add_regime_ellipse(ax, x, y, w, h, title, line1, line2, line3,
                       fc, ec, title_color="black"):
    """
    Horizontal regime node for Panel E.
    Uses an ellipse with separately positioned text lines to avoid overlap.
    """
    ell = Ellipse(
        (x, y), width=w, height=h,
        facecolor=fc, edgecolor=ec, linewidth=1.8,
        zorder=3
    )
    ax.add_patch(ell)

    ax.text(
        x, y + h * 0.16,
        title,
        fontsize=12.5, fontweight="bold",
        ha="center", va="center",
        color=title_color, zorder=5
    )

    ax.text(
        x, y - h * 0.03,
        line1,
        fontsize=11.5,
        ha="center", va="center",
        color="#333333", zorder=5
    )

    ax.text(
        x, y - h * 0.15,
        line2,
        fontsize=10.5,
        ha="center", va="center",
        color="#333333", zorder=5
    )

    ax.text(
        x, y - h * 0.27,
        line3,
        fontsize=10.5,
        ha="center", va="center",
        color="#333333", zorder=5
    )

    return ell


def add_metric_box(ax, x, y, w, h, metric, pattern):
    """
    Compact two-line metric box for Panel E.
    """
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.015,rounding_size=0.018",
        facecolor="#fafafa", edgecolor="#b0b0b0",
        linewidth=1.0, zorder=2
    )
    ax.add_patch(patch)

    ax.text(
        x + w / 2, y + h * 0.67,
        metric,
        fontsize=12.5, fontweight="bold",
        ha="center", va="center",
        color="black", zorder=5
    )

    ax.text(
        x + w / 2, y + h * 0.25,
        pattern,
        fontsize=11.5,
        ha="center", va="center",
        color="#333333", zorder=5
    )

    return patch

# -----------------------------
# Figure canvas
# -----------------------------
fig = plt.figure(figsize=(16, 17.5))

gs = fig.add_gridspec(
    3, 2,
    left=0.04, right=0.98, top=0.96, bottom=0.04,
    wspace=0.12, hspace=0.10,
    height_ratios=[1.0, 1.0, 0.85]
)

axA = fig.add_subplot(gs[0, 0])
axB = fig.add_subplot(gs[0, 1])
axC = fig.add_subplot(gs[1, 0])
axD = fig.add_subplot(gs[1, 1])
axE = fig.add_subplot(gs[2, :])

for ax in [axA, axB, axC, axD, axE]:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

# -----------------------------
# Panel A: Longitudinal study design
# -----------------------------
add_panel_label(axA, "A")

axA.text(
    0.5, 0.92,
    "Longitudinal pediatric leukemia study design",
    fontsize=16, fontweight='bold', ha='center'
)

# Main timeline
y_tl = 0.55
x_positions = [0.16, 0.39, 0.62, 0.85]
stage_labels = [
    "Diagnosis",
    "On-therapy\nresponse states",
    "Remission /\nEnd of induction",
    "Relapse"
]
stage_colors = ["#d9edf7", "#fcf3cf", "#d5f5e3", "#f5c6cb"]

axA.plot([0.12, 0.82], [y_tl, y_tl], color="black", lw=2)

for x, label, c in zip(x_positions, stage_labels, stage_colors):
    circ = Circle((x, y_tl), 0.04, facecolor=c, edgecolor="black", linewidth=1.5, zorder=4)
    axA.add_patch(circ)
    add_wrapped_text(axA, x, y_tl + 0.11, label, width=16, fontsize=12.0, weight='bold')

# Sampling icons / sample blocks
sample_texts = [
    "Baseline\nsingle-cell\nmultiome",
    "Serial therapy-\nlinked profiles",
    "Residual disease /\nrecovery state",
    "Escape /\nprogression state"
]
for x, txt in zip(x_positions, sample_texts):
    add_box(axA, x - 0.05, 0.24, 0.10, 0.18, txt, fc="#ffffff", ec="black", fontsize=9.5, text_width=12)

# Discovery / calibration / validation bands
add_box(axA, 0.11, 0.76, 0.38, 0.08,
        "Discovery: multimodal single-cell cohorts with matched temporal structure",
        fc="#eef5ff", ec="#2e5aac", fontsize=10.5, text_width=34, weight='bold')
add_box(axA, 0.555, 0.76, 0.14, 0.08,
        "Calibration cohorts",
        fc="#f4f6f6", ec="#616a6b", fontsize=10.5, text_width=14, weight='bold')
add_box(axA, 0.76, 0.76, 0.13, 0.08,
        "External validation",
        fc="#f4f6f6", ec="#616a6b", fontsize=10.5, text_width=14, weight='bold')

add_arrow(axA, 0.51, 0.80, 0.54, 0.80, lw=1.5)
add_arrow(axA, 0.715, 0.80, 0.745, 0.80, lw=1.5)

# Footer hypothesis line
add_box(
    axA, 0.07, 0.05, 0.88, 0.09,
    "Serial sampling provides the temporal scaffold needed to infer constrained evolution, treatment-associated destabilization, and relapse-linked escape trajectories.",
    fc="#fafafa", ec="#999999", fontsize=10.5, text_width=87
)

# -----------------------------
# Panel B: Multimodal inputs
# -----------------------------
add_panel_label(axB, "B")

axB.text(
    0.52, 0.92,
    "Multimodal inputs defining disease state and context",
    fontsize=16, fontweight='bold', ha='center'
)

# Top explanatory band
add_box(
    axB, 0.08, 0.79, 0.84, 0.07,
    "Multimodal measurements jointly represent state, while treatment phase provides dynamic biological context.",
    fc="#fafafa", ec="#999999", fontsize=10.5, text_width=80
)

# Central latent-state box
center_x, center_y, center_w, center_h = 0.39, 0.33, 0.22, 0.16
add_box(
    axB, center_x, center_y, center_w, center_h,
    "Integrated latent\ndisease state",
    fc="#fff2cc", ec="#b8860b", lw=2.0,
    fontsize=12, text_width=18, weight='bold'
)

# Surrounding modality boxes
modalities = [
    {
        "xywh": (0.10, 0.58, 0.24, 0.14),
        "text": "Transcriptomic state\n(expression programs,\ncell-state composition)",
        "fc": "#e8f4fd"
    },
    {
        "xywh": (0.66, 0.58, 0.24, 0.14),
        "text": "Regulatory state\n(chromatin,\nTF accessibility)",
        "fc": "#eafaf1"
    },
    {
        "xywh": (0.10, 0.10, 0.24, 0.14),
        "text": "Spatial / ecological\ncontext (niche\nstructure,\nimmune–stromal\norganization)",
        "fc": "#fdebd0"
    },
    {
        "xywh": (0.66, 0.10, 0.24, 0.14),
        "text": "Treatment / clinical\nphase (diagnosis,\non-therapy, EOI,\nrelapse)",
        "fc": "#f9ebea"
    },
]

for m in modalities:
    x, y, w, h = m["xywh"]
    add_box(axB, x, y, w, h, m["text"], fc=m["fc"], ec="black",
            fontsize=10.7, text_width=20)

# Short, symmetric arrows: from outer-box edge to central-box edge
arrow_specs = [
    # top-left  -> central upper-left edge
    ((0.10 + 0.24, 0.50 + 0.14 * 0.52), (center_x,           center_y + center_h * 0.72)),
    # top-right -> central upper-right edge
    ((0.66,        0.50 + 0.14 * 0.52), (center_x + center_w, center_y + center_h * 0.72)),
    # bottom-left -> central lower-left edge
    ((0.10 + 0.24, 0.18 + 0.14 * 0.48), (center_x,           center_y + center_h * 0.27)),
    # bottom-right -> central lower-right edge
    ((0.66,        0.18 + 0.14 * 0.48), (center_x + center_w, center_y + center_h * 0.27)),
]

for (x1, y1), (x2, y2) in arrow_specs:
    add_arrow(
        axB, x1, y1, x2, y2,
        lw=1.5, mutation_scale=14,
        connectionstyle="arc3,rad=0.0"
    )

# -----------------------------
# Panel C: Therapy-aware OU–Lévy–branching model
# -----------------------------
add_panel_label(axC, "C")

axC.text(
    0.5, 0.92,
    "Therapy-aware OU–Lévy–branching dynamics",
    fontsize=16, fontweight='bold', ha='center'
)

# Attractor wells
for x0, label in [(0.24, "Pre-therapy\nattractor"),
                  (0.52, "On-therapy\nshifted attractor"),
                  (0.79, "Relapse /\nescape attractor")]:
    circ = Circle((x0, 0.30), 0.06, facecolor="#d6eaf8", edgecolor="#2e86c1", lw=1.6, alpha=0.95)
    axC.add_patch(circ)
    add_wrapped_text(axC, x0, 0.18, label, width=14, fontsize=10, weight='bold')

# Evolutionary trajectory
traj_x = [0.12, 0.18, 0.25, 0.31, 0.39, 0.46, 0.55, 0.60, 0.66, 0.73, 0.82]
traj_y = [0.60, 0.49, 0.34, 0.43, 0.47, 0.37, 0.32, 0.44, 0.40, 0.52, 0.33]
axC.plot(traj_x, traj_y, color="#2c3e50", lw=2.2, zorder=5)

# Direction arrows along path
for i in [1, 4, 7]:
    add_arrow(axC, traj_x[i], traj_y[i], traj_x[i+1], traj_y[i+1],
              color="#2c3e50", lw=1.8, mutation_scale=12)

# Lévy jump
add_arrow(axC, 0.61, 0.45, 0.77, 0.54, color="#c0392b", lw=2.2,
          linestyle='--', mutation_scale=16)
axC.text(0.69, 0.54, "Punctuated\njump", color="#c0392b",
         fontsize=10, ha='center', fontweight='bold')

# Branching split
add_arrow(axC, 0.33, 0.45, 0.43, 0.55, color="#7d3c98", lw=1.8, mutation_scale=14)
axC.text(0.43, 0.55, "Branch-specific\ndiversification", fontsize=10,
         color="#7d3c98", ha='center', fontweight='bold')

# Therapy modulation box
add_box(
    axC, 0.08, 0.68, 0.40, 0.17,
    "Treatment phase modulates:\n• attractor location\n• stabilizing strength\n• diffusion scale\n• jump propensity",
    fc="#f9ebea", ec="#c0392b", fontsize=10.0, text_width=26, weight='bold'
)

# Equation / concept box
add_box(
    axC, 0.56, 0.68, 0.37, 0.17,
    "Latent state =\nmean reversion + stochastic diffusion\n+ occasional Lévy jumps + branching",
    fc="#eef5ff", ec="#2e5aac", fontsize=10.0, text_width=26, weight='bold'
)

# Footer interpretation
add_box(
    axC, 0.10, 0.03, 0.80, 0.07,
    "Therapy context determines whether trajectories remain constrained, become destabilized, or undergo punctuated escape.",
    fc="#fafafa", ec="#999999", fontsize=10.5, text_width=80
)

# -----------------------------
# Panel D: Analysis workflow
# -----------------------------
add_panel_label(axD, "D")

axD.text(
    0.5, 0.92,
    "Analysis workflow",
    fontsize=16, fontweight='bold', ha='center'
)

workflow = [
    ("Preprocessing", 0.06, 0.68, "#e8f4fd"),
    ("Multimodal\nintegration", 0.28, 0.68, "#eafaf1"),
    ("Latent state\nconstruction", 0.50, 0.68, "#fff2cc"),
    ("Trajectory\nreconstruction", 0.72, 0.68, "#fdebd0"),
    ("Therapy-aware\ndynamic parameter\nestimation", 0.17, 0.36, "#f9ebea"),
    ("Branch-level\nanalysis", 0.44, 0.36, "#f4ecf7"),
    ("External projection\nand validation", 0.71, 0.36, "#f4f6f6"),
]

box_w = 0.16
box_h = 0.14

for label, x, y, fc in workflow:
    add_box(axD, x, y, box_w, box_h, label, fc=fc, ec="black",
            fontsize=10.5, text_width=16, weight='bold')

# Top row arrows
add_arrow(axD, 0.24, 0.75, 0.28, 0.75)
add_arrow(axD, 0.46, 0.75, 0.50, 0.75)
add_arrow(axD, 0.68, 0.75, 0.72, 0.75)

# Down arrows
add_arrow(axD, 0.59, 0.68, 0.26, 0.50, connectionstyle="arc3,rad=0.0")
add_arrow(axD, 0.59, 0.68, 0.53, 0.50, connectionstyle="arc3,rad=0.0")
add_arrow(axD, 0.81, 0.68, 0.80, 0.50, connectionstyle="arc3,rad=0.0")

# Bottom row arrows
add_arrow(axD, 0.35, 0.43, 0.44, 0.43)
add_arrow(axD, 0.62, 0.43, 0.71, 0.43)

# Final summary band
add_box(
    axD, 0.08, 0.11, 0.84, 0.12,
    "Outputs: therapy-modulated OU parameters, jump-sensitive trajectory statistics, branch-specific evolutionary structure, and validation by projection into the learned scaffold.",
    fc="#fafafa", ec="#999999", fontsize=10.5, text_width=80
)

# -----------------------------
# Panel E: Clinical dynamic regimes
# -----------------------------
add_panel_label(axE, "E")

axE.text(
    0.5, 0.94,
    "Therapy-aware disease-state regimes inferred from scaffold dynamics",
    fontsize=16, fontweight='bold', ha='center'
)

# Light background frame
add_box(
    axE, 0.035, 0.13, 0.93, 0.72,
    "",
    fc="#ffffff", ec="#d8d8d8", lw=1.0,
    fontsize=1, text_width=1,
    style="round,pad=0.02,rounding_size=0.02"
)

# Regime nodes
node_y = 0.65
node_w = 0.255
node_h = 0.275

add_regime_ellipse(
    axE, 0.17, node_y, node_w, node_h,
    "Constrained response-like state",
    "diagnosis-anchored",
    "stronger restoration",
    "low jump signal",
    fc="#d9edf7", ec="#2e86c1"
)

add_regime_ellipse(
    axE, 0.50, node_y, node_w, node_h,
    "Residual persistent state",
    "incomplete contraction",
    "retained disease structure",
    "intermediate displacement",
    fc="#d5f5e3", ec="#239b56"
)

add_regime_ellipse(
    axE, 0.83, node_y, node_w, node_h,
    "Relapse escape state",
    "larger displacement",
    "branch switching",
    "elevated jump signal",
    fc="#f5c6cb", ec="#c0392b"
)

# Directional arrows between nodes
add_arrow(
    axE, 0.30, node_y, 0.37, node_y,
    color="#555555", lw=1.8, mutation_scale=15
)
add_arrow(
    axE, 0.63, node_y, 0.70, node_y,
    color="#555555", lw=1.8, mutation_scale=15
)

# Arrow labels, placed above arrows but away from node text
axE.text(
    0.335, 0.775,
    "therapy pressure\npartial response",
    fontsize=11.0, ha='center', va='center', color="#555555"
)

axE.text(
    0.665, 0.775,
    "destabilization\nescape-prone transition",
    fontsize=11.0, ha='center', va='center', color="#555555"
)

# Metric strip
metric_y = 0.205
metric_h = 0.155
metric_w = 0.207

metric_specs = [
    (0.045, "Attractor displacement", "low → intermediate → high"),
    (0.279, "Restoring strength", "strong → incomplete → weak/variable"),
    (0.514, "Branch behavior", "continuous → retained/shifted\n→ switching-enriched"),
    (0.749, "Jump-sensitive signal", "low → mixed → elevated"),
]

for x, metric, pattern in metric_specs:
    add_metric_box(
        axE, x, metric_y, metric_w, metric_h,
        metric, pattern
    )

# Optional subtle label above metric strip
axE.text(
    0.5, 0.41,
    "Dynamic summary metrics",
    fontsize=13.5, fontweight="bold",
    ha="center", va="center", color="#444444"
)

# Final interpretation sentence
axE.text(
    0.5, 0.005,
    "The scaffold converts longitudinal multimodal leukemia profiles into interpretable precision-oncology states\nfor response monitoring, residual persistence, and relapse-escape prioritization.",
    fontsize=13.5, ha='center', va='bottom'
)

# -----------------------------
# Final formatting and save
# -----------------------------
for ax in [axA, axB, axC, axD, axE]:
    for spine in ax.spines.values():
        spine.set_visible(False)

plt.savefig("Figure1_multimodal_therapy_aware_scaffold_full.png", dpi=600, bbox_inches="tight")
plt.savefig("Figure1_multimodal_therapy_aware_scaffold_full.pdf", bbox_inches="tight")
plt.show()
