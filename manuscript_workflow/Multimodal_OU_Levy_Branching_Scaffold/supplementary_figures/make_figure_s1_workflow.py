from pathlib import Path
import textwrap

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle, FancyArrowPatch, Polygon


# ----------------------------
# Global style
# ----------------------------
mpl.rcParams["font.family"] = "DejaVu Sans"
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42

OUTDIR = Path(".")
PNG_OUT = OUTDIR / "Figure_S1_workflow.png"
PDF_OUT = OUTDIR / "Figure_S1_workflow.pdf"


# ----------------------------
# Drawing helpers
# ----------------------------
def rounded_box(ax, x, y, w, h, fc, ec, lw=1.8, radius=0.018, z=1):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.012,rounding_size={radius}",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
        zorder=z,
    )
    ax.add_patch(patch)
    return patch


def circle_label(ax, x, y, label, color, r=0.031):
    circ = Circle((x, y), r, facecolor=color, edgecolor="white", linewidth=2.0, zorder=10)
    ax.add_patch(circ)
    ax.text(
        x, y, str(label),
        ha="center", va="center",
        fontsize=14, fontweight="bold",
        color="white", zorder=11,
    )


def arrow(ax, start, end, rad=0.0, lw=1.8):
    arr = FancyArrowPatch(
        start, end,
        arrowstyle="-|>",
        mutation_scale=13,
        linewidth=lw,
        color="#1f2937",
        connectionstyle=f"arc3,rad={rad}",
        shrinkA=6,
        shrinkB=6,
        zorder=3,
    )
    ax.add_patch(arr)


def wrapped_bullets(ax, x, y, bullets, width=42, fontsize=8.2, line_gap=0.020):
    yy = y
    for b in bullets:
        wrapped = textwrap.wrap(b, width=width)
        for j, line in enumerate(wrapped):
            prefix = "• " if j == 0 else "  "
            ax.text(
                x, yy, prefix + line,
                ha="left", va="top",
                fontsize=fontsize,
                color="#111827",
                zorder=8,
            )
            yy -= line_gap
        yy -= 0.004


def code_chip(ax, x, y, text, w, h=0.026):
    rounded_box(ax, x, y, w, h, fc="#dbeafe", ec="#dbeafe", lw=0.6, radius=0.006, z=6)
    ax.text(
        x + 0.008, y + h / 2, text,
        ha="left", va="center",
        fontsize=7.1,
        family="monospace",
        color="#111827",
        zorder=8,
    )


# ----------------------------
# Simple icons
# ----------------------------
def icon_download(ax, cx, cy, scale=1.0):
    ax.add_patch(Rectangle((cx - 0.036*scale, cy - 0.020*scale), 0.072*scale, 0.052*scale,
                           facecolor="#e5e7eb", edgecolor="#1f2937", lw=1.8, zorder=7))
    ax.add_patch(Rectangle((cx - 0.048*scale, cy - 0.032*scale), 0.096*scale, 0.012*scale,
                           facecolor="#1f2937", edgecolor="#1f2937", zorder=7))
    ax.arrow(cx, cy + 0.025*scale, 0, -0.034*scale,
             width=0.006*scale, head_width=0.026*scale, head_length=0.018*scale,
             color="#38bdf8", length_includes_head=True, zorder=8)


def icon_tar(ax, cx, cy, scale=1.0):
    rounded_box(ax, cx - 0.032*scale, cy - 0.044*scale, 0.064*scale, 0.088*scale,
                fc="#16a34a", ec="#15803d", lw=1.2, radius=0.006, z=7)
    ax.text(cx, cy + 0.012*scale, "RAW", ha="center", va="center",
            fontsize=7.5*scale, color="white", fontweight="bold", zorder=8)
    ax.text(cx, cy - 0.012*scale, ".tar.gz", ha="center", va="center",
            fontsize=7.2*scale, color="white", fontweight="bold", zorder=8)


def icon_folder(ax, cx, cy, scale=1.0):
    ax.add_patch(Rectangle((cx - 0.044*scale, cy - 0.025*scale), 0.088*scale, 0.055*scale,
                           facecolor="#fbbf24", edgecolor="#b45309", lw=1.3, zorder=7))
    ax.add_patch(Rectangle((cx - 0.044*scale, cy + 0.020*scale), 0.050*scale, 0.018*scale,
                           facecolor="#f59e0b", edgecolor="#b45309", lw=1.1, zorder=7))


def icon_gz(ax, cx, cy, scale=1.0):
    rounded_box(ax, cx - 0.032*scale, cy - 0.044*scale, 0.064*scale, 0.088*scale,
                fc="#8b5cf6", ec="#6d28d9", lw=1.2, radius=0.006, z=7)
    ax.text(cx, cy, ".gz", ha="center", va="center",
            fontsize=12*scale, color="white", fontweight="bold", zorder=8)


def icon_anndata(ax, cx, cy, scale=1.0):
    rounded_box(ax, cx - 0.040*scale, cy - 0.042*scale, 0.080*scale, 0.084*scale,
                fc="#f9a8d4", ec="#db2777", lw=1.2, radius=0.008, z=7)
    colors = ["#60a5fa", "#fb7185", "#f97316", "#a78bfa"]
    for i in range(4):
        for j in range(4):
            ax.add_patch(Circle((cx - 0.024*scale + i*0.016*scale,
                                 cy - 0.024*scale + j*0.016*scale),
                                0.0055*scale, color=colors[(i+j) % 4], zorder=8))


def icon_metadata(ax, cx, cy, scale=1.0):
    rounded_box(ax, cx - 0.038*scale, cy - 0.044*scale, 0.076*scale, 0.088*scale,
                fc="#67e8f9", ec="#0e7490", lw=1.2, radius=0.008, z=7)
    ax.add_patch(Circle((cx, cy + 0.012*scale), 0.011*scale, color="#0e7490", zorder=8))
    ax.add_patch(Rectangle((cx - 0.026*scale, cy - 0.030*scale), 0.052*scale, 0.024*scale,
                           facecolor="#0e7490", edgecolor="#0e7490", zorder=8))


def icon_qc(ax, cx, cy, scale=1.0):
    rounded_box(ax, cx - 0.038*scale, cy - 0.044*scale, 0.076*scale, 0.088*scale,
                fc="#67e8f9", ec="#0e7490", lw=1.2, radius=0.008, z=7)
    for i, h in enumerate([0.025, 0.045, 0.065]):
        ax.add_patch(Rectangle((cx - 0.025*scale + i*0.020*scale, cy - 0.030*scale),
                               0.012*scale, h*scale,
                               facecolor="#0e7490", edgecolor="#0e7490", zorder=8))
    ax.add_patch(Circle((cx + 0.030*scale, cy + 0.030*scale), 0.018*scale,
                        facecolor="#14b8a6", edgecolor="white", lw=1.0, zorder=9))
    ax.text(cx + 0.030*scale, cy + 0.030*scale, "✓",
            ha="center", va="center", fontsize=10*scale,
            color="white", fontweight="bold", zorder=10)


def icon_h5ad(ax, cx, cy, scale=1.0):
    rounded_box(ax, cx - 0.030*scale, cy - 0.044*scale, 0.060*scale, 0.088*scale,
                fc="#38bdf8", ec="#0369a1", lw=1.2, radius=0.006, z=7)
    ax.text(cx, cy - 0.002*scale, ".h5ad", ha="center", va="center",
            fontsize=8.5*scale, color="white", fontweight="bold", zorder=8)


def icon_csv(ax, cx, cy, scale=1.0):
    rounded_box(ax, cx - 0.030*scale, cy - 0.044*scale, 0.060*scale, 0.088*scale,
                fc="#22c55e", ec="#15803d", lw=1.2, radius=0.006, z=7)
    ax.text(cx, cy - 0.002*scale, "CSV", ha="center", va="center",
            fontsize=10*scale, color="white", fontweight="bold", zorder=8)


# ----------------------------
# Box constructor
# ----------------------------
def workflow_box(
    ax, x, y, w, h, num, title, bullets, icon_func, fc, ec,
    bullet_width=40,
):
    rounded_box(ax, x, y, w, h, fc=fc, ec=ec, lw=1.8, radius=0.018, z=1)
    circle_label(ax, x - 0.005, y + h - 0.012, num, ec)

    # Icon
    icon_func(ax, x + 0.060, y + h * 0.48, scale=0.95)

    # Title
    ax.text(
        x + 0.120, y + h - 0.023,
        title,
        ha="left", va="center",
        fontsize=11.5,
        fontweight="bold",
        color=ec,
        zorder=8,
    )

    # Bullets only; no code chip
    wrapped_bullets(
        ax,
        x + 0.120,
        y + h - 0.05,
        bullets,
        width=bullet_width,
        fontsize=7.8,
        line_gap=0.018,
    )


# ----------------------------
# Main
# ----------------------------
def main():
    fig, ax = plt.subplots(figsize=(13.0, 10.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Colors
    navy = "#10243f"
    blue_fc, blue_ec = "#eaf5ff", "#2563eb"
    green_fc, green_ec = "#ecfdf5", "#16a34a"
    orange_fc, orange_ec = "#fff7ed", "#f59e0b"
    purple_fc, purple_ec = "#f5f3ff", "#7c3aed"
    pink_fc, pink_ec = "#fdf2f8", "#db2777"
    cyan_fc, cyan_ec = "#ecfeff", "#0891b2"

    # Title
    ax.text(
        0.5, 0.965,
        "RAW GEO ARCHIVE → PER-SAMPLE H5AD FILES & dx_sample_summary.csv",
        ha="center", va="center",
        fontsize=18.5,
        fontweight="bold",
        color=navy,
    )
    ax.text(
        0.5, 0.925,
        "Step-by-step workflow",
        ha="center", va="center",
        fontsize=13.5,
        fontweight="bold",
        color=navy,
    )

    # Layout
    left_x, right_x = 0.07, 0.535
    box_w, box_h = 0.400, 0.115
    left_ys = [0.765, 0.625, 0.485, 0.345]
    right_ys = [0.765, 0.600, 0.435, 0.270]

    # Left column
    workflow_box(
        ax, left_x, left_ys[0], box_w, box_h, 1,
        "Download raw dataset",
        [
            "Identify GEO accession and raw archive.",
            "Download compressed expression files.",
            "Example: GSEXXXXX_RAW.tar.gz",
        ],
        icon_download, blue_fc, blue_ec, bullet_width=40,
    )

    workflow_box(
        ax, left_x, left_ys[1], box_w, box_h, 2,
        "Extract archive",
        [
            "Create a raw-data directory.",
            "Extract per-sample folders.",
            "Preserve sample-level file structure.",
        ],
        icon_tar, green_fc, green_ec, bullet_width=40,
    )
    
    workflow_box(
        ax, left_x, left_ys[2], box_w, box_h, 3,
        "Inspect extracted files",
        [
            "Confirm matrix, gene, and barcode files.",
            "Verify consistent sample naming.",
            "Check expected Matrix Market inputs.",
        ],
        icon_folder, orange_fc, orange_ec, bullet_width=40,
    )
    
    workflow_box(
        ax, left_x, left_ys[3], box_w, box_h, 4,
        "Decompress sample files",
        [
            "Unzip per-sample input files.",
            "Prepare matrix, gene, and barcode tables.",
            "Keep raw counts for h5ad construction.",
        ],
        icon_gz, purple_fc, purple_ec, bullet_width=42,
    )
    
        # Right column
    workflow_box(
        ax, right_x, right_ys[0], box_w, box_h, 5,
        "Create AnnData object",
        [
            "Read Matrix Market count matrix.",
            "Assign barcodes and gene names.",
            "Create one object per sample.",
        ],
        icon_anndata, pink_fc, pink_ec, bullet_width=40,
    )
    
    workflow_box(
        ax, right_x, right_ys[1], box_w, box_h, 6,
        "Annotate sample metadata",
        [
            "Map sample identifiers to metadata.",
            "Add diagnosis and sample labels.",
            "Store annotations in obs.",
        ],
        icon_metadata, cyan_fc, cyan_ec, bullet_width=40,
    )
    
    workflow_box(
        ax, right_x, right_ys[2], box_w, box_h, 7,
        "Optional QC and processing",
        [
            "Calculate n_genes, n_counts, pct_mito.",
            "Filter low-quality cells if required.",
            "Retain raw counts for downstream analyses.",
        ],
        icon_qc, cyan_fc, cyan_ec, bullet_width=45,
    )
    
    workflow_box(
        ax, right_x, right_ys[3], box_w, box_h, 8,
        "Save per-sample h5ad files",
        [
            "Write one annotated object per sample.",
            "Save outputs for downstream analysis.",
            "Record paths in summary table.",
        ],
        icon_h5ad, blue_fc, blue_ec, bullet_width=40,
    )

    # Arrows within columns
    for i in range(3):
        arrow(
            ax,
            (left_x + box_w / 2, left_ys[i]),
            (left_x + box_w / 2, left_ys[i + 1] + box_h),
            lw=1.8,
        )

    for i in range(3):
        arrow(
            ax,
            (right_x + box_w / 2, right_ys[i]),
            (right_x + box_w / 2, right_ys[i + 1] + box_h),
            lw=1.8,
        )

    # Clean cross-column arrow from step 4 to step 5
    arrow(
        ax,
        (left_x + box_w + 0.010, left_ys[3] + box_h * 0.62),
        (right_x - 0.010, right_ys[0] + box_h * 0.45),
        rad=-0.22,
        lw=1.9,
    )

    # Final outputs label
    rounded_box(ax, 0.445, 0.178, 0.110, 0.036, fc="#0f3d66", ec="#0f3d66", lw=1.0, radius=0.012, z=5)
    ax.text(
        0.500, 0.196,
        "Final outputs",
        ha="center", va="center",
        fontsize=9.5,
        fontweight="bold",
        color="white",
        zorder=8,
    )

    # Arrows to final outputs
    arrow(ax, (left_x + box_w * 0.50, left_ys[3]), (0.315, 0.165), rad=0.12, lw=1.8)
    arrow(ax, (right_x + box_w * 0.50, right_ys[3]), (0.700, 0.165), rad=-0.12, lw=1.8)

    # Output boxes
    out_y, out_h = 0.045, 0.115

    rounded_box(ax, 0.145, out_y, 0.350, out_h, fc=purple_fc, ec=purple_ec, lw=1.8, radius=0.018)
    circle_label(ax, 0.155, out_y + out_h - 0.012, "A", purple_ec)
    icon_h5ad(ax, 0.225, out_y + 0.055, scale=1.12)
    ax.text(
        0.285, out_y + out_h - 0.022,
        "Per-sample h5ad files",
        ha="left", va="center",
        fontsize=12.0,
        fontweight="bold",
        color=purple_ec,
    )
    wrapped_bullets(
        ax, 0.285, out_y + 0.073,
        [
            "One annotated h5ad object per sample.",
            "Contains raw counts and cell/gene metadata.",
        ],
        width=34,
        fontsize=8.2,
        line_gap=0.021,
    )

    rounded_box(ax, 0.535, out_y, 0.360, out_h, fc=green_fc, ec=green_ec, lw=1.8, radius=0.018)
    circle_label(ax, 0.545, out_y + out_h - 0.012, "B", green_ec)
    icon_csv(ax, 0.615, out_y + 0.055, scale=1.12)
    ax.text(
        0.675, out_y + out_h - 0.022,
        "dx_sample_summary.csv",
        ha="left", va="center",
        fontsize=12.0,
        fontweight="bold",
        color=green_ec,
    )
    wrapped_bullets(
        ax, 0.675, out_y + 0.073,
        [
            "Machine-readable cohort summary.",
            "Includes sample IDs, labels, and output paths.",
        ],
        width=34,
        fontsize=8.2,
        line_gap=0.021,
    )

    fig.savefig(PNG_OUT, dpi=600, bbox_inches="tight")
    fig.savefig(PDF_OUT, bbox_inches="tight")
    print(f"Saved: {PNG_OUT}")
    print(f"Saved: {PDF_OUT}")


if __name__ == "__main__":
    main()
