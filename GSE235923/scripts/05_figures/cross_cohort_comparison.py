from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PRIMARY_DIR = Path("/GSE235063/derived_dx_primary_training")
SECONDARY_DIR = Path("/GSE235923/derived_secondary_calibration")
OUTDIR = SECONDARY_DIR / "comparison_figure"
OUTDIR.mkdir(parents=True, exist_ok=True)

primary = pd.read_csv(PRIMARY_DIR / "dx_ou_ilr_branch_ready.csv")
secondary = pd.read_csv(SECONDARY_DIR / "gse235923_dx_secondary_calibration_table.csv")

variables = ["PC1", "PC2", "ilr_stem_vs_committed", "log_aux_erybaso"]
panel_titles = {
    "PC1": "A. Ecotype PC1",
    "PC2": "B. Ecotype PC2",
    "ilr_stem_vs_committed": "C. Stem-versus-committed balance",
    "log_aux_erybaso": "D. Erythroid/basophil auxiliary program",
}
ylabels = {
    "PC1": "PC1",
    "PC2": "PC2",
    "ilr_stem_vs_committed": "ilr_stem_vs_committed",
    "log_aux_erybaso": "log_aux_erybaso",
}

compare_rows = []
for var in variables:
    pmin = float(primary[var].min())
    pmax = float(primary[var].max())
    smin = float(secondary[var].min())
    smax = float(secondary[var].max())
    frac = float(((secondary[var] >= pmin) & (secondary[var] <= pmax)).mean())
    compare_rows.append({
        "variable": var,
        "primary_min": pmin,
        "primary_max": pmax,
        "secondary_min": smin,
        "secondary_max": smax,
        "secondary_within_primary_range_frac": frac,
    })

compare_df = pd.DataFrame(compare_rows)
compare_df.to_csv(OUTDIR / "primary_vs_secondary_comparison_ranges.csv", index=False)

rng = np.random.default_rng(42)

fig, axes = plt.subplots(2, 2, figsize=(12.5, 10), constrained_layout=False)
fig.subplots_adjust(left=0.08, right=0.98, bottom=0.08, top=0.90, wspace=0.22, hspace=0.28)
#fig.suptitle("Cross-cohort comparison of projected ecotype and malignant-state summaries", fontsize=18, y=0.975)
#fig.text(
#    0.5, 0.945,
#    "Primary training cohort: GSE235063 (Dx, n=19)   |   Secondary calibration cohort: GSE235923 (Dx, n=19)",
#    ha="center", va="center", fontsize=11
#)

for ax, var in zip(axes.ravel(), variables):
    p = primary[var].dropna().to_numpy()
    s = secondary[var].dropna().to_numpy()

    pmin = p.min()
    pmax = p.max()
    frac = float(((s >= pmin) & (s <= pmax)).mean())

    lo = min(p.min(), s.min())
    hi = max(p.max(), s.max())
    pad = 0.08 * (hi - lo if hi > lo else 1.0)
    lo -= pad
    hi += pad

    # primary-range band
    ax.axhspan(pmin, pmax, alpha=0.12)
    ax.axhline(pmin, linestyle="--", linewidth=1)
    ax.axhline(pmax, linestyle="--", linewidth=1)

    # boxplots
    ax.boxplot(
        [p, s],
        positions=[0, 1],
        widths=0.45,
        patch_artist=False,
        showfliers=False,
    )

    # jittered points
    x_primary = 0 + rng.uniform(-0.10, 0.10, size=len(p))
    x_secondary = 1 + rng.uniform(-0.10, 0.10, size=len(s))
    ax.scatter(x_primary, p, s=28, zorder=3)
    ax.scatter(x_secondary, s, s=28, zorder=3)

    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(lo, hi)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([
        "Primary\nGSE235063",
        "Secondary\nGSE235923",
    ])
    ax.set_ylabel(ylabels[var])
    ax.set_title(panel_titles[var], fontsize=13, pad=8)

    ax.text(
        0.03, 0.97,
        f"Secondary within primary range: {100*frac:.1f}%",
        transform=ax.transAxes,
        ha="left", va="top", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="black", linewidth=0.8)
    )

png_path = OUTDIR / "primary_vs_secondary_comparison_2x2.png"
pdf_path = OUTDIR / "primary_vs_secondary_comparison_2x2.pdf"
fig.savefig(png_path, dpi=300, bbox_inches="tight")
fig.savefig(pdf_path, bbox_inches="tight")
plt.close(fig)

print("Wrote:")
print(OUTDIR / "primary_vs_secondary_comparison_ranges.csv")
print(png_path)
print(pdf_path)
