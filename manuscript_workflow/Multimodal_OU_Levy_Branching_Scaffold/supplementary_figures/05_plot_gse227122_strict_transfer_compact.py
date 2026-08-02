import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def add_panel_label(ax, label: str):
    ax.text(-0.14, 1.05, label, transform=ax.transAxes, fontsize=12, fontweight="bold", va="top")


def natural_key(s: str):
    parts = re.split(r"(\d+)", str(s))
    return [int(p) if p.isdigit() else p for p in parts]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--projected-samples", required=True)
    parser.add_argument("--cross-cohort-table", required=True)
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    proj = pd.read_csv(args.projected_samples)
    cross = pd.read_csv(args.cross_cohort_table)

    proj["sample_id"] = proj["sample_id"].astype(str)
    if "timepoint" in proj.columns:
        proj["timepoint"] = proj["timepoint"].astype(str)

    TIMEPOINT_MAP = {
        "Dx": "DX",
        "dx": "DX",
        "Rel": "REL",
        "rel": "REL",
        "EOI": "EOI",
        "Eoi": "EOI",
        "eoi": "EOI",
    }

    # standardize timepoint values in proj
    if "timepoint" in proj.columns:
        proj["timepoint"] = proj["timepoint"].replace(TIMEPOINT_MAP)

    # standardize embedded sample labels like T11_Rel or T1_Dx
    for col in ["sample_id", "sample_label", "label"]:
        if col in proj.columns:
            proj[col] = (
                proj[col].astype(str)
                .str.replace(r'(?<=_)Dx\b', "DX", regex=True)
                .str.replace(r'(?<=_)Rel\b', "REL", regex=True)
                .str.replace(r'(?<=_)dx\b', "DX", regex=True)
                .str.replace(r'(?<=_)rel\b', "REL", regex=True)
            )

    # -----------------------------
    # Panel A: cross-cohort ecotype space
    # -----------------------------
    cross_plot = cross.copy()
    cross_plot = cross_plot.loc[cross_plot["mode"].isin(["reference", "strict_transfer"])].copy()
    cross_plot = cross_plot.dropna(subset=["PC1", "PC2"])

    # -----------------------------
    # Panel B/C/D use GSE227122 strict transfer
    # -----------------------------
    g227 = proj.copy()
    g227 = g227.dropna(subset=["PC1_strict", "PC2_strict"]).copy()

    fig, axes = plt.subplots(2, 2, figsize=(13, 11))

    # =============================
    # A. Cross-cohort scatter
    # =============================
    ax = axes[0, 0]
    cohort_markers = {
        "GSE235063": "o",
        "GSE235923": "^",
        "GSE227122": "s",
    }
    cohort_colors = {
        "GSE235063": "tab:blue",
        "GSE235923": "tab:orange",
        "GSE227122": "tab:green",
    }
    
    shown = set()
    for _, row in cross_plot.iterrows():
        cohort = row["cohort"]
        label = cohort if cohort not in shown else None
        shown.add(cohort)
        ax.scatter(
            row["PC1"], row["PC2"],
            marker=cohort_markers.get(cohort, "o"),
            color=cohort_colors.get(cohort, "gray"),
            s=55,
            alpha=0.85,
            label=label,
        )

    ax.axhline(0, linewidth=0.8, alpha=0.5)
    ax.axvline(0, linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Transferred PC1")
    ax.set_ylabel("Transferred PC2")
    ax.set_title("Cross-cohort strict ecotype transfer")
    ax.legend(frameon=False, fontsize=8, loc="best")
    add_panel_label(ax, "A")

    # =============================
    # B. GSE227122 scatter by timepoint
    # =============================
    ax = axes[0, 1]
    
    df_panelB = g227.copy()
    
    timepoint_order = ["DX", "EOI", "REL"]
    timepoint_markers = {"DX": "o", "EOI": "^", "REL": "*"}
    timepoint_colors = {"DX": "tab:blue", "EOI": "tab:orange", "REL": "tab:green"}
    
    for tp in timepoint_order:
        sub = df_panelB[df_panelB["timepoint"] == tp]
        ax.scatter(
            sub["PC1_strict"], sub["PC2_strict"],
            marker=timepoint_markers[tp],
            color=timepoint_colors[tp],
            s=90,
            label=tp
        )
    
    ax.axhline(0, linewidth=0.8, alpha=0.5)
    ax.axvline(0, linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Transferred PC1")
    ax.set_ylabel("Transferred PC2")
    ax.set_title("GSE227122 strict transfer by timepoint")
    ax.legend(frameon=False, fontsize=8, loc="best")
    add_panel_label(ax, "B")

    # =============================
    # C. Paired longitudinal trajectories
    # =============================
    ax = axes[1, 0]
    
    if {"patient_id", "timepoint"}.issubset(g227.columns):
        for pid in sorted(g227["patient_id"].dropna().astype(str).unique(), key=natural_key):
            sub = g227.loc[g227["patient_id"].astype(str) == pid].copy()
    
            dx = sub.loc[sub["timepoint"] == "DX"]
            eoi = sub.loc[sub["timepoint"] == "EOI"]
            rel = sub.loc[sub["timepoint"] == "REL"]
    
            if len(dx) > 0 and len(eoi) > 0:
                dxr = dx.iloc[0]
                eoir = eoi.iloc[0]
    
                ax.scatter(dxr["PC1_strict"], dxr["PC2_strict"], s=60, marker="o", color="tab:blue")
                ax.scatter(eoir["PC1_strict"], eoir["PC2_strict"], s=70, marker="^", color="tab:orange")
    
                ax.annotate(
                    "",
                    xy=(eoir["PC1_strict"], eoir["PC2_strict"]),
                    xytext=(dxr["PC1_strict"], dxr["PC2_strict"]),
                    arrowprops=dict(arrowstyle="->", lw=1.2),
                )
    
                mx = 0.5 * (dxr["PC1_strict"] + eoir["PC1_strict"])
                my = 0.5 * (dxr["PC2_strict"] + eoir["PC2_strict"])
                ax.text(mx, my, pid, fontsize=8)
    
            if len(rel) > 0:
                rr = rel.iloc[0]
                ax.scatter(
                    rr["PC1_strict"], rr["PC2_strict"],
                    s=130, marker="*", color="tab:green",
                    label="REL" if pid == "T11" else None
                )
                ax.text(rr["PC1_strict"], rr["PC2_strict"], f" {rr['sample_id']}", fontsize=8, va="center")
    
    ax.axhline(0, linewidth=0.8, alpha=0.5)
    ax.axvline(0, linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Transferred PC1")
    ax.set_ylabel("Transferred PC2")
    ax.set_title("Paired longitudinal trajectories in GSE227122")
    add_panel_label(ax, "C")

    # =============================
    # D. Normal cells used for transfer
    # =============================
    ax = axes[1, 1]
    
    plot_df = g227[["sample_id", "timepoint", "normal_cells_used_for_transfer"]].copy()
    plot_df = plot_df.sort_values("sample_id", key=lambda s: s.map(natural_key))
    
    plot_df["sample_id"] = (
        plot_df["sample_id"].astype(str)
        .str.replace(r'(?<=_)Dx\b', "DX", regex=True)
        .str.replace(r'(?<=_)Rel\b', "REL", regex=True)
        .str.replace(r'(?<=_)dx\b', "DX", regex=True)
        .str.replace(r'(?<=_)rel\b', "REL", regex=True)
    )
    
    x = np.arange(len(plot_df))
    ax.bar(x, plot_df["normal_cells_used_for_transfer"].values)
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["sample_id"].tolist(), rotation=45, ha="right")
    ax.set_ylabel("Normal cells used for transfer")
    ax.set_title("Per-sample normal-cell support")
    add_panel_label(ax, "D")

    pdf_path = outdir / "figure_gse227122_strict_transfer_compact.pdf"
    png_path = outdir / "figure_gse227122_strict_transfer_compact.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=600, bbox_inches="tight")
    plt.close(fig)

    print("\n=== WRITTEN ===")
    print(pdf_path)
    print(png_path)


if __name__ == "__main__":
    main()
