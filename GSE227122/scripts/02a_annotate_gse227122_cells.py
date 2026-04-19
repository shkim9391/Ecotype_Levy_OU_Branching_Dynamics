from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse


# --------------------------------------------------
# Marker sets
# --------------------------------------------------
MARKER_SETS: Dict[str, List[str]] = {
    # Canonical non-blast lineages
    "T_cell": ["CD3D", "CD3E", "IL7R", "LTB", "TRBC1", "TRBC2", "MAL"],
    "NK": ["NKG7", "GNLY", "KLRD1", "CTSW", "PRF1"],
    "B_cell": ["CD19", "CD79A", "CD79B", "MS4A1", "HLA-DRA"],
    "Myeloid": ["FCN1", "LYZ", "S100A8", "S100A9", "CTSS", "FCER1G", "CD14", "CD68"],
    "DC": ["LILRA4", "FCER1A", "CLEC10A", "GZMB", "IRF7"],
    "Erythroid": ["HBB", "HBD", "HBA1", "HBA2", "AHSP", "ALAS2"],
    "HSPC": ["CD34", "SPINK2", "GATA2", "KIT", "PROM1", "AVP", "HOPX"],

    # Auxiliary
    "Proliferating": ["MKI67", "TOP2A", "TYMS", "UBE2C", "CENPF", "HMGB2"],

    # T-ALL blast / lymphoblast program
    "Blast_TALL": ["SOX4", "STMN1", "JUN", "HES4", "CDK6", "ARMH1", "CD99", "TUBA1B", "DNTT", "MAL"],
}

NORMAL_LINEAGES = ["T_cell", "NK", "B_cell", "Myeloid", "DC", "Erythroid", "HSPC"]

BROAD_MAP = {
    "T_cell": "T_NK",
    "NK": "T_NK",
    "T_NK": "T_NK",
    "B_cell": "B_Plasma",
    "B_Plasma": "B_Plasma",
    "Myeloid": "Myeloid_APC",
    "DC": "Myeloid_APC",
    "Myeloid_APC": "Myeloid_APC",
    "Erythroid": "Erythroid_Baso",
    "Erythroid_Baso": "Erythroid_Baso",
    "HSPC": "HSPC_Prog",
    "HSPC_Prog": "HSPC_Prog",
    "Proliferating": "HSPC_Prog",
    "Blast_TALL": "Blast_TALL",
    "Unknown": "Unknown",
}


def to_dense_1d(x) -> np.ndarray:
    if sparse.issparse(x):
        x = x.A
    x = np.asarray(x).reshape(-1)
    return x


def present_genes(adata_obj: ad.AnnData, genes: List[str]) -> List[str]:
    upper_to_orig = {}
    for g in adata_obj.var_names.astype(str):
        gu = g.upper()
        if gu not in upper_to_orig:
            upper_to_orig[gu] = g
    out = []
    for g in genes:
        if g.upper() in upper_to_orig:
            out.append(upper_to_orig[g.upper()])
    return out


def mean_module_score_z(adata_obj: ad.AnnData, genes: List[str]) -> np.ndarray:
    keep = present_genes(adata_obj, genes)
    if len(keep) == 0:
        return np.full(adata_obj.n_obs, np.nan, dtype=float)

    X = adata_obj[:, keep].X
    vals = np.asarray(X.mean(axis=1)).reshape(-1)
    sd = np.nanstd(vals)
    if not np.isfinite(sd) or sd == 0:
        return np.zeros_like(vals, dtype=float)
    return (vals - np.nanmean(vals)) / sd


def choose_prelim_label(row: pd.Series) -> str:
    score_cols = {k: row[f"score_{k}"] for k in MARKER_SETS.keys()}
    finite = {k: v for k, v in score_cols.items() if pd.notna(v)}
    if len(finite) == 0:
        return "Unknown"
    best_label = max(finite, key=finite.get)
    best_score = finite[best_label]
    if best_score < 0.10:
        return "Unknown"
    return best_label


def cluster_top_fraction(sub: pd.Series) -> float:
    vc = sub.astype(str).value_counts(normalize=True)
    if len(vc) == 0:
        return np.nan
    return float(vc.iloc[0])


def extract_rank_genes_groups(adata_obj: ad.AnnData, groupby: str, top_n: int = 20) -> pd.DataFrame:
    result = []
    rg = adata_obj.uns["rank_genes_groups"]
    groups = rg["names"].dtype.names
    for grp in groups:
        names = rg["names"][grp][:top_n]
        scores = rg["scores"][grp][:top_n] if "scores" in rg else [np.nan] * len(names)
        pvals_adj = rg["pvals_adj"][grp][:top_n] if "pvals_adj" in rg else [np.nan] * len(names)
        logfc = rg["logfoldchanges"][grp][:top_n] if "logfoldchanges" in rg else [np.nan] * len(names)
        for rank, (n, s, p, lfc) in enumerate(zip(names, scores, pvals_adj, logfc), start=1):
            result.append(
                {
                    groupby: str(grp),
                    "rank": rank,
                    "gene": str(n),
                    "score": float(s) if pd.notna(s) else np.nan,
                    "logfoldchange": float(lfc) if pd.notna(lfc) else np.nan,
                    "pvals_adj": float(p) if pd.notna(p) else np.nan,
                }
            )
    return pd.DataFrame(result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="results/gse227122_transfer/gse227122_qc.h5ad")
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--n-top-genes", type=int, default=3000)
    parser.add_argument("--n-pcs", type=int, default=30)
    parser.add_argument("--n-neighbors", type=int, default=15)
    parser.add_argument("--leiden-resolution", type=float, default=0.60)
    parser.add_argument("--random-state", type=int, default=0)

    # Heuristic thresholds
    parser.add_argument("--blast-z-min", type=float, default=0.35)
    parser.add_argument("--blast-margin", type=float, default=0.15)
    parser.add_argument("--dxrel-min", type=float, default=0.70)
    parser.add_argument("--top-sample-min", type=float, default=0.35)
    parser.add_argument("--min-cluster-size", type=int, default=50)
    parser.add_argument("--skip-rank-genes", action="store_true")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    adata_obj = ad.read_h5ad(args.input)

    # Use gene symbols for marker scoring when available.
    # The ingested GSE227122 object keeps gene IDs in var_names and gene symbols in adata.var["gene_symbols"].
    if "gene_symbols" in adata_obj.var.columns:
        adata_obj.var["gene_id"] = adata_obj.var_names.astype(str)
        adata_obj.var["gene_symbol_raw"] = adata_obj.var["gene_symbols"].astype(str)
        adata_obj.var_names = adata_obj.var["gene_symbols"].astype(str).values

    # Standardize metadata types
    for col in ["sample_id", "barcode", "patient_id", "timepoint", "transfer_set", "mrd_status", "batch"]:
        if col in adata_obj.obs.columns:
            adata_obj.obs[col] = adata_obj.obs[col].astype(str)

    if not adata_obj.obs_names.is_unique:
        adata_obj.obs_names_make_unique()
    if not adata_obj.var_names.is_unique:
        adata_obj.var_names_make_unique()

    # Preserve counts
    adata_obj.layers["counts"] = adata_obj.X.copy()

    # Basic preprocessing
    sc.pp.normalize_total(adata_obj, target_sum=1e4)
    sc.pp.log1p(adata_obj)

    sc.pp.highly_variable_genes(
        adata_obj,
        n_top_genes=args.n_top_genes,
        flavor="seurat",
        subset=False,
        inplace=True,
    )

    n_hvg = int(adata_obj.var["highly_variable"].sum()) if "highly_variable" in adata_obj.var.columns else 0
    use_hvg = n_hvg >= 200

    sc.pp.pca(
        adata_obj,
        n_comps=args.n_pcs,
        use_highly_variable=use_hvg,
        svd_solver="arpack",
    )
    sc.pp.neighbors(adata_obj, n_neighbors=args.n_neighbors, n_pcs=args.n_pcs, random_state=args.random_state)
    sc.tl.umap(adata_obj, random_state=args.random_state)
    sc.tl.leiden(adata_obj, resolution=args.leiden_resolution, random_state=args.random_state, key_added="leiden")

    # Marker / program scores
    score_meta = []
    for name, genes in MARKER_SETS.items():
        used = present_genes(adata_obj, genes)
        adata_obj.obs[f"score_{name}"] = mean_module_score_z(adata_obj, genes)
        score_meta.append(
            {
                "set_name": name,
                "n_requested": len(genes),
                "n_found": len(used),
                "genes_found": ",".join(used),
            }
        )

    score_meta_df = pd.DataFrame(score_meta)
    score_meta_df.to_csv(outdir / "gse227122_marker_sets_used.csv", index=False)

    # Preliminary cell labels
    adata_obj.obs["prelim_label"] = adata_obj.obs.apply(choose_prelim_label, axis=1)

    normal_score_cols = [f"score_{k}" for k in NORMAL_LINEAGES]
    adata_obj.obs["score_normal_best"] = adata_obj.obs[normal_score_cols].max(axis=1)
    adata_obj.obs["score_blast_margin"] = adata_obj.obs["score_Blast_TALL"] - adata_obj.obs["score_normal_best"]

    # Cluster summary
    summary_rows = []
    for cluster_id, sub in adata_obj.obs.groupby("leiden"):
        row = {
            "leiden": str(cluster_id),
            "n_cells": int(len(sub)),
            "top_sample_frac": cluster_top_fraction(sub["sample_id"]) if "sample_id" in sub.columns else np.nan,
            "dominant_prelim_label": sub["prelim_label"].astype(str).value_counts().idxmax(),
            "dominant_prelim_frac": float(sub["prelim_label"].astype(str).value_counts(normalize=True).iloc[0]),
        }

        if "timepoint" in sub.columns:
            tp = sub["timepoint"].astype(str).value_counts(normalize=True)
            row["dx_frac"] = float(tp.get("Dx", 0.0))
            row["eoi_frac"] = float(tp.get("EOI", 0.0))
            row["rel_frac"] = float(tp.get("Rel", 0.0))
            row["dxrel_frac"] = float(tp.get("Dx", 0.0) + tp.get("Rel", 0.0))
        else:
            row["dx_frac"] = np.nan
            row["eoi_frac"] = np.nan
            row["rel_frac"] = np.nan
            row["dxrel_frac"] = np.nan

        for set_name in MARKER_SETS.keys():
            row[f"median_score_{set_name}"] = float(np.nanmedian(sub[f"score_{set_name}"].values))

        # Lineage call from normal marker medians
        normal_medians = {
            k: row[f"median_score_{k}"]
            for k in NORMAL_LINEAGES
            if pd.notna(row[f"median_score_{k}"])
        }
        if len(normal_medians) == 0:
            lineage = "Unknown"
            row["median_score_normal_best"] = np.nan
            row["median_score_blast_margin"] = np.nan
        else:
            lineage = max(normal_medians, key=normal_medians.get)
            row["median_score_normal_best"] = max(normal_medians.values())
            if pd.notna(row["median_score_Blast_TALL"]):
                row["median_score_blast_margin"] = row["median_score_Blast_TALL"] - row["median_score_normal_best"]
            else:
                row["median_score_blast_margin"] = np.nan

        row["cluster_lineage_call"] = lineage

        # Malignant heuristic
        malignant = False
        if (
            row["median_score_Blast_TALL"] >= args.blast_z_min
            and row["median_score_blast_margin"] >= args.blast_margin
            and (
                (pd.notna(row["dxrel_frac"]) and row["dxrel_frac"] >= args.dxrel_min)
                or (pd.notna(row["top_sample_frac"]) and row["top_sample_frac"] >= args.top_sample_min)
            )
            and lineage not in {"B_cell", "Myeloid", "DC", "Erythroid"}
        ):
            malignant = True

        if (
            row["dominant_prelim_label"] == "Proliferating"
            and row["median_score_Blast_TALL"] >= (args.blast_z_min + 0.15)
            and row["cluster_lineage_call"] in {"T_cell", "HSPC", "NK"}
        ):
            malignant = True

        row["cluster_is_malignant"] = malignant

        # Final labels
        if malignant:
            fine = "Blast_TALL"
        else:
            fine = lineage

        if fine == "T_cell" and row["median_score_NK"] >= row["median_score_T_cell"] - 0.10:
            fine = "T_NK"

        if fine == "NK":
            fine = "T_NK"

        row["cluster_fine_annotation"] = fine
        row["Broad_Cellgroup"] = BROAD_MAP.get(fine, "Unknown")

        # Review flag
        review_flag = False
        if row["n_cells"] < args.min_cluster_size:
            review_flag = True
        if row["dominant_prelim_frac"] < 0.55:
            review_flag = True
        if abs(row["median_score_blast_margin"]) < 0.10:
            review_flag = True

        row["review_flag"] = review_flag
        summary_rows.append(row)

    cluster_summary = pd.DataFrame(summary_rows).sort_values("leiden").reset_index(drop=True)
    cluster_summary.to_csv(outdir / "gse227122_cluster_annotation_summary.csv", index=False)

    # Map cluster labels back to cells
    cluster_to_fine = cluster_summary.set_index("leiden")["cluster_fine_annotation"].to_dict()
    cluster_to_broad = cluster_summary.set_index("leiden")["Broad_Cellgroup"].to_dict()
    cluster_to_malignant = cluster_summary.set_index("leiden")["cluster_is_malignant"].to_dict()
    cluster_to_review = cluster_summary.set_index("leiden")["review_flag"].to_dict()

    adata_obj.obs["fine_annotation"] = adata_obj.obs["leiden"].astype(str).map(cluster_to_fine)
    adata_obj.obs["Broad_Cellgroup"] = adata_obj.obs["leiden"].astype(str).map(cluster_to_broad)
    adata_obj.obs["is_malignant"] = adata_obj.obs["leiden"].astype(str).map(cluster_to_malignant).fillna(False)
    adata_obj.obs["is_normal"] = ~adata_obj.obs["is_malignant"]
    adata_obj.obs["annotation_review_flag"] = adata_obj.obs["leiden"].astype(str).map(cluster_to_review).fillna(True)

    # Optional cluster marker table
    if not args.skip_rank_genes:
        sc.tl.rank_genes_groups(adata_obj, "leiden", method="wilcoxon")
        markers = extract_rank_genes_groups(adata_obj, "leiden", top_n=20)
        markers.to_csv(outdir / "gse227122_cluster_top_markers.csv", index=False)

    # Write annotation table
    keep_obs_cols = [
        c for c in [
            "sample_id", "barcode", "patient_id", "timepoint", "transfer_set", "mrd_status",
            "leiden", "prelim_label", "fine_annotation", "Broad_Cellgroup",
            "is_normal", "is_malignant", "annotation_review_flag",
            "score_T_cell", "score_NK", "score_B_cell", "score_Myeloid", "score_DC",
            "score_Erythroid", "score_HSPC", "score_Proliferating", "score_Blast_TALL",
            "score_normal_best", "score_blast_margin",
        ] if c in adata_obj.obs.columns
    ]
    ann = adata_obj.obs[keep_obs_cols].copy()
    ann.insert(0, "cell_id", adata_obj.obs_names.astype(str))
    ann.to_csv(outdir / "gse227122_cell_annotations.csv", index=False)

    # Save parameters + marker sets
    params = {
        "n_top_genes": args.n_top_genes,
        "n_pcs": args.n_pcs,
        "n_neighbors": args.n_neighbors,
        "leiden_resolution": args.leiden_resolution,
        "random_state": args.random_state,
        "blast_z_min": args.blast_z_min,
        "blast_margin": args.blast_margin,
        "dxrel_min": args.dxrel_min,
        "top_sample_min": args.top_sample_min,
        "min_cluster_size": args.min_cluster_size,
        "marker_sets": MARKER_SETS,
    }
    with open(outdir / "gse227122_annotation_parameters.json", "w") as f:
        json.dump(params, f, indent=2)

    # Store metadata in uns and write h5ad
    adata_obj.uns["annotation_parameters"] = params
    adata_obj.write_h5ad(outdir / "gse227122_qc_annotated.h5ad", compression="gzip")

    # UMAP plots
    sc.pl.umap(adata_obj, color="leiden", show=False)
    plt.savefig(outdir / "gse227122_umap_leiden.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Fine annotation: avoid on-data overlap by moving labels to right margin
    sc.pl.umap(
        adata_obj,
        color="fine_annotation",
        legend_loc="right margin",
        legend_fontsize=10,
        show=False
    )
    plt.savefig(outdir / "gse227122_umap_fine_annotation.png", dpi=300, bbox_inches="tight")
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    sc.pl.umap(
        adata_obj,
        color="Broad_Cellgroup",
        ax=axes[0],
        legend_loc="right margin",
        legend_fontsize=9,
        show=False
    )
    
    sc.pl.umap(
        adata_obj,
        color="is_malignant",
        ax=axes[1],
        legend_loc="right margin",
        legend_fontsize=9,
        show=False
    )
    
    axes[1].set_ylabel("")
    fig.subplots_adjust(wspace=0.55, right=0.95)
    
    plt.savefig(outdir / "gse227122_umap_broad_and_malignant.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("\n=== WRITTEN ===")
    print(outdir / "gse227122_qc_annotated.h5ad")
    print(outdir / "gse227122_cell_annotations.csv")
    print(outdir / "gse227122_cluster_annotation_summary.csv")
    print(outdir / "gse227122_marker_sets_used.csv")
    print(outdir / "gse227122_annotation_parameters.json")
    if not args.skip_rank_genes:
        print(outdir / "gse227122_cluster_top_markers.csv")

    print("\n=== DIAGNOSTICS ===")
    print("Cells:", adata_obj.n_obs)
    print("Genes:", adata_obj.n_vars)
    print("Clusters:", adata_obj.obs['leiden'].nunique())
    print("Putative malignant cells:", int(adata_obj.obs["is_malignant"].sum()))
    print("Putative normal cells:", int(adata_obj.obs["is_normal"].sum()))
    print("\nFine annotations:")
    print(adata_obj.obs["fine_annotation"].value_counts(dropna=False).to_string())
    print("\nBroad groups:")
    print(adata_obj.obs["Broad_Cellgroup"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
