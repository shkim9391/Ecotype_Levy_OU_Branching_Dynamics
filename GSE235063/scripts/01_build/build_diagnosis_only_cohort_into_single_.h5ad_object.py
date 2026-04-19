python3 - <<'PY'
from pathlib import Path
import gzip
import numpy as np
import pandas as pd
from scipy.io import mmread
from scipy import sparse
import anndata as ad

ROOT = Path("/GSE235063/GSE235063_RAW")
MANIFEST = ROOT / "primary_training_cohort_manifest_dx_processed.csv"
OUTDIR = ROOT.parent / "derived_dx_primary_training"
OUTDIR.mkdir(parents=True, exist_ok=True)

manifest = pd.read_csv(MANIFEST).sort_values(["sample", "gsm"]).reset_index(drop=True)

def read_mtx_gz(path):
    with gzip.open(path, "rb") as f:
        X = mmread(f)
    if sparse.isspmatrix(X):
        return X.tocsr()
    return sparse.csr_matrix(X)

def choose_barcode_col(df):
    for c in ["Cell_Barcode", "cell_barcode", "barcode", "Barcode"]:
        if c in df.columns:
            return c
    return df.columns[0]

adatas = []
sample_summaries = []
union_cols = set()
intersect_cols = None

for _, row in manifest.iterrows():
    gsm = row["gsm"]
    sample = row["sample"]
    tp = row["timepoint"]

    barcodes = pd.read_csv(
        ROOT / row["barcodes_file"],
        sep="\t",
        header=None,
        compression="gzip"
    )[0].astype(str)

    genes = pd.read_csv(
        ROOT / row["genes_file"],
        sep="\t",
        header=None,
        compression="gzip"
    )

    meta = pd.read_csv(
        ROOT / row["metadata_file"],
        sep="\t",
        compression="gzip"
    )

    bc_col = choose_barcode_col(meta)
    union_cols.update(meta.columns)
    intersect_cols = set(meta.columns) if intersect_cols is None else intersect_cols.intersection(meta.columns)

    X = read_mtx_gz(ROOT / row["matrix_file"])

    # 10x-style matrices are usually genes x cells; AnnData expects cells x genes
    if X.shape[1] == len(barcodes):
        X = X.T.tocsr()
    elif X.shape[0] == len(barcodes):
        X = X.tocsr()
    else:
        raise ValueError(
            f"{sample}: matrix shape {X.shape} is incompatible with {len(barcodes)} barcodes"
        )

    # genes.tsv can have 1, 2, or 3 columns depending on the export
    if genes.shape[1] == 1:
        gene_id = genes.iloc[:, 0].astype(str).values
        gene_symbol = genes.iloc[:, 0].astype(str).values
        feature_type = np.array(["Gene Expression"] * len(gene_id), dtype=object)
    elif genes.shape[1] == 2:
        gene_id = genes.iloc[:, 0].astype(str).values
        gene_symbol = genes.iloc[:, 1].astype(str).values
        feature_type = np.array(["Gene Expression"] * len(gene_id), dtype=object)
    else:
        gene_id = genes.iloc[:, 0].astype(str).values
        gene_symbol = genes.iloc[:, 1].astype(str).values
        feature_type = genes.iloc[:, 2].astype(str).values

    var = pd.DataFrame(
        {
            "gene_id": gene_id,
            "gene_symbol": gene_symbol,
            "feature_type": feature_type,
        },
        index=pd.Index(gene_id, name="gene_id"),
    )
    var = var[~var.index.duplicated(keep="first")]

    if X.shape[1] != var.shape[0]:
        raise ValueError(
            f"{sample}: matrix has {X.shape[1]} genes but genes.tsv has {var.shape[0]}"
        )

    meta[bc_col] = meta[bc_col].astype(str)
    meta = meta.drop_duplicates(subset=[bc_col], keep="first").set_index(bc_col)

    obs = pd.DataFrame(index=pd.Index(barcodes.values, name="barcode"))
    obs = obs.join(meta, how="left")

    missing_meta = int(obs.isna().all(axis=1).sum())

    obs["gsm"] = gsm
    obs["sample"] = sample
    obs["timepoint"] = tp
    obs["sample_id"] = f"{sample}_{tp}"
    obs["barcode_raw"] = obs.index.astype(str)

    # make cell names unique across samples
    obs.index = obs["sample_id"] + ":" + obs["barcode_raw"]

    adata = ad.AnnData(X=X, obs=obs, var=var)
    adatas.append(adata)

    sample_summaries.append(
        {
            "gsm": gsm,
            "sample": sample,
            "timepoint": tp,
            "n_cells": adata.n_obs,
            "n_genes": adata.n_vars,
            "metadata_rows": meta.shape[0],
            "cells_missing_metadata": missing_meta,
            "matrix_nnz": int(adata.X.nnz),
        }
    )

cohort = ad.concat(adatas, axis=0, join="outer", merge="same", index_unique=None)

for col in ["sample", "timepoint", "sample_id"]:
    if col in cohort.obs.columns:
        cohort.obs[col] = cohort.obs[col].astype("category")

summary_df = pd.DataFrame(sample_summaries).sort_values(["sample", "gsm"])
summary_df.to_csv(OUTDIR / "dx_primary_training_sample_summary.csv", index=False)

with open(OUTDIR / "dx_metadata_columns_union.txt", "w") as f:
    for c in sorted(union_cols):
        f.write(f"{c}\n")

with open(OUTDIR / "dx_metadata_columns_intersection.txt", "w") as f:
    for c in sorted(intersect_cols):
        f.write(f"{c}\n")

cohort.write_h5ad(OUTDIR / "dx_primary_training_allcells.h5ad", compression="gzip")

# Optional: save a labeled-cell subset if that column exists
if "Classified_Celltype" in cohort.obs.columns:
    keep = cohort.obs["Classified_Celltype"].notna() & (
        cohort.obs["Classified_Celltype"].astype(str).str.strip() != ""
    )
    cohort_labeled = cohort[keep].copy()
    cohort_labeled.write_h5ad(
        OUTDIR / "dx_primary_training_labeledcells.h5ad",
        compression="gzip"
    )
    labeled_n = cohort_labeled.n_obs
else:
    labeled_n = None

print("\n=== DX PRIMARY TRAINING COHORT BUILT ===")
print("Samples:", cohort.obs["sample"].nunique())
print("Cells:", cohort.n_obs)
print("Genes:", cohort.n_vars)
print("Output directory:", OUTDIR)

print("\n=== SAMPLE SUMMARY (first 10) ===")
print(summary_df.head(10).to_string(index=False))

print("\n=== FIRST OBS COLUMNS ===")
print(list(cohort.obs.columns[:30]))

if labeled_n is not None:
    print("\nSaved labeled-cell subset with", labeled_n, "cells")
PY
