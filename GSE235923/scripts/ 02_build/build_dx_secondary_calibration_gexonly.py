python - <<'PY'
from pathlib import Path
import gzip
import numpy as np
import pandas as pd
from scipy.io import mmread
from scipy import sparse
import anndata as ad

ROOT = Path("/GSE235923/GSE235923_RAW")
MANIFEST = ROOT / "gse235923_manifest_dx.csv"
OUTDIR = ROOT.parent / "derived_secondary_calibration"
OUTDIR.mkdir(parents=True, exist_ok=True)

manifest = pd.read_csv(MANIFEST).sort_values(["sample_base", "gsm"]).reset_index(drop=True)

def read_mtx_gz(path):
    with gzip.open(path, "rb") as f:
        X = mmread(f)
    if sparse.isspmatrix(X):
        return X.tocsr()
    return sparse.csr_matrix(X)

adatas = []
sample_rows = []

for _, row in manifest.iterrows():
    gsm = row["gsm"]
    sample_base = row["sample_base"]
    timepoint = row["timepoint"]
    sample_id = row["sample_id"]

    barcodes = pd.read_csv(
        ROOT / row["barcodes_file"],
        sep="\t",
        header=None,
        compression="gzip"
    )[0].astype(str)

    features = pd.read_csv(
        ROOT / row["features_file"],
        sep="\t",
        header=None,
        compression="gzip"
    )

    X = read_mtx_gz(ROOT / row["matrix_file"])

    # Convert to cells x genes if needed
    if X.shape[1] == len(barcodes):
        X = X.T.tocsr()
    elif X.shape[0] == len(barcodes):
        X = X.tocsr()
    else:
        raise ValueError(f"{sample_id}: matrix shape {X.shape} incompatible with {len(barcodes)} barcodes")

    if features.shape[1] == 1:
        gene_id = features.iloc[:, 0].astype(str).values
        gene_symbol = features.iloc[:, 0].astype(str).values
        feature_type = np.array(["Gene Expression"] * len(gene_id), dtype=object)
    elif features.shape[1] == 2:
        gene_id = features.iloc[:, 0].astype(str).values
        gene_symbol = features.iloc[:, 1].astype(str).values
        feature_type = np.array(["Gene Expression"] * len(gene_id), dtype=object)
    else:
        gene_id = features.iloc[:, 0].astype(str).values
        gene_symbol = features.iloc[:, 1].astype(str).values
        feature_type = features.iloc[:, 2].astype(str).values

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
        raise ValueError(f"{sample_id}: matrix has {X.shape[1]} genes but features.tsv has {var.shape[0]}")

    obs = pd.DataFrame(index=pd.Index(barcodes.values, name="barcode"))
    obs["gsm"] = gsm
    obs["sample_base"] = sample_base
    obs["timepoint"] = timepoint
    obs["sample_id"] = sample_id
    obs["barcode_raw"] = obs.index.astype(str)
    obs.index = obs["sample_id"] + ":" + obs["barcode_raw"]

    adata = ad.AnnData(X=X, obs=obs, var=var)
    adatas.append(adata)

    sample_rows.append({
        "gsm": gsm,
        "sample_base": sample_base,
        "timepoint": timepoint,
        "sample_id": sample_id,
        "n_cells": adata.n_obs,
        "n_genes": adata.n_vars,
        "matrix_nnz": int(adata.X.nnz),
    })

cohort = ad.concat(adatas, axis=0, join="outer", merge="same", index_unique=None)

for col in ["sample_base", "timepoint", "sample_id"]:
    cohort.obs[col] = cohort.obs[col].astype("category")

summary = pd.DataFrame(sample_rows).sort_values(["sample_base", "gsm"]).reset_index(drop=True)

summary_csv = OUTDIR / "gse235923_dx_secondary_calibration_sample_summary.csv"
h5ad_path = OUTDIR / "gse235923_dx_secondary_calibration_unlabeled.h5ad"

summary.to_csv(summary_csv, index=False)
cohort.write_h5ad(h5ad_path, compression="gzip")

print("\n=== GSE235923 DX SECONDARY CALIBRATION COHORT BUILT ===")
print("Samples:", cohort.obs["sample_id"].nunique())
print("Cells:", cohort.n_obs)
print("Genes:", cohort.n_vars)
print("Output:", h5ad_path)

print("\n=== SAMPLE SUMMARY (first 20) ===")
print(summary.head(20).to_string(index=False))

print("\nSaved:")
print(summary_csv)
print(h5ad_path)
PY
