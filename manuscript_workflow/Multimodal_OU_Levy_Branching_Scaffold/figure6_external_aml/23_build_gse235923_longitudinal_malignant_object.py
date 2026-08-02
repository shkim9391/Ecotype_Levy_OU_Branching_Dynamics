from __future__ import annotations

from pathlib import Path
import gzip

import numpy as np
import pandas as pd
import anndata as ad
from scipy.io import mmread
from scipy import sparse


# ============================================================
# 1. CONFIG
# ============================================================
MANIFEST_CSV = Path("/Ecotype_OU_Branching/GSE235923/GSE235923_RAW/gse235923_manifest_all.csv")
RAW_DIR = MANIFEST_CSV.parent

PROJECT_DIR = Path("/Multimodal_OU_Lévy_Branching_Scaffold/Figure_6")
DERIVED_DIR = PROJECT_DIR / "derived"

OUT_H5AD = DERIVED_DIR / "gse235923_longitudinal_allcells_raw.h5ad"
OUT_QC = DERIVED_DIR / "gse235923_longitudinal_sample_qc.tsv"
OUT_MANIFEST = DERIVED_DIR / "gse235923_manifest_harmonized.tsv"

REQUIRED_MANIFEST_COLS = [
    "gsm",
    "sample_base",
    "timepoint",
    "sample_id",
    "barcodes_file",
    "features_file",
    "matrix_file",
]

TIMEPOINT_MAP = {
    "DX": "DX",
    "EOI": "EOI_REM",
    "REL": "REL",
}


# ============================================================
# 2. HELPERS
# ============================================================
def ensure_dirs() -> None:
    DERIVED_DIR.mkdir(parents=True, exist_ok=True)


def validate_manifest(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_MANIFEST_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Manifest missing required columns: {missing}")

    bad = sorted(set(df["timepoint"].astype(str)) - set(TIMEPOINT_MAP.keys()))
    if bad:
        raise ValueError(
            f"Manifest contains unsupported timepoint values: {bad}. "
            f"Expected subset of {sorted(TIMEPOINT_MAP.keys())}."
        )


def read_barcodes(fp: Path) -> list[str]:
    df = pd.read_csv(fp, sep="\t", header=None, compression="gzip")
    return df.iloc[:, 0].astype(str).tolist()


def read_features(fp: Path) -> pd.DataFrame:
    df = pd.read_csv(fp, sep="\t", header=None, compression="gzip")

    if df.shape[1] >= 2:
        gene_ids = df.iloc[:, 0].astype(str).tolist()
        gene_names = df.iloc[:, 1].astype(str).tolist()
    else:
        gene_ids = df.iloc[:, 0].astype(str).tolist()
        gene_names = df.iloc[:, 0].astype(str).tolist()

    out = pd.DataFrame({
        "gene_id": gene_ids,
        "gene_name": gene_names,
    })
    return out


def make_var_names_unique(df: pd.DataFrame) -> pd.DataFrame:
    names = pd.Index(df["gene_name"].astype(str))
    seen: dict[str, int] = {}
    unique = []

    for g in names:
        if g not in seen:
            seen[g] = 0
            unique.append(g)
        else:
            seen[g] += 1
            unique.append(f"{g}-{seen[g]}")

    out = df.copy()
    out["gene_name_unique"] = unique
    return out


def read_matrix(fp: Path) -> sparse.csr_matrix:
    with gzip.open(fp, "rb") as fh:
        m = mmread(fh)

    if not sparse.issparse(m):
        m = sparse.csr_matrix(m)
    else:
        m = m.tocsr()

    return m


def build_adata_from_row(row: pd.Series) -> ad.AnnData:
    bc_fp = RAW_DIR / str(row["barcodes_file"])
    ft_fp = RAW_DIR / str(row["features_file"])
    mx_fp = RAW_DIR / str(row["matrix_file"])

    for fp in [bc_fp, ft_fp, mx_fp]:
        if not fp.exists():
            raise FileNotFoundError(fp)

    barcodes = read_barcodes(bc_fp)
    features = make_var_names_unique(read_features(ft_fp))
    matrix = read_matrix(mx_fp)

    # Most GEO 10x matrices are genes x cells; transpose to cells x genes
    if matrix.shape == (len(features), len(barcodes)):
        X = matrix.T.tocsr()
    elif matrix.shape == (len(barcodes), len(features)):
        X = matrix.tocsr()
    else:
        raise ValueError(
            f"Matrix dimensions {matrix.shape} do not match "
            f"features={len(features)} and barcodes={len(barcodes)}"
        )

    obs = pd.DataFrame(index=pd.Index(barcodes, name="barcode"))
    obs["barcode_raw"] = obs.index.astype(str)
    obs["cell_id"] = [f"{row['sample_id']}::{bc}" for bc in obs["barcode_raw"]]
    obs.index = pd.Index(obs["cell_id"].astype(str), name="cell_id")

    obs["gsm"] = str(row["gsm"])
    obs["sample_base"] = str(row["sample_base"])
    obs["patient_id"] = str(row["sample_base"])
    obs["sample_id"] = str(row["sample_id"])
    obs["clinical_timepoint_raw"] = str(row["timepoint"])
    obs["clinical_timepoint_coarse"] = TIMEPOINT_MAP[str(row["timepoint"])]

    # Placeholders to be filled later in Step 24
    obs["is_malignant_known"] = False
    obs["is_malignant"] = "unknown"
    obs["Classified_Celltype"] = "Unlabeled"

    var = features.copy()
    var.index = pd.Index(var["gene_name_unique"].astype(str), name="gene_name_unique")

    adata = ad.AnnData(X=X, obs=obs, var=var)
    adata.var["gene_id"] = var["gene_id"].values
    adata.var["gene_name"] = var["gene_name"].values

    return adata

def sanitize_obs_for_h5ad(obs: pd.DataFrame) -> pd.DataFrame:
    obs = obs.copy()
    for c in obs.columns:
        if pd.api.types.is_object_dtype(obs[c]) or pd.api.types.is_string_dtype(obs[c]):
            obs[c] = obs[c].astype("string").fillna("NA").astype(str)
    return obs

# ============================================================
# 3. MAIN
# ============================================================
def main() -> None:
    ensure_dirs()

    manifest = pd.read_csv(MANIFEST_CSV)
    validate_manifest(manifest)

    manifest = manifest.copy()
    manifest["clinical_timepoint_raw"] = manifest["timepoint"].astype(str)
    manifest["clinical_timepoint_coarse"] = manifest["timepoint"].map(TIMEPOINT_MAP)
    manifest["patient_id"] = manifest["sample_base"].astype(str)

    # Save harmonized manifest
    manifest.to_csv(OUT_MANIFEST, sep="\t", index=False)

    qc_rows = []
    adatas = []

    for i, row in manifest.iterrows():
        print(f"[INFO] Loading {i+1}/{len(manifest)}: {row['sample_id']}")

        adata = build_adata_from_row(row)

        qc_rows.append({
            "gsm": row["gsm"],
            "patient_id": row["patient_id"],
            "sample_id": row["sample_id"],
            "clinical_timepoint_raw": row["clinical_timepoint_raw"],
            "clinical_timepoint_coarse": row["clinical_timepoint_coarse"],
            "n_cells": adata.n_obs,
            "n_genes": adata.n_vars,
        })

        adatas.append(adata)

    combined = ad.concat(
        adatas,
        join="outer",
        merge="same",
        label="source_sample",
        keys=[a.obs["sample_id"].iloc[0] for a in adatas],
        index_unique=None,
    )

    combined.uns["timepoint_order"] = ["DX", "EOI_REM", "REL"]
    combined.uns["external_cohort"] = "GSE235923"
    combined.uns["build_note"] = (
        "All-cells longitudinal query object built from gse235923_manifest_all.csv. "
        "Malignant filtering and label transfer are deferred to Step 24 because "
        "the manifest does not include per-cell metadata."
    )

    qc = pd.DataFrame(qc_rows)
    qc.to_csv(OUT_QC, sep="\t", index=False)
    combined.obs = sanitize_obs_for_h5ad(combined.obs)
    combined.write_h5ad(OUT_H5AD)

    print(f"\n[DONE] Saved main object: {OUT_H5AD}")
    print(f"[DONE] Saved QC table:    {OUT_QC}")
    print(f"[DONE] Saved manifest:     {OUT_MANIFEST}")

    print("\n[SUMMARY: sample counts by phase]")
    print(qc["clinical_timepoint_coarse"].value_counts(dropna=False).sort_index())

    print("\n[SUMMARY: cells by phase]")
    print(
        qc.groupby("clinical_timepoint_coarse")["n_cells"]
          .sum()
          .sort_index()
    )

    print("\n[SUMMARY: patient longitudinal structure]")
    seq = (
        manifest.groupby("patient_id")["clinical_timepoint_coarse"]
        .apply(list)
        .reset_index(name="timepoints")
    )
    print(seq.to_string(index=False))


if __name__ == "__main__":
    main()
