from __future__ import annotations

from pathlib import Path
import gzip
import warnings

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from scipy.io import mmread
from scipy import sparse


# ============================================================
# 1. CONFIG
# ============================================================
RAW_DIR = Path("/Ecotype_OU_Branching/GSE235063/GSE235063_RAW")
MANIFEST_CSV = RAW_DIR / "longitudinal_cohort_manifest_processed.csv"

PROJECT_DIR = Path("/Multimodal_OU_Lévy_Branching_Scaffold/Figure_3")
DERIVED_DIR = PROJECT_DIR / "derived"
OUT_H5AD = DERIVED_DIR / "gse235063_longitudinal_malignant_raw.h5ad"
OUT_QC = DERIVED_DIR / "gse235063_longitudinal_malignant_sample_qc.tsv"

REQUIRED_MANIFEST_COLS = [
    "gsm",
    "sample_id",
    "patient_id",
    "clinical_timepoint_raw",
    "clinical_timepoint_coarse",
    "barcodes_file",
    "genes_file",
    "matrix_file",
    "metadata_file",
]

VALID_TIMEPOINTS = {"DX", "EOI_REM", "REL"}

PRINT_FIRST_METADATA_SCHEMA = False

FORCE_MALIGNANT_COLUMN: str | None = "Malignant"
FORCE_MALIGNANT_POSITIVE_VALUES: set[str] | None = {"Malignant"}


# ============================================================
# 2. HELPERS
# ============================================================
def ensure_dirs() -> None:
    DERIVED_DIR.mkdir(parents=True, exist_ok=True)


def validate_manifest(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_MANIFEST_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Manifest missing required columns: {missing}")

    bad = sorted(set(df["clinical_timepoint_coarse"]) - VALID_TIMEPOINTS)
    if bad:
        raise ValueError(
            f"clinical_timepoint_coarse contains invalid labels: {bad}. "
            f"Expected subset of {sorted(VALID_TIMEPOINTS)}"
        )


def read_barcodes(fp: Path) -> list[str]:
    df = pd.read_csv(fp, sep="\t", header=None, compression="gzip")
    return df.iloc[:, 0].astype(str).tolist()


def read_genes(fp: Path) -> pd.DataFrame:
    df = pd.read_csv(fp, sep="\t", header=None, compression="gzip")
    if df.shape[1] >= 2:
        gene_ids = df.iloc[:, 0].astype(str).tolist()
        gene_names = df.iloc[:, 1].astype(str).tolist()
    else:
        gene_ids = df.iloc[:, 0].astype(str).tolist()
        gene_names = df.iloc[:, 0].astype(str).tolist()

    out = pd.DataFrame({"gene_id": gene_ids, "gene_name": gene_names})
    return out


def read_matrix(fp: Path) -> sparse.csr_matrix:
    with gzip.open(fp, "rb") as fh:
        m = mmread(fh)

    if not sparse.issparse(m):
        m = sparse.csr_matrix(m)
    else:
        m = m.tocsr()

    return m


def read_metadata(fp: Path) -> pd.DataFrame:
    return pd.read_csv(fp, sep="\t", compression="gzip")


def detect_barcode_column(meta: pd.DataFrame, barcodes: list[str]) -> str | None:
    barcode_set = set(barcodes)
    candidates = []

    for c in meta.columns:
        c_low = str(c).lower()
        if any(tok in c_low for tok in ["barcode", "cell", "cell_id", "cellid"]):
            candidates.append(c)

    # Try explicit candidates first
    for c in candidates:
        vals = meta[c].astype(str)
        overlap = vals.isin(barcode_set).mean()
        if overlap > 0.5:
            return c

    # Then try any column heuristically
    for c in meta.columns:
        vals = meta[c].astype(str)
        overlap = vals.isin(barcode_set).mean()
        if overlap > 0.5:
            return c

    return None


def align_metadata_to_barcodes(meta: pd.DataFrame, barcodes: list[str]) -> pd.DataFrame:
    bc_col = detect_barcode_column(meta, barcodes)

    if bc_col is not None:
        meta = meta.copy()
        meta[bc_col] = meta[bc_col].astype(str)
        meta = meta.drop_duplicates(subset=[bc_col], keep="first").set_index(bc_col)
        meta = meta.reindex(barcodes)
    else:
        if meta.shape[0] != len(barcodes):
            raise ValueError(
                "Could not infer barcode column in metadata, and metadata row count "
                f"({meta.shape[0]}) does not match number of barcodes ({len(barcodes)})."
            )
        meta = meta.copy()
        meta.index = pd.Index(barcodes, name="barcode")

    meta.index = meta.index.astype(str)
    return meta


def make_var_names_unique(df: pd.DataFrame) -> pd.DataFrame:
    gene_names = pd.Index(df["gene_name"].astype(str))
    seen: dict[str, int] = {}
    out = []

    for g in gene_names:
        if g not in seen:
            seen[g] = 0
            out.append(g)
        else:
            seen[g] += 1
            out.append(f"{g}-{seen[g]}")

    df = df.copy()
    df["gene_name_unique"] = out
    return df


def build_adata_from_row(row: pd.Series) -> ad.AnnData:
    bc_fp = RAW_DIR / str(row["barcodes_file"])
    gn_fp = RAW_DIR / str(row["genes_file"])
    mt_fp = RAW_DIR / str(row["matrix_file"])
    md_fp = RAW_DIR / str(row["metadata_file"])

    for fp in [bc_fp, gn_fp, mt_fp, md_fp]:
        if not fp.exists():
            raise FileNotFoundError(fp)

    barcodes = read_barcodes(bc_fp)
    genes = make_var_names_unique(read_genes(gn_fp))
    matrix = read_matrix(mt_fp)
    meta = align_metadata_to_barcodes(read_metadata(md_fp), barcodes)

    # Most GEO 10x matrices are genes x cells; transpose to cells x genes
    if matrix.shape == (len(genes), len(barcodes)):
        X = matrix.T.tocsr()
    elif matrix.shape == (len(barcodes), len(genes)):
        X = matrix.tocsr()
    else:
        raise ValueError(
            f"Matrix dimensions {matrix.shape} do not match genes={len(genes)} and barcodes={len(barcodes)}"
        )

    obs = meta.copy()
    obs["barcode"] = obs.index.astype(str)
    obs["cell_id"] = [f"{row['sample_id']}::{bc}" for bc in obs["barcode"]]
    obs.index = pd.Index(obs["cell_id"].astype(str), name="cell_id")

    if "Patient_ID" in obs.columns:
        obs["patient_id_internal"] = obs["Patient_ID"].astype(str)

    if "Library_ID" in obs.columns:
        obs["library_id_internal"] = obs["Library_ID"].astype(str)

    if "Subgroup" in obs.columns:
        obs["disease_subgroup"] = obs["Subgroup"].astype(str)
    else:
        obs["disease_subgroup"] = "AML"

    obs["gsm"] = str(row["gsm"])
    obs["sample_id"] = str(row["sample_id"])
    obs["patient_id"] = str(row["patient_id"])
    obs["clinical_timepoint_raw"] = str(row["clinical_timepoint_raw"])
    obs["clinical_timepoint_coarse"] = str(row["clinical_timepoint_coarse"])

    var = genes.copy()
    var.index = pd.Index(var["gene_name_unique"].astype(str), name="gene_name_unique")

    adata = ad.AnnData(X=X, obs=obs, var=var)
    adata.var["gene_id"] = var["gene_id"].values
    adata.var["gene_name"] = var["gene_name"].values

    return adata


def infer_malignant_mask_from_boolean(meta: pd.DataFrame) -> tuple[pd.Series | None, str | None]:
    for c in meta.columns:
        c_low = str(c).lower()
        if not any(tok in c_low for tok in ["malig", "leuk", "blast", "tumor"]):
            continue

        s = meta[c]
        if pd.api.types.is_bool_dtype(s):
            return s.fillna(False).astype(bool), f"boolean column: {c}"

        # numeric 0/1-like
        vals = pd.to_numeric(s, errors="coerce")
        uniq = set(vals.dropna().unique().tolist())
        if uniq and uniq.issubset({0, 1}):
            return vals.fillna(0).astype(int).astype(bool), f"0/1 numeric column: {c}"

    return None, None


def infer_malignant_mask_from_text(meta: pd.DataFrame) -> tuple[pd.Series | None, str | None]:
    positive_patterns = [
        "malignant",
        "leuk",
        "blast",
        "aml blast",
        "tumor",
    ]

    malignant_state_values = {
        "HSC", "Prog", "GMP", "MonoDC", "Mono/DC",
        "HSC-like", "Prog-like", "GMP-like", "Mono/DC-like",
        "Progenitor-like", "Progenitor"
    }

    candidate_cols = []
    for c in meta.columns:
        c_low = str(c).lower()
        if any(tok in c_low for tok in ["annotation", "cell_type", "celltype", "class", "state", "label", "cluster"]):
            candidate_cols.append(c)

    # first pass: direct malignant keywords
    for c in candidate_cols:
        s = meta[c].astype(str)
        s_low = s.str.lower()

        mask = pd.Series(False, index=meta.index)
        for patt in positive_patterns:
            mask = mask | s_low.str.contains(patt, na=False)

        if mask.sum() > 0:
            return mask, f"text-matched malignant keywords in column: {c}"

    # second pass: explicit malignant-state labels
    for c in candidate_cols:
        s = meta[c].astype(str)
        mask = s.isin(malignant_state_values)
        if mask.sum() > 0:
            return mask, f"state-label malignant mapping in column: {c}"

    return None, None


def infer_malignant_mask(meta: pd.DataFrame) -> tuple[pd.Series, str]:
    if FORCE_MALIGNANT_COLUMN is not None:
        if FORCE_MALIGNANT_COLUMN not in meta.columns:
            raise ValueError(f"FORCE_MALIGNANT_COLUMN={FORCE_MALIGNANT_COLUMN} not found in metadata.")

        s = meta[FORCE_MALIGNANT_COLUMN]

        if FORCE_MALIGNANT_POSITIVE_VALUES is not None:
            mask = s.astype(str).isin(FORCE_MALIGNANT_POSITIVE_VALUES)
            return mask, f"forced categorical mapping from {FORCE_MALIGNANT_COLUMN}"

        if pd.api.types.is_bool_dtype(s):
            return s.fillna(False).astype(bool), f"forced boolean column {FORCE_MALIGNANT_COLUMN}"

        vals = pd.to_numeric(s, errors="coerce")
        uniq = set(vals.dropna().unique().tolist())
        if uniq.issubset({0, 1}):
            return vals.fillna(0).astype(int).astype(bool), f"forced numeric column {FORCE_MALIGNANT_COLUMN}"

        raise ValueError(
            "FORCE_MALIGNANT_COLUMN was set, but no FORCE_MALIGNANT_POSITIVE_VALUES were provided "
            "and the column is not boolean/0-1."
        )

    mask, info = infer_malignant_mask_from_boolean(meta)
    if mask is not None:
        return mask, info

    mask, info = infer_malignant_mask_from_text(meta)
    if mask is not None:
        return mask, info

    raise ValueError(
        "Could not infer malignant cells from metadata automatically. "
        "Inspect metadata columns and then set FORCE_MALIGNANT_COLUMN "
        "and optionally FORCE_MALIGNANT_POSITIVE_VALUES in the script."
    )


def subset_malignant_cells(adata: ad.AnnData) -> tuple[ad.AnnData, str, int, int]:
    n_before = adata.n_obs
    mask, info = infer_malignant_mask(adata.obs)

    mask = pd.Series(mask, index=adata.obs_names).fillna(False).astype(bool)
    adata_sub = adata[mask.values].copy()
    n_after = adata_sub.n_obs

    return adata_sub, info, n_before, n_after


def maybe_print_metadata_schema(row: pd.Series) -> None:
    if not PRINT_FIRST_METADATA_SCHEMA:
        return
    md_fp = RAW_DIR / str(row["metadata_file"])
    meta = read_metadata(md_fp)
    print("\n[INFO] First metadata file preview:")
    print(f"  file: {md_fp.name}")
    print(f"  columns: {meta.columns.tolist()[:40]}")
    print(meta.head(3).to_string(index=False))


# ============================================================
# 3. MAIN
# ============================================================
def main() -> None:
    ensure_dirs()

    manifest = pd.read_csv(MANIFEST_CSV)
    validate_manifest(manifest)

    if manifest.empty:
        raise ValueError("Manifest is empty.")

    maybe_print_metadata_schema(manifest.iloc[0])

    adatas: list[ad.AnnData] = []
    qc_rows: list[dict] = []

    for i, row in manifest.iterrows():
        print(f"[INFO] Loading {i+1}/{len(manifest)}: {row['sample_id']}")

        adata = build_adata_from_row(row)
        adata_mal, filter_info, n_before, n_after = subset_malignant_cells(adata)

        qc_rows.append({
            "gsm": row["gsm"],
            "sample_id": row["sample_id"],
            "patient_id": row["patient_id"],
            "clinical_timepoint_raw": row["clinical_timepoint_raw"],
            "clinical_timepoint_coarse": row["clinical_timepoint_coarse"],
            "n_cells_before_filter": n_before,
            "n_cells_after_filter": n_after,
            "malignant_filter_source": filter_info,
            "n_genes": adata.n_vars,
            "kept_in_malignant_object": int(n_after > 0),
        })

        if n_after == 0:
            print(f"[WARN] Skipping {row['sample_id']}: zero malignant cells after filtering ({filter_info})")
            continue

        adatas.append(adata_mal)

    if len(adatas) == 0:
        raise ValueError("No samples with malignant cells were retained.")

    combined = ad.concat(
        adatas,
        join="outer",
        merge="same",
        label="source_sample",
        keys=[a.obs["sample_id"].iloc[0] for a in adatas],
        index_unique=None,
    )

    combined.uns["timepoint_order"] = ["DX", "EOI_REM", "REL"]
    combined.uns["figure3_build_note"] = (
        "Built from longitudinal_cohort_manifest_processed.csv and filtered to malignant cells."
    )

    qc = pd.DataFrame(qc_rows)

    print("\n[SUMMARY: samples kept vs skipped]")
    print(qc.groupby(["clinical_timepoint_coarse", "kept_in_malignant_object"]).size())

    qc.to_csv(OUT_QC, sep="\t", index=False)
    combined.write_h5ad(OUT_H5AD)

    print(f"\n[DONE] Saved main object: {OUT_H5AD}")
    print(f"[DONE] Saved QC table:    {OUT_QC}")
    print("\n[SUMMARY: malignant cells by timepoint]")
    print(qc.groupby("clinical_timepoint_coarse")["n_cells_after_filter"].sum())

if __name__ == "__main__":
    main()
