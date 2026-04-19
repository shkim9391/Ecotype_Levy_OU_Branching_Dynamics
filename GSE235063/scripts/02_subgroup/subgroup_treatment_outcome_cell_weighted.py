python - <<'PY'
import anndata as ad
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path("/GSE235063/derived_dx_primary_training")
adata = ad.read_h5ad(ROOT / "dx_primary_training_allcells.h5ad")

# -----------------------------
# Standardize key columns
# -----------------------------
for col in ["Malignant", "Classified_Celltype", "sample", "sample_id",
            "Patient_ID", "Subgroup", "Expected_Driving_Aberration",
            "Biopsy_Origin", "Treatment_Outcome", "gsm"]:
    if col in adata.obs.columns:
        adata.obs[col] = adata.obs[col].astype(str)

adata.obs["is_malignant"] = adata.obs["Malignant"].eq("Malignant")
adata.obs["is_normal"] = adata.obs["Malignant"].eq("Normal")

# -----------------------------
# Save core subsets
# -----------------------------
adata_mal = adata[adata.obs["is_malignant"]].copy()
adata_norm = adata[adata.obs["is_normal"]].copy()

adata_mal.write_h5ad(ROOT / "dx_primary_training_malignant.h5ad", compression="gzip")
adata_norm.write_h5ad(ROOT / "dx_primary_training_normal.h5ad", compression="gzip")

# -----------------------------
# Sample-level summary table
# -----------------------------
meta_cols = [c for c in [
    "gsm", "sample", "sample_id", "Patient_ID", "Biopsy_Origin",
    "Subgroup", "Expected_Driving_Aberration", "Treatment_Outcome"
] if c in adata.obs.columns]

sample_meta = (
    adata.obs[meta_cols]
    .drop_duplicates(subset=["sample_id"])
    .set_index("sample_id")
)

sample_counts = pd.DataFrame({
    "total_cells": adata.obs.groupby("sample_id").size(),
    "malignant_cells": adata.obs.groupby("sample_id")["is_malignant"].sum(),
    "normal_cells": adata.obs.groupby("sample_id")["is_normal"].sum(),
})

sample_counts["malignant_frac"] = sample_counts["malignant_cells"] / sample_counts["total_cells"]
sample_counts["normal_frac"] = sample_counts["normal_cells"] / sample_counts["total_cells"]

sample_summary = sample_meta.join(sample_counts).reset_index()
sample_summary = sample_summary.sort_values(["sample", "gsm"])
sample_summary.to_csv(ROOT / "dx_primary_training_sample_level_summary.csv", index=False)

# -----------------------------
# Fine-grained cell-type composition
# -----------------------------
ct_all_counts = pd.crosstab(adata.obs["sample_id"], adata.obs["Classified_Celltype"])
ct_all_frac = ct_all_counts.div(ct_all_counts.sum(axis=1), axis=0).fillna(0)

ct_norm_counts = pd.crosstab(adata_norm.obs["sample_id"], adata_norm.obs["Classified_Celltype"])
ct_norm_frac = ct_norm_counts.div(ct_norm_counts.sum(axis=1), axis=0).fillna(0)

ct_all_counts.to_csv(ROOT / "dx_allcells_celltype_counts_by_sample.csv")
ct_all_frac.to_csv(ROOT / "dx_allcells_celltype_fractions_by_sample.csv")
ct_norm_counts.to_csv(ROOT / "dx_normal_celltype_counts_by_sample.csv")
ct_norm_frac.to_csv(ROOT / "dx_normal_celltype_fractions_by_sample.csv")

# -----------------------------
# Broad normal-cell ecotype matrix
# -----------------------------
broad_map = {
    "CD4.Naive": "T_NK",
    "CD4.Memory": "T_NK",
    "CD8.Naive": "T_NK",
    "CD8.Memory": "T_NK",
    "CD8.Effector": "T_NK",
    "NK": "T_NK",

    "B.Cell": "B_Plasma",
    "Pre.B.Cell": "B_Plasma",
    "Plasma": "B_Plasma",

    "Monocytes": "Myeloid_APC",
    "CD16.Monocytes": "Myeloid_APC",
    "cDC": "Myeloid_APC",
    "pDC": "Myeloid_APC",

    "HSC": "HSPC_Prog",
    "CLP": "HSPC_Prog",
    "GMP": "HSPC_Prog",
    "Progenitor": "HSPC_Prog",

    "Early.Erythrocyte": "Erythroid_Baso",
    "Late.Erythrocyte": "Erythroid_Baso",
    "Early.Basophil": "Erythroid_Baso",

    "Unknown": "Unknown"
}

norm_obs = adata_norm.obs.copy()
norm_obs["Broad_Cellgroup"] = norm_obs["Classified_Celltype"].map(broad_map).fillna("Other")

broad_counts = pd.crosstab(norm_obs["sample_id"], norm_obs["Broad_Cellgroup"])
broad_frac = broad_counts.div(broad_counts.sum(axis=1), axis=0).fillna(0)

broad_counts.to_csv(ROOT / "dx_normal_broad_cellgroup_counts_by_sample.csv")
broad_frac.to_csv(ROOT / "dx_normal_broad_cellgroup_fractions_by_sample.csv")

# -----------------------------
# Patient-level version too, if needed
# -----------------------------
if "Patient_ID" in norm_obs.columns:
    patient_broad_counts = pd.crosstab(norm_obs["Patient_ID"], norm_obs["Broad_Cellgroup"])
    patient_broad_frac = patient_broad_counts.div(patient_broad_counts.sum(axis=1), axis=0).fillna(0)
    patient_broad_counts.to_csv(ROOT / "dx_normal_broad_cellgroup_counts_by_patient.csv")
    patient_broad_frac.to_csv(ROOT / "dx_normal_broad_cellgroup_fractions_by_patient.csv")

# -----------------------------
# Print key diagnostics
# -----------------------------
print("\n=== SAVED SUBSETS ===")
print(ROOT / "dx_primary_training_malignant.h5ad")
print(ROOT / "dx_primary_training_normal.h5ad")

print("\n=== MASTER SUMMARY ===")
print("All DX cells:", adata.n_obs)
print("Malignant DX cells:", adata_mal.n_obs)
print("Normal DX cells:", adata_norm.n_obs)

print("\n=== SAMPLE-LEVEL SUMMARY (first 10) ===")
print(sample_summary.head(10).to_string(index=False))

print("\n=== SAMPLES WITH LOW NORMAL-CELL COUNTS (<200) ===")
low_norm = sample_summary.loc[sample_summary["normal_cells"] < 200,
                              ["sample", "gsm", "Patient_ID", "Biopsy_Origin", "normal_cells", "total_cells"]]
if len(low_norm) == 0:
    print("None")
else:
    print(low_norm.to_string(index=False))

print("\n=== NORMAL BROAD CELLGROUP FRACTIONS (first 10 samples) ===")
print(broad_frac.head(10).round(4).to_string())

print("\n=== FILES WRITTEN ===")
for fn in [
    "dx_primary_training_sample_level_summary.csv",
    "dx_allcells_celltype_counts_by_sample.csv",
    "dx_allcells_celltype_fractions_by_sample.csv",
    "dx_normal_celltype_counts_by_sample.csv",
    "dx_normal_celltype_fractions_by_sample.csv",
    "dx_normal_broad_cellgroup_counts_by_sample.csv",
    "dx_normal_broad_cellgroup_fractions_by_sample.csv",
]:
    print(ROOT / fn)
PY
