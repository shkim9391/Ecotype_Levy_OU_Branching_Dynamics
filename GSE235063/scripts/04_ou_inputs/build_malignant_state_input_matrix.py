python - <<'PY'
from pathlib import Path
import numpy as np
import pandas as pd
import anndata as ad

ROOT = Path("/GSE235063/derived_dx_primary_training")

# --------------------------------------------------
# inputs
# --------------------------------------------------
adata_mal = ad.read_h5ad(ROOT / "dx_primary_training_malignant.h5ad")
cov = pd.read_csv(ROOT / "dx_ecotype_continuous_covariates_for_ou.csv")

cov["sample_id"] = cov["sample_id"].astype(str)
adata_mal.obs["sample_id"] = adata_mal.obs["sample_id"].astype(str)
adata_mal.obs["Classified_Celltype"] = adata_mal.obs["Classified_Celltype"].astype(str)

train_samples = cov["sample_id"].tolist()
adata_mal = adata_mal[adata_mal.obs["sample_id"].isin(train_samples)].copy()

# --------------------------------------------------
# basic summary
# --------------------------------------------------
print("\n=== TRAINING MALIGNANT COHORT ===")
print("Training samples:", len(train_samples))
print("Malignant cells retained:", adata_mal.n_obs)
print("Genes:", adata_mal.n_vars)

# --------------------------------------------------
# malignant state counts/fractions by sample
# --------------------------------------------------
state_counts = pd.crosstab(adata_mal.obs["sample_id"], adata_mal.obs["Classified_Celltype"])
state_frac = state_counts.div(state_counts.sum(axis=1), axis=0).fillna(0)

state_counts.to_csv(ROOT / "dx_ou_malignant_state_counts_by_sample_full.csv")
state_frac.to_csv(ROOT / "dx_ou_malignant_state_fractions_by_sample_full.csv")

# --------------------------------------------------
# choose informative malignant states
# rules:
#   - exclude Unknown
#   - state fraction >=2% in at least 3 samples
#   - and at least 50 cells in at least 3 samples
# --------------------------------------------------
candidate_states = [c for c in state_frac.columns if c != "Unknown"]

state_cols = [
    c for c in candidate_states
    if ((state_frac[c] >= 0.02).sum() >= 3) and ((state_counts[c] >= 50).sum() >= 3)
]

# fallback if filtering is too strict
if len(state_cols) < 3:
    state_cols = [
        c for c in candidate_states
        if ((state_frac[c] >= 0.01).sum() >= 2)
    ]

print("\n=== INFORMATIVE MALIGNANT STATES USED ===")
print(state_cols)

state_counts_f = state_counts[state_cols].copy()
state_frac_f = state_frac[state_cols].copy()

state_counts_f.to_csv(ROOT / "dx_ou_malignant_state_counts_by_sample_filtered.csv")
state_frac_f.to_csv(ROOT / "dx_ou_malignant_state_fractions_by_sample_filtered.csv")

# --------------------------------------------------
# CLR transform for compositional modeling
# --------------------------------------------------
X_frac = state_frac_f.div(state_frac_f.sum(axis=1), axis=0).fillna(0)
eps = 1e-4
X_clr = np.log(X_frac + eps)
X_clr = X_clr.sub(X_clr.mean(axis=1), axis=0)

X_clr.to_csv(ROOT / "dx_ou_malignant_state_clr_by_sample.csv")

# --------------------------------------------------
# dominant malignant state per sample
# --------------------------------------------------
dominant_state = state_frac_f.idxmax(axis=1).rename("dominant_malignant_state")
dominant_frac = state_frac_f.max(axis=1).rename("dominant_malignant_state_frac")

# --------------------------------------------------
# final OU design matrix
# --------------------------------------------------
cov2 = cov.set_index("sample_id").copy()
design = cov2.join(dominant_state).join(dominant_frac)
design = design.join(state_frac_f.add_prefix("malfrac_"))
design = design.join(X_clr.add_prefix("malclr_"))

design.to_csv(ROOT / "dx_ou_training_design_matrix.csv")

# --------------------------------------------------
# optional: compact per-cell metadata for downstream state models
# --------------------------------------------------
cell_meta_cols = [
    c for c in [
        "sample_id", "sample", "gsm", "Patient_ID", "Biopsy_Origin",
        "Subgroup", "Expected_Driving_Aberration", "Treatment_Outcome",
        "Classified_Celltype", "Seurat_Cluster", "Counts", "Features",
        "nCount_RNA", "nFeature_RNA"
    ] if c in adata_mal.obs.columns
]

cell_meta = adata_mal.obs[cell_meta_cols].copy()
cell_meta.index.name = "cell_id"
cell_meta.to_csv(ROOT / "dx_ou_malignant_cell_metadata.csv.gz", compression="gzip")

# --------------------------------------------------
# sample-level printout
# --------------------------------------------------
sample_summary = design.reset_index()[[
    "sample_id", "sample", "Biopsy_Origin", "Subgroup",
    "malignant_cells", "normal_cells", "PC1", "PC2",
    "ecotype_label", "dominant_malignant_state", "dominant_malignant_state_frac"
]]

print("\n=== SAMPLE-LEVEL OU DESIGN SUMMARY ===")
print(sample_summary.sort_values("sample").to_string(index=False))

print("\n=== OVERALL MALIGNANT CELLTYPE COUNTS ===")
print(adata_mal.obs["Classified_Celltype"].value_counts().to_string())

print("\n=== FILES WRITTEN ===")
for fn in [
    "dx_ou_malignant_state_counts_by_sample_full.csv",
    "dx_ou_malignant_state_fractions_by_sample_full.csv",
    "dx_ou_malignant_state_counts_by_sample_filtered.csv",
    "dx_ou_malignant_state_fractions_by_sample_filtered.csv",
    "dx_ou_malignant_state_clr_by_sample.csv",
    "dx_ou_training_design_matrix.csv",
    "dx_ou_malignant_cell_metadata.csv.gz",
]:
    print(ROOT / fn)
PY
