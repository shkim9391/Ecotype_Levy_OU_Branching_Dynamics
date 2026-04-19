python - <<'PY'
from pathlib import Path
import numpy as np
import pandas as pd
import scanpy as sc

PRIMARY = Path("/GSE235063/derived_dx_primary_training/dx_primary_training_allcells.h5ad")
SECONDARY = Path("/GSE235923/derived_secondary_calibration/gse235923_dx_secondary_calibration_unlabeled_gexonly_autoname.h5ad")
OUTDIR = SECONDARY.parent

sc.settings.verbosity = 2

ref = sc.read_h5ad(PRIMARY)
qry = sc.read_h5ad(SECONDARY)

def norm_names(x):
    s = pd.Series(x, dtype="object").astype(str).str.strip()
    s = s.str.replace(r"\.\d+$", "", regex=True)
    s = s.replace({
        "": np.nan, "nan": np.nan, "NaN": np.nan,
        "None": np.nan, "NONE": np.nan, "NA": np.nan, "N/A": np.nan
    })
    s = s.str.upper()
    return s

ref.var_names = pd.Index(norm_names(ref.var_names).fillna(""), name="feature_name")
qry.var_names = pd.Index(norm_names(qry.var_names).fillna(""), name="feature_name")
ref = ref[:, ref.var_names != ""].copy()
qry = qry[:, qry.var_names != ""].copy()
ref.var_names_make_unique()
qry.var_names_make_unique()

shared = ref.var_names.intersection(qry.var_names)

print("\n=== GENE OVERLAP AFTER AUTONAME GEX-ONLY REBUILD ===")
print("Reference genes:", ref.n_vars)
print("Query genes:", qry.n_vars)
print("Shared genes:", len(shared))

if len(shared) < 2000:
    print("\n=== FIRST 20 REFERENCE var_names ===")
    print(ref.var_names[:20].tolist())
    print("\n=== FIRST 20 QUERY var_names ===")
    print(qry.var_names[:20].tolist())
    raise ValueError(f"Too few shared genes after autoname rebuild: {len(shared)}")

ref = ref[:, shared].copy()
qry = qry[:, shared].copy()

ref.obs["label_malignant"] = ref.obs["Malignant"].astype(str).astype("category")
ref.obs["label_celltype"] = ref.obs["Classified_Celltype"].astype(str).astype("category")

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
    "Unknown": "Unknown",
}
ref.obs["label_broad"] = (
    ref.obs["label_celltype"].astype(str).map(broad_map).fillna("Other").astype("category")
)

for adata in [ref, qry]:
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

sc.pp.highly_variable_genes(ref, n_top_genes=3000, flavor="seurat")
hvg = ref.var["highly_variable"].fillna(False).values

ref = ref[:, hvg].copy()
qry = qry[:, ref.var_names].copy()

sc.pp.scale(ref, max_value=10)
sc.pp.scale(qry, max_value=10)

sc.tl.pca(ref, n_comps=50, svd_solver="arpack")
sc.pp.neighbors(ref, n_neighbors=15, n_pcs=min(30, ref.obsm["X_pca"].shape[1]))
sc.tl.umap(ref, min_dist=0.4)

sc.tl.ingest(qry, ref, obs=["label_malignant", "label_celltype", "label_broad"])

qry.obs = qry.obs.rename(columns={
    "label_malignant": "pred_malignant",
    "label_celltype": "pred_celltype",
    "label_broad": "pred_broad",
})

def coarse_malignant_state(ct):
    if ct == "HSC":
        return "state_HSC"
    elif ct == "Progenitor":
        return "state_Prog"
    elif ct == "GMP":
        return "state_GMP"
    elif ct in {"Monocytes", "cDC"}:
        return "state_MonoDC"
    elif ct in {"Early.Erythrocyte", "Early.Basophil"}:
        return "aux_EryBaso"
    elif ct == "CLP":
        return "aux_CLP"
    else:
        return "other_state"

qry.obs["pred_malignant_coarse"] = "not_malignant"
is_mal = qry.obs["pred_malignant"].astype(str).eq("Malignant")
qry.obs.loc[is_mal, "pred_malignant_coarse"] = (
    qry.obs.loc[is_mal, "pred_celltype"].astype(str).map(coarse_malignant_state).fillna("other_state")
)

for col in ["pred_malignant", "pred_celltype", "pred_broad", "pred_malignant_coarse"]:
    qry.obs[col] = qry.obs[col].astype("category")

labeled_h5ad = OUTDIR / "gse235923_dx_secondary_calibration_labeled_by_gse235063.h5ad"
qry.write_h5ad(labeled_h5ad, compression="gzip")

obs = qry.obs.copy()
obs["sample_id"] = obs["sample_id"].astype(str)

sample_meta = (
    obs[["gsm", "sample_base", "timepoint", "sample_id"]]
    .drop_duplicates(subset=["sample_id"])
    .set_index("sample_id")
)

sample_summary = pd.DataFrame({
    "total_cells": obs.groupby("sample_id").size(),
    "pred_malignant_cells": obs.assign(is_m=obs["pred_malignant"].astype(str).eq("Malignant")).groupby("sample_id")["is_m"].sum(),
})
sample_summary["pred_normal_cells"] = sample_summary["total_cells"] - sample_summary["pred_malignant_cells"]
sample_summary["pred_malignant_frac"] = sample_summary["pred_malignant_cells"] / sample_summary["total_cells"]
sample_summary["pred_normal_frac"] = sample_summary["pred_normal_cells"] / sample_summary["total_cells"]
sample_summary = sample_meta.join(sample_summary).reset_index()

sample_summary_csv = OUTDIR / "gse235923_dx_predicted_sample_summary.csv"
sample_summary.to_csv(sample_summary_csv, index=False)

print("\n=== SECONDARY CALIBRATION LABEL TRANSFER COMPLETE ===")
print("Query cells:", qry.n_obs)
print("Shared genes used before HVG selection:", len(shared))
print("HVGs used for ingest:", qry.n_vars)
print("Output labeled h5ad:", labeled_h5ad)

print("\n=== OVERALL PREDICTED MALIGNANT ===")
print(obs["pred_malignant"].astype(str).value_counts().to_string())

print("\n=== OVERALL PREDICTED CELLTYPE (top 20) ===")
print(obs["pred_celltype"].astype(str).value_counts().head(20).to_string())

print("\n=== OVERALL PREDICTED COARSE MALIGNANT STATE ===")
print(obs["pred_malignant_coarse"].astype(str).value_counts().to_string())

print("\n=== SAMPLE SUMMARY (first 20) ===")
print(sample_summary.head(20).to_string(index=False))

print("\nSaved:")
print(sample_summary_csv)
print(labeled_h5ad)
PY
