python - <<'PY'
from pathlib import Path
import numpy as np
import pandas as pd
import anndata as ad
from sklearn.decomposition import PCA

PRIMARY_DIR = Path("/GSE235063/derived_dx_primary_training")
SECONDARY_DIR = Path("/GSE235923/derived_secondary_calibration")

primary_broad = pd.read_csv(PRIMARY_DIR / "dx_normal_broad_cellgroup_fractions_by_sample.csv", index_col=0)
primary_core4 = pd.read_csv(PRIMARY_DIR / "dx_ou_malignant_core4_fractions.csv", index_col=0)
secondary = ad.read_h5ad(SECONDARY_DIR / "gse235923_dx_secondary_calibration_labeled_by_gse235063.h5ad")

# --------------------------------------------------
# Primary ecotype PCA model from broad normal fractions
# --------------------------------------------------
broad_features = [c for c in ["B_Plasma", "Erythroid_Baso", "HSPC_Prog", "Myeloid_APC", "T_NK"] if c in primary_broad.columns]

X_primary = primary_broad[broad_features].copy()
X_primary = X_primary.div(X_primary.sum(axis=1), axis=0).fillna(0)

eps = 1e-4
X_primary_clr = np.log(X_primary + eps)
X_primary_clr = X_primary_clr.sub(X_primary_clr.mean(axis=1), axis=0)

clr_mean = X_primary_clr.mean(axis=0)
clr_sd = X_primary_clr.std(axis=0, ddof=0).replace(0, 1.0)

X_primary_z = (X_primary_clr - clr_mean) / clr_sd

pca = PCA(n_components=2)
primary_pcs = pca.fit_transform(X_primary_z.values)

primary_pc_df = pd.DataFrame(
    primary_pcs,
    index=X_primary.index,
    columns=["PC1", "PC2"]
)
primary_pc_df.to_csv(SECONDARY_DIR / "primary_ecotype_pca_reference_scores.csv")

loadings = pd.DataFrame(
    pca.components_.T,
    index=broad_features,
    columns=["PC1_loading", "PC2_loading"]
)
loadings.to_csv(SECONDARY_DIR / "primary_ecotype_pca_loadings.csv")

# --------------------------------------------------
# Secondary normal broad fractions from transferred labels
# --------------------------------------------------
obs = secondary.obs.copy()
obs["sample_id"] = obs["sample_id"].astype(str)

is_normal = obs["pred_malignant"].astype(str).eq("Normal")
obs_norm = obs.loc[is_normal].copy()

sec_broad_counts = pd.crosstab(obs_norm["sample_id"], obs_norm["pred_broad"].astype(str))
sec_broad_counts = sec_broad_counts.reindex(columns=broad_features, fill_value=0)

sec_broad_frac = sec_broad_counts.div(sec_broad_counts.sum(axis=1), axis=0).fillna(0)
sec_broad_frac.to_csv(SECONDARY_DIR / "gse235923_dx_pred_broad_fractions_by_sample_restricted.csv")

X_sec_clr = np.log(sec_broad_frac + eps)
X_sec_clr = X_sec_clr.sub(X_sec_clr.mean(axis=1), axis=0)
X_sec_z = (X_sec_clr - clr_mean) / clr_sd

sec_pcs = pca.transform(X_sec_z.values)
sec_pc_df = pd.DataFrame(
    sec_pcs,
    index=sec_broad_frac.index,
    columns=["PC1", "PC2"]
)
sec_pc_df.to_csv(SECONDARY_DIR / "gse235923_dx_projected_ecotype_pcs.csv")

# --------------------------------------------------
# Secondary malignant coarse fractions
# --------------------------------------------------
is_malignant = obs["pred_malignant"].astype(str).eq("Malignant")
obs_mal = obs.loc[is_malignant].copy()

coarse_counts = pd.crosstab(obs_mal["sample_id"], obs_mal["pred_malignant_coarse"].astype(str))
for c in ["state_HSC", "state_Prog", "state_GMP", "state_MonoDC", "aux_EryBaso", "aux_CLP"]:
    if c not in coarse_counts.columns:
        coarse_counts[c] = 0

coarse_counts = coarse_counts[["state_HSC", "state_Prog", "state_GMP", "state_MonoDC", "aux_EryBaso", "aux_CLP"]]
coarse_frac = coarse_counts.div(coarse_counts.sum(axis=1), axis=0).fillna(0)
coarse_frac.to_csv(SECONDARY_DIR / "gse235923_dx_pred_malignant_coarse_fractions_by_sample.csv")

core4 = coarse_frac[["state_HSC", "state_Prog", "state_GMP", "state_MonoDC"]].copy()
core4 = core4.div(core4.sum(axis=1), axis=0).fillna(0)
core4.to_csv(SECONDARY_DIR / "gse235923_dx_pred_core4_fractions_by_sample.csv")

# --------------------------------------------------
# Secondary ILR / branch outcomes
# --------------------------------------------------
core = core4.copy()
core = core + eps
core = core.div(core.sum(axis=1), axis=0)

sec_outcomes = pd.DataFrame(index=core.index)
sec_outcomes["state_HSC"] = core4["state_HSC"]
sec_outcomes["state_Prog"] = core4["state_Prog"]
sec_outcomes["state_GMP"] = core4["state_GMP"]
sec_outcomes["state_MonoDC"] = core4["state_MonoDC"]
sec_outcomes["aux_EryBaso"] = coarse_frac["aux_EryBaso"]
sec_outcomes["aux_CLP"] = coarse_frac["aux_CLP"]

sec_outcomes["ilr_stem_vs_committed"] = np.sqrt(3/4) * np.log(
    core["state_HSC"] /
    ((core["state_Prog"] * core["state_GMP"] * core["state_MonoDC"]) ** (1/3))
)

sec_outcomes["log_aux_erybaso"] = np.log(sec_outcomes["aux_EryBaso"] + eps)
sec_outcomes["log_aux_clp"] = np.log(sec_outcomes["aux_CLP"] + eps)

sec_outcomes.to_csv(SECONDARY_DIR / "gse235923_dx_secondary_outcomes.csv")

# --------------------------------------------------
# Sample metadata and final calibration table
# --------------------------------------------------
sample_summary = pd.read_csv(SECONDARY_DIR / "gse235923_dx_predicted_sample_summary.csv")
sample_summary = sample_summary.set_index("sample_id")

calibration = sample_summary.join(sec_pc_df, how="left").join(sec_outcomes, how="left")
calibration = calibration.reset_index()

calibration_csv = SECONDARY_DIR / "gse235923_dx_secondary_calibration_table.csv"
calibration.to_csv(calibration_csv, index=False)

# --------------------------------------------------
# Compare to primary ranges
# --------------------------------------------------
primary_outcomes = pd.read_csv(PRIMARY_DIR / "dx_ou_ilr_branch_ready.csv").set_index("sample_id")

compare_rows = []
for col in ["PC1", "PC2", "ilr_stem_vs_committed", "log_aux_erybaso"]:
    pmin = float(primary_outcomes[col].min())
    pmax = float(primary_outcomes[col].max())
    smin = float(calibration[col].min())
    smax = float(calibration[col].max())
    compare_rows.append({
        "variable": col,
        "primary_min": pmin,
        "primary_max": pmax,
        "secondary_min": smin,
        "secondary_max": smax,
        "secondary_within_primary_range_frac": float(((calibration[col] >= pmin) & (calibration[col] <= pmax)).mean()),
    })

compare_df = pd.DataFrame(compare_rows)
compare_df.to_csv(SECONDARY_DIR / "gse235923_dx_secondary_vs_primary_range_comparison.csv", index=False)

# --------------------------------------------------
# Print summary
# --------------------------------------------------
print("\n=== SECONDARY CALIBRATION TABLE BUILT ===")
print("Samples:", calibration.shape[0])

print("\n=== PROJECTED SECONDARY ECOTYPE PCS (first 10) ===")
print(sec_pc_df.head(10).round(4).to_string())

print("\n=== SECONDARY CORE4 FRACTIONS (first 10) ===")
print(core4.head(10).round(4).to_string())

print("\n=== SECONDARY OUTCOMES (first 10) ===")
print(sec_outcomes[["ilr_stem_vs_committed", "log_aux_erybaso", "log_aux_clp"]].head(10).round(4).to_string())

print("\n=== PRIMARY VS SECONDARY RANGE COMPARISON ===")
print(compare_df.round(4).to_string(index=False))

print("\n=== FINAL CALIBRATION TABLE (first 20) ===")
print(calibration.head(20).round(4).to_string(index=False))

print("\nSaved:")
print(SECONDARY_DIR / "primary_ecotype_pca_reference_scores.csv")
print(SECONDARY_DIR / "primary_ecotype_pca_loadings.csv")
print(SECONDARY_DIR / "gse235923_dx_projected_ecotype_pcs.csv")
print(SECONDARY_DIR / "gse235923_dx_pred_broad_fractions_by_sample_restricted.csv")
print(SECONDARY_DIR / "gse235923_dx_pred_malignant_coarse_fractions_by_sample.csv")
print(SECONDARY_DIR / "gse235923_dx_pred_core4_fractions_by_sample.csv")
print(SECONDARY_DIR / "gse235923_dx_secondary_outcomes.csv")
print(SECONDARY_DIR / "gse235923_dx_secondary_calibration_table.csv")
print(SECONDARY_DIR / "gse235923_dx_secondary_vs_primary_range_comparison.csv")
PY
