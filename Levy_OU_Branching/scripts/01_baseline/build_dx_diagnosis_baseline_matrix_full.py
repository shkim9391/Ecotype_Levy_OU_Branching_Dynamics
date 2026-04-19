from pathlib import Path
import pandas as pd

output_base = Path("/Lévy_OU_Branching")
output_base.mkdir(parents=True, exist_ok=True)

broad_file = output_base / "dx_broad_cellgroup_fractions_by_sample.csv"
meta_file = Path(
    "/GSE235063/derived_dx_primary_training/"
    "dx_primary_training_sample_level_summary_frozen_transfer.csv"
)

out_file = output_base / "dx_diagnosis_baseline_matrix_full.csv"

broad = pd.read_csv(broad_file)
meta = pd.read_csv(meta_file)

broad["sample_id"] = broad["sample_id"].astype(str).str.strip()
meta["sample_id"] = meta["sample_id"].astype(str).str.strip()

print("broad shape:", broad.shape)
print("meta shape:", meta.shape)
print("duplicate broad sample_id:", broad["sample_id"].duplicated().sum())
print("duplicate meta sample_id:", meta["sample_id"].duplicated().sum())

missing_in_meta = sorted(set(broad["sample_id"]) - set(meta["sample_id"]))
missing_in_broad = sorted(set(meta["sample_id"]) - set(broad["sample_id"]))

print("samples in broad but not meta:", len(missing_in_meta))
print("samples in meta but not broad:", len(missing_in_broad))
if missing_in_meta:
    print("missing in meta:", missing_in_meta)
if missing_in_broad:
    print("missing in broad:", missing_in_broad)

merged = meta.merge(
    broad,
    on="sample_id",
    how="inner",
    validate="one_to_one",
)

# consistency check from sample_id
merged["sample_from_id"] = merged["sample_id"].str.replace(r"_DX$", "", regex=True)

front_cols = [
    c for c in [
        "sample_id",
        "sample",
        "sample_from_id",
        "Patient_ID",
        "gsm",
        "timepoint",
        "Biopsy_Origin",
        "Subgroup",
        "Expected_Driving_Aberration",
        "Treatment_Outcome",
        "total_cells",
        "malignant_cells",
        "normal_cells",
        "malignant_frac",
        "normal_frac",
        "PC1",
        "PC2",
        "ecotype_label",
        "state_HSC",
        "state_Prog",
        "state_GMP",
        "state_MonoDC",
        "aux_EryBaso",
        "aux_CLP",
        "ilr_stem_vs_committed",
        "ilr_prog_vs_mature",
        "ilr_gmp_vs_monodc",
        "log_aux_erybaso",
        "log_aux_clp",
        "patient_id",
        "timepoint_label",
        "time_index",
        "B_Plasma",
        "Erythroid_Baso",
        "HSPC_Prog",
        "Myeloid_APC",
        "T_NK",
        "Unknown",
        "known_total",
        "B_Plasma_given_known",
        "Erythroid_Baso_given_known",
        "HSPC_Prog_given_known",
        "Myeloid_APC_given_known",
        "T_NK_given_known",
        "B_Plasma_given_known_z",
        "Erythroid_Baso_given_known_z",
        "HSPC_Prog_given_known_z",
        "Myeloid_APC_given_known_z",
        "T_NK_given_known_z",
    ] if c in merged.columns
]

other_cols = [c for c in merged.columns if c not in front_cols]
merged = merged[front_cols + other_cols]

print("merged shape:", merged.shape)
print("rows with missing malignant_cells:", merged["malignant_cells"].isna().sum())
print("rows with zero malignant_cells:", (merged["malignant_cells"] == 0).sum())
print("rows with missing transfer PC1:", merged["PC1"].isna().sum())
print("rows with missing ecotype_label:", merged["ecotype_label"].isna().sum())

# optional model-ready subset: samples with malignant-state features available
model_ready = merged[
    merged["malignant_cells"].fillna(0) > 0
].copy()

model_ready_file = output_base / "dx_diagnosis_baseline_matrix_model_ready.csv"

merged.to_csv(out_file, index=False)
model_ready.to_csv(model_ready_file, index=False)

print(f"saved: {out_file}")
print(f"saved: {model_ready_file}")
print(merged.head())
