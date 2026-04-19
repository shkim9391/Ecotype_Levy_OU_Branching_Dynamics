from pathlib import Path
import pandas as pd

base = Path("/Lévy_OU_Branching")
infile = base / "dx_diagnosis_baseline_matrix_full.csv"
outfile = base / "dx_diagnosis_state_ready_matrix.csv"
excluded_file = base / "dx_diagnosis_state_ready_excluded_samples.csv"

df = pd.read_csv(infile)

# --------------------------------------------------
# Define first-pass continuous state vector
# --------------------------------------------------
state_cols = [
    "ilr_stem_vs_committed",
    "ilr_prog_vs_mature",
    "ilr_gmp_vs_monodc",
    "T_NK_given_known_z",
    "Myeloid_APC_given_known_z",
    "B_Plasma_given_known_z",
]

required_cols = [
    "sample_id",
    "sample",
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
] + state_cols

missing_required = [c for c in required_cols if c not in df.columns]
if missing_required:
    raise ValueError(f"Missing required columns: {missing_required}")

# --------------------------------------------------
# Mark exclusion reasons
# --------------------------------------------------
df["exclude_zero_malignant"] = df["malignant_cells"].fillna(0) <= 0
df["exclude_missing_state"] = df[state_cols].isna().any(axis=1)
df["exclude_any"] = df["exclude_zero_malignant"] | df["exclude_missing_state"]

# --------------------------------------------------
# Split included/excluded
# --------------------------------------------------
included = df.loc[~df["exclude_any"], required_cols].copy()
excluded = df.loc[df["exclude_any"]].copy()

# sample-derived consistency check
included["sample_from_id"] = included["sample_id"].str.replace(r"_DX$", "", regex=True)

# --------------------------------------------------
# Save
# --------------------------------------------------
included.to_csv(outfile, index=False)
excluded.to_csv(excluded_file, index=False)

print("full shape:", df.shape)
print("included shape:", included.shape)
print("excluded shape:", excluded.shape)
print("zero malignant excluded:", int(df["exclude_zero_malignant"].sum()))
print("missing state excluded:", int(df["exclude_missing_state"].sum()))
print("excluded sample_ids:", excluded["sample_id"].tolist())

print(f"saved: {outfile}")
print(f"saved: {excluded_file}")

print("\nIncluded samples:")
print(included[["sample_id", "malignant_cells"] + state_cols].head())

print("\nExcluded samples:")
print(excluded[["sample_id", "malignant_cells", "exclude_zero_malignant", "exclude_missing_state"]])
