from pathlib import Path
import pandas as pd

base = Path("/Lévy_OU_Branching")
infile = base / "dx_diagnosis_baseline_matrix_full.csv"
outfile = base / "dx_state_missingness_report.csv"

df = pd.read_csv(infile)

state_cols = [
    "ilr_stem_vs_committed",
    "ilr_prog_vs_mature",
    "ilr_gmp_vs_monodc",
    "T_NK_given_known_z",
    "Myeloid_APC_given_known_z",
    "B_Plasma_given_known_z",
]

# Missingness by sample
report = df[[
    "sample_id", "sample", "Patient_ID", "malignant_cells", "total_cells",
    "Treatment_Outcome", "Subgroup"
] + state_cols].copy()

for c in state_cols:
    report[f"missing_{c}"] = report[c].isna()

report["n_missing_state_cols"] = report[[f"missing_{c}" for c in state_cols]].sum(axis=1)

report = report.sort_values(
    by=["n_missing_state_cols", "malignant_cells"],
    ascending=[False, True]
)

print("Missingness by column:")
print(report[[f"missing_{c}" for c in state_cols]].sum())

print("\nSamples with any missing state column:")
print(
    report.loc[report["n_missing_state_cols"] > 0,
               ["sample_id", "malignant_cells", "n_missing_state_cols"] +
               [f"missing_{c}" for c in state_cols]]
    .to_string(index=False)
)

report.to_csv(outfile, index=False)
print(f"\nsaved: {outfile}")
