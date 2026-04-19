from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/GSE235063/derived_dx_primary_training")

df = pd.read_csv(ROOT / "dx_ou_training_design_matrix_core4.csv")

core_cols = ["state_HSC", "state_Prog", "state_GMP", "state_MonoDC"]
eps = 1e-4

# clean and renormalize
for c in core_cols + ["aux_EryBaso", "aux_CLP"]:
    df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0).clip(lower=0.0)

core = df[core_cols].copy()
core = core.div(core.sum(axis=1), axis=0).fillna(0.0) + eps
core = core.div(core.sum(axis=1), axis=0)

# interpretable ILR coordinates
df["ilr_stem_vs_committed"] = np.sqrt(3/4) * np.log(
    core["state_HSC"] /
    ((core["state_Prog"] * core["state_GMP"] * core["state_MonoDC"]) ** (1/3))
)

df["ilr_prog_vs_mature"] = np.sqrt(2/3) * np.log(
    core["state_Prog"] /
    np.sqrt(core["state_GMP"] * core["state_MonoDC"])
)

df["ilr_gmp_vs_monodc"] = np.sqrt(1/2) * np.log(
    core["state_GMP"] / core["state_MonoDC"]
)

# branch programs
df["log_aux_erybaso"] = np.log(df["aux_EryBaso"] + eps)
df["log_aux_clp"] = np.log(df["aux_CLP"] + eps)

# simple modeling covariates
df["is_blood"] = (df["Biopsy_Origin"].astype(str) == "Blood").astype(int)

out_cols = [
    "sample_id", "sample", "gsm", "Patient_ID",
    "Biopsy_Origin", "is_blood",
    "Subgroup", "Expected_Driving_Aberration", "Treatment_Outcome",
    "malignant_cells", "normal_cells",
    "PC1", "PC2", "ecotype_label",
    "state_HSC", "state_Prog", "state_GMP", "state_MonoDC",
    "aux_EryBaso", "aux_CLP",
    "ilr_stem_vs_committed", "ilr_prog_vs_mature", "ilr_gmp_vs_monodc",
    "log_aux_erybaso", "log_aux_clp",
]

df[out_cols].to_csv(ROOT / "dx_ou_ilr_branch_ready.csv", index=False)

print("\n=== READY MATRIX SAVED ===")
print(ROOT / "dx_ou_ilr_branch_ready.csv")

print("\n=== FIRST 10 ROWS ===")
print(df[out_cols].head(10).to_string(index=False))
