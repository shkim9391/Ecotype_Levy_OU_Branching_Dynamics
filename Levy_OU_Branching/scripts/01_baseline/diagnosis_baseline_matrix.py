from pathlib import Path
import pandas as pd
import numpy as np

base = Path("/GSE235063/derived_dx_primary_training")
infile = base / "dx_allcells_celltype_fractions_by_sample.csv"

df = pd.read_csv(infile)

# -----------------------------
# Basic checks
# -----------------------------
fine_cols = [c for c in df.columns if c != "sample_id"]
row_sums = df[fine_cols].sum(axis=1)

print("shape:", df.shape)
print("max abs(row_sum - 1):", np.abs(row_sums - 1).max())

if np.abs(row_sums - 1).max() > 1e-8:
    raise ValueError("Row fractions do not sum to 1. Please inspect the input file.")

# -----------------------------
# Parse IDs
# -----------------------------
out = pd.DataFrame({
    "sample_id": df["sample_id"],
    "patient_id": df["sample_id"].str.replace(r"_DX$", "", regex=True),
    "timepoint_label": "DX",
    "time_index": 0
})

# -----------------------------
# Collapse to broad groups
# -----------------------------
out["B_Plasma"] = df["B.Cell"] + df["Plasma"] + df["Pre.B.Cell"]

out["Erythroid_Baso"] = (
    df["Early.Basophil"] +
    df["Early.Erythrocyte"] +
    df["Late.Erythrocyte"]
)

out["HSPC_Prog"] = (
    df["HSC"] +
    df["Progenitor"] +
    df["GMP"] +
    df["CLP"]
)

out["Myeloid_APC"] = (
    df["Monocytes"] +
    df["CD16.Monocytes"] +
    df["cDC"] +
    df["pDC"]
)

out["T_NK"] = (
    df["CD4.Memory"] +
    df["CD4.Naive"] +
    df["CD8.Effector"] +
    df["CD8.Memory"] +
    df["CD8.Naive"] +
    df["NK"]
)

out["Unknown"] = df["Unknown"]

broad_cols = ["B_Plasma", "Erythroid_Baso", "HSPC_Prog", "Myeloid_APC", "T_NK", "Unknown"]
out["broad_sum"] = out[broad_cols].sum(axis=1)

print("max abs(broad_sum - 1):", np.abs(out["broad_sum"] - 1).max())

# -----------------------------
# Known-cell conditional fractions
# -----------------------------
out["known_total"] = 1.0 - out["Unknown"]

known_cols = ["B_Plasma", "Erythroid_Baso", "HSPC_Prog", "Myeloid_APC", "T_NK"]
for col in known_cols:
    out[f"{col}_given_known"] = np.where(
        out["known_total"] > 0,
        out[col] / out["known_total"],
        np.nan
    )

# -----------------------------
# Z-score the known-cell fractions
# -----------------------------
for col in known_cols:
    gcol = f"{col}_given_known"
    zcol = f"{gcol}_z"
    mu = out[gcol].mean()
    sd = out[gcol].std(ddof=0)
    out[zcol] = (out[gcol] - mu) / sd if sd > 0 else 0.0

# -----------------------------
# Save outputs
# -----------------------------
keep_main = [
    "sample_id", "patient_id", "timepoint_label", "time_index",
    "B_Plasma", "Erythroid_Baso", "HSPC_Prog", "Myeloid_APC", "T_NK", "Unknown",
    "known_total",
    "B_Plasma_given_known", "Erythroid_Baso_given_known", "HSPC_Prog_given_known",
    "Myeloid_APC_given_known", "T_NK_given_known",
    "B_Plasma_given_known_z", "Erythroid_Baso_given_known_z", "HSPC_Prog_given_known_z",
    "Myeloid_APC_given_known_z", "T_NK_given_known_z"
]

out_main = out[keep_main].copy()

outfile = base / "dx_broad_cellgroup_fractions_by_sample.csv"
out_main.to_csv(outfile, index=False)

print(f"saved: {outfile}")

# Optional summary
summary = out_main[
    ["B_Plasma", "Erythroid_Baso", "HSPC_Prog", "Myeloid_APC", "T_NK", "Unknown"]
].describe().T
summary_file = base / "dx_broad_cellgroup_fractions_summary.csv"
summary.to_csv(summary_file)

print(f"saved: {summary_file}")
print(summary)
