from pathlib import Path
import pandas as pd

output_base = Path("/Lévy_OU_Branching")
output_base.mkdir(parents=True, exist_ok=True)

broad_file = output_base / "dx_broad_cellgroup_fractions_by_sample.csv"
meta_file = Path("/GSE235063/derived_dx_primary_training/rebuilt_transfer_expression/gse235063_malignant_pseudobulk_sample_metadata.csv")

out_file = output_base / "dx_diagnosis_baseline_matrix_minimal.csv"
missing_file = output_base / "dx_missing_samples_report.txt"

broad = pd.read_csv(broad_file)
meta = pd.read_csv(meta_file)

broad["sample_id"] = broad["sample_id"].astype(str).str.strip()
meta["sample_id"] = meta["sample_id"].astype(str).str.strip()

meta = meta.rename(columns={"n_cells": "malignant_n_cells"})

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

merged = broad.merge(
    meta[["sample_id", "malignant_n_cells"]],
    on="sample_id",
    how="left",
    validate="one_to_one",
)

# simple consistency check
merged["patient_id_from_sample"] = merged["sample_id"].str.replace(r"_DX$", "", regex=True)

front_cols = [
    "sample_id",
    "patient_id",
    "patient_id_from_sample",
    "timepoint_label",
    "time_index",
    "malignant_n_cells",
]

other_cols = [c for c in merged.columns if c not in front_cols]
merged = merged[front_cols + other_cols]

print("merged shape:", merged.shape)
print("rows with missing malignant_n_cells:", merged["malignant_n_cells"].isna().sum())

merged.to_csv(out_file, index=False)

with open(missing_file, "w") as f:
    f.write("samples in broad but not meta:\n")
    for x in missing_in_meta:
        f.write(f"{x}\n")
    f.write("\n")
    f.write("samples in meta but not broad:\n")
    for x in missing_in_broad:
        f.write(f"{x}\n")

print(f"saved: {out_file}")
print(f"saved: {missing_file}")
print(merged.head())
