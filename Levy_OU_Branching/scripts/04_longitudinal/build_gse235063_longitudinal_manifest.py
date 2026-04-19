from pathlib import Path
import pandas as pd
import gzip
import re

raw_dir = Path("/GSE235063/GSE235063_RAW")
out_dir = Path("/Lévy_OU_Branching")
out_dir.mkdir(parents=True, exist_ok=True)

out_manifest = out_dir / "gse235063_longitudinal_manifest_raw.csv"
out_columns = out_dir / "gse235063_longitudinal_metadata_columns_summary.txt"

pattern = re.compile(r"^(GSM\d+)_(AML\d+)_(DX|REM|REL)_processed_metadata\.tsv\.gz$")

metadata_files = sorted(raw_dir.glob("*_processed_metadata.tsv.gz"))

rows = []
all_cols = {}

for f in metadata_files:
    m = pattern.match(f.name)
    if not m:
        continue

    gsm, sample, timepoint = m.groups()
    sample_id = f"{sample}_{timepoint}"

    try:
        df = pd.read_csv(f, sep="\t", compression="gzip")
    except Exception as e:
        print(f"ERROR reading {f.name}: {e}")
        continue

    cols = list(df.columns)
    all_cols[f.name] = cols

    row = {
        "gsm": gsm,
        "sample": sample,
        "timepoint": timepoint,
        "sample_id": sample_id,
        "metadata_file": f.name,
        "n_cells": len(df),
    }

    # carry through first non-null unique value if present
    preferred_single_value_cols = [
        "Patient_ID",
        "Patient_Sample",
        "Biopsy_Origin",
        "Subgroup",
        "Expected_Driving_Aberration",
        "Treatment_Outcome",
        "Clinical_Blast_Percent",
        "Disease_free_days",
        "Lambo_et_al_ID",
        "GEO_ID",
        "Library_ID",
    ]

    for c in preferred_single_value_cols:
        if c in df.columns:
            vals = df[c].dropna().astype(str).unique()
            row[c] = vals[0] if len(vals) > 0 else pd.NA

    # cell-level counts if present
    if "Malignant" in df.columns:
        mal = df["Malignant"].astype(str).str.lower()
        malignant_mask = mal.isin(["true", "1", "yes"])
        row["malignant_cells_from_metadata"] = int(malignant_mask.sum())
        row["normal_cells_from_metadata"] = int((~malignant_mask).sum())
        row["malignant_frac_from_metadata"] = (
            row["malignant_cells_from_metadata"] / row["n_cells"] if row["n_cells"] > 0 else pd.NA
        )
        row["normal_frac_from_metadata"] = (
            row["normal_cells_from_metadata"] / row["n_cells"] if row["n_cells"] > 0 else pd.NA
        )

    row["has_Patient_ID"] = "Patient_ID" in df.columns
    row["has_Malignant"] = "Malignant" in df.columns
    row["n_metadata_columns"] = len(df.columns)

    rows.append(row)

manifest = pd.DataFrame(rows).sort_values(["sample", "timepoint", "gsm"]).reset_index(drop=True)

# helpful ordering
tp_order = {"DX": 0, "REM": 1, "REL": 2}
manifest["time_index"] = manifest["timepoint"].map(tp_order)

front_cols = [
    "gsm", "sample", "timepoint", "time_index", "sample_id", "metadata_file", "n_cells",
    "Patient_ID", "Patient_Sample", "Biopsy_Origin", "Subgroup",
    "Expected_Driving_Aberration", "Treatment_Outcome", "Clinical_Blast_Percent",
    "Disease_free_days", "malignant_cells_from_metadata", "normal_cells_from_metadata",
    "malignant_frac_from_metadata", "normal_frac_from_metadata",
    "has_Patient_ID", "has_Malignant", "n_metadata_columns"
]
front_cols = [c for c in front_cols if c in manifest.columns]
other_cols = [c for c in manifest.columns if c not in front_cols]
manifest = manifest[front_cols + other_cols]

manifest.to_csv(out_manifest, index=False)

with open(out_columns, "w") as f:
    for fname, cols in all_cols.items():
        f.write(f"{fname}\n")
        for c in cols:
            f.write(f"  {c}\n")
        f.write("\n")

print(f"processed metadata files found: {len(metadata_files)}")
print(f"manifest shape: {manifest.shape}")
print(f"saved: {out_manifest}")
print(f"saved: {out_columns}")

print("\nTimepoint counts:")
print(manifest["timepoint"].value_counts(dropna=False).to_string())

print("\nFirst 15 rows:")
print(manifest.head(15).to_string(index=False))
