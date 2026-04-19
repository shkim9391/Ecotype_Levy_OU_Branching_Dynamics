from pathlib import Path
import pandas as pd
import numpy as np
import re

raw_dir = Path("/GSE235063/GSE235063_RAW")
out_dir = Path("/Lévy_OU_Branching")
out_dir.mkdir(parents=True, exist_ok=True)

dx_train_file = out_dir / "dx_broad_cellgroup_fractions_by_sample.csv"

out_manifest = out_dir / "gse235063_longitudinal_manifest_fixed.csv"
out_all_counts = out_dir / "gse235063_allcells_celltype_counts_by_sample.csv"
out_all_fracs = out_dir / "gse235063_allcells_celltype_fractions_by_sample.csv"
out_broad = out_dir / "gse235063_broad_cellgroup_fractions_by_sample.csv"
out_normal_counts = out_dir / "gse235063_normal_celltype_counts_by_sample.csv"
out_normal_fracs = out_dir / "gse235063_normal_celltype_fractions_by_sample.csv"
out_normal_broad = out_dir / "gse235063_normal_broad_cellgroup_fractions_by_sample.csv"

pattern = re.compile(r"^(GSM\d+)_(AML\d+)_(DX|REM|REL)_processed_metadata\.tsv\.gz$")

broad_map = {
    "B_Plasma": ["B.Cell", "Plasma", "Pre.B.Cell"],
    "Erythroid_Baso": ["Early.Basophil", "Early.Erythrocyte", "Late.Erythrocyte"],
    "HSPC_Prog": ["HSC", "Progenitor", "GMP", "CLP"],
    "Myeloid_APC": ["Monocytes", "CD16.Monocytes", "cDC", "pDC"],
    "T_NK": ["CD4.Memory", "CD4.Naive", "CD8.Effector", "CD8.Memory", "CD8.Naive", "NK"],
    "Unknown": ["Unknown"],
}

known_cols = ["B_Plasma", "Erythroid_Baso", "HSPC_Prog", "Myeloid_APC", "T_NK"]

# diagnosis-trained means/sds for projection-compatible z-scoring
dx_train = pd.read_csv(dx_train_file)
train_stats = {}
for col in known_cols:
    gcol = f"{col}_given_known"
    train_stats[gcol] = {
        "mean": dx_train[gcol].mean(),
        "sd": dx_train[gcol].std(ddof=0),
    }

manifest_rows = []
all_counts_rows = []
normal_counts_rows = []

metadata_files = sorted(raw_dir.glob("*_processed_metadata.tsv.gz"))

for f in metadata_files:
    m = pattern.match(f.name)
    if not m:
        continue

    gsm, sample, timepoint = m.groups()
    sample_id_from_filename = f"{sample}_{timepoint}"

    df = pd.read_csv(f, sep="\t", compression="gzip")

    # robust identifiers
    library_ids = df["Library_ID"].dropna().astype(str).str.strip().unique() if "Library_ID" in df.columns else []
    sample_id = library_ids[0] if len(library_ids) > 0 else sample_id_from_filename

    patient_ids = df["Patient_ID"].dropna().astype(str).str.strip().unique() if "Patient_ID" in df.columns else []
    patient_id = patient_ids[0] if len(patient_ids) > 0 else pd.NA

    stage_vals = df["Patient_Sample"].dropna().astype(str).str.strip().unique() if "Patient_Sample" in df.columns else []
    metadata_stage_label = stage_vals[0] if len(stage_vals) > 0 else pd.NA

    malignant_vals = df["Malignant"].astype(str).str.strip() if "Malignant" in df.columns else pd.Series(index=df.index, dtype=str)
    mal_mask = malignant_vals.eq("Malignant")
    norm_mask = malignant_vals.eq("Normal")

    row = {
        "gsm": gsm,
        "sample": sample,
        "timepoint": timepoint,
        "sample_id": sample_id,
        "sample_id_from_filename": sample_id_from_filename,
        "metadata_file": f.name,
        "metadata_stage_label": metadata_stage_label,
        "n_cells": len(df),
        "malignant_cells": int(mal_mask.sum()),
        "normal_cells": int(norm_mask.sum()),
        "malignant_frac": float(mal_mask.mean()) if len(df) > 0 else np.nan,
        "normal_frac": float(norm_mask.mean()) if len(df) > 0 else np.nan,
        "has_stage_mismatch": str(metadata_stage_label).upper() != {"DX": "DIAGNOSIS", "REM": "REMISSION", "REL": "RELAPSE"}[timepoint],
    }

    preferred_cols = [
        "Patient_ID",
        "Biopsy_Origin",
        "Age_Months",
        "Disease_free_days",
        "Clinical_Blast_Percent",
        "Expected_Driving_Aberration",
        "Subgroup",
        "Color_Subgroup",
        "Known_CNVs",
        "Treatment_Outcome",
        "Lambo_et_al_ID",
        "GEO_ID",
        "Library_ID",
    ]

    for c in preferred_cols:
        if c in df.columns:
            vals = df[c].dropna().astype(str).unique()
            row[c] = vals[0] if len(vals) > 0 else pd.NA

    manifest_rows.append(row)

    # all-cell counts
    all_counts = df["Classified_Celltype"].astype(str).value_counts()
    count_row = {"sample_id": sample_id, "gsm": gsm, "sample": sample, "timepoint": timepoint}
    for ct, n in all_counts.items():
        count_row[ct] = int(n)
    all_counts_rows.append(count_row)

    # normal-only counts
    normal_df = df.loc[norm_mask].copy()
    normal_counts = normal_df["Classified_Celltype"].astype(str).value_counts() if len(normal_df) > 0 else pd.Series(dtype=int)
    normal_row = {"sample_id": sample_id, "gsm": gsm, "sample": sample, "timepoint": timepoint}
    for ct, n in normal_counts.items():
        normal_row[ct] = int(n)
    normal_counts_rows.append(normal_row)

manifest = pd.DataFrame(manifest_rows).sort_values(["sample", "timepoint", "gsm"]).reset_index(drop=True)

tp_order = {"DX": 0, "REM": 1, "REL": 2}
manifest["time_index"] = manifest["timepoint"].map(tp_order)

# build count matrices
all_counts_df = pd.DataFrame(all_counts_rows).fillna(0)
normal_counts_df = pd.DataFrame(normal_counts_rows).fillna(0)

id_cols = ["sample_id", "gsm", "sample", "timepoint"]
all_celltype_cols = [c for c in all_counts_df.columns if c not in id_cols]
normal_celltype_cols = [c for c in normal_counts_df.columns if c not in id_cols]

all_counts_df[all_celltype_cols] = all_counts_df[all_celltype_cols].astype(int)
normal_counts_df[normal_celltype_cols] = normal_counts_df[normal_celltype_cols].astype(int)

# fractions
all_fracs_df = all_counts_df.copy()
all_row_sums = all_fracs_df[all_celltype_cols].sum(axis=1)
all_fracs_df[all_celltype_cols] = all_fracs_df[all_celltype_cols].div(all_row_sums, axis=0)

normal_fracs_df = normal_counts_df.copy()
normal_row_sums = normal_fracs_df[normal_celltype_cols].sum(axis=1).replace(0, np.nan)
normal_fracs_df[normal_celltype_cols] = normal_fracs_df[normal_celltype_cols].div(normal_row_sums, axis=0)

def build_broad_fraction_table(frac_df):
    out = frac_df[id_cols].copy()
    for broad_name, members in broad_map.items():
        present = [m for m in members if m in frac_df.columns]
        out[broad_name] = frac_df[present].sum(axis=1) if present else 0.0

    out["broad_sum"] = out[list(broad_map.keys())].sum(axis=1)
    out["known_total"] = 1.0 - out["Unknown"]

    for col in known_cols:
        gcol = f"{col}_given_known"
        out[gcol] = np.where(out["known_total"] > 0, out[col] / out["known_total"], np.nan)

        zcol = f"{gcol}_z_dxtrain"
        mu = train_stats[gcol]["mean"]
        sd = train_stats[gcol]["sd"]
        out[zcol] = (out[gcol] - mu) / sd if sd > 0 else 0.0

    return out

broad_df = build_broad_fraction_table(all_fracs_df)
normal_broad_df = build_broad_fraction_table(normal_fracs_df)

# merge manifest first columns into summary tables
merge_cols = [
    "sample_id", "Patient_ID", "Biopsy_Origin", "Subgroup",
    "Expected_Driving_Aberration", "Treatment_Outcome",
    "n_cells", "malignant_cells", "normal_cells",
    "malignant_frac", "normal_frac"
]
merge_cols = [c for c in merge_cols if c in manifest.columns]

broad_df = manifest[["sample_id", "gsm", "sample", "timepoint", "time_index"] + merge_cols[1:]].merge(
    broad_df, on=["sample_id", "gsm", "sample", "timepoint"], how="left"
)

normal_broad_df = manifest[["sample_id", "gsm", "sample", "timepoint", "time_index"] + merge_cols[1:]].merge(
    normal_broad_df, on=["sample_id", "gsm", "sample", "timepoint"], how="left"
)

# save
manifest.to_csv(out_manifest, index=False)
all_counts_df.to_csv(out_all_counts, index=False)
all_fracs_df.to_csv(out_all_fracs, index=False)
broad_df.to_csv(out_broad, index=False)
normal_counts_df.to_csv(out_normal_counts, index=False)
normal_fracs_df.to_csv(out_normal_fracs, index=False)
normal_broad_df.to_csv(out_normal_broad, index=False)

print("processed metadata files:", len(metadata_files))
print("manifest shape:", manifest.shape)
print("all-cell counts shape:", all_counts_df.shape)
print("normal-cell counts shape:", normal_counts_df.shape)
print("broad fractions shape:", broad_df.shape)
print("normal broad fractions shape:", normal_broad_df.shape)

print("\nTimepoint counts:")
print(manifest["timepoint"].value_counts().to_string())

print("\nStage mismatches between filename timepoint and metadata Patient_Sample:")
print(int(manifest["has_stage_mismatch"].sum()))

print("\nFirst 12 manifest rows:")
print(manifest.head(12).to_string(index=False))

print(f"\nsaved: {out_manifest}")
print(f"saved: {out_all_counts}")
print(f"saved: {out_all_fracs}")
print(f"saved: {out_broad}")
print(f"saved: {out_normal_counts}")
print(f"saved: {out_normal_fracs}")
print(f"saved: {out_normal_broad}")
