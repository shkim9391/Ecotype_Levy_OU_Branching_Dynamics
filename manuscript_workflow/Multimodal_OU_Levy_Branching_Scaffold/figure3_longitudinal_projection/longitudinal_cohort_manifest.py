from pathlib import Path
import re
import pandas as pd

root = Path("/Ecotype_OU_Branching/GSE235063/GSE235063_RAW")

pat = re.compile(
    r'^(GSM\d+)_(.+?)_(DX|REM|REL)_(processed|raw)_(barcodes\.tsv|genes\.tsv|matrix\.mtx|metadata\.tsv)\.gz$'
)

rows = []
unmatched = []

for f in sorted(root.glob("*.gz")):
    m = pat.match(f.name)
    if m:
        gsm, sample, timepoint, layer, kind = m.groups()
        rows.append({
            "file": f.name,
            "gsm": gsm,
            "sample": sample,
            "timepoint": timepoint,
            "layer": layer,
            "kind": kind,
        })
    else:
        unmatched.append(f.name)

df = pd.DataFrame(rows)

print("\n=== BASIC COUNTS ===")
print("Matched files:", len(df))
print("Unmatched files:", len(unmatched))

if len(df) == 0:
    raise SystemExit("No matching GEO-style .gz files found.")

print("\n=== COUNTS BY TIMEPOINT / LAYER / KIND ===")
print(df.groupby(["timepoint", "layer", "kind"]).size().sort_index())

# Keep all processed longitudinal samples
proc = df[df["layer"] == "processed"].copy()

manifest = (
    proc.pivot_table(
        index=["gsm", "sample", "timepoint"],
        columns="kind",
        values="file",
        aggfunc="first"
    )
    .reset_index()
)

manifest = manifest.rename(columns={
    "sample": "patient_id",
    "timepoint": "clinical_timepoint_raw",
    "barcodes.tsv": "barcodes_file",
    "genes.tsv": "genes_file",
    "matrix.mtx": "matrix_file",
    "metadata.tsv": "metadata_file",
})

manifest["sample_id"] = manifest["patient_id"].astype(str) + "_" + manifest["clinical_timepoint_raw"].astype(str)

time_map = {
    "DX": "DX",
    "REM": "EOI_REM",
    "REL": "REL",
}
manifest["clinical_timepoint_coarse"] = manifest["clinical_timepoint_raw"].map(time_map)

# Reorder columns
front = [
    "gsm",
    "sample_id",
    "patient_id",
    "clinical_timepoint_raw",
    "clinical_timepoint_coarse",
    "barcodes_file",
    "genes_file",
    "matrix_file",
    "metadata_file",
]
manifest = manifest[front]

out_csv = root / "longitudinal_cohort_manifest_processed.csv"
manifest.to_csv(out_csv, index=False)

print("\n=== LONGITUDINAL PROCESSED COHORT ===")
print("Number of processed longitudinal samples:", len(manifest))
print(manifest.head(30).to_string(index=False))

print(f"\nSaved manifest to: {out_csv}")

if unmatched:
    print("\n=== FIRST 30 UNMATCHED FILENAMES ===")
    for x in unmatched[:30]:
        print(x)
