python3 - <<'PY'
from pathlib import Path
import pandas as pd
import re

root = Path("/GSE235923/GSE235923_RAW")

# only the actual GEO bundle files
files = sorted([p.name for p in root.glob("GSM*_*.gz")])

bundle_pat = re.compile(
    r'^(GSM\d+)_(Sample\d+)(Dx|D|E|R)_(barcodes\.tsv|features\.tsv|matrix\.mtx)\.gz$',
    re.IGNORECASE
)

tp_map = {
    "D": "DX",
    "Dx": "DX",
    "E": "EOI",
    "R": "REL",
}

rows = []
unmatched = []

for f in files:
    m = bundle_pat.match(f)
    if not m:
        unmatched.append(f)
        continue

    gsm, sample_base, tp_code, kind = m.groups()
    rows.append({
        "file": f,
        "gsm": gsm,
        "sample_base": sample_base,
        "tp_code": tp_code,
        "timepoint": tp_map[tp_code],
        "sample_id": f"{sample_base}_{tp_map[tp_code]}",
        "kind": kind,
    })

df = pd.DataFrame(rows)

print("\n=== MATCHED GEO BUNDLE FILES ===")
print(len(df))

print("\n=== UNMATCHED GEO BUNDLE FILES ===")
print(len(unmatched))
if unmatched:
    for x in unmatched[:20]:
        print(x)

print("\n=== COUNTS BY TIMEPOINT ===")
print(df[["sample_id", "timepoint"]].drop_duplicates()["timepoint"].value_counts().to_string())

print("\n=== COUNTS BY KIND ===")
print(df["kind"].value_counts().to_string())

bundle = (
    df.assign(present=1)
      .pivot_table(
          index=["gsm", "sample_base", "timepoint", "sample_id"],
          columns="kind",
          values="present",
          aggfunc="max",
          fill_value=0
      )
      .reset_index()
)

expected_cols = ["barcodes.tsv", "features.tsv", "matrix.mtx"]
for c in expected_cols:
    if c not in bundle.columns:
        bundle[c] = 0

incomplete = bundle[(bundle[expected_cols].min(axis=1) == 0)].copy()

print("\n=== INCOMPLETE BUNDLES ===")
if len(incomplete) == 0:
    print("None")
else:
    print(incomplete.to_string(index=False))

manifest = (
    df.pivot_table(
        index=["gsm", "sample_base", "timepoint", "sample_id"],
        columns="kind",
        values="file",
        aggfunc="first"
    )
    .reset_index()
    .rename(columns={
        "barcodes.tsv": "barcodes_file",
        "features.tsv": "features_file",
        "matrix.mtx": "matrix_file",
    })
    .sort_values(["timepoint", "sample_base", "gsm"])
    .reset_index(drop=True)
)

manifest_all = root / "gse235923_manifest_all.csv"
manifest_dx = root / "gse235923_manifest_dx.csv"
manifest_eoi = root / "gse235923_manifest_eoi.csv"
manifest_rel = root / "gse235923_manifest_rel.csv"

manifest.to_csv(manifest_all, index=False)
manifest[manifest["timepoint"] == "DX"].to_csv(manifest_dx, index=False)
manifest[manifest["timepoint"] == "EOI"].to_csv(manifest_eoi, index=False)
manifest[manifest["timepoint"] == "REL"].to_csv(manifest_rel, index=False)

print("\n=== MANIFEST COUNTS ===")
print(manifest["timepoint"].value_counts().to_string())

print("\nSaved:")
print(manifest_all)
print(manifest_dx)
print(manifest_eoi)
print(manifest_rel)
PY
