python3 - <<'PY'
from pathlib import Path
import pandas as pd
import re

root = Path("/GSE235923/GSE235923_RAW")

files = sorted([p.name for p in root.iterdir() if p.is_file()])

print("\n=== TOTAL FILES ===")
print(len(files))

print("\n=== FIRST 80 FILENAMES ===")
for x in files[:80]:
    print(x)

# Generic parsing attempts
rows = []
for f in files:
    rec = {"file": f}

    m_gsm = re.search(r'(GSM\d+)', f)
    rec["gsm"] = m_gsm.group(1) if m_gsm else None

    fl = f.lower()

    if "diagnosis" in fl or re.search(r'(^|[_\-])dx([_\-\.]|$)', fl):
        rec["timepoint"] = "DX"
    elif "end_of_induction" in fl or "end-of-induction" in fl or "eoi" in fl:
        rec["timepoint"] = "EOI"
    elif "relapse" in fl or re.search(r'(^|[_\-])rel([_\-\.]|$)', fl):
        rec["timepoint"] = "REL"
    else:
        rec["timepoint"] = None

    if "matrix.mtx" in fl:
        rec["kind"] = "matrix.mtx"
    elif "barcodes.tsv" in fl:
        rec["kind"] = "barcodes.tsv"
    elif "genes.tsv" in fl:
        rec["kind"] = "genes.tsv"
    elif "features.tsv" in fl:
        rec["kind"] = "features.tsv"
    elif "metadata" in fl:
        rec["kind"] = "metadata"
    else:
        rec["kind"] = "other"

    rows.append(rec)

df = pd.DataFrame(rows)

print("\n=== COUNTS BY KIND ===")
print(df["kind"].value_counts(dropna=False).to_string())

print("\n=== COUNTS BY TIMEPOINT ===")
print(df["timepoint"].value_counts(dropna=False).to_string())

if df["gsm"].notna().any():
    print("\n=== FILE COUNTS BY GSM ===")
    print(df.groupby("gsm").size().sort_values(ascending=False).to_string())

out_csv = root / "gse235923_file_inventory.csv"
df.to_csv(out_csv, index=False)

print(f"\nSaved inventory to: {out_csv}")
PY
