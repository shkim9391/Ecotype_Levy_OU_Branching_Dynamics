from pathlib import Path
import pandas as pd

raw_dir = Path("/GSE235063/GSE235063_RAW")

files = [
    raw_dir / "GSM7494285_AML1_DX_processed_metadata.tsv.gz",
    raw_dir / "GSM7494284_AML1_REM_processed_metadata.tsv.gz",
    raw_dir / "GSM7494292_AML10_DX_processed_metadata.tsv.gz",
    raw_dir / "GSM7494293_AML10_REL_processed_metadata.tsv.gz",
    raw_dir / "GSM7494294_AML10_REM_processed_metadata.tsv.gz",
]

for f in files:
    print("\n" + "=" * 100)
    print(f.name)
    df = pd.read_csv(f, sep="\t", compression="gzip")
    print("shape:", df.shape)
    print("columns:")
    for c in df.columns:
        print(" ", c)

    for c in ["Malignant", "Patient_Sample", "Classified_Celltype", "Biopsy_Origin", "Treatment_Outcome"]:
        if c in df.columns:
            print(f"\nValue counts for {c}:")
            print(df[c].astype(str).value_counts(dropna=False).head(20).to_string())

    print("\nhead:")
    print(df.head(5).to_string(index=False))
