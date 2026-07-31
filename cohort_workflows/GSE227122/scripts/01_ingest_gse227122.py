from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import pandas as pd

from transfer_utils import (
    apply_fixed_qc,
    compute_qc_metrics,
    make_var_names_unique_from_symbol,
    read_sample_adata,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, help="CSV with sample_id,patient_id,timepoint,transfer_set,raw_path,mrd_status")
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--gene-symbol-col", default=None)
    parser.add_argument("--min-genes", type=int, required=True)
    parser.add_argument("--max-genes", type=int, default=None)
    parser.add_argument("--max-mt-pct", type=float, default=None)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(args.manifest)
    adatas = []

    for _, row in manifest.iterrows():
        adata = read_sample_adata(Path(row["raw_path"]))
        adata = make_var_names_unique_from_symbol(adata, args.gene_symbol_col)
        adata.obs["barcode"] = adata.obs_names.astype(str)
        adata.obs["sample_id"] = row["sample_id"]
        adata.obs["patient_id"] = row["patient_id"]
        adata.obs["timepoint"] = row["timepoint"]
        adata.obs["transfer_set"] = row["transfer_set"]
        adata.obs["mrd_status"] = row.get("mrd_status", "")
        adata.obs_names = [f'{row["sample_id"]}__{bc}' for bc in adata.obs_names]
        adatas.append(adata)

    merged = ad.concat(adatas, join="outer", merge="same", label="batch", keys=manifest["sample_id"].tolist())
    merged.write_h5ad(outdir / "gse227122_raw.h5ad")

    merged = compute_qc_metrics(merged)
    merged.obs.to_csv(outdir / "gse227122_cells_qc.csv")

    qc = apply_fixed_qc(
        merged,
        min_genes=args.min_genes,
        max_genes=args.max_genes,
        max_mt_pct=args.max_mt_pct,
    )
    qc.write_h5ad(outdir / "gse227122_qc.h5ad")


if __name__ == "__main__":
    main()
