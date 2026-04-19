import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def paired_delta_table(df: pd.DataFrame, pc_cols: list[str]) -> pd.DataFrame:
    rows = []
    patients = sorted(df["patient_id"].dropna().astype(str).unique())

    for pid in patients:
        sub = df.loc[df["patient_id"].astype(str) == pid].copy()
        dx = sub.loc[sub["timepoint"] == "Dx"]
        eoi = sub.loc[sub["timepoint"] == "EOI"]

        if len(dx) == 0 or len(eoi) == 0:
            continue

        dx_row = dx.iloc[0]
        eoi_row = eoi.iloc[0]

        row = {
            "patient_id": pid,
            "dx_sample_id": dx_row["sample_id"],
            "eoi_sample_id": eoi_row["sample_id"],
        }
        for c in pc_cols:
            row[f"{c}_dx"] = dx_row.get(c, np.nan)
            row[f"{c}_eoi"] = eoi_row.get(c, np.nan)
            row[f"{c}_delta_eoi_minus_dx"] = row[f"{c}_eoi"] - row[f"{c}_dx"]
        rows.append(row)

    return pd.DataFrame(rows)


def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [
        a if (b == "" or str(b).startswith("Unnamed")) else f"{a}_{b}"
        for a, b in df.columns.to_flat_index()
    ]
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--projected-samples", required=True)
    parser.add_argument("--cross-cohort-table", required=True)
    parser.add_argument(
        "--gse227122-program-summary",
        default=None,
        help="Optional CSV with sample_id and program columns such as ilr_stem_vs_committed/log_aux_erybaso",
    )
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    proj = pd.read_csv(args.projected_samples)
    cross = pd.read_csv(args.cross_cohort_table)

    proj["sample_id"] = proj["sample_id"].astype(str)

    if args.gse227122_program_summary is not None:
        prog = pd.read_csv(args.gse227122_program_summary)
        if "sample_id" not in prog.columns:
            raise ValueError("Program summary CSV must contain sample_id")
        prog["sample_id"] = prog["sample_id"].astype(str)
        proj = proj.merge(prog, on="sample_id", how="left")

    dx_df = proj.loc[proj["timepoint"] == "Dx"].copy() if "timepoint" in proj.columns else proj.copy()
    eoi_df = proj.loc[proj["timepoint"] == "EOI"].copy() if "timepoint" in proj.columns else proj.iloc[0:0].copy()
    rel_df = proj.loc[proj["timepoint"] == "Rel"].copy() if "timepoint" in proj.columns else proj.iloc[0:0].copy()

    pc_cols = [c for c in ["PC1_strict", "PC2_strict", "PC1_cal", "PC2_cal"] if c in proj.columns]

    dx_df.to_csv(outdir / "gse227122_dx_projected_samples.csv", index=False)
    eoi_df.to_csv(outdir / "gse227122_eoi_projected_samples.csv", index=False)
    rel_df.to_csv(outdir / "gse227122_rel_projected_samples.csv", index=False)

    paired = pd.DataFrame()
    if {"patient_id", "timepoint"}.issubset(proj.columns):
        paired = paired_delta_table(proj, pc_cols)
        paired.to_csv(outdir / "gse227122_dx_eoi_paired_pc_deltas.csv", index=False)

    cross_summary = cross.copy()

    stats = (
        cross_summary
        .groupby(["cohort", "mode"], dropna=False)[["PC1", "PC2"]]
        .agg(["count", "mean", "std", "median"])
        .reset_index()
    )
    stats = flatten_columns(stats)
    stats.to_csv(outdir / "cross_cohort_ecotype_summary_stats.csv", index=False)

    if {"cohort", "mode", "timepoint", "PC1", "PC2"}.issubset(cross_summary.columns):
        timepoint_stats = (
            cross_summary
            .groupby(["cohort", "mode", "timepoint"], dropna=False)[["PC1", "PC2"]]
            .agg(["count", "mean", "std", "median"])
            .reset_index()
        )
        timepoint_stats = flatten_columns(timepoint_stats)
        timepoint_stats.to_csv(outdir / "cross_cohort_ecotype_timepoint_stats.csv", index=False)

    proj.to_csv(outdir / "gse227122_projected_samples_with_optional_programs.csv", index=False)

    print("\n=== WRITTEN ===")
    print(outdir / "gse227122_dx_projected_samples.csv")
    print(outdir / "gse227122_eoi_projected_samples.csv")
    print(outdir / "gse227122_rel_projected_samples.csv")
    print(outdir / "gse227122_dx_eoi_paired_pc_deltas.csv")
    print(outdir / "cross_cohort_ecotype_summary_stats.csv")
    print(outdir / "gse227122_projected_samples_with_optional_programs.csv")

    print("\n=== DIAGNOSTICS ===")
    print("Dx samples:", len(dx_df))
    print("EOI samples:", len(eoi_df))
    print("Rel samples:", len(rel_df))
    print("Cross-cohort rows:", len(cross_summary))
    if not paired.empty:
        print("Paired Dx→EOI rows:", len(paired))


if __name__ == "__main__":
    main()
