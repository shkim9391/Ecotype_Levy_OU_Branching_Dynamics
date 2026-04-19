import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def first_present(columns, candidates):
    for c in candidates:
        if c in columns:
            return c
    return None


def load_feature_order(asset_dir: Path) -> list[str]:
    with open(asset_dir / "ecotype_feature_order.json", "r") as f:
        return json.load(f)["feature_order"]


def load_calibration_map(path: Path | None) -> pd.DataFrame | None:
    if path is None:
        return None
    path = Path(path)
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if not {"axis", "alpha", "beta"}.issubset(df.columns):
        raise ValueError(f"Calibration CSV must contain axis, alpha, beta: {path}")
    return df


def apply_axiswise_calibration(df: pd.DataFrame, cal_df: pd.DataFrame | None) -> pd.DataFrame:
    out = df.copy()
    if cal_df is None:
        return out

    for axis in ["PC1", "PC2"]:
        strict_col = f"{axis}_strict"
        cal_col = f"{axis}_cal"
        row = cal_df.loc[cal_df["axis"] == axis]
        if len(row) == 0:
            continue
        alpha = float(row.iloc[0]["alpha"])
        beta = float(row.iloc[0]["beta"])
        out[cal_col] = alpha + beta * out[strict_col]
    return out


def assign_nearest_ecotype(df: pd.DataFrame, aml_summary: pd.DataFrame) -> pd.DataFrame:
    ref = aml_summary[["PC1", "PC2", "ecotype_label"]].dropna().copy()
    if ref.empty:
        df["ecotype_label_pred_strict"] = np.nan
        return df

    centroids = ref.groupby("ecotype_label")[["PC1", "PC2"]].mean()

    labels = []
    for _, row in df.iterrows():
        x = np.array([row["PC1_strict"], row["PC2_strict"]], dtype=float)
        d2 = ((centroids.values - x[None, :]) ** 2).sum(axis=1)
        labels.append(centroids.index[int(np.argmin(d2))])

    out = df.copy()
    out["ecotype_label_pred_strict"] = labels
    return out


def standardize_reference_summary(df: pd.DataFrame, cohort_name: str) -> pd.DataFrame:
    cols = df.columns.tolist()

    sample_id_col = first_present(cols, ["sample_id", "sample"])
    if sample_id_col is None:
        raise ValueError(f"{cohort_name}: could not find sample_id/sample column")

    pc1_col = first_present(cols, ["PC1", "PC1_cal", "PC1_strict"])
    pc2_col = first_present(cols, ["PC2", "PC2_cal", "PC2_strict"])

    out = pd.DataFrame()
    out["sample_id"] = df[sample_id_col].astype(str)

    for c in ["sample", "gsm", "Patient_ID", "patient_id", "Biopsy_Origin", "timepoint", "transfer_set", "mrd_status",
              "ecotype_label", "ilr_stem_vs_committed", "log_aux_erybaso"]:
        if c in cols:
            out[c] = df[c]

    if pc1_col is not None:
        out["PC1"] = df[pc1_col]
    if pc2_col is not None:
        out["PC2"] = df[pc2_col]

    out["cohort"] = cohort_name
    out["mode"] = "reference"
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--broad-fractions", required=True)
    parser.add_argument("--sample-summary", required=True)
    parser.add_argument("--frozen-asset-dir", required=True)
    parser.add_argument("--secondary-summary", default=None, help="Optional GSE235923 sample summary CSV")
    parser.add_argument("--calibration-csv", default=None, help="Optional axis_calibration_map.csv")
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    asset_dir = Path(args.frozen_asset_dir)

    broad_frac = pd.read_csv(args.broad_fractions, index_col=0)
    broad_frac.index = broad_frac.index.astype(str)

    sample_summary = pd.read_csv(args.sample_summary)
    sample_summary["sample_id"] = sample_summary["sample_id"].astype(str)

    feature_order = load_feature_order(asset_dir)
    scaler_mean = np.load(asset_dir / "ecotype_scaler_mean.npy")
    scaler_scale = np.load(asset_dir / "ecotype_scaler_scale.npy")
    pca_components = np.load(asset_dir / "ecotype_pca_components.npy")

    X = broad_frac.reindex(columns=feature_order, fill_value=0.0).values
    safe_scale = np.where(scaler_scale == 0, 1.0, scaler_scale)
    X_scaled = (X - scaler_mean[None, :]) / safe_scale[None, :]
    pcs = X_scaled @ pca_components.T

    proj = pd.DataFrame(
        pcs,
        index=broad_frac.index,
        columns=["PC1_strict", "PC2_strict"],
    ).reset_index().rename(columns={"index": "sample_id"})

    cal_df = load_calibration_map(Path(args.calibration_csv) if args.calibration_csv else None)
    proj = apply_axiswise_calibration(proj, cal_df)

    aml_summary_path = asset_dir / "training_sample_summary_with_transfer_columns.csv"
    if not aml_summary_path.exists():
        raise FileNotFoundError(f"Missing AML frozen summary: {aml_summary_path}")
    aml_summary = pd.read_csv(aml_summary_path)
    aml_summary["sample_id"] = aml_summary["sample_id"].astype(str)

    proj = assign_nearest_ecotype(proj, aml_summary)

    gse227122 = sample_summary.merge(proj, on="sample_id", how="left")
    gse227122["cohort"] = "GSE227122"

    # Cross-cohort reference table
    aml_ref = standardize_reference_summary(aml_summary, "GSE235063")

    cross = [aml_ref]

    if args.secondary_summary is not None:
        sec_path = Path(args.secondary_summary)
        if sec_path.exists():
            sec_df = pd.read_csv(sec_path)
            sec_ref = standardize_reference_summary(sec_df, "GSE235923")
            cross.append(sec_ref)

    g227_ref = pd.DataFrame({
        "sample_id": gse227122["sample_id"].astype(str),
        "sample": gse227122["sample_id"].astype(str),
        "timepoint": gse227122["timepoint"] if "timepoint" in gse227122.columns else np.nan,
        "transfer_set": gse227122["transfer_set"] if "transfer_set" in gse227122.columns else np.nan,
        "mrd_status": gse227122["mrd_status"] if "mrd_status" in gse227122.columns else np.nan,
        "PC1": gse227122["PC1_strict"],
        "PC2": gse227122["PC2_strict"],
        "ecotype_label": gse227122["ecotype_label_pred_strict"],
        "cohort": "GSE227122",
        "mode": "strict_transfer",
    })
    cross.append(g227_ref)

    if {"PC1_cal", "PC2_cal"}.issubset(gse227122.columns):
        g227_cal = g227_ref.copy()
        g227_cal["PC1"] = gse227122["PC1_cal"]
        g227_cal["PC2"] = gse227122["PC2_cal"]
        g227_cal["mode"] = "calibrated_transfer"
        cross.append(g227_cal)

    cross_df = pd.concat(cross, ignore_index=True, sort=False)

    gse227122.to_csv(outdir / "gse227122_projected_sample_scores.csv", index=False)
    cross_df.to_csv(outdir / "cross_cohort_reference_table.csv", index=False)

    print("\n=== WRITTEN ===")
    print(outdir / "gse227122_projected_sample_scores.csv")
    print(outdir / "cross_cohort_reference_table.csv")

    print("\n=== DIAGNOSTICS ===")
    print("Projected GSE227122 samples:", gse227122["sample_id"].nunique())
    print("Strict PC1/PC2 non-null:",
          int(gse227122["PC1_strict"].notna().sum()),
          int(gse227122["PC2_strict"].notna().sum()))
    if {"PC1_cal", "PC2_cal"}.issubset(gse227122.columns):
        print("Calibrated PC1/PC2 non-null:",
              int(gse227122["PC1_cal"].notna().sum()),
              int(gse227122["PC2_cal"].notna().sum()))


if __name__ == "__main__":
    main()
