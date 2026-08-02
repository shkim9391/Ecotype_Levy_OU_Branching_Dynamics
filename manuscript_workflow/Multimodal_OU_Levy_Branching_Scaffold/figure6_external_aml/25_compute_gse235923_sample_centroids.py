from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import scanpy as sc


# ============================================================
# 1. CONFIG
# ============================================================
FIG6_DIR = Path("/Multimodal_OU_Lévy_Branching_Scaffold/Figure_6")
FIG6_INPUTS = FIG6_DIR / "inputs"
FIG6_DERIVED = FIG6_DIR / "derived"

IN_H5AD = FIG6_INPUTS / "gse235923_longitudinal_malignant_projected.h5ad"

OUT_CSV = FIG6_DERIVED / "gse235923_sample_centroids.csv"

TIME_ORDER = {"DX": 0, "EOI_REM": 1, "REL": 2}


# ============================================================
# 2. HELPERS
# ============================================================
def ensure_dirs() -> None:
    FIG6_DERIVED.mkdir(parents=True, exist_ok=True)


def dominant_value(s: pd.Series) -> str:
    s = s.dropna().astype(str)
    if s.empty:
        return "Unknown"
    return s.value_counts().idxmax()


def dominant_fraction(s: pd.Series) -> float:
    s = s.dropna().astype(str)
    if s.empty:
        return np.nan
    vc = s.value_counts(normalize=True)
    return float(vc.iloc[0])


def safe_numeric_median(s: pd.Series) -> float:
    s = pd.to_numeric(s, errors="coerce")
    return float(np.nanmedian(s)) if s.notna().any() else np.nan


def assert_requirements(adata) -> None:
    obs_req = [
        "sample_id",
        "patient_id",
        "clinical_timepoint_raw",
        "clinical_timepoint_coarse",
        "branch_id",
        "ecotype_label",
        "branch_maxprob",
        "branch_entropy",
    ]
    obsm_req = ["X_fig2", "X_scaffold"]

    miss_obs = [c for c in obs_req if c not in adata.obs.columns]
    miss_obsm = [k for k in obsm_req if k not in adata.obsm.keys()]

    if miss_obs:
        raise ValueError(f"Missing required obs columns: {miss_obs}")
    if miss_obsm:
        raise ValueError(f"Missing required obsm keys: {miss_obsm}")


# ============================================================
# 3. MAIN
# ============================================================
def main() -> None:
    ensure_dirs()

    adata = sc.read_h5ad(IN_H5AD)
    assert_requirements(adata)

    xy = np.asarray(adata.obsm["X_fig2"])
    hd = np.asarray(adata.obsm["X_scaffold"])

    tmp = adata.obs.copy()
    tmp["x2d"] = xy[:, 0]
    tmp["y2d"] = xy[:, 1]

    hd_cols = [f"hd_{i+1}" for i in range(hd.shape[1])]
    hd_df = pd.DataFrame(hd, index=adata.obs_names, columns=hd_cols)
    tmp = tmp.join(hd_df)

    rows = []
    group_cols = ["patient_id", "sample_id", "clinical_timepoint_coarse"]

    for keys, sub in tmp.groupby(group_cols, sort=False):
        patient_id, sample_id, tp = keys

        rec = {
            "patient_id": patient_id,
            "sample_id": sample_id,
            "clinical_timepoint_coarse": tp,
            "time_order": TIME_ORDER.get(tp, 999),
            "n_cells": int(sub.shape[0]),
            "x2d": float(np.nanmedian(sub["x2d"])),
            "y2d": float(np.nanmedian(sub["y2d"])),
            "clinical_timepoint_raw": dominant_value(sub["clinical_timepoint_raw"]),
            "branch_id_dominant": dominant_value(sub["branch_id"]),
            "branch_id_dominant_frac": dominant_fraction(sub["branch_id"]),
            "ecotype_dominant": dominant_value(sub["ecotype_label"]),
            "branch_maxprob_median": safe_numeric_median(sub["branch_maxprob"]),
            "branch_entropy_median": safe_numeric_median(sub["branch_entropy"]),
        }

        for c in hd_cols:
            rec[c] = float(np.nanmean(pd.to_numeric(sub[c], errors="coerce")))

        if "PC1" in sub.columns:
            rec["PC1"] = safe_numeric_median(sub["PC1"])
        else:
            rec["PC1"] = np.nan

        if "PC2" in sub.columns:
            rec["PC2"] = safe_numeric_median(sub["PC2"])
        else:
            rec["PC2"] = np.nan

        rows.append(rec)

    out = pd.DataFrame(rows)
    out = out.sort_values(["patient_id", "time_order", "sample_id"]).reset_index(drop=True)
    out.to_csv(OUT_CSV, index=False)

    print(f"[DONE] Saved {OUT_CSV}")

    print("\n[SUMMARY: sample centroids by phase]")
    print(out["clinical_timepoint_coarse"].value_counts(dropna=False).sort_index())

    print("\n[SUMMARY: patients with longitudinal structure]")
    seq = (
        out.groupby("patient_id")["clinical_timepoint_coarse"]
           .apply(list)
           .reset_index(name="timepoints")
    )
    print(seq.to_string(index=False))

    print("\n[SUMMARY: dominant branch counts]")
    print(out["branch_id_dominant"].value_counts(dropna=False).to_string())

    print("\n[SUMMARY: dominant ecotype counts]")
    print(out["ecotype_dominant"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
