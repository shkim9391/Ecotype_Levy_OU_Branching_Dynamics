from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import scanpy as sc


# ============================================================
# 1. CONFIG
# ============================================================
PROJECT_DIR = Path("/Multimodal_OU_Lévy_Branching_Scaffold/Figure_3")
INPUTS_DIR = PROJECT_DIR / "inputs"
DERIVED_DIR = PROJECT_DIR / "derived"

IN_H5AD = INPUTS_DIR / "gse235063_longitudinal_malignant_projected.h5ad"
MAIN_PATIENTS_TXT = DERIVED_DIR / "figure3_main_analysis_patients.txt"

OUT_ALL = DERIVED_DIR / "patient_timepoint_centroids_all.csv"
OUT_MAIN = DERIVED_DIR / "patient_timepoint_centroids_main.csv"

TIME_ORDER = {"DX": 0, "EOI_REM": 1, "REL": 2}
MIN_CELLS_PER_TIMEPOINT = 50
EXCLUDE_PATIENTS_FROM_MAIN = {"AML1"}


# ============================================================
# 2. HELPERS
# ============================================================
def load_main_patients(fp: Path) -> set[str]:
    pts = []
    with open(fp, "r") as f:
        for line in f:
            p = line.strip()
            if p:
                pts.append(p)
    return set(pts)


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
        "disease_subgroup",
        "branch_id",
        "ecotype_label",
        "reg_program_score" if "reg_program_score" in adata.obs.columns else None,
    ]
    obs_req = [x for x in obs_req if x is not None]

    miss_obs = [c for c in obs_req if c not in adata.obs.columns]
    miss_obsm = [k for k in ["X_fig2", "X_scaffold"] if k not in adata.obsm.keys()]

    if miss_obs:
        raise ValueError(f"Missing required obs columns: {miss_obs}")
    if miss_obsm:
        raise ValueError(f"Missing required obsm keys: {miss_obsm}")


def build_cell_table(adata) -> pd.DataFrame:
    xy = np.asarray(adata.obsm["X_fig2"])
    hd = np.asarray(adata.obsm["X_scaffold"])

    tmp = adata.obs.copy()
    tmp["x2d"] = xy[:, 0]
    tmp["y2d"] = xy[:, 1]

    hd_cols = [f"hd_{i+1}" for i in range(hd.shape[1])]
    hd_df = pd.DataFrame(hd, index=adata.obs_names, columns=hd_cols)
    tmp = tmp.join(hd_df)

    return tmp, hd_cols


def compute_centroids(df: pd.DataFrame, hd_cols: list[str]) -> pd.DataFrame:
    rows = []

    group_cols = ["patient_id", "sample_id", "clinical_timepoint_coarse"]
    for keys, sub in df.groupby(group_cols, sort=False):
        patient_id, sample_id, tp = keys

        rec = {
            "patient_id": patient_id,
            "sample_id": sample_id,
            "clinical_timepoint_coarse": tp,
            "time_order": TIME_ORDER.get(tp, 999),
            "n_cells": int(sub.shape[0]),
            "x2d": float(np.nanmedian(sub["x2d"])),
            "y2d": float(np.nanmedian(sub["y2d"])),
        }

        if "clinical_timepoint_raw" in sub.columns:
            rec["clinical_timepoint_raw"] = dominant_value(sub["clinical_timepoint_raw"])
        else:
            rec["clinical_timepoint_raw"] = tp

        if "disease_subgroup" in sub.columns:
            rec["disease_subgroup"] = dominant_value(sub["disease_subgroup"])
        else:
            rec["disease_subgroup"] = "Unknown"

        if "branch_id" in sub.columns:
            rec["branch_id_dominant"] = dominant_value(sub["branch_id"])
            rec["branch_id_dominant_frac"] = dominant_fraction(sub["branch_id"])
        else:
            rec["branch_id_dominant"] = "Unknown"
            rec["branch_id_dominant_frac"] = np.nan

        if "ecotype_label" in sub.columns:
            rec["ecotype_dominant"] = dominant_value(sub["ecotype_label"])
        else:
            rec["ecotype_dominant"] = "Unknown"

        if "reg_program_score" in sub.columns:
            rec["reg_program_score_median"] = safe_numeric_median(sub["reg_program_score"])
        else:
            rec["reg_program_score_median"] = np.nan

        for c in hd_cols:
            rec[c] = float(np.nanmean(pd.to_numeric(sub[c], errors="coerce")))

        rows.append(rec)

    out = pd.DataFrame(rows)
    out = out.sort_values(["patient_id", "time_order", "sample_id"]).reset_index(drop=True)
    return out


def filter_main_analysis(cent: pd.DataFrame, main_patients: set[str]) -> pd.DataFrame:
    out = cent.copy()

    out = out[out["patient_id"].isin(main_patients)].copy()
    out = out[~out["patient_id"].isin(EXCLUDE_PATIENTS_FROM_MAIN)].copy()
    out = out[out["n_cells"] >= MIN_CELLS_PER_TIMEPOINT].copy()

    # Keep only patients that still have both DX and REL after filtering
    keep_patients = []
    for patient_id, sub in out.groupby("patient_id", sort=False):
        tps = set(sub["clinical_timepoint_coarse"].astype(str))
        if "DX" in tps and "REL" in tps:
            keep_patients.append(patient_id)

    out = out[out["patient_id"].isin(keep_patients)].copy()
    out = out.sort_values(["patient_id", "time_order", "sample_id"]).reset_index(drop=True)
    return out


# ============================================================
# 3. MAIN
# ============================================================
def main() -> None:
    DERIVED_DIR.mkdir(parents=True, exist_ok=True)

    adata = sc.read_h5ad(IN_H5AD)
    assert_requirements(adata)

    main_patients = load_main_patients(MAIN_PATIENTS_TXT)

    cell_df, hd_cols = build_cell_table(adata)
    cent_all = compute_centroids(cell_df, hd_cols)
    cent_main = filter_main_analysis(cent_all, main_patients)

    cent_all.to_csv(OUT_ALL, index=False)
    cent_main.to_csv(OUT_MAIN, index=False)

    print(f"[DONE] Saved all centroids:  {OUT_ALL}")
    print(f"[DONE] Saved main centroids: {OUT_MAIN}")

    print("\n[SUMMARY: all centroids by timepoint]")
    print(cent_all["clinical_timepoint_coarse"].value_counts(dropna=False).sort_index())

    print("\n[SUMMARY: main-analysis centroids by timepoint]")
    print(cent_main["clinical_timepoint_coarse"].value_counts(dropna=False).sort_index())

    print("\n[SUMMARY: main-analysis patients]")
    print(sorted(cent_main['patient_id'].unique().tolist()))

    if "AML21" in set(cent_main["patient_id"].astype(str)):
        print("\n[INFO] AML21 retained as three-timepoint case:")
        print(
            cent_main[cent_main["patient_id"] == "AML21"][
                ["patient_id", "sample_id", "clinical_timepoint_coarse", "n_cells", "x2d", "y2d", "branch_id_dominant"]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()
