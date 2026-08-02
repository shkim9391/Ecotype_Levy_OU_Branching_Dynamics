from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import scanpy as sc


# ============================================================
# 1. CONFIG
# ============================================================
FIG4_DIR = Path("/Multimodal_OU_Lévy_Branching_Scaffold/Figure_4")
FIG6_DIR = Path("/Multimodal_OU_Lévy_Branching_Scaffold/Figure_6")

FIG4_DERIVED = FIG4_DIR / "derived"
FIG6_INPUTS = FIG6_DIR / "inputs"
FIG6_DERIVED = FIG6_DIR / "derived"

IN_H5AD = FIG6_INPUTS / "gse235923_longitudinal_malignant_projected.h5ad"
IN_DX_ATTRACTOR = FIG4_DERIVED / "dx_attractor_scaffold.tsv"

OUT_CSV = FIG6_DERIVED / "gse235923_sample_dynamic_parameters.csv"

SCAFFOLD_COLS = [
    "state_HSC",
    "state_Prog",
    "state_GMP",
    "state_MonoDC",
    "aux_EryBaso",
    "aux_CLP",
]

REQUIRED_OBS = [
    "sample_id",
    "patient_id",
    "clinical_timepoint_raw",
    "clinical_timepoint_coarse",
    "branch_id",
    "branch_maxprob",
    "branch_entropy",
    "ecotype_label",
] + SCAFFOLD_COLS

MAX_BRANCH_ENTROPY = np.log(4.0)  # 4 main coarse states


# ============================================================
# 2. HELPERS
# ============================================================
def ensure_dirs() -> None:
    FIG6_DERIVED.mkdir(parents=True, exist_ok=True)


def assert_requirements(adata) -> None:
    missing = [c for c in REQUIRED_OBS if c not in adata.obs.columns]
    if missing:
        raise ValueError(f"Projected external object missing required obs columns: {missing}")


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


def robust_z(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    if not np.isfinite(mad) or mad == 0:
        return pd.Series(np.zeros(len(x)), index=s.index, dtype=float)
    return 0.6745 * (x - med) / mad


def euclidean(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.sum((a - b) ** 2)))


def collapse_to_sample_table(adata) -> pd.DataFrame:
    """
    Collapse cell-level projected object to one row per sample.
    Sample-level projected scaffold features are duplicated across malignant cells,
    so median/first are effectively equivalent; we use robust aggregation for clarity.
    """
    obs = adata.obs.copy()

    rows = []
    for sample_id, sub in obs.groupby("sample_id", sort=False):
        rec = {
            "sample_id": str(sample_id),
            "patient_id": dominant_value(sub["patient_id"]),
            "clinical_timepoint_raw": dominant_value(sub["clinical_timepoint_raw"]),
            "clinical_timepoint_coarse": dominant_value(sub["clinical_timepoint_coarse"]),
            "n_cells": int(sub.shape[0]),
            "branch_id_dominant": dominant_value(sub["branch_id"]),
            "branch_id_dominant_frac": dominant_fraction(sub["branch_id"]),
            "ecotype_label": dominant_value(sub["ecotype_label"]),
            "branch_maxprob": float(np.nanmedian(pd.to_numeric(sub["branch_maxprob"], errors="coerce"))),
            "branch_entropy": float(np.nanmedian(pd.to_numeric(sub["branch_entropy"], errors="coerce"))),
        }

        for c in SCAFFOLD_COLS:
            rec[c] = float(np.nanmedian(pd.to_numeric(sub[c], errors="coerce")))

        if "PC1" in sub.columns:
            rec["PC1"] = safe_numeric_median(sub["PC1"])
        else:
            rec["PC1"] = np.nan

        if "PC2" in sub.columns:
            rec["PC2"] = safe_numeric_median(sub["PC2"])
        else:
            rec["PC2"] = np.nan

        rows.append(rec)

    return pd.DataFrame(rows)


def load_discovery_dx_attractor(fp: Path) -> np.ndarray:
    dx = pd.read_csv(fp, sep="\t")
    req = {"feature", "dx_attractor_value"}
    if not req.issubset(dx.columns):
        raise ValueError(f"dx_attractor_scaffold.tsv missing required columns: {sorted(req)}")

    dx = dx.copy()
    dx["feature"] = dx["feature"].astype(str)
    dx = dx.set_index("feature")

    missing = [c for c in SCAFFOLD_COLS if c not in dx.index]
    if missing:
        raise ValueError(f"DX attractor table missing scaffold features: {missing}")

    return dx.loc[SCAFFOLD_COLS, "dx_attractor_value"].to_numpy(dtype=float)


def add_effective_parameters(sample_df: pd.DataFrame, dx_attractor_vec: np.ndarray) -> pd.DataFrame:
    out = sample_df.copy()

    # Same definitions as Figure 4
    out["theta_eff"] = pd.to_numeric(out["branch_maxprob"], errors="coerce")
    out["sigma_eff"] = pd.to_numeric(out["branch_entropy"], errors="coerce") / MAX_BRANCH_ENTROPY

    mu_shift = []
    for _, row in out.iterrows():
        x = row[SCAFFOLD_COLS].to_numpy(dtype=float)
        mu_shift.append(euclidean(x, dx_attractor_vec))
    out["mu_shift_from_dx"] = mu_shift

    out["theta_eff_z"] = robust_z(out["theta_eff"])
    out["sigma_eff_z"] = robust_z(out["sigma_eff"])
    out["mu_shift_from_dx_z"] = robust_z(out["mu_shift_from_dx"])

    out["theta_sigma_ratio"] = out["theta_eff"] / (out["sigma_eff"] + 1e-6)

    phase_order = {"DX": 0, "EOI_REM": 1, "REL": 2}
    out["phase_order"] = out["clinical_timepoint_coarse"].map(phase_order).fillna(999).astype(int)

    # Helpful external longitudinal flags
    tp_counts = out.groupby("patient_id")["clinical_timepoint_coarse"].nunique().to_dict()
    out["n_timepoints_for_patient"] = out["patient_id"].map(tp_counts).astype(int)
    out["has_longitudinal_followup"] = out["n_timepoints_for_patient"] >= 2

    return out


# ============================================================
# 3. MAIN
# ============================================================
def main() -> None:
    ensure_dirs()

    adata = sc.read_h5ad(IN_H5AD)
    assert_requirements(adata)

    sample_df = collapse_to_sample_table(adata)
    dx_attractor_vec = load_discovery_dx_attractor(IN_DX_ATTRACTOR)
    out = add_effective_parameters(sample_df, dx_attractor_vec)

    out = out.sort_values(["patient_id", "phase_order", "sample_id"]).reset_index(drop=True)
    out.to_csv(OUT_CSV, index=False)

    print(f"[DONE] Saved {OUT_CSV}")

    print("\n[SUMMARY: sample counts by phase]")
    print(out["clinical_timepoint_coarse"].value_counts(dropna=False).sort_index())

    print("\n[SUMMARY: theta_eff by phase]")
    print(
        out.groupby("clinical_timepoint_coarse")["theta_eff"]
           .describe()
           .round(4)
           .to_string()
    )

    print("\n[SUMMARY: sigma_eff by phase]")
    print(
        out.groupby("clinical_timepoint_coarse")["sigma_eff"]
           .describe()
           .round(4)
           .to_string()
    )

    print("\n[SUMMARY: mu_shift_from_dx by phase]")
    print(
        out.groupby("clinical_timepoint_coarse")["mu_shift_from_dx"]
           .describe()
           .round(4)
           .to_string()
    )

    print("\n[SUMMARY: patients with >=2 timepoints]")
    print(
        out.loc[out["has_longitudinal_followup"], ["patient_id", "n_timepoints_for_patient"]]
           .drop_duplicates()
           .sort_values(["n_timepoints_for_patient", "patient_id"], ascending=[False, True])
           .to_string(index=False)
    )

    triplets = out.groupby("patient_id")["clinical_timepoint_coarse"].apply(list).reset_index(name="timepoints")
    triplets = triplets[triplets["timepoints"].apply(lambda x: set(x) == {"DX", "EOI_REM", "REL"})]
    if not triplets.empty:
        print("\n[SUMMARY: full DX->EOI_REM->REL patients]")
        print(triplets.to_string(index=False))


if __name__ == "__main__":
    main()
