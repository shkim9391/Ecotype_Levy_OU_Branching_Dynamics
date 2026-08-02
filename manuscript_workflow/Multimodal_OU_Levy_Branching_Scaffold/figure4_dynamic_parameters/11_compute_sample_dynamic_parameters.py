from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import scanpy as sc


# ============================================================
# 1. CONFIG
# ============================================================
FIG3_DIR = Path("/Multimodal_OU_Lévy_Branching_Scaffold/Figure_3")
FIG4_DIR = Path("/Multimodal_OU_Lévy_Branching_Scaffold/Figure_4")

FIG3_INPUTS = FIG3_DIR / "inputs"
FIG3_DERIVED = FIG3_DIR / "derived"
FIG4_DERIVED = FIG4_DIR / "derived"

IN_H5AD = FIG3_INPUTS / "gse235063_longitudinal_malignant_projected.h5ad"
MAIN_PATIENTS_TXT = FIG3_DERIVED / "figure3_main_analysis_patients.txt"

OUT_CSV = FIG4_DERIVED / "sample_dynamic_parameters.csv"
OUT_ATTRACTOR = FIG4_DERIVED / "dx_attractor_scaffold.tsv"

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
    "clinical_timepoint_coarse",
    "branch_id",
    "branch_maxprob",
    "branch_entropy",
    "ecotype_label",
] + SCAFFOLD_COLS

EXPLORATORY_PATIENTS = {"AML1"}
MIN_MAIN_CELLS = 50
MAX_BRANCH_ENTROPY = np.log(4.0)  # 4 main coarse states


# ============================================================
# 2. HELPERS
# ============================================================
def ensure_dirs() -> None:
    FIG4_DERIVED.mkdir(parents=True, exist_ok=True)


def load_main_patients(fp: Path) -> set[str]:
    pts = []
    with open(fp, "r") as f:
        for line in f:
            p = line.strip()
            if p:
                pts.append(p)
    return set(pts)


def assert_requirements(adata) -> None:
    missing = [c for c in REQUIRED_OBS if c not in adata.obs.columns]
    if missing:
        raise ValueError(f"Projected object missing required obs columns: {missing}")


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
    Since projected scaffold features are sample-level and replicated across cells,
    median/first are equivalent; we use robust aggregation for clarity.
    """
    obs = adata.obs.copy()

    rows = []
    for sample_id, sub in obs.groupby("sample_id", sort=False):
        rec = {
            "sample_id": str(sample_id),
            "patient_id": dominant_value(sub["patient_id"]),
            "clinical_timepoint_coarse": dominant_value(sub["clinical_timepoint_coarse"]),
            "n_cells": int(sub.shape[0]),
            "branch_id_dominant": dominant_value(sub["branch_id"]),
            "branch_id_dominant_frac": dominant_fraction(sub["branch_id"]),
            "ecotype_label": dominant_value(sub["ecotype_label"]),
            "branch_maxprob": float(np.nanmedian(pd.to_numeric(sub["branch_maxprob"], errors="coerce"))),
            "branch_entropy": float(np.nanmedian(pd.to_numeric(sub["branch_entropy"], errors="coerce"))),
        }

        # Carry scaffold coordinates
        for c in SCAFFOLD_COLS:
            rec[c] = float(np.nanmedian(pd.to_numeric(sub[c], errors="coerce")))

        # Carry 2D projected coordinates if present in obs
        if "PC1" in sub.columns:
            rec["PC1"] = float(np.nanmedian(pd.to_numeric(sub["PC1"], errors="coerce")))
        else:
            rec["PC1"] = np.nan

        if "PC2" in sub.columns:
            rec["PC2"] = float(np.nanmedian(pd.to_numeric(sub["PC2"], errors="coerce")))
        else:
            rec["PC2"] = np.nan

        rows.append(rec)

    out = pd.DataFrame(rows)
    return out


def compute_dx_attractor(sample_df: pd.DataFrame) -> pd.Series:
    dx = sample_df[sample_df["clinical_timepoint_coarse"] == "DX"].copy()
    if dx.empty:
        raise ValueError("No DX samples found; cannot compute diagnosis attractor.")

    attractor = dx[SCAFFOLD_COLS].median(axis=0)
    attractor.name = "dx_attractor"
    return attractor


def add_effective_parameters(sample_df: pd.DataFrame, dx_attractor: pd.Series, main_patients: set[str]) -> pd.DataFrame:
    out = sample_df.copy()

    # Effective restoring strength proxy
    out["theta_eff"] = pd.to_numeric(out["branch_maxprob"], errors="coerce")

    # Effective instability / diffusion proxy
    out["sigma_eff"] = pd.to_numeric(out["branch_entropy"], errors="coerce") / MAX_BRANCH_ENTROPY

    # Attractor displacement from robust DX center
    dx_vec = dx_attractor[SCAFFOLD_COLS].to_numpy(dtype=float)

    mu_shift = []
    for _, row in out.iterrows():
        x = row[SCAFFOLD_COLS].to_numpy(dtype=float)
        mu_shift.append(euclidean(x, dx_vec))
    out["mu_shift_from_dx"] = mu_shift

    # Main-analysis flags
    out["is_main_analysis_patient"] = out["patient_id"].astype(str).isin(main_patients)
    out["is_qc_flagged_patient"] = out["patient_id"].astype(str).isin(EXPLORATORY_PATIENTS)
    out["is_main_analysis_sample"] = (
        out["is_main_analysis_patient"]
        & (~out["is_qc_flagged_patient"])
        & (pd.to_numeric(out["n_cells"], errors="coerce") >= MIN_MAIN_CELLS)
    )

    # Optional robust z-scores for downstream plotting
    out["theta_eff_z"] = robust_z(out["theta_eff"])
    out["sigma_eff_z"] = robust_z(out["sigma_eff"])
    out["mu_shift_from_dx_z"] = robust_z(out["mu_shift_from_dx"])

    # Helpful regime-oriented summaries
    out["theta_sigma_ratio"] = out["theta_eff"] / (out["sigma_eff"] + 1e-6)

    # Phase ordering
    phase_order = {"DX": 0, "EOI_REM": 1, "REL": 2}
    out["phase_order"] = out["clinical_timepoint_coarse"].map(phase_order).fillna(999).astype(int)

    return out


# ============================================================
# 3. MAIN
# ============================================================
def main() -> None:
    ensure_dirs()

    adata = sc.read_h5ad(IN_H5AD)
    assert_requirements(adata)

    main_patients = load_main_patients(MAIN_PATIENTS_TXT)

    sample_df = collapse_to_sample_table(adata)
    dx_attractor = compute_dx_attractor(sample_df)
    out = add_effective_parameters(sample_df, dx_attractor, main_patients)

    # Save outputs
    out = out.sort_values(["patient_id", "phase_order", "sample_id"]).reset_index(drop=True)
    out.to_csv(OUT_CSV, index=False)

    dx_attr_df = pd.DataFrame({
        "feature": SCAFFOLD_COLS,
        "dx_attractor_value": dx_attractor[SCAFFOLD_COLS].to_numpy(dtype=float),
    })
    dx_attr_df.to_csv(OUT_ATTRACTOR, sep="\t", index=False)

    print(f"[DONE] Saved {OUT_CSV}")
    print(f"[DONE] Saved {OUT_ATTRACTOR}")

    print("\n[SUMMARY: sample counts by phase]")
    print(out["clinical_timepoint_coarse"].value_counts(dropna=False).sort_index())

    print("\n[SUMMARY: main-analysis sample counts by phase]")
    print(
        out.loc[out["is_main_analysis_sample"], "clinical_timepoint_coarse"]
           .value_counts(dropna=False)
           .sort_index()
    )

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

    if "AML21" in set(out["patient_id"].astype(str)):
        print("\n[INFO] AML21 sample dynamic parameters]")
        print(
            out[out["patient_id"] == "AML21"][
                [
                    "patient_id",
                    "sample_id",
                    "clinical_timepoint_coarse",
                    "n_cells",
                    "theta_eff",
                    "sigma_eff",
                    "mu_shift_from_dx",
                    "branch_id_dominant",
                    "ecotype_label",
                ]
            ].to_string(index=False)
        )

    if "AML1" in set(out["patient_id"].astype(str)):
        print("\n[INFO] AML1 exploratory sample dynamic parameters]")
        print(
            out[out["patient_id"] == "AML1"][
                [
                    "patient_id",
                    "sample_id",
                    "clinical_timepoint_coarse",
                    "n_cells",
                    "theta_eff",
                    "sigma_eff",
                    "mu_shift_from_dx",
                    "branch_id_dominant",
                    "ecotype_label",
                    "is_qc_flagged_patient",
                ]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()
