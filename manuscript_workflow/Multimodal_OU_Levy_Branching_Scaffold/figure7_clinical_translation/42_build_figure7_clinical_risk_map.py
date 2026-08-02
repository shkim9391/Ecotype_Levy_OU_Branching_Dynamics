from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


# ============================================================
# 1. CONFIG
# ============================================================
BASE = Path("/Multimodal_OU_Lévy_Branching_Scaffold")

FIG4_DERIVED = BASE / "Figure_4" / "derived"
FIG6_DERIVED = BASE / "Figure_6" / "derived"
FIG7_DERIVED = BASE / "Figure_7" / "derived"

IN_DISC = FIG4_DERIVED / "sample_dynamic_parameters.csv"
IN_EXT = FIG6_DERIVED / "gse235923_sample_dynamic_parameters.csv"

OUT_CSV = FIG7_DERIVED / "figure7_clinical_risk_map.csv"
OUT_SUMMARY = FIG7_DERIVED / "figure7_clinical_risk_map_summary.tsv"

REQUIRED_DISC = [
    "sample_id",
    "patient_id",
    "clinical_timepoint_coarse",
    "n_cells",
    "theta_eff",
    "sigma_eff",
    "mu_shift_from_dx",
    "branch_id_dominant",
    "ecotype_label",
    "is_main_analysis_sample",
]

REQUIRED_EXT = [
    "sample_id",
    "patient_id",
    "clinical_timepoint_coarse",
    "n_cells",
    "theta_eff",
    "sigma_eff",
    "mu_shift_from_dx",
    "branch_id_dominant",
    "ecotype_label",
]

PHASE_ORDER = {"DX": 0, "EOI_REM": 1, "REL": 2}
COHORT_ORDER = {"discovery": 0, "external_aml": 1}

RISK_ZONE_ORDER = {
    "Constrained / response-like": 0,
    "Residual persistent / unstable": 1,
    "Escape-prone / displaced": 2,
}

RISK_TIER_MAP = {
    "Constrained / response-like": "Low-risk constrained",
    "Residual persistent / unstable": "Intermediate residual",
    "Escape-prone / displaced": "High-risk escape-prone",
}


# ============================================================
# 2. HELPERS
# ============================================================
def ensure_dirs() -> None:
    FIG7_DERIVED.mkdir(parents=True, exist_ok=True)


def assert_columns(df: pd.DataFrame, required: list[str], label: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{label} missing required columns: {missing}")


def to_bool_series(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False)

    x = s.astype(str).str.strip().str.lower()
    return x.isin(["true", "1", "t", "yes"])


def robust_center_scale(x: pd.Series) -> tuple[float, float]:
    vals = pd.to_numeric(x, errors="coerce").to_numpy(dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.nan, np.nan

    med = float(np.nanmedian(vals))
    mad = float(np.nanmedian(np.abs(vals - med)))
    return med, mad


def apply_robust_z(x: pd.Series, center: float, scale: float) -> pd.Series:
    vals = pd.to_numeric(x, errors="coerce")
    if not np.isfinite(center):
        return pd.Series(np.nan, index=x.index, dtype=float)
    if (not np.isfinite(scale)) or scale == 0:
        return pd.Series(np.zeros(len(vals)), index=x.index, dtype=float)
    return 0.6745 * (vals - center) / scale


def harmonize_discovery(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    assert_columns(df, REQUIRED_DISC, "sample_dynamic_parameters.csv")

    df["is_main_analysis_sample"] = to_bool_series(df["is_main_analysis_sample"])
    df = df[df["is_main_analysis_sample"]].copy()

    df["cohort"] = "GSE235063"
    df["source_group"] = "discovery"
    df["has_longitudinal_followup"] = True  # discovery main set is longitudinal by construction
    df["usable_for_summary"] = True

    keep = [
        "cohort",
        "source_group",
        "sample_id",
        "patient_id",
        "clinical_timepoint_coarse",
        "n_cells",
        "theta_eff",
        "sigma_eff",
        "mu_shift_from_dx",
        "branch_id_dominant",
        "ecotype_label",
        "has_longitudinal_followup",
        "usable_for_summary",
    ]
    return df[keep].copy()


def harmonize_external(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    assert_columns(df, REQUIRED_EXT, "gse235923_sample_dynamic_parameters.csv")

    df["cohort"] = "GSE235923"
    df["source_group"] = "external_aml"

    if "has_longitudinal_followup" not in df.columns:
        df["has_longitudinal_followup"] = False
    else:
        df["has_longitudinal_followup"] = to_bool_series(df["has_longitudinal_followup"])

    if "usable_for_summary" not in df.columns:
        df["usable_for_summary"] = True
    else:
        df["usable_for_summary"] = to_bool_series(df["usable_for_summary"])

    keep = [
        "cohort",
        "source_group",
        "sample_id",
        "patient_id",
        "clinical_timepoint_coarse",
        "n_cells",
        "theta_eff",
        "sigma_eff",
        "mu_shift_from_dx",
        "branch_id_dominant",
        "ecotype_label",
        "has_longitudinal_followup",
        "usable_for_summary",
    ]
    return df[keep].copy()


def assign_risk_zone(
    mu_shift: float,
    theta_eff: float,
    sigma_eff: float,
    mu_q75: float,
    theta_med: float,
    sigma_med: float,
) -> str:
    if not np.isfinite(mu_shift) or not np.isfinite(theta_eff) or not np.isfinite(sigma_eff):
        return "Residual persistent / unstable"

    if mu_shift >= mu_q75:
        return "Escape-prone / displaced"

    if (theta_eff >= theta_med) and (sigma_eff <= sigma_med):
        return "Constrained / response-like"

    return "Residual persistent / unstable"


# ============================================================
# 3. MAIN
# ============================================================
def main() -> None:
    ensure_dirs()

    disc = pd.read_csv(IN_DISC)
    ext = pd.read_csv(IN_EXT)

    disc = harmonize_discovery(disc)
    ext = harmonize_external(ext)

    combined = pd.concat([disc, ext], axis=0, ignore_index=True)

    # numeric coercion
    for c in ["n_cells", "theta_eff", "sigma_eff", "mu_shift_from_dx"]:
        combined[c] = pd.to_numeric(combined[c], errors="coerce")

    combined["clinical_timepoint_coarse"] = combined["clinical_timepoint_coarse"].astype(str)
    combined["branch_id_dominant"] = combined["branch_id_dominant"].astype(str)
    combined["ecotype_label"] = combined["ecotype_label"].astype(str)

    # --------------------------------------------------------
    # Discovery-anchored thresholds for clinical interpretation
    # --------------------------------------------------------
    ref = combined[combined["source_group"] == "discovery"].copy()
    if ref.empty:
        raise ValueError("No discovery reference samples available after harmonization.")

    mu_med = float(np.nanmedian(ref["mu_shift_from_dx"]))
    mu_q75 = float(np.nanquantile(ref["mu_shift_from_dx"], 0.75))
    theta_med = float(np.nanmedian(ref["theta_eff"]))
    sigma_med = float(np.nanmedian(ref["sigma_eff"]))

    # robust score components anchored to discovery
    mu_ctr, mu_mad = robust_center_scale(ref["mu_shift_from_dx"])
    th_ctr, th_mad = robust_center_scale(ref["theta_eff"])
    sg_ctr, sg_mad = robust_center_scale(ref["sigma_eff"])

    combined["mu_shift_z_ref"] = apply_robust_z(combined["mu_shift_from_dx"], mu_ctr, mu_mad)
    combined["theta_eff_z_ref"] = apply_robust_z(combined["theta_eff"], th_ctr, th_mad)
    combined["sigma_eff_z_ref"] = apply_robust_z(combined["sigma_eff"], sg_ctr, sg_mad)

    # higher score = more clinically concerning
    combined["clinical_risk_score"] = (
        combined["mu_shift_z_ref"].fillna(0.0)
        - combined["theta_eff_z_ref"].fillna(0.0)
        + combined["sigma_eff_z_ref"].fillna(0.0)
    )

    combined["risk_zone"] = [
        assign_risk_zone(mu, th, sg, mu_q75, theta_med, sigma_med)
        for mu, th, sg in zip(
            combined["mu_shift_from_dx"],
            combined["theta_eff"],
            combined["sigma_eff"],
        )
    ]
    combined["risk_tier"] = combined["risk_zone"].map(RISK_TIER_MAP)

    combined["phase_order"] = combined["clinical_timepoint_coarse"].map(PHASE_ORDER).fillna(999).astype(int)
    combined["cohort_order"] = combined["source_group"].map(COHORT_ORDER).fillna(999).astype(int)
    combined["risk_zone_order"] = combined["risk_zone"].map(RISK_ZONE_ORDER).fillna(999).astype(int)

    # helpful display label for later panels
    combined["display_label"] = combined["sample_id"].astype(str)

    combined = combined.sort_values(
        ["cohort_order", "phase_order", "patient_id", "sample_id"]
    ).reset_index(drop=True)

    combined.to_csv(OUT_CSV, index=False)

    # --------------------------------------------------------
    # Summary table for later panels
    # --------------------------------------------------------
    summary_rows = [
        {
            "section": "thresholds",
            "item": "mu_shift_median_ref",
            "value": f"{mu_med:.6f}",
        },
        {
            "section": "thresholds",
            "item": "mu_shift_q75_ref",
            "value": f"{mu_q75:.6f}",
        },
        {
            "section": "thresholds",
            "item": "theta_eff_median_ref",
            "value": f"{theta_med:.6f}",
        },
        {
            "section": "thresholds",
            "item": "sigma_eff_median_ref",
            "value": f"{sigma_med:.6f}",
        },
    ]

    zone_counts = (
        combined.groupby(["cohort", "clinical_timepoint_coarse", "risk_zone"])
        .size()
        .reset_index(name="n_samples")
    )
    for _, r in zone_counts.iterrows():
        summary_rows.append({
            "section": "zone_counts",
            "item": f"{r['cohort']} | {r['clinical_timepoint_coarse']} | {r['risk_zone']}",
            "value": str(int(r["n_samples"])),
        })

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT_SUMMARY, sep="\t", index=False)

    print(f"[DONE] Saved {OUT_CSV}")
    print(f"[DONE] Saved {OUT_SUMMARY}")

    print("\n[SUMMARY: discovery thresholds]")
    print(f"mu_shift median = {mu_med:.4f}")
    print(f"mu_shift q75    = {mu_q75:.4f}")
    print(f"theta median    = {theta_med:.4f}")
    print(f"sigma median    = {sigma_med:.4f}")

    print("\n[SUMMARY: risk zones by cohort and phase]")
    print(
        zone_counts.sort_values(["cohort", "clinical_timepoint_coarse", "risk_zone"])
        .to_string(index=False)
    )

    print("\n[SUMMARY: representative highest-risk samples]")
    rep = combined.sort_values("clinical_risk_score", ascending=False).head(10)
    print(
        rep[
            [
                "cohort",
                "sample_id",
                "patient_id",
                "clinical_timepoint_coarse",
                "theta_eff",
                "sigma_eff",
                "mu_shift_from_dx",
                "clinical_risk_score",
                "risk_zone",
                "risk_tier",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
