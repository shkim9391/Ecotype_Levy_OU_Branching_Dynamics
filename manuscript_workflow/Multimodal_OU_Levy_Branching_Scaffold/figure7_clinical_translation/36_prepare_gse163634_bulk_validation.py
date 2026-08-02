from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


# ============================================================
# 1. CONFIG
# ============================================================
SRC_DIR = Path("/Ecotype_OU_Branching/GSE163634")
FIG4_DIR = Path("/Multimodal_OU_Lévy_Branching_Scaffold/Figure_4")
FIG7_DIR = Path("/Multimodal_OU_Lévy_Branching_Scaffold/Figure_7")

IN_META = SRC_DIR / "derived_bulk_start" / "gse163634_sample_metadata.csv"
IN_SCORE = SRC_DIR / "derived_pc12_recovery" / "gse163634_bulk_score_matrix_with_pc12.csv"
IN_DELTAS = SRC_DIR / "derived_pc12_recovery" / "gse163634_bulk_serial_deltas_with_pc12.csv"
IN_LVC = SRC_DIR / "derived_bulk_validation_with_pc12" / "gse163634_bulk_leukemia_vs_control_stats.csv"
IN_DX_ATTRACTOR = FIG4_DIR / "derived" / "dx_attractor_scaffold.tsv"

OUT_STATE = FIG7_DIR / "derived" / "gse163634_bulk_state_variables.csv"
OUT_SUMMARY = FIG7_DIR / "derived" / "gse163634_bulk_validation_summary.csv"
OUT_REPL = FIG7_DIR / "derived" / "gse163634_calibration_summary.tsv"

STATE_REQ = [
    "sample_id",
    "patient_id",
    "stage",
    "stage_order",
    "is_control",
    "is_leukemia",
    "pair_group",
    "ilr_stem_vs_committed_cal",
    "ilr_prog_vs_mature_cal",
    "ilr_gmp_vs_monodc_cal",
    "log_aux_clp_cal",
    "log_aux_erybaso_cal",
    "pc1_cal",
    "pc2_cal",
]

DX_ATTRACTOR_FEATURES = [
    "state_HSC",
    "state_Prog",
    "state_GMP",
    "state_MonoDC",
    "aux_EryBaso",
    "aux_CLP",
]


# ============================================================
# 2. HELPERS
# ============================================================
def ensure_dirs() -> None:
    (FIG7_DIR / "derived").mkdir(parents=True, exist_ok=True)


def assert_columns(df: pd.DataFrame, required: list[str], label: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{label} missing required columns: {missing}")


def load_discovery_dx_attractor(fp: Path) -> np.ndarray:
    dx = pd.read_csv(fp, sep="\t")
    req = {"feature", "dx_attractor_value"}
    if not req.issubset(dx.columns):
        raise ValueError(f"dx_attractor_scaffold.tsv missing required columns: {sorted(req)}")

    dx = dx.copy()
    dx["feature"] = dx["feature"].astype(str)
    dx = dx.set_index("feature")

    missing = [c for c in DX_ATTRACTOR_FEATURES if c not in dx.index]
    if missing:
        raise ValueError(f"DX attractor table missing scaffold features: {missing}")

    return dx.loc[DX_ATTRACTOR_FEATURES, "dx_attractor_value"].to_numpy(dtype=float)


def euclidean(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.sum((a - b) ** 2)))


def stage_to_group(stage: str, is_control: bool, is_leukemia: bool) -> str:
    if bool(is_control):
        return "control"
    s = str(stage).lower()
    if s == "dx":
        return "dx_leukemia"
    if s == "r1":
        return "response_r1"
    if s == "r2":
        return "response_r2"
    return "other"


def stage_to_timepoint(stage: str) -> str:
    s = str(stage).lower()
    if s == "dx":
        return "DX"
    if s in {"r1", "r2"}:
        return "EOI_REM"
    return str(stage).upper()


# ============================================================
# 3. MAIN
# ============================================================
def main() -> None:
    ensure_dirs()

    meta = pd.read_csv(IN_META)
    score = pd.read_csv(IN_SCORE)
    deltas = pd.read_csv(IN_DELTAS)
    lvc = pd.read_csv(IN_LVC)

    assert_columns(meta, ["sample_id", "patient_id", "stage", "stage_order", "is_control", "is_leukemia", "pair_group"], "gse163634_sample_metadata.csv")
    assert_columns(score, STATE_REQ, "gse163634_bulk_score_matrix_with_pc12.csv")
    assert_columns(deltas, ["patient_id", "transition", "from_sample", "to_sample", "from_stage", "to_stage"], "gse163634_bulk_serial_deltas_with_pc12.csv")
    assert_columns(lvc, ["axis", "score_column", "n_leukemia", "n_control", "median_leukemia", "median_control", "median_difference", "auroc_leukemia_vs_control", "mannwhitney_p", "fdr_q"], "gse163634_bulk_leukemia_vs_control_stats.csv")

    dx_attractor = load_discovery_dx_attractor(IN_DX_ATTRACTOR)

    # ---------------------------------------------------------
    # Build standardized bulk state-variable table
    # ---------------------------------------------------------
    df = score.copy()

    # map calibrated bulk axes to scaffold-like 6D state space
    df["state_HSC"] = pd.to_numeric(df["ilr_stem_vs_committed_cal"], errors="coerce")
    df["state_Prog"] = pd.to_numeric(df["ilr_prog_vs_mature_cal"], errors="coerce")
    df["state_GMP"] = pd.to_numeric(df["ilr_gmp_vs_monodc_cal"], errors="coerce")
    df["state_MonoDC"] = -pd.to_numeric(df["ilr_gmp_vs_monodc_cal"], errors="coerce")
    df["aux_EryBaso"] = pd.to_numeric(df["log_aux_erybaso_cal"], errors="coerce")
    df["aux_CLP"] = pd.to_numeric(df["log_aux_clp_cal"], errors="coerce")

    df["clinical_group"] = [
        stage_to_group(stage, ic, il)
        for stage, ic, il in zip(df["stage"], df["is_control"], df["is_leukemia"])
    ]
    df["clinical_timepoint_coarse"] = df["stage"].map(stage_to_timepoint)

    # discovery-anchored distance in the same 6D summary space
    scaffold = df[DX_ATTRACTOR_FEATURES].to_numpy(dtype=float)
    mu_shift = []
    for row in scaffold:
        if np.isfinite(row).all():
            mu_shift.append(euclidean(row, dx_attractor))
        else:
            mu_shift.append(np.nan)
    df["mu_shift_from_dx"] = mu_shift

    state_out = df[
        [
            "sample_id",
            "patient_id",
            "stage",
            "stage_order",
            "clinical_timepoint_coarse",
            "clinical_group",
            "is_control",
            "control_type",
            "donor_id",
            "is_leukemia",
            "pair_group",
            "state_HSC",
            "state_Prog",
            "state_GMP",
            "state_MonoDC",
            "aux_EryBaso",
            "aux_CLP",
            "pc1_cal",
            "pc2_cal",
            "mu_shift_from_dx",
        ]
    ].copy()
    state_out = state_out.rename(columns={"pc1_cal": "PC1", "pc2_cal": "PC2"})
    state_out.to_csv(OUT_STATE, index=False)

    # ---------------------------------------------------------
    # Build simplified validation summary table
    # ---------------------------------------------------------
    summary = state_out[
        [
            "sample_id",
            "patient_id",
            "clinical_group",
            "clinical_timepoint_coarse",
            "stage",
            "stage_order",
            "pair_group",
            "mu_shift_from_dx",
            "state_HSC",
            "state_Prog",
            "state_GMP",
            "state_MonoDC",
            "aux_EryBaso",
            "aux_CLP",
            "PC1",
            "PC2",
        ]
    ].copy()
    summary.to_csv(OUT_SUMMARY, index=False)

    # ---------------------------------------------------------
    # Calibration summary table for Figure 7D
    # ---------------------------------------------------------
    rows = []

    # leukemia vs control from ready-made stats
    for _, r in lvc.iterrows():
        rows.append({
            "cohort": "GSE163634",
            "metric": str(r["axis"]),
            "comparison": "leukemia_vs_control",
            "effect_direction": "higher_in_leukemia" if float(r["median_difference"]) > 0 else "lower_in_leukemia",
            "estimate": float(r["median_difference"]),
            "evidence_level": f"AUROC={float(r['auroc_leukemia_vs_control']):.3f}",
            "interpretation": f"p={float(r['mannwhitney_p']):.3g}, q={float(r['fdr_q']):.3g}",
        })

    # serial summaries from deltas
    delta_cols = [
        "delta_ilr_stem_vs_committed_cal",
        "delta_ilr_prog_vs_mature_cal",
        "delta_ilr_gmp_vs_monodc_cal",
        "delta_log_aux_clp_cal",
        "delta_log_aux_erybaso_cal",
        "delta_pc1_cal",
        "delta_pc2_cal",
    ]
    for c in delta_cols:
        vals = pd.to_numeric(deltas[c], errors="coerce")
        rows.append({
            "cohort": "GSE163634",
            "metric": c,
            "comparison": "serial_delta",
            "effect_direction": "positive_median_delta" if float(np.nanmedian(vals)) > 0 else "negative_median_delta",
            "estimate": float(np.nanmedian(vals)),
            "evidence_level": f"n={int(vals.notna().sum())}",
            "interpretation": "median paired serial delta",
        })

    repl = pd.DataFrame(rows)
    repl.to_csv(OUT_REPL, sep="\t", index=False)

    print(f"[DONE] Saved {OUT_STATE}")
    print(f"[DONE] Saved {OUT_SUMMARY}")
    print(f"[DONE] Saved {OUT_REPL}")

    print("\n[SUMMARY: clinical groups]")
    print(state_out["clinical_group"].value_counts(dropna=False).to_string())

    print("\n[SUMMARY: coarse timepoints]")
    print(state_out["clinical_timepoint_coarse"].value_counts(dropna=False).to_string())

    print("\n[SUMMARY: mu_shift_from_dx by clinical group]")
    print(
        state_out.groupby("clinical_group")["mu_shift_from_dx"]
        .describe()
        .round(4)
        .to_string()
    )

    print("\n[SUMMARY: leukemia vs control stats]")
    print(lvc.to_string(index=False))

    print("\n[SUMMARY: serial delta medians]")
    print(
        pd.Series({
            c: float(np.nanmedian(pd.to_numeric(deltas[c], errors="coerce")))
            for c in delta_cols
        }).to_string()
    )


if __name__ == "__main__":
    main()
