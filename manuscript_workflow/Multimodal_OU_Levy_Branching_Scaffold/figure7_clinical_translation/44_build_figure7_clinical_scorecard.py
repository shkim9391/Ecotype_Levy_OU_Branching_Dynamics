from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


# ============================================================
# 1. CONFIG
# ============================================================
BASE = Path("/Multimodal_OU_Lévy_Branching_Scaffold")

FIG3_DERIVED = BASE / "Figure_3" / "derived"
FIG7_DERIVED = BASE / "Figure_7" / "derived"

IN_RISK_MAP = FIG7_DERIVED / "figure7_clinical_risk_map.csv"
IN_JUMP = FIG3_DERIVED / "relapse_jump_candidates.csv"

OUT_CSV = FIG7_DERIVED / "figure7_clinical_scorecard.csv"
OUT_SUMMARY = FIG7_DERIVED / "figure7_clinical_scorecard_summary.tsv"

REQUIRED_RISK = [
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
    "clinical_risk_score",
    "risk_zone",
    "risk_tier",
]

REQUIRED_JUMP = [
    "patient_id",
    "sample_start",
    "sample_end",
    "jump_score",
    "branch_switch",
]

# Representative clinical scorecard rows to construct
ROW_PLAN = [
    "discovery_dx_constrained",
    "discovery_eoi_residual",
    "discovery_rel_escape",
    "external_dx_constrained",
    "external_eoi_residual",
    "external_rel_escape_1",
    "external_rel_escape_2",
]


# ============================================================
# 2. HELPERS
# ============================================================
def ensure_dirs() -> None:
    FIG7_DERIVED.mkdir(parents=True, exist_ok=True)


def assert_columns(df: pd.DataFrame, required: list[str], label: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{label} missing required columns: {missing}")


def safe_bool_series(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False)
    x = s.astype(str).str.strip().str.lower()
    return x.isin(["true", "1", "t", "yes"])


def add_discovery_jump_context(risk_map: pd.DataFrame, jump_df: pd.DataFrame) -> pd.DataFrame:
    """
    Attach jump / branch-switch context to discovery samples where possible.

    Logic:
    - discovery DX samples can inherit jump context from matching sample_start
    - discovery REL samples can inherit jump context from matching sample_end
    - other samples remain NA
    """
    out = risk_map.copy()

    start_map = (
        jump_df[["sample_start", "jump_score", "branch_switch"]]
        .drop_duplicates(subset=["sample_start"])
        .rename(columns={
            "sample_start": "sample_id",
            "jump_score": "jump_score_from_start",
            "branch_switch": "branch_switch_from_start",
        })
    )

    end_map = (
        jump_df[["sample_end", "jump_score", "branch_switch"]]
        .drop_duplicates(subset=["sample_end"])
        .rename(columns={
            "sample_end": "sample_id",
            "jump_score": "jump_score_from_end",
            "branch_switch": "branch_switch_from_end",
        })
    )

    out = out.merge(start_map, on="sample_id", how="left")
    out = out.merge(end_map, on="sample_id", how="left")

    out["jump_score"] = np.nan
    out["branch_switch"] = np.nan
    out["jump_context_role"] = "NA"

    dx_mask = (out["source_group"] == "discovery") & (out["clinical_timepoint_coarse"] == "DX")
    rel_mask = (out["source_group"] == "discovery") & (out["clinical_timepoint_coarse"] == "REL")

    out.loc[dx_mask, "jump_score"] = out.loc[dx_mask, "jump_score_from_start"]
    out.loc[dx_mask, "branch_switch"] = out.loc[dx_mask, "branch_switch_from_start"]
    out.loc[dx_mask, "jump_context_role"] = "interval_start"

    out.loc[rel_mask, "jump_score"] = out.loc[rel_mask, "jump_score_from_end"]
    out.loc[rel_mask, "branch_switch"] = out.loc[rel_mask, "branch_switch_from_end"]
    out.loc[rel_mask, "jump_context_role"] = "interval_end"

    out = out.drop(columns=[
        "jump_score_from_start", "branch_switch_from_start",
        "jump_score_from_end", "branch_switch_from_end",
    ])

    return out


def choose_row(df: pd.DataFrame, *, order_cols: list[str], ascending: list[bool], selection_reason: str):
    if df.empty:
        return None
    row = df.sort_values(order_cols, ascending=ascending).iloc[0].copy()
    row["selection_reason"] = selection_reason
    return row


def choose_rel_rows(df: pd.DataFrame, n: int, selection_reason_prefix: str):
    if df.empty:
        return []

    sub = df.sort_values(
        ["clinical_risk_score", "mu_shift_from_dx", "theta_eff"],
        ascending=[False, False, True]
    ).head(n).copy()

    rows = []
    for i, (_, r) in enumerate(sub.iterrows(), start=1):
        rr = r.copy()
        rr["selection_reason"] = f"{selection_reason_prefix}_{i}"
        rows.append(rr)
    return rows


def finalize_scorecard(rows: list[pd.Series]) -> pd.DataFrame:
    keep = []
    for r in rows:
        if r is not None:
            keep.append(r)

    if not keep:
        raise ValueError("No representative rows were selected for the clinical scorecard.")

    out = pd.DataFrame(keep).copy()

    # Stable display order
    out["display_order"] = np.arange(1, out.shape[0] + 1)

    # Friendly group labels for later plotting
    def row_group(x: str) -> str:
        mapping = {
            "discovery_dx_constrained": "Discovery DX",
            "discovery_eoi_residual": "Discovery EOI/Residual",
            "discovery_rel_escape": "Discovery REL",
            "external_dx_constrained": "External AML DX",
            "external_eoi_residual": "External AML EOI",
            "external_rel_escape_1": "External AML REL",
            "external_rel_escape_2": "External AML REL",
        }
        return mapping.get(x, "Representative")

    out["row_group"] = out["selection_reason"].map(row_group).fillna("Representative")

    # readable jump fields
    out["jump_score_display"] = pd.to_numeric(out["jump_score"], errors="coerce")
    out["branch_switch_display"] = pd.to_numeric(out["branch_switch"], errors="coerce")

    cols_front = [
        "display_order",
        "row_group",
        "selection_reason",
        "cohort",
        "source_group",
        "sample_id",
        "patient_id",
        "clinical_timepoint_coarse",
        "n_cells",
        "theta_eff",
        "sigma_eff",
        "mu_shift_from_dx",
        "clinical_risk_score",
        "risk_zone",
        "risk_tier",
        "branch_id_dominant",
        "ecotype_label",
        "jump_score_display",
        "branch_switch_display",
        "jump_context_role",
    ]
    rest = [c for c in out.columns if c not in cols_front]
    out = out[cols_front + rest]
    
    out["risk_tier_short"] = out["risk_tier"].map({
        "Low-risk constrained": "Constrained",
        "Intermediate residual": "Residual",
        "High-risk escape-prone": "Escape-prone",
    }).fillna(out["risk_tier"])

    return out


# ============================================================
# 3. MAIN
# ============================================================
def main() -> None:
    ensure_dirs()

    risk = pd.read_csv(IN_RISK_MAP)
    jump = pd.read_csv(IN_JUMP)

    assert_columns(risk, REQUIRED_RISK, "figure7_clinical_risk_map.csv")
    assert_columns(jump, REQUIRED_JUMP, "relapse_jump_candidates.csv")

    for c in ["n_cells", "theta_eff", "sigma_eff", "mu_shift_from_dx", "clinical_risk_score"]:
        risk[c] = pd.to_numeric(risk[c], errors="coerce")

    jump["jump_score"] = pd.to_numeric(jump["jump_score"], errors="coerce")
    jump["branch_switch"] = pd.to_numeric(jump["branch_switch"], errors="coerce")

    risk["clinical_timepoint_coarse"] = risk["clinical_timepoint_coarse"].astype(str)
    risk["source_group"] = risk["source_group"].astype(str)
    risk["risk_zone"] = risk["risk_zone"].astype(str)
    risk["risk_tier"] = risk["risk_tier"].astype(str)

    risk = add_discovery_jump_context(risk, jump)

    rows = []

    # --------------------------------------------------------
    # Discovery representatives
    # --------------------------------------------------------
    disc_dx = risk[
        (risk["source_group"] == "discovery") &
        (risk["clinical_timepoint_coarse"] == "DX") &
        (risk["risk_zone"] == "Constrained / response-like")
    ].copy()
    rows.append(
        choose_row(
            disc_dx,
            order_cols=["clinical_risk_score", "theta_eff", "mu_shift_from_dx"],
            ascending=[True, False, True],
            selection_reason="discovery_dx_constrained",
        )
    )

    disc_eoi = risk[
        (risk["source_group"] == "discovery") &
        (risk["clinical_timepoint_coarse"] == "EOI_REM")
    ].copy()

    # Prefer AML21_REM if present because it is a known residual-like illustrative case
    if "AML21_REM" in set(disc_eoi["sample_id"]):
        row = disc_eoi[disc_eoi["sample_id"] == "AML21_REM"].iloc[0].copy()
        row["selection_reason"] = "discovery_eoi_residual"
        rows.append(row)
    else:
        rows.append(
            choose_row(
                disc_eoi,
                order_cols=["clinical_risk_score", "sigma_eff", "theta_eff"],
                ascending=[False, False, True],
                selection_reason="discovery_eoi_residual",
            )
        )

    disc_rel = risk[
        (risk["source_group"] == "discovery") &
        (risk["clinical_timepoint_coarse"] == "REL")
    ].copy()

    # Prefer highest-risk REL in the escape zone; fallback to highest-risk REL overall
    disc_rel_escape = disc_rel[disc_rel["risk_zone"] == "Escape-prone / displaced"].copy()
    if disc_rel_escape.empty:
        disc_rel_escape = disc_rel.copy()

    rows.append(
        choose_row(
            disc_rel_escape,
            order_cols=["clinical_risk_score", "mu_shift_from_dx", "theta_eff"],
            ascending=[False, False, True],
            selection_reason="discovery_rel_escape",
        )
    )

    # --------------------------------------------------------
    # External AML representatives
    # --------------------------------------------------------
    ext_dx = risk[
        (risk["source_group"] == "external_aml") &
        (risk["clinical_timepoint_coarse"] == "DX") &
        (risk["risk_zone"] == "Constrained / response-like")
    ].copy()
    rows.append(
        choose_row(
            ext_dx,
            order_cols=["clinical_risk_score", "theta_eff", "mu_shift_from_dx"],
            ascending=[True, False, True],
            selection_reason="external_dx_constrained",
        )
    )

    ext_eoi = risk[
        (risk["source_group"] == "external_aml") &
        (risk["clinical_timepoint_coarse"] == "EOI_REM")
    ].copy()
    ext_eoi_resid = ext_eoi[ext_eoi["risk_zone"] == "Residual persistent / unstable"].copy()
    if ext_eoi_resid.empty:
        ext_eoi_resid = ext_eoi.copy()

    rows.append(
        choose_row(
            ext_eoi_resid,
            order_cols=["clinical_risk_score", "sigma_eff", "theta_eff"],
            ascending=[False, False, True],
            selection_reason="external_eoi_residual",
        )
    )

    ext_rel = risk[
        (risk["source_group"] == "external_aml") &
        (risk["clinical_timepoint_coarse"] == "REL")
    ].copy()

    out = finalize_scorecard(rows)
    out.to_csv(OUT_CSV, index=False)

    summary_rows = []
    for _, r in out.iterrows():
        summary_rows.append({
            "section": "selected_representative",
            "item": f"{r['display_order']}. {r['sample_id']}",
            "value": (
                f"{r['row_group']} | {r['risk_tier']} | "
                f"theta={r['theta_eff']:.3f}, sigma={r['sigma_eff']:.3f}, "
                f"mu_shift={r['mu_shift_from_dx']:.3f}"
            ),
        })

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT_SUMMARY, sep="\t", index=False)

    print(f"[DONE] Saved {OUT_CSV}")
    print(f"[DONE] Saved {OUT_SUMMARY}")

    print("\n[SUMMARY: selected scorecard rows]")
    print(
        out[
            [
                "display_order",
                "row_group",
                "sample_id",
                "clinical_timepoint_coarse",
                "theta_eff",
                "sigma_eff",
                "mu_shift_from_dx",
                "clinical_risk_score",
                "risk_zone",
                "risk_tier",
                "branch_id_dominant",
                "jump_score_display",
                "branch_switch_display",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
