from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


# ============================================================
# 1. CONFIG
# ============================================================
FIG3_DIR = Path("/Multimodal_OU_Lévy_Branching_Scaffold/Figure_3")
FIG4_DIR = Path("/Multimodal_OU_Lévy_Branching_Scaffold/Figure_4")
FIG5_DIR = Path("/Multimodal_OU_Lévy_Branching_Scaffold/Figure_5")

FIG3_DERIVED = FIG3_DIR / "derived"
FIG4_DERIVED = FIG4_DIR / "derived"
FIG5_DERIVED = FIG5_DIR / "derived"

IN_INTERVAL = FIG3_DERIVED / "patient_interval_metrics_main.csv"
IN_JUMP = FIG3_DERIVED / "relapse_jump_candidates.csv"
IN_SAMPLE = FIG4_DERIVED / "sample_dynamic_parameters.csv"

OUT_TRANSITION = FIG5_DERIVED / "branch_transition_table.csv"
OUT_ECOLOGY = FIG5_DERIVED / "branch_ecology_summary.csv"
OUT_PROGRAM = FIG5_DERIVED / "branch_scaffold_program_summary.csv"
OUT_RISK = FIG5_DERIVED / "branch_escape_risk_summary.csv"
OUT_STATS = FIG5_DERIVED / "figure5_stats_summary.tsv"

BRANCH_ORDER = [
    "HSC-like basin",
    "Progenitor-like basin",
    "GMP-like basin",
    "Mono/DC-like basin",
]

SCAFFOLD_COLS = [
    "state_HSC",
    "state_Prog",
    "state_GMP",
    "state_MonoDC",
    "aux_EryBaso",
    "aux_CLP",
]

SAMPLE_REQUIRED = [
    "sample_id",
    "patient_id",
    "clinical_timepoint_coarse",
    "branch_id_dominant",
    "ecotype_label",
    "theta_eff",
    "sigma_eff",
    "mu_shift_from_dx",
    "is_main_analysis_sample",
    "PC1",
    "PC2",
] + SCAFFOLD_COLS

INTERVAL_REQUIRED = [
    "patient_id",
    "interval_class",
    "sample_start",
    "sample_end",
    "branch_start",
    "branch_end",
    "branch_switch",
    "displacement_hd",
]

JUMP_REQUIRED = [
    "patient_id",
    "sample_start",
    "sample_end",
    "jump_score",
]


# ============================================================
# 2. HELPERS
# ============================================================
def ensure_dirs() -> None:
    FIG5_DERIVED.mkdir(parents=True, exist_ok=True)


def assert_columns(df: pd.DataFrame, required: list[str], label: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{label} missing required columns: {missing}")


def zscore_series(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    mu = np.nanmean(x)
    sd = np.nanstd(x, ddof=0)
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(np.zeros(len(x)), index=s.index, dtype=float)
    return (x - mu) / sd


def safe_mode(s: pd.Series) -> str:
    s = s.dropna().astype(str)
    if s.empty:
        return "Unknown"
    return s.value_counts().idxmax()


def sort_branch(df: pd.DataFrame, col: str) -> pd.DataFrame:
    out = df.copy()
    out["_branch_order"] = out[col].map({b: i for i, b in enumerate(BRANCH_ORDER)}).fillna(999)
    out = out.sort_values(["_branch_order", col]).drop(columns="_branch_order")
    return out


# ============================================================
# 3. BUILD TABLES
# ============================================================
def build_transition_table(interval_df: pd.DataFrame, jump_df: pd.DataFrame, sample_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build row-level DX->REL transition table with start/end sample annotations.
    """
    dxrel = interval_df[interval_df["interval_class"] == "DX_to_REL"].copy()

    # join jump score
    jump_cols = ["patient_id", "sample_start", "sample_end", "jump_score"]
    jump_sub = jump_df[jump_cols].drop_duplicates().copy()

    out = dxrel.merge(
        jump_sub,
        on=["patient_id", "sample_start", "sample_end"],
        how="left",
        validate="one_to_one",
    )

    # start/end sample annotations from sample_dynamic_parameters
    sample_cols = [
        "sample_id",
        "clinical_timepoint_coarse",
        "branch_id_dominant",
        "ecotype_label",
        "theta_eff",
        "sigma_eff",
        "mu_shift_from_dx",
        "PC1",
        "PC2",
    ] + SCAFFOLD_COLS

    samp = sample_df[sample_cols].drop_duplicates("sample_id").copy()

    start_map = samp.rename(columns={c: f"{c}_start" for c in sample_cols if c != "sample_id"})
    end_map = samp.rename(columns={c: f"{c}_end" for c in sample_cols if c != "sample_id"})

    out = out.merge(
        start_map,
        left_on="sample_start",
        right_on="sample_id",
        how="left",
    ).drop(columns=["sample_id"])

    out = out.merge(
        end_map,
        left_on="sample_end",
        right_on="sample_id",
        how="left",
    ).drop(columns=["sample_id"])

    out["transition_label"] = out["branch_start"].astype(str) + " -> " + out["branch_end"].astype(str)
    out["transition_class"] = np.where(
        out["branch_switch"].astype(int) == 1,
        "Branch-switching",
        "Branch-continuous",
    )

    # convenient flags
    if "tail_flag_dx_rel_q90" in out.columns:
        out["tail_flag_dx_rel_q90"] = out["tail_flag_dx_rel_q90"].astype(bool)
    else:
        q90 = float(np.nanquantile(pd.to_numeric(out["displacement_hd"], errors="coerce"), 0.90))
        out["tail_threshold_dx_rel_q90"] = q90
        out["tail_flag_dx_rel_q90"] = pd.to_numeric(out["displacement_hd"], errors="coerce") >= q90

    # reorder
    front = [
        "patient_id",
        "interval_class",
        "sample_start",
        "sample_end",
        "branch_start",
        "branch_end",
        "branch_switch",
        "transition_label",
        "transition_class",
        "displacement_hd",
        "jump_score",
        "tail_flag_dx_rel_q90",
        "ecotype_label_start",
        "ecotype_label_end",
        "theta_eff_start",
        "theta_eff_end",
        "sigma_eff_start",
        "sigma_eff_end",
        "mu_shift_from_dx_start",
        "mu_shift_from_dx_end",
        "PC1_start",
        "PC2_start",
        "PC1_end",
        "PC2_end",
    ]
    rest = [c for c in out.columns if c not in front]
    out = out[front + rest]

    return out


def build_branch_ecology_summary(sample_df: pd.DataFrame) -> pd.DataFrame:
    """
    Branch-by-ecotype composition for main-analysis sample-timepoints.
    """
    s = sample_df[sample_df["is_main_analysis_sample"] == True].copy()
    s = s[s["branch_id_dominant"].notna()].copy()

    branch_totals = (
        s.groupby("branch_id_dominant")
         .size()
         .rename("branch_total_samples")
         .reset_index()
    )

    eco = (
        s.groupby(["branch_id_dominant", "ecotype_label"])
         .size()
         .rename("n_samples")
         .reset_index()
    )

    eco = eco.merge(branch_totals, on="branch_id_dominant", how="left")
    eco["fraction_within_branch"] = eco["n_samples"] / eco["branch_total_samples"]

    # branch-level summaries repeated for convenience
    branch_summary = (
        s.groupby("branch_id_dominant")[["PC1", "PC2", "theta_eff", "sigma_eff", "mu_shift_from_dx"]]
         .mean()
         .reset_index()
         .rename(columns={
             "PC1": "PC1_mean",
             "PC2": "PC2_mean",
             "theta_eff": "theta_eff_mean",
             "sigma_eff": "sigma_eff_mean",
             "mu_shift_from_dx": "mu_shift_from_dx_mean",
         })
    )

    eco = eco.merge(branch_summary, on="branch_id_dominant", how="left")
    eco = sort_branch(eco, "branch_id_dominant").sort_values(
        ["branch_id_dominant", "fraction_within_branch", "ecotype_label"],
        ascending=[True, False, True]
    )
    return eco


def build_branch_scaffold_program_summary(sample_df: pd.DataFrame) -> pd.DataFrame:
    """
    Branch-level means and z-scored means for scaffold state programs.
    """
    s = sample_df[sample_df["is_main_analysis_sample"] == True].copy()
    s = s[s["branch_id_dominant"].notna()].copy()

    group_cols = ["branch_id_dominant"]
    agg_cols = SCAFFOLD_COLS + ["theta_eff", "sigma_eff", "mu_shift_from_dx"]

    summary = (
        s.groupby(group_cols)[agg_cols]
         .mean()
         .reset_index()
    )

    counts = (
        s.groupby("branch_id_dominant")
         .size()
         .rename("n_samples")
         .reset_index()
    )
    summary = summary.merge(counts, on="branch_id_dominant", how="left")

    # rename means
    mean_rename = {c: f"{c}_mean" for c in agg_cols}
    summary = summary.rename(columns=mean_rename)

    # z-score the scaffold program means across branches
    for c in SCAFFOLD_COLS:
        mean_col = f"{c}_mean"
        z_col = f"{c}_z"
        summary[z_col] = zscore_series(summary[mean_col])

    # optional z for dynamics too
    for c in ["theta_eff_mean", "sigma_eff_mean", "mu_shift_from_dx_mean"]:
        summary[f"{c}_z"] = zscore_series(summary[c])

    summary = sort_branch(summary, "branch_id_dominant")
    return summary


def build_branch_escape_risk_summary(transition_df: pd.DataFrame) -> pd.DataFrame:
    """
    Branch-start-specific jump and switching summaries.
    """
    t = transition_df.copy()

    out = (
        t.groupby("branch_start")
         .agg(
             n_intervals=("patient_id", "size"),
             n_switch=("branch_switch", lambda x: int(np.sum(pd.to_numeric(x, errors="coerce").fillna(0).astype(int) == 1))),
             n_stable=("branch_switch", lambda x: int(np.sum(pd.to_numeric(x, errors="coerce").fillna(0).astype(int) == 0))),
             switch_fraction=("branch_switch", lambda x: float(np.mean(pd.to_numeric(x, errors="coerce").fillna(0).astype(int) == 1))),
             mean_jump_score=("jump_score", lambda x: float(np.nanmean(pd.to_numeric(x, errors="coerce")))),
             median_jump_score=("jump_score", lambda x: float(np.nanmedian(pd.to_numeric(x, errors="coerce")))),
             mean_displacement_hd=("displacement_hd", lambda x: float(np.nanmean(pd.to_numeric(x, errors="coerce")))),
             median_displacement_hd=("displacement_hd", lambda x: float(np.nanmedian(pd.to_numeric(x, errors="coerce")))),
             tail_fraction_dx_rel_q90=("tail_flag_dx_rel_q90", lambda x: float(np.mean(pd.Series(x).astype(bool)))),
         )
         .reset_index()
         .rename(columns={"branch_start": "branch_id_start"})
    )

    out["mean_jump_score_z"] = zscore_series(out["mean_jump_score"])
    out["mean_displacement_hd_z"] = zscore_series(out["mean_displacement_hd"])
    out = sort_branch(out, "branch_id_start")
    return out


def build_stats_summary(transition_df: pd.DataFrame, eco_df: pd.DataFrame, risk_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compact long-form summary table for manuscript reference.
    """
    rows = []

    # transition counts
    trans_counts = (
        transition_df.groupby(["branch_start", "branch_end"])
        .size()
        .rename("count")
        .reset_index()
    )
    for _, r in trans_counts.iterrows():
        rows.append({
            "section": "transition_count",
            "item": f"{r['branch_start']} -> {r['branch_end']}",
            "value": float(r["count"]),
        })

    # ecology dominant labels by branch
    top_eco = (
        eco_df.sort_values(["branch_id_dominant", "fraction_within_branch"], ascending=[True, False])
             .groupby("branch_id_dominant")
             .head(1)
    )
    for _, r in top_eco.iterrows():
        rows.append({
            "section": "dominant_ecotype_by_branch",
            "item": f"{r['branch_id_dominant']}",
            "value": f"{r['ecotype_label']} ({r['fraction_within_branch']:.3f})",
        })

    # escape risk
    for _, r in risk_df.iterrows():
        rows.append({
            "section": "branch_escape_risk",
            "item": f"{r['branch_id_start']}",
            "value": f"mean_jump={r['mean_jump_score']:.3f}; switch_frac={r['switch_fraction']:.3f}",
        })

    return pd.DataFrame(rows)


# ============================================================
# 4. MAIN
# ============================================================
def main() -> None:
    ensure_dirs()

    interval_df = pd.read_csv(IN_INTERVAL)
    jump_df = pd.read_csv(IN_JUMP)
    sample_df = pd.read_csv(IN_SAMPLE)

    assert_columns(interval_df, INTERVAL_REQUIRED, "patient_interval_metrics_main.csv")
    assert_columns(jump_df, JUMP_REQUIRED, "relapse_jump_candidates.csv")
    assert_columns(sample_df, SAMPLE_REQUIRED, "sample_dynamic_parameters.csv")

    # Restrict to current main-analysis sample-timepoints
    transition_df = build_transition_table(interval_df, jump_df, sample_df)
    ecology_df = build_branch_ecology_summary(sample_df)
    program_df = build_branch_scaffold_program_summary(sample_df)
    risk_df = build_branch_escape_risk_summary(transition_df)
    stats_df = build_stats_summary(transition_df, ecology_df, risk_df)

    transition_df.to_csv(OUT_TRANSITION, index=False)
    ecology_df.to_csv(OUT_ECOLOGY, index=False)
    program_df.to_csv(OUT_PROGRAM, index=False)
    risk_df.to_csv(OUT_RISK, index=False)
    stats_df.to_csv(OUT_STATS, sep="\t", index=False)

    print(f"[DONE] Saved {OUT_TRANSITION}")
    print(f"[DONE] Saved {OUT_ECOLOGY}")
    print(f"[DONE] Saved {OUT_PROGRAM}")
    print(f"[DONE] Saved {OUT_RISK}")
    print(f"[DONE] Saved {OUT_STATS}")

    print("\n[SUMMARY: branch transitions]")
    print(
        transition_df.groupby(["branch_start", "branch_end"])
        .size()
        .sort_values(ascending=False)
        .to_string()
    )

    print("\n[SUMMARY: dominant ecotype by branch]")
    top_eco = (
        ecology_df.sort_values(["branch_id_dominant", "fraction_within_branch"], ascending=[True, False])
                 .groupby("branch_id_dominant")
                 .head(1)
    )
    print(
        top_eco[["branch_id_dominant", "ecotype_label", "fraction_within_branch"]]
        .to_string(index=False)
    )

    print("\n[SUMMARY: branch escape risk]")
    print(
        risk_df[["branch_id_start", "n_intervals", "switch_fraction", "mean_jump_score", "mean_displacement_hd"]]
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
