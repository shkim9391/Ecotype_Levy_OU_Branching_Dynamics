from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


# ============================================================
# 1. CONFIG
# ============================================================
BASE = Path("/Multimodal_OU_Lévy_Branching_Scaffold")
FIG5_DERIVED = BASE / "Figure_5" / "derived"
FIG6_DERIVED = BASE / "Figure_6" / "derived"
FIG7_DIR = BASE / "Figure_7"
FIG7_DERIVED = FIG7_DIR / "derived"
FIG7_PANELS = FIG7_DIR / "panels"

IN_DISC_TRANS = FIG5_DERIVED / "branch_transition_table.csv"
IN_DISC_RISK = FIG5_DERIVED / "branch_escape_risk_summary.csv"

IN_EXT_AML_DYN = FIG6_DERIVED / "gse235923_sample_dynamic_parameters.csv"
IN_EXT_AML_CENT = FIG6_DERIVED / "gse235923_sample_centroids.csv"

IN_BULK_SUM = FIG7_DERIVED / "gse163634_bulk_validation_summary.csv"
IN_RISK_MAP = FIG7_DERIVED / "figure7_clinical_risk_map.csv"
IN_SCORECARD = FIG7_DERIVED / "figure7_clinical_scorecard.csv"

OUT_PNG = FIG7_PANELS / "Figure7D_clinical_translation_summary.png"
OUT_PDF = FIG7_PANELS / "Figure7D_clinical_translation_summary.pdf"
OUT_TSV = FIG7_DERIVED / "figure7_clinical_translation_summary.tsv"

BOX_COLORS = {
    "discovery": "#E8EEF6",
    "external_aml": "#F8E8E8",
    "bulk": "#F3F3F3",
    "clinical": "#E7F4EE",
}


# ============================================================
# 2. HELPERS
# ============================================================
def ensure_dirs() -> None:
    FIG7_PANELS.mkdir(parents=True, exist_ok=True)
    FIG7_DERIVED.mkdir(parents=True, exist_ok=True)


def style_axis(ax, panel_label: str, title: str,
               panel_fontsize: int = 18,
               title_fontsize: int = 12,
               title_x: float = 0.10) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

    ax.text(
        0.00, 1.03, panel_label,
        transform=ax.transAxes,
        fontsize=panel_fontsize,
        fontweight="bold",
        ha="left",
        va="bottom"
    )
    ax.text(
        title_x, 1.03, title,
        transform=ax.transAxes,
        fontsize=title_fontsize,
        fontweight="bold",
        ha="left",
        va="bottom"
    )


def draw_summary_box(ax, x, y, w, h, title, body, facecolor):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        facecolor=facecolor,
        edgecolor="#777777",
        linewidth=1.1,
        alpha=0.95,
    )
    ax.add_patch(box)

    pad_x = 0.025

    ax.text(
        x + pad_x,
        y + h - 0.030,
        title,
        fontsize=11.0,
        fontweight="bold",
        ha="left",
        va="top",
        color="#222222",
    )

    ax.text(
        x + pad_x,
        y + h - 0.090,
        body,
        fontsize=8.5,
        ha="left",
        va="top",
        color="#333333",
        linespacing=1.28,
    )


# ============================================================
# 3. SUMMARIES
# ============================================================
def summarize_discovery() -> dict:
    trans = pd.read_csv(IN_DISC_TRANS)
    risk = pd.read_csv(IN_DISC_RISK)

    trans["branch_switch"] = pd.to_numeric(trans["branch_switch"], errors="coerce").fillna(0).astype(int)

    n_total = int(trans.shape[0])
    n_switch = int((trans["branch_switch"] == 1).sum())
    n_continuous = int((trans["branch_switch"] == 0).sum())

    top = risk.sort_values("mean_jump_score", ascending=False).iloc[0]
    top_branch = str(top["branch_id_start"]).replace("-like basin", "")
    top_jump = float(top["mean_jump_score"])

    title = "Discovery cohort (GSE235063)"
    body = (
        f"{n_total} DX→REL intervals support both\n"
        f"branch-continuous and switching routes.\n\n"
        f"Branch-continuous = {n_continuous}\n"
        f"Branch-switching = {n_switch}\n\n"
        f"Top escape context:\n"
        f"{top_branch} branch\n"
        f"mean jump = {top_jump:.2f}"
    )

    row = {
        "cohort": "GSE235063",
        "clinical_message": "Discovery cohort identifies constrained, residual, and escape-prone dynamic states.",
        "supporting_metric": "DX→REL intervals",
        "supporting_value": f"branch_continuous={n_continuous}; branch_switching={n_switch}; top_escape_branch={top_branch}",
    }
    return {"title": title, "body": body, "row": row}


def summarize_external_aml() -> dict:
    dyn = pd.read_csv(IN_EXT_AML_DYN)
    cent = pd.read_csv(IN_EXT_AML_CENT)

    dyn["clinical_timepoint_coarse"] = dyn["clinical_timepoint_coarse"].astype(str)

    dx_mu = float(np.nanmedian(dyn.loc[dyn["clinical_timepoint_coarse"] == "DX", "mu_shift_from_dx"]))
    eoi_mu = float(np.nanmedian(dyn.loc[dyn["clinical_timepoint_coarse"] == "EOI_REM", "mu_shift_from_dx"]))
    rel_mu = float(np.nanmedian(dyn.loc[dyn["clinical_timepoint_coarse"] == "REL", "mu_shift_from_dx"]))

    seq = cent.groupby("patient_id")["clinical_timepoint_coarse"].apply(list).reset_index(name="timepoints")
    triads = seq[seq["timepoints"].apply(lambda x: set(x) == {"DX", "EOI_REM", "REL"})]
    n_triads = int(triads.shape[0])

    title = "External AML calibration (GSE235923)"
    body = (
        "An independent AML cohort reproduces\n"
        "the treatment-aware dynamic logic.\n\n"
        f"Median μ-shift\n"
        f"DX = {dx_mu:.3f}\n"
        f"EOI/REM = {eoi_mu:.3f}\n"
        f"REL = {rel_mu:.3f}\n\n"
        f"Full triads = {n_triads}"
    )

    row = {
        "cohort": "GSE235923",
        "clinical_message": "External AML calibration preserves directional treatment-aware structure.",
        "supporting_metric": "μ-shift by phase",
        "supporting_value": f"DX={dx_mu:.3f}; EOI/REM={eoi_mu:.3f}; REL={rel_mu:.3f}; triads={n_triads}",
    }
    return {"title": title, "body": body, "row": row}


def summarize_bulk() -> dict:
    bulk = pd.read_csv(IN_BULK_SUM)
    bulk["clinical_group"] = bulk["clinical_group"].astype(str)

    ctrl_mu = float(np.nanmedian(bulk.loc[bulk["clinical_group"] == "control", "mu_shift_from_dx"]))
    dx_mu = float(np.nanmedian(bulk.loc[bulk["clinical_group"] == "dx_leukemia", "mu_shift_from_dx"]))
    r1_mu = float(np.nanmedian(bulk.loc[bulk["clinical_group"] == "response_r1", "mu_shift_from_dx"]))
    r2_mu = float(np.nanmedian(bulk.loc[bulk["clinical_group"] == "response_r2", "mu_shift_from_dx"]))

    title = "Serial bulk validation (GSE163634)"
    body = (
        "A lower-resolution, more deployable assay\n"
        "still preserves directional state structure.\n\n"
        f"Median μ-shift\n"
        f"Control = {ctrl_mu:.3f}\n"
        f"DX = {dx_mu:.3f}\n"
        f"r1 = {r1_mu:.3f}\n"
        f"r2 = {r2_mu:.3f}"
    )

    row = {
        "cohort": "GSE163634",
        "clinical_message": "Conservative serial bulk validation supports deployable response monitoring.",
        "supporting_metric": "bulk μ-shift",
        "supporting_value": f"control={ctrl_mu:.3f}; dx={dx_mu:.3f}; r1={r1_mu:.3f}; r2={r2_mu:.3f}",
    }
    return {"title": title, "body": body, "row": row}


def summarize_clinical_translation() -> dict:
    risk = pd.read_csv(IN_RISK_MAP)
    score = pd.read_csv(IN_SCORECARD)

    risk["risk_tier"] = risk["risk_tier"].astype(str)
    counts = risk["risk_tier"].value_counts()

    n_constrained = int(counts.get("Low-risk constrained", 0))
    n_residual = int(counts.get("Intermediate residual", 0))
    n_escape = int(counts.get("High-risk escape-prone", 0))

    exemplar_map = {}
    for tier in ["Low-risk constrained", "Intermediate residual", "High-risk escape-prone"]:
        sub = score[score["risk_tier"] == tier]
        exemplar_map[tier] = ", ".join(sub["sample_id"].astype(str).tolist()) if not sub.empty else "NA"

    title = "Clinical translation takeaway"
    body = (
        "Dynamic states support response\n"
        "monitoring and residual-state triage.\n\n"
        f"Constrained: {n_constrained}\n"
        f"Residual: {n_residual}\n"
        f"Escape-prone: {n_escape}\n\n"
        f"Examples:\n"
        f"Residual → {exemplar_map['Intermediate residual']}\n"
        f"Escape → {exemplar_map['High-risk escape-prone']}"
    )

    row = {
        "cohort": "clinical_translation",
        "clinical_message": "OU–Lévy–Branching provides interpretable dynamic states for response monitoring and residual-state triage.",
        "supporting_metric": "state tiers",
        "supporting_value": f"constrained={n_constrained}; residual={n_residual}; escape_prone={n_escape}",
    }

    return {"title": title, "body": body, "row": row}


# ============================================================
# 4. MAIN
# ============================================================
def main() -> None:
    ensure_dirs()

    disc = summarize_discovery()
    ext = summarize_external_aml()
    bulk = summarize_bulk()
    clin = summarize_clinical_translation()

    summary_df = pd.DataFrame([
        disc["row"],
        ext["row"],
        bulk["row"],
        clin["row"],
    ])
    summary_df.to_csv(OUT_TSV, sep="\t", index=False)

    fig, ax = plt.subplots(figsize=(12.8, 7.2))
    style_axis(ax, "D", "Clinical translation summary")

    box_w = 0.39
    box_h = 0.33
    x_left = 0.045
    x_right = 0.505
    y_top = 0.60
    y_bot = 0.18

    draw_summary_box(ax, x_left, y_top, box_w, box_h, disc["title"], disc["body"], BOX_COLORS["discovery"])
    draw_summary_box(ax, x_right, y_top, box_w, box_h, ext["title"], ext["body"], BOX_COLORS["external_aml"])
    draw_summary_box(ax, x_left, y_bot, box_w, box_h, bulk["title"], bulk["body"], BOX_COLORS["bulk"])
    draw_summary_box(ax, x_right, y_bot, box_w, box_h, clin["title"], clin["body"], BOX_COLORS["clinical"])

    ax.text(
        0.04, 0.01,
        "Together, these results position OU–Lévy–Branching as a clinically interpretable framework for\n"
        "dynamic response monitoring, residual-state interpretation, and relapse-risk assessment\nin pediatric leukemia.",
        fontsize=13.0,
        ha="left",
        va="bottom",
        color="#333333",
    )

    fig.savefig(OUT_PNG, dpi=600, bbox_inches="tight")
    fig.savefig(OUT_PDF, dpi=600, bbox_inches="tight")
    plt.close(fig)

    print(f"[DONE] Saved {OUT_PNG}")
    print(f"[DONE] Saved {OUT_PDF}")
    print(f"[DONE] Saved {OUT_TSV}")

    print("\n[SUMMARY]")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
