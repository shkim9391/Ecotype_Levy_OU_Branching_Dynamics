from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

base = Path("/Lévy_OU_Branching")
infile = base / "gse235063_longitudinal_branch_projection.csv"

out_counts = base / "gse235063_branch_counts_by_timepoint_thresholds.csv"
out_transition = base / "gse235063_dx_to_rel_transition_summary_by_threshold.csv"
out_patient = base / "gse235063_patient_branch_table_by_threshold.csv"
out_switch = base / "gse235063_dx_to_rel_switch_rates_by_threshold.csv"
out_text = base / "gse235063_transition_summary_threshold_sensitivity.txt"

fig_switch = base / "gse235063_dx_to_rel_switch_rates_by_threshold.png"
fig_counts = base / "gse235063_branch_counts_dx_rel_by_threshold.png"

df = pd.read_csv(infile)

required_cols = [
    "sample", "Patient_ID", "sample_id", "timepoint", "time_index",
    "malignant_cells", "projection_eligible",
    "dist_B1", "dist_B2", "dist_B3"
]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

branch_dist_cols = ["dist_B1", "dist_B2", "dist_B3"]
state_cols = [
    "ilr_stem_vs_committed",
    "ilr_prog_vs_mature",
    "ilr_gmp_vs_monodc",
    "T_NK_given_known_z",
    "Myeloid_APC_given_known_z",
    "B_Plasma_given_known_z",
]

for c in state_cols + branch_dist_cols + ["malignant_cells"]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

thresholds = [20, 50, 100]

# --------------------------------------------------
# Recompute threshold-specific projected branches
# --------------------------------------------------
for thr in thresholds:
    elig_col = f"eligible_ge{thr}"
    proj_col = f"branch_ge{thr}"

    df[elig_col] = (
        df["malignant_cells"].fillna(0) >= thr
    ) & (
        ~df[state_cols].isna().any(axis=1)
    ) & (
        ~df[branch_dist_cols].isna().any(axis=1)
    )

    df[proj_col] = pd.NA
    df.loc[df[elig_col], proj_col] = (
        df.loc[df[elig_col], branch_dist_cols]
        .idxmin(axis=1)
        .str.replace("dist_", "", regex=False)
    )

# --------------------------------------------------
# Branch counts by timepoint and threshold
# --------------------------------------------------
count_rows = []
for thr in thresholds:
    proj_col = f"branch_ge{thr}"
    tab = pd.crosstab(df["timepoint"], df[proj_col], dropna=False)

    for tp in sorted(df["timepoint"].dropna().unique()):
        row = {
            "threshold": thr,
            "timepoint": tp,
            "n_total_samples": int((df["timepoint"] == tp).sum()),
            "n_eligible": int(df.loc[df["timepoint"] == tp, f"eligible_ge{thr}"].sum()),
            "B1": int(tab.loc[tp, "B1"]) if "B1" in tab.columns and tp in tab.index else 0,
            "B2": int(tab.loc[tp, "B2"]) if "B2" in tab.columns and tp in tab.index else 0,
            "B3": int(tab.loc[tp, "B3"]) if "B3" in tab.columns and tp in tab.index else 0,
            "NA": int(tab.loc[tp, np.nan]) if tp in tab.index and np.nan in tab.columns else 0,
        }
        count_rows.append(row)

counts_df = pd.DataFrame(count_rows)
counts_df.to_csv(out_counts, index=False)

# --------------------------------------------------
# Build patient-level branch table for each threshold
# --------------------------------------------------
def first_or_na(series):
    s = series.dropna()
    return s.iloc[0] if len(s) > 0 else pd.NA

patient_rows = []
transition_rows = []
switch_rows = []

for thr in thresholds:
    proj_col = f"branch_ge{thr}"
    elig_col = f"eligible_ge{thr}"

    sub = df.copy()

    # choose one row per patient-timepoint
    pt = (
        sub.sort_values(["sample", "time_index", "sample_id"])
        .groupby(["sample", "timepoint"], as_index=False)
        .agg({
            "Patient_ID": first_or_na,
            "Biopsy_Origin": first_or_na if "Biopsy_Origin" in sub.columns else "first",
            "Subgroup": first_or_na if "Subgroup" in sub.columns else "first",
            "Treatment_Outcome": first_or_na if "Treatment_Outcome" in sub.columns else "first",
            "malignant_cells": "first",
            elig_col: "first",
            proj_col: first_or_na,
        })
    )

    wide = pt.pivot(index="sample", columns="timepoint", values=proj_col)
    wide.columns = [f"{c}_branch_ge{thr}" for c in wide.columns]
    wide = wide.reset_index()

    meta = (
        pt.groupby("sample", as_index=False)
        .agg({
            "Patient_ID": first_or_na,
            "Biopsy_Origin": first_or_na if "Biopsy_Origin" in pt.columns else "first",
            "Subgroup": first_or_na if "Subgroup" in pt.columns else "first",
            "Treatment_Outcome": first_or_na if "Treatment_Outcome" in pt.columns else "first",
        })
    )

    patient_tab = meta.merge(wide, on="sample", how="left")
    patient_tab["threshold"] = thr

    # switch summaries for DX -> REL
    dx_col = f"DX_branch_ge{thr}"
    rel_col = f"REL_branch_ge{thr}"

    patient_tab["dx_rel_pair_available"] = (
        patient_tab[dx_col].notna() & patient_tab[rel_col].notna()
        if dx_col in patient_tab.columns and rel_col in patient_tab.columns
        else False
    )

    if dx_col in patient_tab.columns and rel_col in patient_tab.columns:
        patient_tab["dx_to_rel_switch"] = np.where(
            patient_tab["dx_rel_pair_available"],
            patient_tab[dx_col] != patient_tab[rel_col],
            pd.NA
        )
    else:
        patient_tab["dx_to_rel_switch"] = pd.NA

    patient_rows.append(patient_tab)

    # transition matrix
    if dx_col in patient_tab.columns and rel_col in patient_tab.columns:
        pair = patient_tab.loc[patient_tab["dx_rel_pair_available"], [dx_col, rel_col]].copy()
        if len(pair) > 0:
            trans = pd.crosstab(pair[dx_col], pair[rel_col], dropna=False)
            for dx_branch in ["B1", "B2", "B3"]:
                for rel_branch in ["B1", "B2", "B3"]:
                    transition_rows.append({
                        "threshold": thr,
                        "dx_branch": dx_branch,
                        "rel_branch": rel_branch,
                        "count": int(trans.loc[dx_branch, rel_branch]) if dx_branch in trans.index and rel_branch in trans.columns else 0
                    })

            n_pairs = int(len(pair))
            n_switch = int((pair[dx_col] != pair[rel_col]).sum())
            switch_rows.append({
                "threshold": thr,
                "n_dx_rel_pairs": n_pairs,
                "n_switch": n_switch,
                "n_stable": n_pairs - n_switch,
                "switch_rate": n_switch / n_pairs if n_pairs > 0 else np.nan
            })
        else:
            switch_rows.append({
                "threshold": thr,
                "n_dx_rel_pairs": 0,
                "n_switch": 0,
                "n_stable": 0,
                "switch_rate": np.nan
            })

patient_df = pd.concat(patient_rows, ignore_index=True)
patient_df.to_csv(out_patient, index=False)

transition_df = pd.DataFrame(transition_rows)
transition_df.to_csv(out_transition, index=False)

switch_df = pd.DataFrame(switch_rows).sort_values("threshold")
switch_df.to_csv(out_switch, index=False)

# --------------------------------------------------
# Write text summary
# --------------------------------------------------
lines = []
lines.append("GSE235063 transition summary and threshold sensitivity")
lines.append("=" * 70)
lines.append("")

for thr in thresholds:
    lines.append(f"Threshold >= {thr} malignant cells")
    lines.append("-" * 40)

    csub = counts_df[counts_df["threshold"] == thr].copy()
    lines.append("Branch counts by timepoint:")
    lines.append(csub.to_string(index=False))
    lines.append("")

    ssub = switch_df[switch_df["threshold"] == thr]
    if len(ssub) > 0:
        lines.append("DX -> REL switch summary:")
        lines.append(ssub.to_string(index=False))
        lines.append("")

    tsub = transition_df[transition_df["threshold"] == thr].copy()
    if len(tsub) > 0:
        mat = tsub.pivot(index="dx_branch", columns="rel_branch", values="count").fillna(0).astype(int)
        lines.append("DX -> REL transition matrix:")
        lines.append(mat.to_string())
        lines.append("")

    psub = patient_df[patient_df["threshold"] == thr].copy()
    keep_cols = [c for c in [
        "sample", "Patient_ID", "Treatment_Outcome", "Subgroup",
        f"DX_branch_ge{thr}", f"REM_branch_ge{thr}", f"REL_branch_ge{thr}",
        "dx_rel_pair_available", "dx_to_rel_switch"
    ] if c in psub.columns]
    lines.append("Patient-level branch table:")
    lines.append(psub[keep_cols].sort_values("sample").to_string(index=False))
    lines.append("")

with open(out_text, "w") as f:
    f.write("\n".join(lines))

# --------------------------------------------------
# Figures
# --------------------------------------------------
plt.figure(figsize=(6, 4))
plt.plot(switch_df["threshold"], switch_df["switch_rate"], marker="o")
plt.xlabel("Malignant-cell threshold")
plt.ylabel("DX→REL switch rate")
plt.title("DX→REL switch-rate sensitivity")
plt.tight_layout()
plt.savefig(fig_switch, dpi=300, bbox_inches="tight")
plt.close()

# stacked DX/REL counts by threshold
plot_rows = []
for thr in thresholds:
    for tp in ["DX", "REL"]:
        sub = counts_df[(counts_df["threshold"] == thr) & (counts_df["timepoint"] == tp)]
        if len(sub) == 0:
            continue
        row = sub.iloc[0].to_dict()
        plot_rows.append(row)

plot_df = pd.DataFrame(plot_rows)

fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
for ax, tp in zip(axes, ["DX", "REL"]):
    sub = plot_df[plot_df["timepoint"] == tp].sort_values("threshold")
    x = np.arange(len(sub))
    ax.bar(x, sub["B1"], label="B1")
    ax.bar(x, sub["B2"], bottom=sub["B1"], label="B2")
    ax.bar(x, sub["B3"], bottom=sub["B1"] + sub["B2"], label="B3")
    ax.set_xticks(x)
    ax.set_xticklabels(sub["threshold"].astype(str))
    ax.set_xlabel("Threshold")
    ax.set_title(tp)
axes[0].set_ylabel("Eligible projected samples")
axes[1].legend()
fig.suptitle("Projected branch counts by threshold")
fig.tight_layout()
fig.savefig(fig_counts, dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"saved: {out_counts}")
print(f"saved: {out_transition}")
print(f"saved: {out_patient}")
print(f"saved: {out_switch}")
print(f"saved: {out_text}")
print(f"saved: {fig_switch}")
print(f"saved: {fig_counts}")

print("\nDX -> REL switch rates:")
print(switch_df.to_string(index=False))

print("\nProjected branch counts by timepoint and threshold:")
print(counts_df.to_string(index=False))
