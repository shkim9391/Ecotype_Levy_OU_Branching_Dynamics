from pathlib import Path
import pandas as pd

base = Path("/Lévy_OU_Branching")
ecotype_file = Path(
    "/GSE235063/derived_dx_primary_training/"
    "dx_ecotype_association_eligible_samples.csv"
)

full_file = base / "dx_diagnosis_baseline_matrix_full.csv"
state_ready_file = base / "dx_diagnosis_state_ready_matrix.csv"

out_prov = base / "dx_cohort_provenance_table.csv"
out_excluded = base / "dx_transfer_missing_after_eligibility.csv"
out_summary = base / "dx_included_vs_excluded_summary.txt"

full = pd.read_csv(full_file)
eligible = pd.read_csv(ecotype_file)
state_ready = pd.read_csv(state_ready_file)

full["sample_id"] = full["sample_id"].astype(str).str.strip()
eligible["sample_id"] = eligible["sample_id"].astype(str).str.strip()
state_ready["sample_id"] = state_ready["sample_id"].astype(str).str.strip()

prov = full.copy()

prov["in_full_dx"] = True
prov["in_ecotype_eligible"] = prov["sample_id"].isin(set(eligible["sample_id"]))
prov["in_transfer_state_ready"] = prov["sample_id"].isin(set(state_ready["sample_id"]))

prov["excluded_pre_eligibility"] = ~prov["in_ecotype_eligible"]
prov["excluded_post_eligibility"] = prov["in_ecotype_eligible"] & (~prov["in_transfer_state_ready"])

def assign_stage(row):
    if row["in_transfer_state_ready"]:
        return "state_ready_19"
    if row["excluded_post_eligibility"]:
        return "eligible_but_missing_transfer_6"
    return "excluded_pre_eligibility_2"

prov["cohort_stage"] = prov.apply(assign_stage, axis=1)

# Save provenance table
prov.to_csv(out_prov, index=False)

# Save the 6 key samples that passed eligibility but lost transfer-state features
excluded6 = prov.loc[prov["excluded_post_eligibility"]].copy()
excluded6.to_csv(out_excluded, index=False)

# Build a text summary
lines = []
lines.append("Diagnosis cohort provenance summary")
lines.append("=" * 60)
lines.append(f"Full DX samples: {len(prov)}")
lines.append(f"Ecotype-eligible: {int(prov['in_ecotype_eligible'].sum())}")
lines.append(f"Transfer/state-ready: {int(prov['in_transfer_state_ready'].sum())}")
lines.append(f"Excluded pre-eligibility: {int(prov['excluded_pre_eligibility'].sum())}")
lines.append(f"Excluded post-eligibility: {int(prov['excluded_post_eligibility'].sum())}")
lines.append("")

for stage in ["state_ready_19", "eligible_but_missing_transfer_6", "excluded_pre_eligibility_2"]:
    sub = prov.loc[prov["cohort_stage"] == stage].copy()
    lines.append(stage)
    lines.append("-" * len(stage))
    lines.append(f"n = {len(sub)}")
    if len(sub) == 0:
        lines.append("")
        continue

    if "Treatment_Outcome" in sub.columns:
        lines.append("Treatment_Outcome:")
        lines.append(sub["Treatment_Outcome"].value_counts(dropna=False).to_string())
        lines.append("")

    if "Subgroup" in sub.columns:
        lines.append("Subgroup:")
        lines.append(sub["Subgroup"].value_counts(dropna=False).to_string())
        lines.append("")

    if "Biopsy_Origin" in sub.columns:
        lines.append("Biopsy_Origin:")
        lines.append(sub["Biopsy_Origin"].value_counts(dropna=False).to_string())
        lines.append("")

    for col in ["malignant_cells", "malignant_frac", "normal_cells", "normal_frac"]:
        if col in sub.columns:
            lines.append(f"{col}:")
            lines.append(sub[col].describe().to_string())
            lines.append("")

    lines.append("sample_ids:")
    lines.append(", ".join(sub["sample_id"].tolist()))
    lines.append("")

with open(out_summary, "w") as f:
    f.write("\n".join(lines))

print(f"saved: {out_prov}")
print(f"saved: {out_excluded}")
print(f"saved: {out_summary}")

print("\nCohort stage counts:")
print(prov["cohort_stage"].value_counts())

print("\nPost-eligibility transfer-missing samples:")
print(
    prov.loc[prov["excluded_post_eligibility"],
             ["sample_id", "Treatment_Outcome", "Subgroup", "Biopsy_Origin", "malignant_cells", "malignant_frac"]]
    .to_string(index=False)
)
