from pathlib import Path
import pandas as pd
import numpy as np

base = Path("/Lévy_OU_Branching")

infile = base / "gse235063_dx_rel_displacement_table_threshold50.csv"
outfile = base / "gse235063_dx_rel_jump_candidate_table_threshold50.csv"
summary_file = base / "gse235063_dx_rel_jump_candidate_summary_threshold50.txt"

df = pd.read_csv(infile)

required_cols = [
    "sample",
    "Patient_ID",
    "Subgroup",
    "DX_branch_ge50",
    "REL_branch_ge50",
    "dx_to_rel_switch",
    "dx_malignant_cells",
    "rel_malignant_cells",
    "disp_total_6d",
    "disp_malignant_3d",
    "disp_tme_3d",
]

missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

# -----------------------------
# Basic type cleanup
# -----------------------------
df["dx_to_rel_switch"] = df["dx_to_rel_switch"].astype(bool)

for c in [
    "dx_malignant_cells",
    "rel_malignant_cells",
    "disp_total_6d",
    "disp_malignant_3d",
    "disp_tme_3d",
]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# -----------------------------
# Rank / percentile features
# -----------------------------
df = df.sort_values("disp_total_6d", ascending=False).reset_index(drop=True)
df["disp_total_rank_desc"] = np.arange(1, len(df) + 1)
df["disp_total_percentile"] = df["disp_total_6d"].rank(method="average", pct=True)

# top quartile cutoff
q75_total = df["disp_total_6d"].quantile(0.75)
q75_malignant = df["disp_malignant_3d"].quantile(0.75)
q75_tme = df["disp_tme_3d"].quantile(0.75)

df["high_disp_total_q75"] = df["disp_total_6d"] >= q75_total
df["high_disp_malignant_q75"] = df["disp_malignant_3d"] >= q75_malignant
df["high_disp_tme_q75"] = df["disp_tme_3d"] >= q75_tme

# -----------------------------
# Jump-candidate taxonomy
# Tier 1: switching + high total displacement
# Tier 2: stable + high total displacement
# Tier 3: all remaining pairs
# -----------------------------
def assign_tier(row):
    if row["dx_to_rel_switch"] and row["high_disp_total_q75"]:
        return "Tier1_switching_high_disp"
    if (not row["dx_to_rel_switch"]) and row["high_disp_total_q75"]:
        return "Tier2_stable_high_disp"
    return "Tier3_other"

df["jump_candidate_tier"] = df.apply(assign_tier, axis=1)

# Optional finer labels
def assign_subtype(row):
    if row["dx_to_rel_switch"] and row["high_disp_malignant_q75"] and row["high_disp_tme_q75"]:
        return "switch_high_malignant_and_tme"
    if row["dx_to_rel_switch"] and row["high_disp_malignant_q75"]:
        return "switch_high_malignant"
    if row["dx_to_rel_switch"] and row["high_disp_tme_q75"]:
        return "switch_high_tme"
    if (not row["dx_to_rel_switch"]) and row["high_disp_total_q75"]:
        return "stable_high_disp"
    return "background"

df["jump_candidate_subtype"] = df.apply(assign_subtype, axis=1)

# -----------------------------
# Branch transition label
# -----------------------------
df["dx_rel_transition"] = (
    df["DX_branch_ge50"].astype(str) + "->" + df["REL_branch_ge50"].astype(str)
)

# -----------------------------
# Select / order columns
# -----------------------------
front_cols = [
    "sample",
    "Patient_ID",
    "Subgroup",
    "DX_branch_ge50",
    "REL_branch_ge50",
    "dx_rel_transition",
    "dx_to_rel_switch",
    "jump_candidate_tier",
    "jump_candidate_subtype",
    "disp_total_rank_desc",
    "disp_total_percentile",
    "disp_total_6d",
    "disp_malignant_3d",
    "disp_tme_3d",
    "high_disp_total_q75",
    "high_disp_malignant_q75",
    "high_disp_tme_q75",
    "dx_malignant_cells",
    "rel_malignant_cells",
]

other_cols = [c for c in df.columns if c not in front_cols]
df = df[front_cols + other_cols]

# -----------------------------
# Save table
# -----------------------------
df.to_csv(outfile, index=False)

# -----------------------------
# Build text summary
# -----------------------------
tier_counts = df["jump_candidate_tier"].value_counts().sort_index()
transition_counts = df["dx_rel_transition"].value_counts().sort_values(ascending=False)
subtype_counts = df["jump_candidate_subtype"].value_counts().sort_values(ascending=False)

lines = []
lines.append("DX→REL jump-candidate table summary (threshold = 50)")
lines.append("=" * 70)
lines.append("")
lines.append(f"n_pairs = {len(df)}")
lines.append(f"q75_total_displacement = {q75_total:.6f}")
lines.append(f"q75_malignant_displacement = {q75_malignant:.6f}")
lines.append(f"q75_tme_displacement = {q75_tme:.6f}")
lines.append("")

lines.append("Tier counts:")
lines.append(tier_counts.to_string())
lines.append("")

lines.append("Subtype counts:")
lines.append(subtype_counts.to_string())
lines.append("")

lines.append("Branch transition counts:")
lines.append(transition_counts.to_string())
lines.append("")

lines.append("Top-ranked pairs by total displacement:")
lines.append(
    df[
        [
            "sample",
            "Patient_ID",
            "Subgroup",
            "dx_rel_transition",
            "dx_to_rel_switch",
            "jump_candidate_tier",
            "disp_total_6d",
            "disp_malignant_3d",
            "disp_tme_3d",
        ]
    ]
    .head(10)
    .to_string(index=False)
)
lines.append("")

lines.append("Full jump-candidate table:")
lines.append(df.to_string(index=False))

with open(summary_file, "w") as f:
    f.write("\n".join(lines))

# -----------------------------
# Console output
# -----------------------------
print(f"saved: {outfile}")
print(f"saved: {summary_file}")

print("\nTier counts:")
print(tier_counts.to_string())

print("\nTop 10 by total displacement:")
print(
    df[
        [
            "sample",
            "DX_branch_ge50",
            "REL_branch_ge50",
            "dx_to_rel_switch",
            "jump_candidate_tier",
            "disp_total_6d",
            "disp_malignant_3d",
            "disp_tme_3d",
        ]
    ]
    .head(10)
    .to_string(index=False)
)
