from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

base = Path("/Lévy_OU_Branching")
infile = base / "dx_branch_assignments_k3.csv"

out_means = base / "dx_branch_state_means_k3.csv"
out_sds = base / "dx_branch_state_sds_k3.csv"
out_summary = base / "dx_branch_characterization_k3.txt"
fig_heatmap = base / "dx_branch_state_means_heatmap_k3.png"

df = pd.read_csv(infile)

state_cols = [
    "ilr_stem_vs_committed",
    "ilr_prog_vs_mature",
    "ilr_gmp_vs_monodc",
    "T_NK_given_known_z",
    "Myeloid_APC_given_known_z",
    "B_Plasma_given_known_z",
]

group_col = "branch_label"

means = df.groupby(group_col)[state_cols].mean().sort_index()
sds = df.groupby(group_col)[state_cols].std().sort_index()

means.to_csv(out_means)
sds.to_csv(out_sds)

# text summary
lines = []
lines.append("Diagnosis branch characterization (k=3)")
lines.append("=" * 60)
lines.append("")
lines.append("Branch sizes:")
lines.append(df[group_col].value_counts().sort_index().to_string())
lines.append("")
lines.append("Mean state coordinates by branch:")
lines.append(means.to_string())
lines.append("")
lines.append("SD of state coordinates by branch:")
lines.append(sds.to_string())
lines.append("")
lines.append("Branch by outcome:")
lines.append(pd.crosstab(df[group_col], df["Treatment_Outcome"]).to_string())
lines.append("")
lines.append("Branch by subgroup:")
lines.append(pd.crosstab(df[group_col], df["Subgroup"]).to_string())
lines.append("")
lines.append("Branch members:")
lines.append(
    df[["sample_id", "branch_label", "Treatment_Outcome", "Subgroup", "Biopsy_Origin", "malignant_cells", "malignant_frac"]]
    .sort_values(["branch_label", "sample_id"])
    .to_string(index=False)
)

with open(out_summary, "w") as f:
    f.write("\n".join(lines))

# heatmap of branch means
plt.figure(figsize=(8, 4))
im = plt.imshow(means.values, aspect="auto")
plt.colorbar(im)
plt.xticks(range(len(state_cols)), state_cols, rotation=45, ha="right")
plt.yticks(range(len(means.index)), means.index)
plt.title("Mean state coordinates by provisional branch (k=3)")
plt.tight_layout()
plt.savefig(fig_heatmap, dpi=300, bbox_inches="tight")
plt.close()

print(f"saved: {out_means}")
print(f"saved: {out_sds}")
print(f"saved: {out_summary}")
print(f"saved: {fig_heatmap}")

print("\nMean state coordinates by branch:")
print(means.to_string())
