from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

base = Path("/Lévy_OU_Branching")

patient_file = base / "gse235063_patient_branch_table_by_threshold.csv"
long_file = base / "gse235063_longitudinal_branch_projection.csv"

threshold = 50

out_matrix_csv = base / f"gse235063_dx_rel_transition_matrix_threshold{threshold}.csv"
out_pairs_csv = base / f"gse235063_dx_rel_patient_pairs_threshold{threshold}.csv"
out_pca_csv = base / f"gse235063_dx_rel_pca_scores_threshold{threshold}.csv"

fig_matrix = base / f"gse235063_dx_rel_transition_matrix_threshold{threshold}.png"
fig_traj = base / f"gse235063_dx_rel_patient_trajectory_figure_threshold{threshold}.png"

# --------------------------------------------------
# Load
# --------------------------------------------------
patient = pd.read_csv(patient_file)
long = pd.read_csv(long_file)

patient = patient[patient["threshold"] == threshold].copy()

dx_col = f"DX_branch_ge{threshold}"
rel_col = f"REL_branch_ge{threshold}"

required_patient = ["sample", "Patient_ID", dx_col, rel_col]
missing_patient = [c for c in required_patient if c not in patient.columns]
if missing_patient:
    raise ValueError(f"Missing columns in patient table: {missing_patient}")

state_cols = [
    "ilr_stem_vs_committed",
    "ilr_prog_vs_mature",
    "ilr_gmp_vs_monodc",
    "T_NK_given_known_z",
    "Myeloid_APC_given_known_z",
    "B_Plasma_given_known_z",
]

required_long = ["sample", "sample_id", "Patient_ID", "timepoint", "malignant_cells"] + state_cols
missing_long = [c for c in required_long if c not in long.columns]
if missing_long:
    raise ValueError(f"Missing columns in longitudinal table: {missing_long}")

for c in ["malignant_cells"] + state_cols:
    long[c] = pd.to_numeric(long[c], errors="coerce")

# --------------------------------------------------
# Patient pairs and transition matrix
# --------------------------------------------------
pairs = patient.copy()
pairs["dx_rel_pair_available"] = pairs[dx_col].notna() & pairs[rel_col].notna()
pairs = pairs[pairs["dx_rel_pair_available"]].copy()
pairs["dx_to_rel_switch"] = pairs[dx_col] != pairs[rel_col]

branches = ["B1", "B2", "B3"]
mat = pd.crosstab(pairs[dx_col], pairs[rel_col], dropna=False)
mat = mat.reindex(index=branches, columns=branches, fill_value=0)
mat.to_csv(out_matrix_csv)

pairs.to_csv(out_pairs_csv, index=False)

# --------------------------------------------------
# Heatmap figure
# --------------------------------------------------
plt.figure(figsize=(5.5, 4.5))
im = plt.imshow(mat.values, aspect="auto")
plt.colorbar(im)

plt.xticks(range(len(branches)), branches)
plt.yticks(range(len(branches)), branches)
plt.xlabel("REL branch")
plt.ylabel("DX branch")
plt.title(f"DX→REL transition matrix (threshold ≥ {threshold})")

for i in range(mat.shape[0]):
    for j in range(mat.shape[1]):
        plt.text(j, i, str(int(mat.iloc[i, j])), ha="center", va="center")

plt.tight_layout()
plt.savefig(fig_matrix, dpi=300, bbox_inches="tight")
plt.close()

# --------------------------------------------------
# Trajectory figure in PCA space
# Fit PCA on all eligible DX/REL samples at this threshold
# --------------------------------------------------
plot_df = long[long["timepoint"].isin(["DX", "REL"])].copy()
plot_df["eligible_ge_thr"] = (
    plot_df["malignant_cells"].fillna(0) >= threshold
) & (
    ~plot_df[state_cols].isna().any(axis=1)
)

plot_df = plot_df[plot_df["eligible_ge_thr"]].copy()

pca = PCA(n_components=2)
pcs = pca.fit_transform(plot_df[state_cols].to_numpy(dtype=float))
plot_df["PC1"] = pcs[:, 0]
plot_df["PC2"] = pcs[:, 1]
plot_df.to_csv(out_pca_csv, index=False)

# only patients with both DX and REL eligible
paired_samples = set(pairs["sample"])
traj_df = plot_df[plot_df["sample"].isin(paired_samples)].copy()

# make sure one row per sample/timepoint
traj_df = (
    traj_df.sort_values(["sample", "timepoint", "sample_id"])
    .groupby(["sample", "timepoint"], as_index=False)
    .first()
)

plt.figure(figsize=(8, 6))

# background: all eligible DX/REL samples
for tp, marker in [("DX", "o"), ("REL", "^")]:
    sub = plot_df[plot_df["timepoint"] == tp]
    plt.scatter(sub["PC1"], sub["PC2"], marker=marker, alpha=0.30, label=f"{tp} eligible")

# arrows for paired patients
for sample in sorted(paired_samples):
    sub = traj_df[traj_df["sample"] == sample].copy()
    if set(sub["timepoint"]) >= {"DX", "REL"}:
        dx = sub[sub["timepoint"] == "DX"].iloc[0]
        rel = sub[sub["timepoint"] == "REL"].iloc[0]

        plt.annotate(
            "",
            xy=(rel["PC1"], rel["PC2"]),
            xytext=(dx["PC1"], dx["PC2"]),
            arrowprops=dict(arrowstyle="->", lw=1.5),
        )

        plt.scatter([dx["PC1"]], [dx["PC2"]], s=60, marker="o")
        plt.scatter([rel["PC1"]], [rel["PC2"]], s=60, marker="^")

        midx = 0.5 * (dx["PC1"] + rel["PC1"])
        midy = 0.5 * (dx["PC2"] + rel["PC2"])
        plt.text(midx, midy, sample, fontsize=8)

plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
plt.title(f"DX→REL patient trajectories in projected state space (threshold ≥ {threshold})")
plt.legend()
plt.tight_layout()
plt.savefig(fig_traj, dpi=300, bbox_inches="tight")
plt.close()

# --------------------------------------------------
# Console summary
# --------------------------------------------------
print(f"saved: {out_matrix_csv}")
print(f"saved: {out_pairs_csv}")
print(f"saved: {out_pca_csv}")
print(f"saved: {fig_matrix}")
print(f"saved: {fig_traj}")

print("\nDX→REL transition matrix:")
print(mat.to_string())

print("\nDX→REL patient pairs:")
keep_cols = [c for c in ["sample", "Patient_ID", "Treatment_Outcome", "Subgroup", dx_col, rel_col, "dx_to_rel_switch"] if c in pairs.columns]
print(pairs[keep_cols].sort_values("sample").to_string(index=False))

print("\nSwitch summary:")
print({
    "threshold": threshold,
    "n_pairs": int(len(pairs)),
    "n_switch": int(pairs["dx_to_rel_switch"].sum()),
    "n_stable": int((~pairs["dx_to_rel_switch"]).sum()),
    "switch_rate": float(pairs["dx_to_rel_switch"].mean()) if len(pairs) > 0 else np.nan,
})
