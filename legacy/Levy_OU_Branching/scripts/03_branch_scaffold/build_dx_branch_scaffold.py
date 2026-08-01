from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

base = Path("/Lévy_OU_Branching")
infile = base / "dx_diagnosis_state_ready_matrix.csv"

out_eval = base / "dx_branch_kmeans_model_selection.csv"
out_assign = base / "dx_branch_assignments_k3.csv"
out_centroids = base / "dx_branch_centroids_k3.csv"
out_summary = base / "dx_branch_membership_summary_k3.txt"

fig_pca = base / "dx_state_pca_by_branch_k3.png"
fig_sizes = base / "dx_branch_sizes_k3.png"

df = pd.read_csv(infile)

state_cols = [
    "ilr_stem_vs_committed",
    "ilr_prog_vs_mature",
    "ilr_gmp_vs_monodc",
    "T_NK_given_known_z",
    "Myeloid_APC_given_known_z",
    "B_Plasma_given_known_z",
]

meta_cols = [
    "sample_id",
    "sample",
    "Patient_ID",
    "Treatment_Outcome",
    "Subgroup",
    "Biopsy_Origin",
    "malignant_cells",
    "malignant_frac",
]

missing = [c for c in state_cols + meta_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

X = df[state_cols].copy()

# --------------------------------------------------
# Standardize state variables for clustering
# --------------------------------------------------
scaler = StandardScaler()
Xz = scaler.fit_transform(X.values)

# --------------------------------------------------
# Model selection scan for k = 2..5
# --------------------------------------------------
rows = []
for k in [2, 3, 4, 5]:
    km = KMeans(n_clusters=k, random_state=42, n_init=50)
    labels = km.fit_predict(Xz)
    inertia = km.inertia_
    sil = silhouette_score(Xz, labels) if k < len(df) else np.nan
    rows.append({
        "k": k,
        "inertia": inertia,
        "silhouette_score": sil,
    })

eval_df = pd.DataFrame(rows)
eval_df.to_csv(out_eval, index=False)

print("K-means model selection:")
print(eval_df.to_string(index=False))

# --------------------------------------------------
# Fit final scaffold with k = 3
# --------------------------------------------------
k_final = 3
km = KMeans(n_clusters=k_final, random_state=42, n_init=50)
raw_labels = km.fit_predict(Xz)

# PCA on standardized state space for visualization and stable label ordering
pca = PCA(n_components=2)
pcs = pca.fit_transform(Xz)

pca_df = df[meta_cols].copy()
pca_df["PC1"] = pcs[:, 0]
pca_df["PC2"] = pcs[:, 1]
pca_df["raw_cluster"] = raw_labels

# Order clusters by centroid location on PCA PC1 for stable B1/B2/B3 labels
cluster_pc1 = (
    pca_df.groupby("raw_cluster", as_index=False)["PC1"]
    .mean()
    .sort_values("PC1")
    .reset_index(drop=True)
)
cluster_pc1["branch_label"] = [f"B{i+1}" for i in range(len(cluster_pc1))]
label_map = dict(zip(cluster_pc1["raw_cluster"], cluster_pc1["branch_label"]))

pca_df["branch_label"] = pca_df["raw_cluster"].map(label_map)

# --------------------------------------------------
# Build assignment table
# --------------------------------------------------
assign_df = df[meta_cols + state_cols].copy()
assign_df["branch_label"] = pca_df["branch_label"].values
assign_df["PC1"] = pca_df["PC1"].values
assign_df["PC2"] = pca_df["PC2"].values
assign_df.to_csv(out_assign, index=False)

# --------------------------------------------------
# Branch centroids in original and standardized coordinates
# --------------------------------------------------
centroids_z = pd.DataFrame(
    km.cluster_centers_,
    columns=[f"{c}_zcluster" for c in state_cols]
)
centroids_z["raw_cluster"] = range(k_final)
centroids_z["branch_label"] = centroids_z["raw_cluster"].map(label_map)

centroids_orig = pd.DataFrame(
    scaler.inverse_transform(km.cluster_centers_),
    columns=state_cols
)
centroids_orig["raw_cluster"] = range(k_final)
centroids_orig["branch_label"] = centroids_orig["raw_cluster"].map(label_map)

centroids = centroids_orig.merge(
    centroids_z[["raw_cluster", "branch_label"] + [f"{c}_zcluster" for c in state_cols]],
    on=["raw_cluster", "branch_label"],
    how="left"
).sort_values("branch_label")

centroids.to_csv(out_centroids, index=False)

# --------------------------------------------------
# Summaries
# --------------------------------------------------
branch_counts = assign_df["branch_label"].value_counts().sort_index()
branch_by_outcome = pd.crosstab(assign_df["branch_label"], assign_df["Treatment_Outcome"])
branch_by_subgroup = pd.crosstab(assign_df["branch_label"], assign_df["Subgroup"])
branch_by_biopsy = pd.crosstab(assign_df["branch_label"], assign_df["Biopsy_Origin"])

lines = []
lines.append("Diagnosis branch scaffold summary (k = 3)")
lines.append("=" * 60)
lines.append("")
lines.append("Model selection:")
lines.append(eval_df.to_string(index=False))
lines.append("")
lines.append("Branch sizes:")
lines.append(branch_counts.to_string())
lines.append("")
lines.append("Branch by Treatment_Outcome:")
lines.append(branch_by_outcome.to_string())
lines.append("")
lines.append("Branch by Subgroup:")
lines.append(branch_by_subgroup.to_string())
lines.append("")
lines.append("Branch by Biopsy_Origin:")
lines.append(branch_by_biopsy.to_string())
lines.append("")
lines.append("Branch centroids (original coordinates):")
lines.append(
    centroids[
        ["branch_label"] + state_cols
    ].sort_values("branch_label").to_string(index=False)
)
lines.append("")
lines.append("Sample assignments:")
lines.append(
    assign_df[
        ["sample_id", "branch_label", "Treatment_Outcome", "Subgroup", "Biopsy_Origin", "malignant_cells", "malignant_frac"]
    ].sort_values(["branch_label", "sample_id"]).to_string(index=False)
)

with open(out_summary, "w") as f:
    f.write("\n".join(lines))

# --------------------------------------------------
# Plot PCA colored by branch
# --------------------------------------------------
plt.figure(figsize=(7, 6))
for branch in sorted(assign_df["branch_label"].unique()):
    sub = assign_df[assign_df["branch_label"] == branch]
    plt.scatter(sub["PC1"], sub["PC2"], label=branch)
    for _, row in sub.iterrows():
        plt.text(row["PC1"], row["PC2"], row["sample"], fontsize=8)

plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
plt.title("Diagnosis state-space PCA colored by provisional branch")
plt.legend()
plt.tight_layout()
plt.savefig(fig_pca, dpi=300, bbox_inches="tight")
plt.close()

# --------------------------------------------------
# Plot branch sizes
# --------------------------------------------------
branch_counts = branch_counts.sort_index()
plt.figure(figsize=(5, 4))
plt.bar(branch_counts.index, branch_counts.values)
plt.ylabel("Number of samples")
plt.title("Provisional diagnosis branch sizes (k=3)")
plt.tight_layout()
plt.savefig(fig_sizes, dpi=300, bbox_inches="tight")
plt.close()

print(f"saved: {out_eval}")
print(f"saved: {out_assign}")
print(f"saved: {out_centroids}")
print(f"saved: {out_summary}")
print(f"saved: {fig_pca}")
print(f"saved: {fig_sizes}")

print("\nBranch sizes:")
print(branch_counts.to_string())

print("\nAssignments:")
print(
    assign_df[
        ["sample_id", "branch_label", "Treatment_Outcome", "Subgroup", "Biopsy_Origin", "malignant_cells", "malignant_frac"]
    ].sort_values(["branch_label", "sample_id"]).to_string(index=False)
)
