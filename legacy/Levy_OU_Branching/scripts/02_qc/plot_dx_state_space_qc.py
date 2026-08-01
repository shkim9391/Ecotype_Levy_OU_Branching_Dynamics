from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

base = Path("/Lévy_OU_Branching")
infile = base / "dx_diagnosis_state_ready_matrix.csv"

out_corr_csv = base / "dx_state_correlation_matrix.csv"
out_pca_csv = base / "dx_state_pca_scores.csv"
out_dist_csv = base / "dx_state_centroid_distance_ranking.csv"

fig_corr = base / "dx_state_correlation_heatmap.png"
fig_pca_outcome = base / "dx_state_pca_by_outcome.png"
fig_pca_subgroup = base / "dx_state_pca_by_subgroup.png"
fig_dist = base / "dx_state_centroid_distance.png"

df = pd.read_csv(infile)

state_cols = [
    "ilr_stem_vs_committed",
    "ilr_prog_vs_mature",
    "ilr_gmp_vs_monodc",
    "T_NK_given_known_z",
    "Myeloid_APC_given_known_z",
    "B_Plasma_given_known_z",
]

# --------------------------------------------------
# Basic checks
# --------------------------------------------------
missing = [c for c in state_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing state columns: {missing}")

X = df[state_cols].copy()

print("input shape:", df.shape)
print("state matrix shape:", X.shape)

# --------------------------------------------------
# Correlation matrix
# --------------------------------------------------
corr = X.corr()
corr.to_csv(out_corr_csv)

plt.figure(figsize=(8, 6))
im = plt.imshow(corr.values, aspect="auto")
plt.colorbar(im)
plt.xticks(range(len(state_cols)), state_cols, rotation=45, ha="right")
plt.yticks(range(len(state_cols)), state_cols)
plt.title("Diagnosis state-vector correlation matrix")
plt.tight_layout()
plt.savefig(fig_corr, dpi=300, bbox_inches="tight")
plt.close()

# --------------------------------------------------
# PCA
# --------------------------------------------------
pca = PCA(n_components=2)
pcs = pca.fit_transform(X.values)

pca_df = df[[
    "sample_id", "sample", "Patient_ID", "Treatment_Outcome",
    "Subgroup", "Biopsy_Origin", "malignant_cells", "malignant_frac"
]].copy()

pca_df["PC1"] = pcs[:, 0]
pca_df["PC2"] = pcs[:, 1]
pca_df.to_csv(out_pca_csv, index=False)

print("PCA explained variance ratio:", pca.explained_variance_ratio_)

# helper for category plotting
def scatter_by_category(plot_df, category, outfile, title):
    cats = plot_df[category].fillna("NA").astype(str).unique()
    plt.figure(figsize=(7, 6))
    for cat in cats:
        sub = plot_df[plot_df[category].fillna("NA").astype(str) == cat]
        plt.scatter(sub["PC1"], sub["PC2"], label=cat)
        for _, row in sub.iterrows():
            plt.text(row["PC1"], row["PC2"], row["sample"], fontsize=8)
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    plt.title(title)
    plt.legend(fontsize=8, loc="best")
    plt.tight_layout()
    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close()

scatter_by_category(
    pca_df,
    "Treatment_Outcome",
    fig_pca_outcome,
    "Diagnosis state-space PCA colored by outcome"
)

scatter_by_category(
    pca_df,
    "Subgroup",
    fig_pca_subgroup,
    "Diagnosis state-space PCA colored by subgroup"
)

# --------------------------------------------------
# Distance from cohort centroid
# --------------------------------------------------
centroid = X.mean(axis=0).values
dist = np.sqrt(((X.values - centroid) ** 2).sum(axis=1))

dist_df = df[[
    "sample_id", "sample", "Patient_ID", "Treatment_Outcome",
    "Subgroup", "Biopsy_Origin", "malignant_cells", "malignant_frac"
]].copy()
dist_df["distance_from_centroid"] = dist
dist_df = dist_df.sort_values("distance_from_centroid", ascending=False)
dist_df.to_csv(out_dist_csv, index=False)

plt.figure(figsize=(10, 5))
plt.bar(dist_df["sample"], dist_df["distance_from_centroid"])
plt.xticks(rotation=45, ha="right")
plt.ylabel("Euclidean distance from cohort centroid")
plt.title("Diagnosis state-space displacement from cohort centroid")
plt.tight_layout()
plt.savefig(fig_dist, dpi=300, bbox_inches="tight")
plt.close()

print(f"saved: {out_corr_csv}")
print(f"saved: {out_pca_csv}")
print(f"saved: {out_dist_csv}")
print(f"saved: {fig_corr}")
print(f"saved: {fig_pca_outcome}")
print(f"saved: {fig_pca_subgroup}")
print(f"saved: {fig_dist}")

print("\nTop 10 by distance from centroid:")
print(dist_df.head(10).to_string(index=False))
