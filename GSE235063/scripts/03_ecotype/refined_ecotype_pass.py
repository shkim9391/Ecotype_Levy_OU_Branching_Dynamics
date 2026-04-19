python - <<'PY'
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

ROOT = Path("/GSE235063/derived_dx_primary_training")

sample = pd.read_csv(ROOT / "dx_primary_training_sample_level_summary.csv").set_index("sample_id")
fine = pd.read_csv(ROOT / "dx_normal_celltype_fractions_by_sample.csv", index_col=0)

# align
df = sample.join(fine, how="inner")

# refined discovery set
disc = df[(df["normal_cells"] >= 300) & (df["malignant_cells"] > 0)].copy()

print("\n=== DISCOVERY SET ===")
print("Samples in joined table:", df.shape[0])
print("Samples in refined discovery:", disc.shape[0])

print("\n=== EXCLUDED FROM REFINED DISCOVERY ===")
excluded = df.loc[
    ~df.index.isin(disc.index),
    ["sample", "gsm", "Patient_ID", "Biopsy_Origin", "normal_cells", "malignant_cells"]
]
print("None" if len(excluded) == 0 else excluded.to_string())

# fine features
all_feature_cols = [c for c in fine.columns if c != "Unknown"]

# primary threshold
feature_cols = [c for c in all_feature_cols if (disc[c] >= 0.02).sum() >= 3]

# fallback if too few survive
if len(feature_cols) < 3:
    feature_cols = [c for c in all_feature_cols if (disc[c] >= 0.01).sum() >= 2]

print("\n=== FINE CELLTYPE FEATURES USED ===")
print(feature_cols)
print("Number of features:", len(feature_cols))

if len(feature_cols) < 2:
    raise ValueError("Too few informative fine cell-type features survived filtering.")

X_frac = disc[feature_cols].copy()
X_frac = X_frac.div(X_frac.sum(axis=1), axis=0).fillna(0)

# remove zero-variance columns after renormalization
eps = 1e-4
X_clr = np.log(X_frac + eps)
X_clr = X_clr.sub(X_clr.mean(axis=1), axis=0)

col_sd_raw = X_clr.std(axis=0, ddof=0)
keep_cols = col_sd_raw[col_sd_raw > 0].index.tolist()

if len(keep_cols) < len(feature_cols):
    dropped = [c for c in feature_cols if c not in keep_cols]
    print("\nDropped zero-variance features:", dropped)

feature_cols = keep_cols
X_clr = X_clr[feature_cols]
col_sd = X_clr.std(axis=0, ddof=0)
X = (X_clr - X_clr.mean(axis=0)) / col_sd

if X.shape[0] < 4:
    raise ValueError(f"Too few discovery samples after filtering: {X.shape[0]}")

print("\n=== MATRIX SHAPE FOR CLUSTERING ===")
print("Samples:", X.shape[0])
print("Features:", X.shape[1])

# choose k
sil_rows = []
labels_by_k = {}

max_k = min(4, X.shape[0] - 1)
for k in range(2, max_k + 1):
    model = AgglomerativeClustering(n_clusters=k, linkage="ward")
    labels = model.fit_predict(X.values)
    sil = silhouette_score(X.values, labels)
    sil_rows.append({"k": k, "silhouette": float(sil)})
    labels_by_k[k] = labels

sil_df = pd.DataFrame(sil_rows).sort_values("k")
best_k = int(sil_df.sort_values("silhouette", ascending=False).iloc[0]["k"])

disc["ecotype_cluster"] = labels_by_k[best_k] + 1

# PCA
pca = PCA(n_components=2)
pcs = pca.fit_transform(X.values)
disc["PC1"] = pcs[:, 0]
disc["PC2"] = pcs[:, 1]

# cluster summaries
cluster_means = disc.groupby("ecotype_cluster")[feature_cols].mean()
global_mean = disc[feature_cols].mean()
cluster_delta = cluster_means.sub(global_mean, axis=1)

def clean_name(x):
    return (
        x.replace(".", "")
         .replace(" ", "")
         .replace("/", "")
         .replace("-", "")
    )

label_map = {}
for cid in cluster_delta.index:
    top2 = cluster_delta.loc[cid].sort_values(ascending=False).head(2).index.tolist()
    if len(top2) == 1:
        label_map[cid] = f"E{cid}_{clean_name(top2[0])}"
    else:
        label_map[cid] = f"E{cid}_{clean_name(top2[0])}_{clean_name(top2[1])}"

disc["ecotype_label"] = disc["ecotype_cluster"].map(label_map)

# save outputs
sil_df.to_csv(ROOT / "dx_ecotype_refined_fine_silhouette_scores.csv", index=False)
cluster_means.to_csv(ROOT / "dx_ecotype_refined_fine_cluster_means.csv")
cluster_delta.to_csv(ROOT / "dx_ecotype_refined_fine_cluster_deltas_vs_global.csv")

out_cols = [
    "gsm", "sample", "Patient_ID", "Biopsy_Origin", "Subgroup",
    "Expected_Driving_Aberration", "Treatment_Outcome",
    "total_cells", "malignant_cells", "normal_cells",
    "malignant_frac", "normal_frac",
    "ecotype_cluster", "ecotype_label", "PC1", "PC2"
] + feature_cols

disc.reset_index()[["sample_id"] + out_cols].to_csv(
    ROOT / "dx_ecotype_refined_fine_assignments.csv",
    index=False
)

disc.reset_index()[[
    "sample_id", "sample", "gsm", "Patient_ID", "Biopsy_Origin",
    "Subgroup", "Expected_Driving_Aberration", "Treatment_Outcome",
    "malignant_cells", "normal_cells", "PC1", "PC2",
    "ecotype_cluster", "ecotype_label"
]].to_csv(
    ROOT / "dx_ecotype_continuous_covariates_for_ou.csv",
    index=False
)

print("\n=== SILHOUETTE SCORES ===")
print(sil_df.to_string(index=False))

print(f"\nChosen k = {best_k}")

print("\n=== REFINED ECOTYPE LABELS ===")
print(
    disc[["sample", "Biopsy_Origin", "Subgroup", "ecotype_label"]]
    .sort_values(["ecotype_label", "sample"])
    .to_string()
)

print("\n=== REFINED CLUSTER MEANS ===")
print(cluster_means.round(4).to_string())

print("\n=== DELTA VS GLOBAL MEAN ===")
print(cluster_delta.round(4).to_string())

print("\n=== ECOTYPE x BIOPSY ORIGIN ===")
print(pd.crosstab(disc["ecotype_label"], disc["Biopsy_Origin"]).to_string())

print("\n=== ECOTYPE x SUBGROUP ===")
print(pd.crosstab(disc["ecotype_label"], disc["Subgroup"]).to_string())

print("\n=== FILES WRITTEN ===")
for fn in [
    "dx_ecotype_refined_fine_silhouette_scores.csv",
    "dx_ecotype_refined_fine_cluster_means.csv",
    "dx_ecotype_refined_fine_cluster_deltas_vs_global.csv",
    "dx_ecotype_refined_fine_assignments.csv",
    "dx_ecotype_continuous_covariates_for_ou.csv",
]:
    print(ROOT / fn)
PY
