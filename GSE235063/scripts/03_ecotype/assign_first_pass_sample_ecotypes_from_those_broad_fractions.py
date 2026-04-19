python - <<'PY'
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

ROOT = Path("/GSE235063/derived_dx_primary_training")

sample = pd.read_csv(ROOT / "dx_primary_training_sample_level_summary.csv")
broad = pd.read_csv(ROOT / "dx_normal_broad_cellgroup_fractions_by_sample.csv", index_col=0)

sample = sample.set_index("sample_id")
df = sample.join(broad, how="inner")

# --------------------------------------------------
# flags
# --------------------------------------------------
df["flag_low_normal"] = df["normal_cells"] < 200
df["flag_zero_malignant"] = df["malignant_cells"] == 0

# ecotype discovery set:
# exclude only unstable low-normal samples
eco = df.loc[~df["flag_low_normal"]].copy()

# association set:
# exclude low-normal + zero-malignant samples
assoc = eco.loc[~eco["flag_zero_malignant"]].copy()

features = [c for c in ["B_Plasma", "Erythroid_Baso", "HSPC_Prog", "Myeloid_APC", "T_NK"] if c in eco.columns]

# renormalize after dropping Unknown
X_frac = eco[features].copy()
X_frac = X_frac.div(X_frac.sum(axis=1), axis=0).fillna(0)

# CLR transform
eps = 1e-4
X_clr = np.log(X_frac + eps)
X_clr = X_clr.sub(X_clr.mean(axis=1), axis=0)

# z-score columns
col_sd = X_clr.std(axis=0, ddof=0).replace(0, 1.0)
X = (X_clr - X_clr.mean(axis=0)) / col_sd

# --------------------------------------------------
# choose k by silhouette
# --------------------------------------------------
sil_rows = []
labels_by_k = {}

max_k = min(5, len(X) - 1)
for k in range(2, max_k + 1):
    model = AgglomerativeClustering(n_clusters=k, linkage="ward")
    labels = model.fit_predict(X.values)
    sil = silhouette_score(X.values, labels)
    sil_rows.append({"k": k, "silhouette": sil})
    labels_by_k[k] = labels

sil_df = pd.DataFrame(sil_rows).sort_values("k")
best_k = int(sil_df.sort_values("silhouette", ascending=False).iloc[0]["k"])

eco["ecotype_cluster"] = labels_by_k[best_k] + 1

# --------------------------------------------------
# human-readable labels based on dominant broad group
# --------------------------------------------------
cluster_means = eco.groupby("ecotype_cluster")[features].mean()

pretty = {
    "B_Plasma": "BPlasma-rich",
    "Erythroid_Baso": "ErythroidBaso-rich",
    "HSPC_Prog": "HSPCProg-rich",
    "Myeloid_APC": "MyeloidAPC-rich",
    "T_NK": "TNK-rich",
}

cluster_name_map = {}
for cid in cluster_means.index:
    dom = cluster_means.loc[cid].idxmax()
    cluster_name_map[cid] = f"E{cid}_{pretty[dom]}"

eco["ecotype_label"] = eco["ecotype_cluster"].map(cluster_name_map)

# --------------------------------------------------
# PCA coordinates
# --------------------------------------------------
pca = PCA(n_components=2)
pcs = pca.fit_transform(X.values)
eco["PC1"] = pcs[:, 0]
eco["PC2"] = pcs[:, 1]

# --------------------------------------------------
# save outputs
# --------------------------------------------------
sil_df.to_csv(ROOT / "dx_ecotype_silhouette_scores.csv", index=False)
cluster_means.to_csv(ROOT / "dx_ecotype_cluster_means_rawfractions.csv")

eco_out_cols = [
    "gsm", "sample", "Patient_ID", "Biopsy_Origin", "Subgroup",
    "Expected_Driving_Aberration", "Treatment_Outcome",
    "total_cells", "malignant_cells", "normal_cells",
    "malignant_frac", "normal_frac",
    "flag_low_normal", "flag_zero_malignant",
    "ecotype_cluster", "ecotype_label", "PC1", "PC2"
] + features

eco.reset_index()[["sample_id"] + eco_out_cols].to_csv(
    ROOT / "dx_ecotype_firstpass_assignments.csv",
    index=False
)

assoc.reset_index().to_csv(
    ROOT / "dx_ecotype_association_eligible_samples.csv",
    index=False
)

# --------------------------------------------------
# print summary
# --------------------------------------------------
print("\n=== EXCLUDED FROM ECOTYPE DISCOVERY (low normal cells) ===")
tmp = df.loc[df["flag_low_normal"], ["sample", "gsm", "Patient_ID", "normal_cells", "malignant_cells", "Biopsy_Origin"]]
print("None" if len(tmp) == 0 else tmp.to_string())

print("\n=== FLAGGED AS ZERO-MALIGNANT ===")
tmp = df.loc[df["flag_zero_malignant"], ["sample", "gsm", "Patient_ID", "normal_cells", "malignant_cells", "Biopsy_Origin"]]
print("None" if len(tmp) == 0 else tmp.to_string())

print("\n=== SILHOUETTE SCORES ===")
print(sil_df.to_string(index=False))

print(f"\nChosen k = {best_k}")

print("\n=== ECOTYPE LABELS ===")
print(eco[["sample", "Biopsy_Origin", "Subgroup", "ecotype_label"]].sort_values(["ecotype_label", "sample"]).to_string())

print("\n=== CLUSTER MEANS (raw broad fractions) ===")
print(cluster_means.round(4).to_string())

print("\n=== ECOTYPE x BIOPSY ORIGIN ===")
print(pd.crosstab(eco["ecotype_label"], eco["Biopsy_Origin"]).to_string())

print("\n=== ECOTYPE x SUBGROUP ===")
print(pd.crosstab(eco["ecotype_label"], eco["Subgroup"]).to_string())

print("\n=== FILES WRITTEN ===")
for fn in [
    "dx_ecotype_silhouette_scores.csv",
    "dx_ecotype_cluster_means_rawfractions.csv",
    "dx_ecotype_firstpass_assignments.csv",
    "dx_ecotype_association_eligible_samples.csv",
]:
    print(ROOT / fn)
PY
