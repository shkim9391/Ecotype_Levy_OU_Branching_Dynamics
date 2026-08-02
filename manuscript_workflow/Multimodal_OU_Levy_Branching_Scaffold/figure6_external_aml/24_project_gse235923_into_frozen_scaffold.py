from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import KNeighborsClassifier


# ============================================================
# 1. CONFIG
# ============================================================
FIG2_REF = Path("/Multimodal_OU_Lévy_Branching_Scaffold/Figure_2/dx_primary_training_malignant_frozen_transfer.h5ad")

FIG6_DIR = Path("/Multimodal_OU_Lévy_Branching_Scaffold/Figure_6")
FIG6_DERIVED = FIG6_DIR / "derived"
FIG6_INPUTS = FIG6_DIR / "inputs"

QRY_H5AD = FIG6_DERIVED / "gse235923_longitudinal_allcells_raw.h5ad"
DX_LABEL_H5AD = Path("/Ecotype_OU_Branching/GSE235923/derived_secondary_calibration/gse235923_dx_secondary_calibration_labeled_by_gse235063.h5ad")

OUT_MALIGNANT_H5AD = FIG6_INPUTS / "gse235923_longitudinal_malignant_projected.h5ad"
OUT_SAMPLE_CSV = FIG6_DERIVED / "gse235923_frozen_sample_scores.csv"
OUT_QC = FIG6_DERIVED / "gse235923_label_transfer_qc.tsv"
OUT_ALLCELLS_LABELED = FIG6_DERIVED / "gse235923_longitudinal_allcells_labeled.h5ad"

SCAFFOLD_COLS = [
    "state_HSC",
    "state_Prog",
    "state_GMP",
    "state_MonoDC",
    "aux_EryBaso",
    "aux_CLP",
]
MAIN_STATE_COLS = ["state_HSC", "state_Prog", "state_GMP", "state_MonoDC"]

BRANCH_LABEL_MAP = {
    "state_HSC": "HSC-like basin",
    "state_Prog": "Progenitor-like basin",
    "state_GMP": "GMP-like basin",
    "state_MonoDC": "Mono/DC-like basin",
}

KNN_NEIGHBORS = 15
SVD_COMPONENTS = 50
RANDOM_STATE = 42


# ============================================================
# 2. HELPERS
# ============================================================
def ensure_dirs() -> None:
    FIG6_INPUTS.mkdir(parents=True, exist_ok=True)
    FIG6_DERIVED.mkdir(parents=True, exist_ok=True)


def assert_query_requirements(qry: ad.AnnData) -> None:
    req = ["sample_id", "patient_id", "clinical_timepoint_raw", "clinical_timepoint_coarse", "barcode_raw"]
    missing = [c for c in req if c not in qry.obs.columns]
    if missing:
        raise ValueError(f"Query object missing required obs columns: {missing}")


def assert_dx_label_requirements(dx: ad.AnnData) -> None:
    req = ["sample_id", "barcode_raw", "pred_malignant", "pred_celltype", "pred_broad", "pred_malignant_coarse"]
    missing = [c for c in req if c not in dx.obs.columns]
    if missing:
        raise ValueError(f"DX-labeled anchor missing required obs columns: {missing}")


def load_reference_recipe(ref):
    normal_map = dict(ref.uns["normal_broad_grouping"])
    pca_recipe = ref.uns["ecotype_pca"]

    feature_order = [str(x) for x in pca_recipe["feature_order"].tolist()]
    scaler_mean = np.asarray(pca_recipe["scaler_mean"], dtype=float)
    scaler_scale = np.asarray(pca_recipe["scaler_scale"], dtype=float)
    pca_components = np.asarray(pca_recipe["pca_components"], dtype=float)

    source_csv = Path(str(ref.uns["transfer_frozen"]["frozen_score_source"]))
    return normal_map, feature_order, scaler_mean, scaler_scale, pca_components, source_csv


def load_reference_ecotype_table(source_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(source_csv)
    req = ["sample_id", "PC1", "PC2", "ecotype_label"]
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise ValueError(f"Reference source CSV missing required columns: {missing}")
    return df[req].copy()


def sanitize_obs_for_h5ad(obs: pd.DataFrame) -> pd.DataFrame:
    obs = obs.copy()
    for c in obs.columns:
        if pd.api.types.is_object_dtype(obs[c]) or pd.api.types.is_string_dtype(obs[c]):
            obs[c] = obs[c].astype("string").fillna("NA").astype(str)
    return obs


def standardize_malignant_label(s: pd.Series) -> pd.Series:
    x = s.astype(str).str.strip().str.lower()

    out = pd.Series("Unknown", index=s.index, dtype="object")
    out[x.str.contains("malig", na=False)] = "Malignant"
    out[x.str.contains("normal", na=False)] = "Normal"
    out[x.str.contains("non", na=False)] = "Normal"
    out[x.isin(["0", "false", "f"])] = "Normal"
    out[x.isin(["1", "true", "t"])] = "Malignant"
    return out


def build_gene_lookup(adata: ad.AnnData) -> pd.Series:
    if "gene_name" in adata.var.columns:
        g = adata.var["gene_name"].astype(str).str.upper()
    else:
        g = pd.Index(adata.var_names.astype(str)).str.upper()

    df = pd.DataFrame({
        "gene": g.values,
        "idx": np.arange(adata.n_vars),
    })
    df = df.drop_duplicates(subset=["gene"], keep="first")
    return df.set_index("gene")["idx"]


def get_matrix_for_genes(adata: ad.AnnData, idxs: np.ndarray):
    X = adata.X[:, idxs]
    if not sparse.issparse(X):
        X = sparse.csr_matrix(X)
    else:
        X = X.tocsr()
    return X


def normalize_log1p_sparse(adata: ad.AnnData) -> ad.AnnData:
    adata = adata.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    return adata


def train_label_transfer_models(dx_anchor: ad.AnnData, qry: ad.AnnData):
    """
    Train kNN label-transfer models from the labeled DX anchor to the external query.

    Uses shared genes between:
      - the labeled DX anchor (already processed)
      - the raw all-cells query (normalized/log1p in this function)
    """
    ref_lookup = build_gene_lookup(dx_anchor)
    qry_lookup = build_gene_lookup(qry)

    common = sorted(set(ref_lookup.index) & set(qry_lookup.index))
    if len(common) < 500:
        raise ValueError(f"Too few shared genes for label transfer: {len(common)}")

    ref_idx = ref_lookup.loc[common].to_numpy(dtype=int)
    qry_idx = qry_lookup.loc[common].to_numpy(dtype=int)

    # Anchor assumed already processed/log-transformed
    X_ref = get_matrix_for_genes(dx_anchor, ref_idx)

    # Query needs normalization/log1p
    qry_proc = normalize_log1p_sparse(qry)
    X_qry = get_matrix_for_genes(qry_proc, qry_idx)

    n_comp = int(min(SVD_COMPONENTS, len(common) - 1, X_ref.shape[0] - 1))
    if n_comp < 5:
        raise ValueError(f"Too few effective dimensions for SVD: {n_comp}")

    svd = TruncatedSVD(n_components=n_comp, random_state=RANDOM_STATE)
    Z_ref = svd.fit_transform(X_ref)
    Z_qry = svd.transform(X_qry)

    models = {}
    for col in ["pred_malignant", "pred_celltype", "pred_broad", "pred_malignant_coarse"]:
        y = dx_anchor.obs[col].astype(str).to_numpy()
        clf = KNeighborsClassifier(n_neighbors=KNN_NEIGHBORS, weights="distance")
        clf.fit(Z_ref, y)
        models[col] = clf

    return Z_qry, models


def merge_dx_labels(qry: ad.AnnData, dx_anchor: ad.AnnData) -> ad.AnnData:
    q = qry.obs.copy()
    d = dx_anchor.obs[
        ["sample_id", "barcode_raw", "pred_malignant", "pred_celltype", "pred_broad", "pred_malignant_coarse"]
    ].copy()

    d["sample_id"] = d["sample_id"].astype(str)
    d["barcode_raw"] = d["barcode_raw"].astype(str)

    q["sample_id"] = q["sample_id"].astype(str)
    q["barcode_raw"] = q["barcode_raw"].astype(str)

    # Ensure a stable cell_id column exists without colliding with reset_index()
    if "cell_id" not in q.columns:
        q["cell_id"] = q.index.astype(str)
    else:
        q["cell_id"] = q["cell_id"].astype(str)

    for c in ["pred_malignant", "pred_celltype", "pred_broad", "pred_malignant_coarse"]:
        q[c] = pd.NA

    # Merge on sample_id + barcode_raw; do not reset index into an existing cell_id column
    q = q.merge(
        d,
        on=["sample_id", "barcode_raw"],
        how="left",
        suffixes=("", "_dx"),
    )

    # Use direct DX labels where available
    for c in ["pred_malignant", "pred_celltype", "pred_broad", "pred_malignant_coarse"]:
        q[c] = q[f"{c}_dx"].combine_first(q[c])
        q = q.drop(columns=[f"{c}_dx"])

    q["label_source"] = np.where(q["pred_malignant"].notna(), "direct_dx_anchor", "unlabeled")
    q = q.set_index("cell_id")

    qry.obs = q
    return qry


def transfer_labels_to_unlabeled(qry: ad.AnnData, dx_anchor: ad.AnnData) -> ad.AnnData:
    unlabeled_mask = qry.obs["pred_malignant"].isna().to_numpy()
    if unlabeled_mask.sum() == 0:
        return qry

    qry_unlab = qry[unlabeled_mask].copy()
    _, models = train_label_transfer_models(dx_anchor, qry)

    # recompute Z_qry for all query cells, then select unlabeled rows
    Z_qry, models = train_label_transfer_models(dx_anchor, qry)

    idx_unlab = np.where(unlabeled_mask)[0]
    for col, model in models.items():
        pred = model.predict(Z_qry[idx_unlab])
        qry.obs.iloc[idx_unlab, qry.obs.columns.get_loc(col)] = pred

    qry.obs.iloc[idx_unlab, qry.obs.columns.get_loc("label_source")] = "knn_transfer_from_dx"
    return qry


def finalize_predicted_labels(qry: ad.AnnData) -> ad.AnnData:
    qry.obs["pred_malignant"] = qry.obs["pred_malignant"].astype(str)
    qry.obs["pred_celltype"] = qry.obs["pred_celltype"].astype(str)
    qry.obs["pred_broad"] = qry.obs["pred_broad"].astype(str)
    qry.obs["pred_malignant_coarse"] = qry.obs["pred_malignant_coarse"].astype(str)

    qry.obs["Malignant"] = standardize_malignant_label(qry.obs["pred_malignant"])
    qry.obs["Classified_Celltype"] = qry.obs["pred_celltype"].astype(str)
    qry.obs["patient_id"] = qry.obs["patient_id"].astype(str)
    qry.obs["sample_id"] = qry.obs["sample_id"].astype(str)
    qry.obs["barcode_raw"] = qry.obs["barcode_raw"].astype(str)
    return qry


def compute_normal_broad_group_fractions_from_labeled_cells(qry: ad.AnnData, normal_map: dict[str, str], feature_order: list[str]) -> pd.DataFrame:
    obs = qry.obs.copy()
    obs["Malignant"] = obs["Malignant"].astype(str)
    obs["Classified_Celltype"] = obs["Classified_Celltype"].astype(str)

    rows = []
    for sample_id, sub in obs.groupby("sample_id", sort=False):
        nonmal = sub[sub["Malignant"] != "Malignant"].copy()
        nonmal["normal_broad_group"] = nonmal["Classified_Celltype"].map(normal_map).fillna("Unknown")

        counts = nonmal["normal_broad_group"].value_counts()
        total = counts.reindex(feature_order, fill_value=0).sum()

        rec = {"sample_id": str(sample_id)}
        if total == 0:
            for k in feature_order:
                rec[f"normal_frac__{k}"] = 0.0
        else:
            for k in feature_order:
                rec[f"normal_frac__{k}"] = float(counts.get(k, 0) / total)

        rows.append(rec)

    return pd.DataFrame(rows)


def project_pc_scores(frac_row: pd.Series, feature_order: list[str], scaler_mean: np.ndarray, scaler_scale: np.ndarray, pca_components: np.ndarray) -> tuple[float, float]:
    x = np.array([frac_row[f"normal_frac__{k}"] for k in feature_order], dtype=float)
    scale = scaler_scale.copy()
    scale[scale == 0] = 1.0
    z = (x - scaler_mean) / scale
    pcs = z @ pca_components.T
    return float(pcs[0]), float(pcs[1])


def assign_ecotype_label_from_nearest_reference(pc1: float, pc2: float, ref_scores: pd.DataFrame) -> tuple[str, str, float]:
    arr = ref_scores[["PC1", "PC2"]].to_numpy(dtype=float)
    q = np.array([pc1, pc2], dtype=float)
    d = np.sqrt(np.sum((arr - q[None, :]) ** 2, axis=1))
    idx = int(np.argmin(d))
    row = ref_scores.iloc[idx]
    return str(row["ecotype_label"]), str(row["sample_id"]), float(d[idx])


def compute_external_ecotype_scores(qry: ad.AnnData, normal_map, feature_order, scaler_mean, scaler_scale, pca_components, ref_scores) -> pd.DataFrame:
    frac_df = compute_normal_broad_group_fractions_from_labeled_cells(qry, normal_map, feature_order)

    rows = []
    sample_meta = (
        qry.obs[["sample_id", "patient_id", "clinical_timepoint_raw", "clinical_timepoint_coarse"]]
           .drop_duplicates("sample_id")
           .set_index("sample_id")
    )

    for _, row in frac_df.iterrows():
        pc1, pc2 = project_pc_scores(row, feature_order, scaler_mean, scaler_scale, pca_components)
        eco_label, eco_ref_sample, eco_dist = assign_ecotype_label_from_nearest_reference(pc1, pc2, ref_scores)

        sample_id = str(row["sample_id"])
        meta = sample_meta.loc[sample_id]

        rec = {
            "sample_id": sample_id,
            "patient_id": str(meta["patient_id"]),
            "clinical_timepoint_raw": str(meta["clinical_timepoint_raw"]),
            "clinical_timepoint_coarse": str(meta["clinical_timepoint_coarse"]),
            "PC1": pc1,
            "PC2": pc2,
            "ecotype_label": eco_label,
            "ecotype_nearest_reference_sample": eco_ref_sample,
            "ecotype_nearest_reference_distance": eco_dist,
        }
        for k in feature_order:
            rec[f"normal_frac__{k}"] = float(row[f"normal_frac__{k}"])
        rows.append(rec)

    return pd.DataFrame(rows)


def compute_malignant_sample_level_scores(qry: ad.AnnData) -> pd.DataFrame:
    obs = qry.obs.copy()
    obs["Malignant"] = obs["Malignant"].astype(str)
    obs["Classified_Celltype"] = obs["Classified_Celltype"].astype(str)

    rows = []
    sample_meta = (
        obs[["sample_id", "patient_id", "clinical_timepoint_raw", "clinical_timepoint_coarse"]]
        .drop_duplicates("sample_id")
        .set_index("sample_id")
    )

    for sample_id, sub_all in obs.groupby("sample_id", sort=False):
        sub = sub_all[sub_all["Malignant"] == "Malignant"].copy()

        meta = sample_meta.loc[sample_id]
        rec = {
            "sample_id": str(sample_id),
            "patient_id": str(meta["patient_id"]),
            "clinical_timepoint_raw": str(meta["clinical_timepoint_raw"]),
            "clinical_timepoint_coarse": str(meta["clinical_timepoint_coarse"]),
            "n_total_cells": int(sub_all.shape[0]),
            "n_malignant_cells": int(sub.shape[0]),
        }

        if sub.shape[0] == 0:
            for c in SCAFFOLD_COLS:
                rec[c] = np.nan
            rec["branch_id"] = "Unknown"
            rec["branch_maxprob"] = np.nan
            rec["branch_entropy"] = np.nan
            rows.append(rec)
            continue

        counts = sub["Classified_Celltype"].value_counts()
        total = float(counts.sum())
        frac = counts / total if total > 0 else counts * np.nan

        frac_HSC = float(frac.get("HSC", 0.0))
        frac_Prog = float(frac.get("Progenitor", 0.0))
        frac_GMP = float(frac.get("GMP", 0.0))
        frac_Mono = float(frac.get("Monocytes", 0.0))
        frac_cDC = float(frac.get("cDC", 0.0))
        frac_EarlyBaso = float(frac.get("Early.Basophil", 0.0))
        frac_EarlyEry = float(frac.get("Early.Erythrocyte", 0.0))
        frac_CLP = float(frac.get("CLP", 0.0))

        mono_dc = frac_Mono + frac_cDC
        main_denom = frac_HSC + frac_Prog + frac_GMP + mono_dc

        if main_denom <= 0:
            state_HSC = np.nan
            state_Prog = np.nan
            state_GMP = np.nan
            state_MonoDC = np.nan
        else:
            state_HSC = frac_HSC / main_denom
            state_Prog = frac_Prog / main_denom
            state_GMP = frac_GMP / main_denom
            state_MonoDC = mono_dc / main_denom

        aux_EryBaso = frac_EarlyBaso + frac_EarlyEry
        aux_CLP = frac_CLP

        state_vec = np.array([state_HSC, state_Prog, state_GMP, state_MonoDC], dtype=float)
        if np.isfinite(state_vec).all():
            branch_maxprob = float(np.nanmax(state_vec))
            branch_entropy = float(-(state_vec * np.log(np.clip(state_vec, 1e-12, None))).sum())
            branch_col = MAIN_STATE_COLS[int(np.nanargmax(state_vec))]
            branch_id = BRANCH_LABEL_MAP[branch_col]
        else:
            branch_maxprob = np.nan
            branch_entropy = np.nan
            branch_id = "Unknown"

        rec.update({
            "state_HSC": float(state_HSC) if pd.notna(state_HSC) else np.nan,
            "state_Prog": float(state_Prog) if pd.notna(state_Prog) else np.nan,
            "state_GMP": float(state_GMP) if pd.notna(state_GMP) else np.nan,
            "state_MonoDC": float(state_MonoDC) if pd.notna(state_MonoDC) else np.nan,
            "aux_EryBaso": float(aux_EryBaso),
            "aux_CLP": float(aux_CLP),
            "branch_id": branch_id,
            "branch_maxprob": branch_maxprob,
            "branch_entropy": branch_entropy,
        })
        rows.append(rec)

    return pd.DataFrame(rows)


def attach_sample_scores_to_malignant_cells(qry: ad.AnnData, sample_scores: pd.DataFrame) -> ad.AnnData:
    mal = qry[qry.obs["Malignant"].astype(str) == "Malignant"].copy()

    # Standardize key
    mal.obs["sample_id"] = mal.obs["sample_id"].astype(str)

    ss = sample_scores.copy()
    ss["sample_id"] = ss["sample_id"].astype(str)

    # Keep one row per sample_id
    ss = ss.drop_duplicates(subset=["sample_id"], keep="first")

    # Drop sample-level columns that already exist in mal.obs
    overlap = [c for c in ss.columns if c != "sample_id" and c in mal.obs.columns]
    if overlap:
        ss = ss.drop(columns=overlap)

    ss = ss.set_index("sample_id")

    # Safe join
    mal.obs = mal.obs.join(ss, on="sample_id", how="left")

    required = ["PC1", "PC2"] + SCAFFOLD_COLS + ["ecotype_label", "branch_id", "branch_maxprob", "branch_entropy"]
    miss = [c for c in required if c not in mal.obs.columns]
    if miss:
        raise ValueError(f"Missing projected sample-level columns on malignant subset: {miss}")

    # Keep only malignant cells with valid projected sample scores
    good = mal.obs["PC1"].notna() & mal.obs["PC2"].notna()
    mal = mal[good].copy()

    mal.obs["sample_has_projected_scores"] = True
    mal.obsm["X_fig2"] = mal.obs[["PC1", "PC2"]].to_numpy(dtype=float)
    mal.obsm["X_scaffold"] = mal.obs[SCAFFOLD_COLS].to_numpy(dtype=float)

    mal.uns["external_projection_note"] = (
        "DX labels directly merged from the labeled anchor; EOI/REL labels inferred by kNN transfer from the DX anchor. "
        "Frozen sample-level scores reconstructed using the same ecotype PCA and malignant-state rules as the discovery cohort."
    )
    mal.uns["external_cohort"] = "GSE235923"
    mal.uns["timepoint_order"] = ["DX", "EOI_REM", "REL"]

    return mal


# ============================================================
# 3. MAIN
# ============================================================
def main() -> None:
    ensure_dirs()

    ref = sc.read_h5ad(FIG2_REF)
    qry = sc.read_h5ad(QRY_H5AD)
    dx_anchor = sc.read_h5ad(DX_LABEL_H5AD)

    assert_query_requirements(qry)
    assert_dx_label_requirements(dx_anchor)

    # 1) Directly merge known DX labels
    qry = merge_dx_labels(qry, dx_anchor)

    # 2) Transfer labels to remaining cells
    qry = transfer_labels_to_unlabeled(qry, dx_anchor)

    # 3) Standardize labels
    qry = finalize_predicted_labels(qry)

    # 4) Load frozen reference recipe
    normal_map, feature_order, scaler_mean, scaler_scale, pca_components, source_csv = load_reference_recipe(ref)
    ref_scores = load_reference_ecotype_table(source_csv)

    # 5) Compute sample-level ecology projection and malignant-state summaries
    ecotype_scores = compute_external_ecotype_scores(
        qry, normal_map, feature_order, scaler_mean, scaler_scale, pca_components, ref_scores
    )
    malignant_scores = compute_malignant_sample_level_scores(qry)

    sample_scores = ecotype_scores.merge(
        malignant_scores,
        on=["sample_id", "patient_id", "clinical_timepoint_raw", "clinical_timepoint_coarse"],
        how="outer",
    )

    # 6) QC
    qc = (
        qry.obs.groupby(["clinical_timepoint_coarse", "label_source", "Malignant"])
           .size()
           .rename("n_cells")
           .reset_index()
    )
    qc.to_csv(OUT_QC, sep="\t", index=False)

    # 7) Save sample-level score table
    sample_scores.to_csv(OUT_SAMPLE_CSV, index=False)

    # 8) Save labeled all-cells object (optional but useful)
    qry.obs = sanitize_obs_for_h5ad(qry.obs)
    qry.write_h5ad(OUT_ALLCELLS_LABELED)

    # 9) Build malignant-only projected object
    mal = attach_sample_scores_to_malignant_cells(qry, sample_scores)
    mal.obs = sanitize_obs_for_h5ad(mal.obs)
    mal.write_h5ad(OUT_MALIGNANT_H5AD)

    print(f"[DONE] Saved projected malignant object: {OUT_MALIGNANT_H5AD}")
    print(f"[DONE] Saved sample-level score table: {OUT_SAMPLE_CSV}")
    print(f"[DONE] Saved label-transfer QC: {OUT_QC}")
    print(f"[DONE] Saved labeled all-cells object: {OUT_ALLCELLS_LABELED}")

    print("\n[SUMMARY: labeled cells by phase/source/malignancy]")
    print(qc.to_string(index=False))

    print("\n[SUMMARY: projected malignant samples by phase]")
    print(
        sample_scores.loc[sample_scores["n_malignant_cells"].fillna(0) > 0, "clinical_timepoint_coarse"]
        .value_counts(dropna=False)
        .sort_index()
    )

    print("\n[SUMMARY: malignant cells retained by phase]")
    print(
        mal.obs["clinical_timepoint_coarse"]
           .value_counts(dropna=False)
           .sort_index()
    )


if __name__ == "__main__":
    main()
