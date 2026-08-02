from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import scanpy as sc


# ============================================================
# 1. CONFIG
# ============================================================
REF_H5AD = Path("/Multimodal_OU_Lévy_Branching_Scaffold/Figure_2/dx_primary_training_malignant_frozen_transfer.h5ad")
QRY_H5AD = Path("/Multimodal_OU_Lévy_Branching_Scaffold/Figure_3/derived/gse235063_longitudinal_malignant_raw.h5ad")

RAW_DIR = Path("/Ecotype_OU_Branching/GSE235063/GSE235063_RAW")
MANIFEST_CSV = RAW_DIR / "longitudinal_cohort_manifest_processed.csv"

PROJECT_DIR = Path("/Multimodal_OU_Lévy_Branching_Scaffold/Figure_3")
INPUTS_DIR = PROJECT_DIR / "inputs"
DERIVED_DIR = PROJECT_DIR / "derived"

OUT_H5AD = INPUTS_DIR / "gse235063_longitudinal_malignant_projected.h5ad"
OUT_SAMPLE_CSV = DERIVED_DIR / "longitudinal_frozen_sample_scores.csv"

MAIN_STATE_COLS = ["state_HSC", "state_Prog", "state_GMP", "state_MonoDC"]
SCAFFOLD_COLS = ["state_HSC", "state_Prog", "state_GMP", "state_MonoDC", "aux_EryBaso", "aux_CLP"]

BRANCH_LABEL_MAP = {
    "state_HSC": "HSC-like basin",
    "state_Prog": "Progenitor-like basin",
    "state_GMP": "GMP-like basin",
    "state_MonoDC": "Mono/DC-like basin",
}


# ============================================================
# 2. HELPERS
# ============================================================
def ensure_dirs() -> None:
    INPUTS_DIR.mkdir(parents=True, exist_ok=True)
    DERIVED_DIR.mkdir(parents=True, exist_ok=True)


def assert_required_query_obs(qry) -> None:
    req = ["sample_id", "patient_id", "clinical_timepoint_raw", "clinical_timepoint_coarse", "Classified_Celltype"]
    missing = [c for c in req if c not in qry.obs.columns]
    if missing:
        raise ValueError(f"Query object missing required obs columns: {missing}")


def load_reference_recipe(ref):
    if "normal_broad_grouping" not in ref.uns:
        raise ValueError("Reference object missing ref.uns['normal_broad_grouping']")
    if "ecotype_pca" not in ref.uns:
        raise ValueError("Reference object missing ref.uns['ecotype_pca']")
    if "transfer_frozen" not in ref.uns:
        raise ValueError("Reference object missing ref.uns['transfer_frozen']")

    normal_map = dict(ref.uns["normal_broad_grouping"])
    pca_recipe = ref.uns["ecotype_pca"]

    feature_order = [str(x) for x in pca_recipe["feature_order"].tolist()]
    scaler_mean = np.asarray(pca_recipe["scaler_mean"], dtype=float)
    scaler_scale = np.asarray(pca_recipe["scaler_scale"], dtype=float)
    pca_components = np.asarray(pca_recipe["pca_components"], dtype=float)

    source_csv = Path(str(ref.uns["transfer_frozen"]["frozen_score_source"]))

    return normal_map, feature_order, scaler_mean, scaler_scale, pca_components, source_csv


def read_metadata(fp: Path) -> pd.DataFrame:
    return pd.read_csv(fp, sep="\t", compression="gzip")


def compute_normal_broad_group_fractions(meta: pd.DataFrame, normal_map: dict[str, str], feature_order: list[str]) -> dict[str, float]:
    if "Malignant" not in meta.columns or "Classified_Celltype" not in meta.columns:
        raise ValueError("Metadata must contain 'Malignant' and 'Classified_Celltype' columns.")

    sub = meta.copy()
    sub["Malignant"] = sub["Malignant"].astype(str)
    sub["Classified_Celltype"] = sub["Classified_Celltype"].astype(str)

    # Non-malignant cells only
    sub = sub[sub["Malignant"] != "Malignant"].copy()

    # Map fine labels to broad normal groups
    sub["normal_broad_group"] = sub["Classified_Celltype"].map(normal_map).fillna("Unknown")

    counts = sub["normal_broad_group"].value_counts()
    total = counts.reindex(feature_order, fill_value=0).sum()

    if total == 0:
        # no normal cells available -> zero vector
        frac = pd.Series(0.0, index=feature_order)
    else:
        frac = counts.reindex(feature_order, fill_value=0).astype(float) / float(total)

    return {f"normal_frac__{k}": float(frac[k]) for k in feature_order}


def project_pc_scores(frac_row: pd.Series, feature_order: list[str], scaler_mean: np.ndarray, scaler_scale: np.ndarray, pca_components: np.ndarray) -> tuple[float, float]:
    x = np.array([frac_row[f"normal_frac__{k}"] for k in feature_order], dtype=float)

    scale = scaler_scale.copy()
    scale[scale == 0] = 1.0
    z = (x - scaler_mean) / scale

    pcs = z @ pca_components.T
    return float(pcs[0]), float(pcs[1])


def load_reference_ecotype_table(source_csv: Path) -> pd.DataFrame:
    if not source_csv.exists():
        raise FileNotFoundError(source_csv)

    df = pd.read_csv(source_csv)
    req = ["sample_id", "PC1", "PC2", "ecotype_label"]
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise ValueError(f"Reference source CSV missing required columns: {missing}")

    return df[req].copy()


def assign_ecotype_label_from_nearest_reference(pc1: float, pc2: float, ref_scores: pd.DataFrame) -> tuple[str, str, float]:
    arr = ref_scores[["PC1", "PC2"]].to_numpy(dtype=float)
    q = np.array([pc1, pc2], dtype=float)
    d = np.sqrt(np.sum((arr - q[None, :]) ** 2, axis=1))
    idx = int(np.argmin(d))

    row = ref_scores.iloc[idx]
    return str(row["ecotype_label"]), str(row["sample_id"]), float(d[idx])


def compute_sample_level_ecotype_scores(manifest: pd.DataFrame, qry_sample_ids: set[str], normal_map, feature_order, scaler_mean, scaler_scale, pca_components, ref_scores) -> pd.DataFrame:
    subman = manifest[manifest["sample_id"].astype(str).isin(qry_sample_ids)].copy()
    rows = []

    for _, row in subman.iterrows():
        meta_fp = RAW_DIR / str(row["metadata_file"])
        meta = read_metadata(meta_fp)

        frac_dict = compute_normal_broad_group_fractions(meta, normal_map, feature_order)
        frac_series = pd.Series(frac_dict)

        pc1, pc2 = project_pc_scores(frac_series, feature_order, scaler_mean, scaler_scale, pca_components)
        eco_label, eco_ref_sample, eco_dist = assign_ecotype_label_from_nearest_reference(pc1, pc2, ref_scores)

        rec = {
            "sample_id": str(row["sample_id"]),
            "patient_id_manifest": str(row["patient_id"]),
            "clinical_timepoint_raw_manifest": str(row["clinical_timepoint_raw"]),
            "clinical_timepoint_coarse_manifest": str(row["clinical_timepoint_coarse"]),
            "PC1": pc1,
            "PC2": pc2,
            "ecotype_label": eco_label,
            "ecotype_nearest_reference_sample": eco_ref_sample,
            "ecotype_nearest_reference_distance": eco_dist,
        }
        rec.update(frac_dict)
        rows.append(rec)

    out = pd.DataFrame(rows)
    return out


def compute_malignant_sample_level_scores(qry) -> pd.DataFrame:
    obs = qry.obs.copy()
    obs["Classified_Celltype"] = obs["Classified_Celltype"].astype(str)

    rows = []
    for sample_id, sub in obs.groupby("sample_id", sort=False):
        counts = sub["Classified_Celltype"].value_counts()
        total = float(counts.sum())

        frac = counts / total if total > 0 else counts * np.nan

        # Exact reconstructed mappings
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

        rec = {
            "sample_id": str(sample_id),
            "n_malignant_cells": int(sub.shape[0]),
            "state_HSC": float(state_HSC) if pd.notna(state_HSC) else np.nan,
            "state_Prog": float(state_Prog) if pd.notna(state_Prog) else np.nan,
            "state_GMP": float(state_GMP) if pd.notna(state_GMP) else np.nan,
            "state_MonoDC": float(state_MonoDC) if pd.notna(state_MonoDC) else np.nan,
            "aux_EryBaso": float(aux_EryBaso),
            "aux_CLP": float(aux_CLP),
            "branch_id": branch_id,
            "branch_maxprob": branch_maxprob,
            "branch_entropy": branch_entropy,
        }
        rows.append(rec)

    return pd.DataFrame(rows)


def attach_sample_scores_to_cells(qry, sample_scores: pd.DataFrame):
    sample_scores = sample_scores.set_index("sample_id")
    qry.obs = qry.obs.join(sample_scores, on="sample_id")

    required = ["PC1", "PC2"] + SCAFFOLD_COLS + ["ecotype_label", "branch_id", "branch_maxprob", "branch_entropy"]
    miss = [c for c in required if c not in qry.obs.columns]
    if miss:
        raise ValueError(f"Missing required merged projected columns: {miss}")

    for c in required:
        if qry.obs[c].isna().any():
            bad = qry.obs.loc[qry.obs[c].isna(), "sample_id"].astype(str).unique().tolist()
            raise ValueError(f"Projected column {c} contains missing values for sample_ids: {bad}")

    qry.obs["sample_has_projected_scores"] = True
    qry.obsm["X_fig2"] = qry.obs[["PC1", "PC2"]].to_numpy(dtype=float)
    qry.obsm["X_scaffold"] = qry.obs[SCAFFOLD_COLS].to_numpy(dtype=float)

    qry.uns["figure3_projection_note"] = (
        "PC1/PC2 reconstructed from frozen normal-cell broad-group PCA; "
        "malignant coarse-state summaries reconstructed from malignant Classified_Celltype fractions."
    )
    qry.uns["figure3_scaffold_feature_order"] = SCAFFOLD_COLS
    return qry


# ============================================================
# 3. MAIN
# ============================================================
def main() -> None:
    ensure_dirs()

    ref = sc.read_h5ad(REF_H5AD)
    qry = sc.read_h5ad(QRY_H5AD)
    manifest = pd.read_csv(MANIFEST_CSV)

    assert_required_query_obs(qry)

    normal_map, feature_order, scaler_mean, scaler_scale, pca_components, source_csv = load_reference_recipe(ref)
    ref_scores = load_reference_ecotype_table(source_csv)

    qry_sample_ids = set(qry.obs["sample_id"].astype(str).unique())

    # Sample-level ecology projection from original metadata files
    ecotype_scores = compute_sample_level_ecotype_scores(
        manifest=manifest,
        qry_sample_ids=qry_sample_ids,
        normal_map=normal_map,
        feature_order=feature_order,
        scaler_mean=scaler_mean,
        scaler_scale=scaler_scale,
        pca_components=pca_components,
        ref_scores=ref_scores,
    )

    # Sample-level malignant-state summaries from the malignant-only query object
    malignant_scores = compute_malignant_sample_level_scores(qry)

    sample_scores = ecotype_scores.merge(malignant_scores, on="sample_id", how="inner")

    # Save sample-level table first
    sample_scores.to_csv(OUT_SAMPLE_CSV, index=False)

    # Attach to every malignant cell
    qry = attach_sample_scores_to_cells(qry, sample_scores)

    qry.write_h5ad(OUT_H5AD)

    print(f"[DONE] Saved projected query object: {OUT_H5AD}")
    print(f"[DONE] Saved sample-level score table: {OUT_SAMPLE_CSV}")
    print("\n[SUMMARY] samples with projected scores by timepoint")
    print(
        qry.obs.groupby("clinical_timepoint_coarse")["sample_id"]
           .nunique()
           .sort_index()
    )


if __name__ == "__main__":
    main()
