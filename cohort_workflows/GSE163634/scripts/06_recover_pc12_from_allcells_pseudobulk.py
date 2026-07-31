from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score

try:
    import anndata as ad
except Exception as e:  # pragma: no cover
    raise SystemExit(f"anndata is required: {e}")

AXES = ["pc1", "pc2"]


def read_gene_list(path: Path) -> List[str]:
    genes = [x.strip() for x in path.read_text().splitlines() if x.strip()]
    if not genes:
        raise ValueError(f"Empty gene list: {path}")
    return genes


def infer_sample_col(obs: pd.DataFrame) -> str:
    preferred = [
        "sample_id",
        "sample",
        "orig.ident",
        "orig_ident",
        "sampleID",
        "Sample",
        "donor_id",
        "patient_id",
    ]
    for col in preferred:
        if col in obs.columns:
            return col
    # Fallback: choose a string/object column with small-ish unique count.
    candidates: List[Tuple[str, int]] = []
    for col in obs.columns:
        if obs[col].dtype == object or str(obs[col].dtype).startswith("string"):
            nunique = obs[col].nunique(dropna=True)
            if 1 < nunique < max(500, len(obs) // 2):
                candidates.append((col, nunique))
    if not candidates:
        raise ValueError("Could not infer sample column from h5ad.obs")
    candidates.sort(key=lambda x: x[1])
    return candidates[0][0]


def infer_gene_symbols(adata: ad.AnnData) -> pd.Index:
    var = adata.var.copy()
    for col in ["gene_symbol", "gene_symbols", "symbol", "Gene_symbol", "feature_name"]:
        if col in var.columns:
            vals = var[col].astype(str)
            if vals.notna().any():
                return pd.Index(vals.values, name="gene_symbol")
    # Fallback to var_names.
    return pd.Index(adata.var_names.astype(str), name="gene_symbol")


def maybe_subset_malignant(adata: ad.AnnData) -> Tuple[ad.AnnData, str]:
    obs = adata.obs.copy()
    # Boolean markers first.
    for col in ["is_malignant", "malignant", "is_leukemia", "leukemia"]:
        if col in obs.columns:
            ser = obs[col]
            if ser.dtype == bool or str(ser.dtype).startswith("bool"):
                if ser.sum() > 0:
                    return adata[ser.values].copy(), f"subset by boolean {col}=True"
    # String columns.
    keys = [
        "celltype",
        "cell_type",
        "annotation",
        "cell_annotation",
        "major_celltype",
        "compartment",
        "predicted_celltype",
        "malignant_coarse",
        "lineage",
    ]
    terms = ["malignant", "blast", "leuk", "aml", "tumor"]
    for col in keys:
        if col in obs.columns:
            ser = obs[col].astype(str).str.lower()
            mask = pd.Series(False, index=ser.index)
            for term in terms:
                mask = mask | ser.str.contains(term, na=False)
            if mask.sum() > 0:
                return adata[mask.values].copy(), f"subset by {col} containing malignant-like terms"
    return adata, "used all cells (no malignant subset inferred)"


def matrix_to_dense(X) -> np.ndarray:
    if sparse.issparse(X):
        return X.toarray()
    return np.asarray(X)


def build_pseudobulk_from_h5ad(
    h5ad_path: Path,
    gene_order: Sequence[str],
    use_malignant_subset: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    adata = ad.read_h5ad(h5ad_path)
    subset_note = "used all cells"
    if use_malignant_subset:
        adata, subset_note = maybe_subset_malignant(adata)

    sample_col = infer_sample_col(adata.obs)
    sample_ids = adata.obs[sample_col].astype(str).values
    genes = infer_gene_symbols(adata)

    # Collapse duplicate gene symbols by summing columns.
    X = adata.X
    if sparse.issparse(X):
        X = X.tocsr()
    else:
        X = np.asarray(X)

    sample_to_idx: Dict[str, List[int]] = {}
    for i, sid in enumerate(sample_ids):
        sample_to_idx.setdefault(sid, []).append(i)

    # Build per-sample counts by gene symbol.
    rows = []
    out_samples = []
    unique_symbols = pd.Index(pd.Series(genes).fillna("").astype(str))
    symbol_groups = pd.Series(range(len(unique_symbols)), index=unique_symbols).groupby(level=0).apply(list)
    ordered_unique = list(dict.fromkeys(unique_symbols.tolist()))

    for sid, idxs in sample_to_idx.items():
        if sparse.issparse(X):
            summed = np.asarray(X[idxs].sum(axis=0)).ravel()
        else:
            summed = X[idxs].sum(axis=0)
        symbol_sum: Dict[str, float] = {}
        for sym, val in zip(unique_symbols, summed):
            if not sym or sym == "nan":
                continue
            symbol_sum[sym] = symbol_sum.get(sym, 0.0) + float(val)
        rows.append(symbol_sum)
        out_samples.append(sid)

    df = pd.DataFrame(rows, index=out_samples).fillna(0.0)
    # Convert to log2 CPM.
    lib = df.sum(axis=1)
    lib = lib.replace(0, np.nan)
    cpm = df.div(lib, axis=0) * 1e6
    log2cpm = np.log2(cpm.fillna(0.0) + 1.0)

    ordered = [g for g in gene_order if g in log2cpm.columns]
    mat = log2cpm.reindex(columns=ordered)
    meta = {
        "h5ad_path": str(h5ad_path),
        "sample_col": sample_col,
        "subset_note": subset_note,
        "n_samples": int(mat.shape[0]),
        "n_genes_matched": int(mat.shape[1]),
    }
    return mat, meta


def fit_ridge_loocv(X: np.ndarray, y: np.ndarray, alphas: Sequence[float]) -> Dict[str, object]:
    n = X.shape[0]
    best = None
    for alpha in alphas:
        preds = np.zeros(n, dtype=float)
        for i in range(n):
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            model = Ridge(alpha=float(alpha), fit_intercept=True)
            model.fit(X[mask], y[mask])
            preds[i] = float(model.predict(X[i : i + 1])[0])
        loo_rmse = math.sqrt(mean_squared_error(y, preds))
        if best is None or loo_rmse < best["loo_rmse"]:
            best = {"alpha": float(alpha), "loo_rmse": loo_rmse, "loo_pred": preds}
    assert best is not None
    final = Ridge(alpha=best["alpha"], fit_intercept=True)
    final.fit(X, y)
    train_pred = final.predict(X)
    return {
        "alpha": best["alpha"],
        "model": final,
        "loo_rmse": float(best["loo_rmse"]),
        "train_rmse": float(math.sqrt(mean_squared_error(y, train_pred))),
        "train_r2": float(r2_score(y, train_pred)),
    }


def calibrate_affine(pred: np.ndarray, obs: np.ndarray) -> Tuple[float, float, int]:
    mask = np.isfinite(pred) & np.isfinite(obs)
    if mask.sum() < 3:
        return 0.0, 1.0, int(mask.sum())
    x = pred[mask]
    y = obs[mask]
    A = np.vstack([np.ones_like(x), x]).T
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    intercept, slope = float(beta[0]), float(beta[1])
    return intercept, slope, int(mask.sum())


def apply_model(df: pd.DataFrame, model: Ridge, feature_order: Sequence[str]) -> np.ndarray:
    X = df.reindex(columns=feature_order).fillna(0.0).to_numpy(dtype=float)
    return model.predict(X)


def update_serial_deltas(serial_df: pd.DataFrame, score_df: pd.DataFrame) -> pd.DataFrame:
    score_idx = score_df.set_index("sample_id")
    out = serial_df.copy()
    for axis in AXES:
        raw_col = f"{axis}_raw"
        cal_col = f"{axis}_cal"
        if cal_col not in score_idx.columns:
            continue
        from_vals = score_idx.loc[out["from_sample"], cal_col].to_numpy()
        to_vals = score_idx.loc[out["to_sample"], cal_col].to_numpy()
        out[f"from_{axis}_cal"] = from_vals
        out[f"to_{axis}_cal"] = to_vals
        out[f"delta_{axis}_cal"] = to_vals - from_vals
        from_raw = score_idx.loc[out["from_sample"], raw_col].to_numpy()
        to_raw = score_idx.loc[out["to_sample"], raw_col].to_numpy()
        out[f"from_{axis}_raw"] = from_raw
        out[f"to_{axis}_raw"] = to_raw
        out[f"delta_{axis}_raw"] = to_raw - from_raw
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--gse235063-root", required=True)
    p.add_argument("--gse235923-root", required=True)
    p.add_argument("--gse163634-root", required=True)
    p.add_argument("--gse235063-allcells-h5ad", required=True)
    p.add_argument("--gse235923-allcells-h5ad", required=True)
    p.add_argument("--gse163634-matrix", default=None)
    p.add_argument("--existing-score-matrix", default=None)
    p.add_argument("--existing-serial-deltas", default=None)
    p.add_argument("--outdir", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    g63 = Path(args.gse235063_root)
    g923 = Path(args.gse235923_root)
    g163 = Path(args.gse163634_root)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    gene_order_path = g63 / "derived_dx_primary_training" / "frozen_gse235063_model" / "train_gene_order.txt"
    target_path = g63 / "derived_dx_primary_training" / "frozen_gse235063_model" / "ecotype_pca_scores_fit_samples.csv"
    reference_path = g923 / "derived_secondary_calibration" / "gse235923_dx_projected_ecotype_pcs.csv"
    primary_ref_path = g923 / "derived_secondary_calibration" / "primary_ecotype_pca_reference_scores.csv"
    g163_matrix_path = Path(args.gse163634_matrix) if args.gse163634_matrix else (g163 / "derived_bulk_start" / "gse163634_log2fpkm_frozen_intersection_samples_by_genes.tsv.gz")
    existing_score_matrix = Path(args.existing_score_matrix) if args.existing_score_matrix else (g163 / "derived_transfer_projection" / "gse163634_bulk_score_matrix.csv")
    existing_serial_deltas = Path(args.existing_serial_deltas) if args.existing_serial_deltas else (g163 / "derived_transfer_projection" / "gse163634_bulk_serial_deltas.csv")

    gene_order = read_gene_list(gene_order_path)
    targets = pd.read_csv(target_path)
    targets = targets.rename(columns={"PC1_fit": "pc1", "PC2_fit": "pc2"})
    targets = targets[["sample_id", "pc1", "pc2"]].set_index("sample_id")

    g63_expr, meta63 = build_pseudobulk_from_h5ad(Path(args.gse235063_allcells_h5ad), gene_order, use_malignant_subset=False)
    shared_train_samples = [s for s in targets.index if s in g63_expr.index]
    if len(shared_train_samples) < 8:
        raise ValueError(f"Too few shared GSE235063 samples between targets and all-cells pseudobulk: {len(shared_train_samples)}")
    g63_expr = g63_expr.loc[shared_train_samples]
    targets = targets.loc[shared_train_samples]

    g923_expr, meta923 = build_pseudobulk_from_h5ad(Path(args.gse235923_allcells_h5ad), gene_order, use_malignant_subset=False)
    calib_ref = pd.read_csv(reference_path).rename(columns={"PC1": "pc1", "PC2": "pc2"}).set_index("sample_id")
    shared_calib_samples = [s for s in calib_ref.index if s in g923_expr.index]
    g923_expr = g923_expr.loc[shared_calib_samples]
    calib_ref = calib_ref.loc[shared_calib_samples]

    g163_expr = pd.read_csv(g163_matrix_path, sep=None, engine="python", compression="infer", index_col=0)
    # Intersect features across all matrices.
    common_genes = [g for g in g163_expr.columns if g in g63_expr.columns and g in g923_expr.columns]
    if len(common_genes) < 1000:
        raise ValueError(f"Too few common genes across GSE235063/GSE235923/GSE163634 for pc1/pc2: {len(common_genes)}")
    g63_use = g63_expr[common_genes]
    g923_use = g923_expr[common_genes]
    g163_use = g163_expr[common_genes]

    alphas = np.logspace(-2, 4, 25)
    model_rows = []
    coef_rows = []
    calibration_rows = []
    g163_scores = pd.DataFrame(index=g163_use.index)
    g923_pred_table = pd.DataFrame(index=g923_use.index)

    for axis in AXES:
        y = targets[axis].to_numpy(dtype=float)
        X = g63_use.to_numpy(dtype=float)
        fit = fit_ridge_loocv(X, y, alphas)
        model: Ridge = fit["model"]
        model_rows.append(
            {
                "axis": axis,
                "alpha": fit["alpha"],
                "intercept": float(model.intercept_),
                "n_samples": int(g63_use.shape[0]),
                "n_genes": int(g63_use.shape[1]),
                "train_rmse": fit["train_rmse"],
                "train_r2": fit["train_r2"],
                "loo_rmse": fit["loo_rmse"],
            }
        )
        coef_rows.extend(
            {"axis": axis, "gene": g, "coefficient": float(c)}
            for g, c in zip(common_genes, model.coef_)
        )

        pred923 = model.predict(g923_use.to_numpy(dtype=float))
        obs923 = calib_ref[axis].to_numpy(dtype=float) if axis in calib_ref.columns else np.full_like(pred923, np.nan)
        intercept, slope, nfit = calibrate_affine(pred923, obs923)
        calibration_rows.append(
            {
                "axis": axis,
                "intercept": intercept,
                "slope": slope,
                "n_samples": nfit,
                "source_table": str(reference_path),
            }
        )
        g923_pred_table[f"{axis}_pred"] = pred923
        g923_pred_table[f"{axis}_obs"] = obs923
        g923_pred_table[f"{axis}_cal"] = intercept + slope * pred923

        pred163 = model.predict(g163_use.to_numpy(dtype=float))
        g163_scores[f"{axis}_raw"] = pred163
        g163_scores[f"{axis}_cal"] = intercept + slope * pred163

    # Merge with existing score matrix.
    merged_scores = None
    if existing_score_matrix.exists():
        base_scores = pd.read_csv(existing_score_matrix)
        merged_scores = base_scores.merge(g163_scores.reset_index().rename(columns={"index": "sample_id"}), on="sample_id", how="left")
    else:
        merged_scores = g163_scores.reset_index().rename(columns={"index": "sample_id"})

    merged_deltas = None
    if existing_serial_deltas.exists() and merged_scores is not None:
        serial_df = pd.read_csv(existing_serial_deltas)
        merged_deltas = update_serial_deltas(serial_df, merged_scores)

    # QC compare GSE235063 fit vs reference if available.
    qc_rows = []
    if primary_ref_path.exists():
        primary_ref = pd.read_csv(primary_ref_path).rename(columns={"PC1": "pc1_ref", "PC2": "pc2_ref"}).set_index("sample_id")
        overlap = [s for s in targets.index if s in primary_ref.index]
        for axis in AXES:
            ref_col = f"{axis}_ref"
            if overlap and ref_col in primary_ref.columns:
                corr = float(pd.Series(targets.loc[overlap, axis]).corr(pd.Series(primary_ref.loc[overlap, ref_col]), method="pearson"))
                qc_rows.append({"axis": axis, "n_overlap": len(overlap), "pearson_fit_vs_reference": corr})

    model_summary_path = outdir / "gse235063_pc12_model_summary.csv"
    coef_path = outdir / "gse235063_pc12_coefficients_long.csv"
    calib_path = outdir / "gse235923_pc12_calibration.csv"
    pred923_path = outdir / "gse235923_pc12_pred_vs_obs.csv"
    pc12_score_path = outdir / "gse163634_pc12_score_matrix.csv"
    merged_score_path = outdir / "gse163634_bulk_score_matrix_with_pc12.csv"
    merged_delta_path = outdir / "gse163634_bulk_serial_deltas_with_pc12.csv"
    qc_path = outdir / "gse235063_pc12_fit_vs_reference_qc.csv"
    manifest_path = outdir / "gse163634_pc12_recovery_manifest.json"

    pd.DataFrame(model_rows).to_csv(model_summary_path, index=False)
    pd.DataFrame(coef_rows).to_csv(coef_path, index=False)
    pd.DataFrame(calibration_rows).to_csv(calib_path, index=False)
    g923_pred_table.reset_index().rename(columns={"index": "sample_id"}).to_csv(pred923_path, index=False)
    g163_scores.reset_index().rename(columns={"index": "sample_id"}).to_csv(pc12_score_path, index=False)
    if merged_scores is not None:
        merged_scores.to_csv(merged_score_path, index=False)
    if merged_deltas is not None:
        merged_deltas.to_csv(merged_delta_path, index=False)
    if qc_rows:
        pd.DataFrame(qc_rows).to_csv(qc_path, index=False)

    manifest = {
        "gene_order_file": str(gene_order_path),
        "training_targets": str(target_path),
        "training_expression_h5ad": str(args.gse235063_allcells_h5ad),
        "training_pseudobulk_meta": meta63,
        "calibration_expression_h5ad": str(args.gse235923_allcells_h5ad),
        "calibration_pseudobulk_meta": meta923,
        "calibration_projected_pc_file": str(reference_path),
        "gse163634_matrix": str(g163_matrix_path),
        "n_training_samples_used": int(g63_use.shape[0]),
        "n_calibration_samples_used": int(g923_use.shape[0]),
        "n_gse163634_samples_scored": int(g163_use.shape[0]),
        "n_shared_genes": int(len(common_genes)),
        "axes_modeled": AXES,
        "outputs": {
            "model_summary": str(model_summary_path),
            "coefficients_long": str(coef_path),
            "calibration": str(calib_path),
            "calibration_pred_vs_obs": str(pred923_path),
            "pc12_score_matrix": str(pc12_score_path),
            "merged_score_matrix": str(merged_score_path) if merged_scores is not None else None,
            "merged_serial_deltas": str(merged_delta_path) if merged_deltas is not None else None,
            "fit_vs_reference_qc": str(qc_path) if qc_rows else None,
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print("[OK] pc1/pc2 recovery complete.")
    print(f"Output: {outdir}")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
