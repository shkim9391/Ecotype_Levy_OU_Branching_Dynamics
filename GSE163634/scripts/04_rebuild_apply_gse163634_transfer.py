from __future__ import annotations

import argparse
import gzip
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneOut

AXES = [
    "pc1",
    "pc2",
    "ilr_stem_vs_committed",
    "ilr_prog_vs_mature",
    "ilr_gmp_vs_monodc",
    "log_aux_clp",
    "log_aux_erybaso",
]

TAB_EXTS = {".csv", ".tsv", ".txt", ".gz", ".csv.gz", ".tsv.gz", ".txt.gz"}
PREFERRED_TARGET_FILES = [
    "training_sample_summary_with_transfer_columns.csv",
    "dx_primary_training_sample_level_summary_frozen_transfer.csv",
    "frozen_sample_transfer_scores.csv",
]
PREFERRED_OBS_CALIB_FILES = [
    "gse235923_dx_secondary_calibration_table.csv",
    "gse235923_dx_secondary_outcomes.csv",
]
PREFERRED_GSE163634_MATRIX_FILES = [
    "gse163634_log2fpkm_frozen_intersection_samples_by_genes.tsv.gz",
    "gse163634_log2fpkm_frozen_intersection_samples_by_genes.tsv",
]


def read_gene_list(path: Path) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        genes = [line.strip() for line in f if line.strip()]
    return genes


def all_files(root: Path) -> List[Path]:
    return [p for p in root.rglob("*") if p.is_file()]


def is_tabular(path: Path) -> bool:
    name = path.name.lower()
    return name.endswith((".csv", ".tsv", ".txt", ".csv.gz", ".tsv.gz", ".txt.gz"))


def infer_sep(path: Path) -> str:
    name = path.name.lower()
    if name.endswith(".csv") or name.endswith(".csv.gz"):
        return ","
    return "\t"


def safe_read_table(path: Path, nrows: Optional[int] = None) -> pd.DataFrame:
    sep = infer_sep(path)
    return pd.read_csv(path, sep=sep, nrows=nrows, low_memory=False)


def looks_like_sample_id(value: object) -> bool:
    if not isinstance(value, str):
        return False
    v = value.strip()
    if not v:
        return False
    return (
        v.startswith("AML")
        or v.startswith("ALL_")
        or v.endswith("_B")
        or v.endswith("_T")
        or v.lower().startswith("sample")
    )


def detect_sample_id_column(df: pd.DataFrame) -> Optional[str]:
    preferred = ["sample_id", "sample", "sampleid", "sample_name", "Sample", "Sample_ID"]
    lower_map = {c.lower(): c for c in df.columns}
    for p in preferred:
        if p.lower() in lower_map:
            return lower_map[p.lower()]
    # fallback to first column if it looks like sample IDs
    if df.shape[1] > 0:
        c0 = df.columns[0]
        vals = df[c0].dropna().astype(str).head(20).tolist()
        if vals and sum(looks_like_sample_id(v) for v in vals) >= max(1, len(vals) // 3):
            return c0
    return None


def find_target_table(gse235063_root: Path) -> Path:
    files = all_files(gse235063_root)
    name_map = {p.name: p for p in files}
    for fname in PREFERRED_TARGET_FILES:
        if fname in name_map:
            return name_map[fname]

    scored: List[Tuple[int, Path]] = []
    for path in files:
        if not is_tabular(path):
            continue
        try:
            df = safe_read_table(path, nrows=20)
        except Exception:
            continue
        cols = {str(c) for c in df.columns}
        axis_hits = sum(1 for a in AXES if a in cols)
        if axis_hits:
            score = axis_hits * 10
            if "transfer" in path.name.lower():
                score += 5
            if "sample" in path.name.lower():
                score += 3
            scored.append((score, path))
    if not scored:
        raise FileNotFoundError("Could not find a GSE235063 target table with transfer axis columns.")
    scored.sort(key=lambda x: (-x[0], str(x[1])))
    return scored[0][1]


def preview_table_orientation(path: Path, gene_set: set[str], sample_set: set[str]) -> Optional[Dict[str, object]]:
    try:
        df = safe_read_table(path, nrows=5)
    except Exception:
        return None
    if df.empty or df.shape[1] < 2:
        return None

    col_hits = sum(1 for c in df.columns if str(c) in gene_set)
    col_sample_hits = sum(1 for c in df.columns if str(c) in sample_set)
    first_col = df.columns[0]
    first_vals = set(df[first_col].dropna().astype(str).head(100).tolist())
    row_gene_hits = len(first_vals & gene_set)
    row_sample_hits = len(first_vals & sample_set)

    name_score = 0
    low = path.name.lower()
    for k in ["expr", "expression", "matrix", "pseudobulk", "sample", "frozen", "training", "log2"]:
        if k in low:
            name_score += 1

    # sample-by-gene candidate
    score_sbg = col_hits + row_sample_hits + name_score
    # gene-by-sample candidate
    score_gbs = row_gene_hits + col_sample_hits + name_score

    orientation = None
    score = 0
    if score_sbg >= 100 and row_sample_hits >= 1:
        orientation = "sample_by_gene"
        score = score_sbg
    elif score_gbs >= 100 and col_sample_hits >= 1:
        orientation = "gene_by_sample"
        score = score_gbs
    else:
        # looser fallback
        if col_hits >= 500 and row_sample_hits >= 1:
            orientation = "sample_by_gene"
            score = score_sbg
        elif row_gene_hits >= 500 and col_sample_hits >= 1:
            orientation = "gene_by_sample"
            score = score_gbs

    if orientation is None:
        return None
    return {
        "path": path,
        "orientation": orientation,
        "score": int(score),
        "col_gene_hits": int(col_hits),
        "row_gene_hits": int(row_gene_hits),
        "col_sample_hits": int(col_sample_hits),
        "row_sample_hits": int(row_sample_hits),
    }


def find_training_expression_table(gse235063_root: Path, gene_order: List[str], sample_ids: List[str]) -> Tuple[Path, str]:
    gene_set = set(gene_order)
    sample_set = set(sample_ids)
    candidates: List[Dict[str, object]] = []
    for path in all_files(gse235063_root):
        if not is_tabular(path):
            continue
        result = preview_table_orientation(path, gene_set, sample_set)
        if result is not None:
            candidates.append(result)
    if not candidates:
        raise FileNotFoundError(
            "Could not auto-detect a GSE235063 training expression table. "
            "Please pass one explicitly after inspection."
        )
    candidates.sort(key=lambda x: (-int(x["score"]), str(x["path"])))
    best = candidates[0]
    return best["path"], best["orientation"]


def load_expression_table(path: Path, orientation: str) -> pd.DataFrame:
    df = safe_read_table(path, nrows=None)
    if orientation == "sample_by_gene":
        sample_col = detect_sample_id_column(df)
        if sample_col is None:
            # treat first column as sample ID if unnamed
            sample_col = df.columns[0]
        df = df.copy()
        df[sample_col] = df[sample_col].astype(str)
        X = df.set_index(sample_col)
        if "Unnamed: 0" in X.columns:
            X = X.drop(columns=["Unnamed: 0"])
        X.columns = X.columns.astype(str)
        X = X.apply(pd.to_numeric, errors="coerce")
        return X

    # gene_by_sample
    first_col = df.columns[0]
    gene_col = first_col
    G = df.copy()
    G[gene_col] = G[gene_col].astype(str)
    G = G.set_index(gene_col)
    if "Unnamed: 0" in G.columns:
        G = G.drop(columns=["Unnamed: 0"])
    G.columns = G.columns.astype(str)
    G = G.apply(pd.to_numeric, errors="coerce")
    return G.T


def normalize_sample_ids(index_like: List[str]) -> List[str]:
    return [str(x).strip() for x in index_like]


def subset_training_data(X: pd.DataFrame, targets: pd.DataFrame, axes_present: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    X = X.copy()
    X.index = normalize_sample_ids(list(X.index))
    targets = targets.copy()
    targets.index = normalize_sample_ids(list(targets.index))

    common = sorted(set(X.index) & set(targets.index))
    if not common:
        raise ValueError("No overlapping sample IDs between training expression and target table.")
    X2 = X.loc[common]
    y2 = targets.loc[common, axes_present]
    mask = ~y2.isna().any(axis=1)
    X2 = X2.loc[mask]
    y2 = y2.loc[mask]
    return X2, y2


def find_calibration_observed_table(gse235923_root: Path) -> Path:
    files = all_files(gse235923_root)
    name_map = {p.name: p for p in files}
    for fname in PREFERRED_OBS_CALIB_FILES:
        if fname in name_map:
            return name_map[fname]
    raise FileNotFoundError("Could not find a GSE235923 calibration table.")


def collect_gse235923_axis_tables(gse235923_root: Path) -> List[Tuple[Path, pd.DataFrame]]:
    tables = []
    for path in all_files(gse235923_root):
        if not is_tabular(path):
            continue
        try:
            df = safe_read_table(path, nrows=50)
        except Exception:
            continue
        tables.append((path, df))
    return tables


def build_column_lookup(cols: List[str], axis: str) -> List[str]:
    axis_low = axis.lower()
    return [c for c in cols if axis_low in c.lower()]


def choose_pred_obs_columns(cols: List[str], axis: str) -> Tuple[Optional[str], Optional[str]]:
    matches = build_column_lookup(cols, axis)
    if not matches:
        return None, None
    pred_pref = [
        "pred", "predicted", "project", "projected", "raw", "transfer", "frozen"
    ]
    obs_pref = [
        "obs", "observed", "actual", "target", "secondary", "outcome"
    ]

    pred_col = None
    obs_col = None
    exact_col = axis if axis in cols else None

    for c in matches:
        low = c.lower()
        if any(k in low for k in pred_pref) and pred_col is None:
            pred_col = c
        if any(k in low for k in obs_pref) and obs_col is None:
            obs_col = c

    if obs_col is None and exact_col is not None:
        obs_col = exact_col
    if pred_col is None:
        remaining = [c for c in matches if c != obs_col]
        if remaining:
            pred_col = remaining[0]
    return pred_col, obs_col


def derive_calibration_from_table(path: Path) -> Dict[str, Dict[str, float]]:
    df = safe_read_table(path)
    sample_col = detect_sample_id_column(df)
    if sample_col is not None:
        df = df.set_index(sample_col)
    out: Dict[str, Dict[str, float]] = {}
    cols = [str(c) for c in df.columns]
    for axis in AXES:
        pred_col, obs_col = choose_pred_obs_columns(cols, axis)
        if pred_col is None or obs_col is None or pred_col == obs_col:
            continue
        sub = df[[pred_col, obs_col]].apply(pd.to_numeric, errors="coerce").dropna()
        if sub.shape[0] < 3:
            continue
        lr = LinearRegression().fit(sub[[pred_col]], sub[obs_col])
        out[axis] = {
            "pred_col": pred_col,
            "obs_col": obs_col,
            "intercept": float(lr.intercept_),
            "slope": float(lr.coef_[0]),
            "n_samples": int(sub.shape[0]),
            "source_table": str(path),
        }
    return out


def derive_calibration_from_multiple_tables(gse235923_root: Path, observed_table: Path) -> Dict[str, Dict[str, float]]:
    observed = safe_read_table(observed_table)
    obs_sample_col = detect_sample_id_column(observed)
    if obs_sample_col is None:
        return {}
    observed = observed.set_index(obs_sample_col)
    observed.index = normalize_sample_ids(list(observed.index))
    obs_cols = [str(c) for c in observed.columns]

    results: Dict[str, Dict[str, float]] = {}
    for path, df_small in collect_gse235923_axis_tables(gse235923_root):
        try:
            df = safe_read_table(path)
        except Exception:
            continue
        sample_col = detect_sample_id_column(df)
        if sample_col is None:
            continue
        df = df.set_index(sample_col)
        df.index = normalize_sample_ids(list(df.index))
        shared_samples = sorted(set(df.index) & set(observed.index))
        if len(shared_samples) < 3:
            continue
        for axis in AXES:
            if axis in results:
                continue
            # observed axis must exist in observed table
            if axis not in obs_cols:
                continue
            pred_matches = build_column_lookup(list(df.columns.astype(str)), axis)
            if not pred_matches:
                continue
            # prefer predicted-looking columns
            pred_col = None
            for c in pred_matches:
                low = c.lower()
                if any(k in low for k in ["pred", "project", "raw", "transfer", "frozen"]):
                    pred_col = c
                    break
            if pred_col is None:
                # if exact axis only and file name looks predicted, use it
                if axis in map(str, df.columns) and any(k in path.name.lower() for k in ["pred", "project", "transfer"]):
                    pred_col = axis
                elif len(pred_matches) == 1:
                    pred_col = pred_matches[0]
            if pred_col is None:
                continue
            sub = pd.DataFrame({
                "pred": pd.to_numeric(df.loc[shared_samples, pred_col], errors="coerce"),
                "obs": pd.to_numeric(observed.loc[shared_samples, axis], errors="coerce"),
            }).dropna()
            if sub.shape[0] < 3:
                continue
            lr = LinearRegression().fit(sub[["pred"]], sub["obs"])
            results[axis] = {
                "pred_col": pred_col,
                "obs_col": axis,
                "intercept": float(lr.intercept_),
                "slope": float(lr.coef_[0]),
                "n_samples": int(sub.shape[0]),
                "source_table": str(path),
                "observed_table": str(observed_table),
            }
    return results


def fit_ridge_models(X_train: pd.DataFrame, y_train: pd.DataFrame, genes_for_transfer: List[str]) -> Tuple[Dict[str, Dict[str, object]], pd.DataFrame]:
    alphas = np.logspace(-3, 3, 25)
    X = X_train.loc[:, genes_for_transfer].copy()
    X = X.fillna(0.0)

    models: Dict[str, Dict[str, object]] = {}
    coeff_rows = []
    for axis in y_train.columns:
        y = pd.to_numeric(y_train[axis], errors="coerce")
        valid = ~y.isna()
        Xv = X.loc[valid]
        yv = y.loc[valid]
        if Xv.shape[0] < 4:
            continue
        model = RidgeCV(alphas=alphas, cv=LeaveOneOut()).fit(Xv, yv)
        fitted = model.predict(Xv)
        rmse = float(np.sqrt(mean_squared_error(yv, fitted)))
        r2 = float(r2_score(yv, fitted)) if Xv.shape[0] > 1 else float("nan")
        models[axis] = {
            "alpha": float(model.alpha_),
            "intercept": float(model.intercept_),
            "n_samples": int(Xv.shape[0]),
            "n_genes": int(Xv.shape[1]),
            "rmse_train": rmse,
            "r2_train": r2,
            "genes": genes_for_transfer,
            "coef": model.coef_.astype(float).tolist(),
        }
        coeff_rows.append(
            pd.DataFrame({
                "axis": axis,
                "gene": genes_for_transfer,
                "coefficient": model.coef_.astype(float),
            })
        )
    coeff_df = pd.concat(coeff_rows, ignore_index=True) if coeff_rows else pd.DataFrame(columns=["axis", "gene", "coefficient"])
    return models, coeff_df


def predict_scores(X_new: pd.DataFrame, models: Dict[str, Dict[str, object]], calibration: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    out = pd.DataFrame(index=X_new.index)
    X_new = X_new.fillna(0.0)
    for axis, meta in models.items():
        genes = list(meta["genes"])
        coef = np.asarray(meta["coef"], dtype=float)
        intercept = float(meta["intercept"])
        missing = [g for g in genes if g not in X_new.columns]
        if missing:
            raise ValueError(f"Prediction matrix is missing {len(missing)} genes for axis {axis}.")
        raw = X_new.loc[:, genes].to_numpy(dtype=float) @ coef + intercept
        out[f"{axis}_raw"] = raw
        if axis in calibration:
            slope = float(calibration[axis].get("slope", 1.0))
            cal_intercept = float(calibration[axis].get("intercept", 0.0))
            out[f"{axis}_cal"] = cal_intercept + slope * raw
        else:
            out[f"{axis}_cal"] = raw
    return out


def locate_gse163634_matrix(gse163634_root: Path) -> Path:
    files = all_files(gse163634_root)
    name_map = {p.name: p for p in files}
    for fname in PREFERRED_GSE163634_MATRIX_FILES:
        if fname in name_map:
            return name_map[fname]
    # fallback: any samples_by_genes frozen intersection file
    for p in files:
        low = p.name.lower()
        if "frozen_intersection" in low and "samples_by_genes" in low and is_tabular(p):
            return p
    raise FileNotFoundError("Could not locate the prepared GSE163634 samples-by-genes matrix.")


def load_gse163634_metadata(gse163634_root: Path) -> Optional[pd.DataFrame]:
    path = gse163634_root / "derived_bulk_start" / "gse163634_sample_metadata.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    sample_col = detect_sample_id_column(df)
    if sample_col is None:
        return df
    return df.set_index(sample_col)


def build_serial_deltas(score_df: pd.DataFrame, metadata: Optional[pd.DataFrame]) -> pd.DataFrame:
    if metadata is None:
        return pd.DataFrame()
    meta = metadata.copy()
    if meta.index.name is None:
        sample_col = detect_sample_id_column(meta.reset_index())
        if sample_col is not None and sample_col in meta.columns:
            meta = meta.set_index(sample_col)
    needed = {"patient_id", "stage"}
    if not needed.issubset(set(meta.columns)):
        return pd.DataFrame()
    merged = score_df.join(meta, how="left")
    rows = []
    for patient_id, grp in merged.groupby("patient_id"):
        grp = grp.copy()
        by_stage = {str(r["stage"]): idx for idx, r in grp.iterrows()}
        for from_stage, to_stage, label in [("dx", "r1", "dx_to_r1"), ("r1", "r2", "r1_to_r2")]:
            if from_stage in by_stage and to_stage in by_stage:
                a = grp.loc[by_stage[from_stage]]
                b = grp.loc[by_stage[to_stage]]
                row = {
                    "patient_id": patient_id,
                    "transition": label,
                    "from_sample": by_stage[from_stage],
                    "to_sample": by_stage[to_stage],
                    "from_stage": from_stage,
                    "to_stage": to_stage,
                }
                for axis in AXES:
                    if f"{axis}_cal" in grp.columns:
                        row[f"delta_{axis}_cal"] = float(b[f"{axis}_cal"] - a[f"{axis}_cal"])
                        row[f"from_{axis}_cal"] = float(a[f"{axis}_cal"])
                        row[f"to_{axis}_cal"] = float(b[f"{axis}_cal"])
                rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser(description="Rebuild frozen transfer models from GSE235063 and apply them to GSE163634, with optional GSE235923 calibration.")
    p.add_argument("--gse235063-root", required=True)
    p.add_argument("--gse235923-root", required=True)
    p.add_argument("--gse163634-root", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--gene-order", default=None, help="Optional explicit path to train_gene_order.txt")
    p.add_argument("--target-table", default=None, help="Optional explicit GSE235063 sample-level target table")
    p.add_argument("--train-expression", default=None, help="Optional explicit GSE235063 sample-by-gene or gene-by-sample table")
    p.add_argument("--train-expression-orientation", choices=["sample_by_gene", "gene_by_sample"], default=None)
    p.add_argument("--gse163634-matrix", default=None, help="Optional explicit prepared GSE163634 samples-by-genes matrix")
    p.add_argument("--calibration-table", default=None, help="Optional explicit GSE235923 observed calibration table")
    args = p.parse_args()

    g63 = Path(args.gse235063_root).expanduser().resolve()
    g923 = Path(args.gse235923_root).expanduser().resolve()
    g163 = Path(args.gse163634_root).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    gene_order_path = Path(args.gene_order).expanduser().resolve() if args.gene_order else (g63 / "derived_dx_primary_training" / "frozen_gse235063_model" / "train_gene_order.txt")
    if not gene_order_path.exists():
        raise FileNotFoundError(f"Gene order file not found: {gene_order_path}")
    gene_order = read_gene_list(gene_order_path)

    target_table_path = Path(args.target_table).expanduser().resolve() if args.target_table else find_target_table(g63)
    targets_df = pd.read_csv(target_table_path)
    sample_col = detect_sample_id_column(targets_df)
    if sample_col is None:
        raise ValueError(f"Could not detect sample ID column in target table: {target_table_path}")
    targets_df[sample_col] = targets_df[sample_col].astype(str)
    targets_df = targets_df.set_index(sample_col)
    axes_present = [a for a in AXES if a in targets_df.columns]
    if not axes_present:
        raise ValueError(f"No required axis columns found in target table: {target_table_path}")

    if args.train_expression:
        train_expr_path = Path(args.train_expression).expanduser().resolve()
        if args.train_expression_orientation is None:
            raise ValueError("When --train-expression is provided, --train-expression-orientation is also required.")
        orientation = args.train_expression_orientation
    else:
        train_expr_path, orientation = find_training_expression_table(g63, gene_order, list(targets_df.index))

    X_train_full = load_expression_table(train_expr_path, orientation)
    X_train_full.index = normalize_sample_ids(list(X_train_full.index))
    X_train_full.columns = X_train_full.columns.astype(str)

    g163_matrix_path = Path(args.gse163634_matrix).expanduser().resolve() if args.gse163634_matrix else locate_gse163634_matrix(g163)
    X_163 = pd.read_csv(g163_matrix_path, sep=infer_sep(g163_matrix_path), index_col=0)
    X_163.index = normalize_sample_ids(list(X_163.index))
    X_163.columns = X_163.columns.astype(str)

    genes_for_transfer = [g for g in gene_order if g in X_train_full.columns and g in X_163.columns]
    if len(genes_for_transfer) < 1000:
        raise ValueError(f"Too few shared genes for transfer: {len(genes_for_transfer)}")

    X_train, y_train = subset_training_data(X_train_full, targets_df, axes_present)

    calibration_table_path = Path(args.calibration_table).expanduser().resolve() if args.calibration_table else find_calibration_observed_table(g923)
    calibration = derive_calibration_from_table(calibration_table_path)
    if not calibration:
        calibration = derive_calibration_from_multiple_tables(g923, calibration_table_path)

    models, coeff_df = fit_ridge_models(X_train, y_train, genes_for_transfer)
    if not models:
        raise RuntimeError("No models were fit. Check training table alignment and gene overlap.")

    missing_model_axes = [a for a in AXES if a not in models]
    score_df = predict_scores(X_163, models, calibration)
    score_df.index.name = "sample_id"

    metadata = load_gse163634_metadata(g163)
    if metadata is not None:
        score_matrix = score_df.join(metadata, how="left")
    else:
        score_matrix = score_df.copy()
    serial_deltas = build_serial_deltas(score_df, metadata)

    calibration_rows = []
    for axis in AXES:
        row = {"axis": axis}
        if axis in calibration:
            row.update(calibration[axis])
        else:
            row.update({"intercept": 0.0, "slope": 1.0, "source_table": "identity_fallback", "n_samples": 0})
        calibration_rows.append(row)
    calibration_df = pd.DataFrame(calibration_rows)

    model_rows = []
    for axis, meta in models.items():
        model_rows.append({
            "axis": axis,
            "alpha": meta["alpha"],
            "intercept": meta["intercept"],
            "n_samples": meta["n_samples"],
            "n_genes": meta["n_genes"],
            "rmse_train": meta["rmse_train"],
            "r2_train": meta["r2_train"],
        })
    model_summary_df = pd.DataFrame(model_rows)

    score_matrix.to_csv(outdir / "gse163634_bulk_score_matrix.csv")
    serial_deltas.to_csv(outdir / "gse163634_bulk_serial_deltas.csv", index=False)
    calibration_df.to_csv(outdir / "gse235923_inferred_axis_calibration.csv", index=False)
    model_summary_df.to_csv(outdir / "gse235063_rebuilt_transfer_model_summary.csv", index=False)
    coeff_df.to_csv(outdir / "gse235063_rebuilt_transfer_coefficients_long.csv", index=False)

    manifest = {
        "gene_order_file": str(gene_order_path),
        "target_table": str(target_table_path),
        "train_expression": str(train_expr_path),
        "train_expression_orientation": orientation,
        "gse163634_matrix": str(g163_matrix_path),
        "calibration_table": str(calibration_table_path),
        "n_axes_in_target_table": len(axes_present),
        "axes_in_target_table": axes_present,
        "axes_modeled": sorted(models.keys()),
        "axes_missing_model": missing_model_axes,
        "n_shared_genes_transfer": len(genes_for_transfer),
        "n_training_samples_used": int(X_train.shape[0]),
        "n_gse163634_samples_scored": int(X_163.shape[0]),
        "calibration_axes_inferred": sorted(calibration.keys()),
        "outputs": {
            "score_matrix": str(outdir / "gse163634_bulk_score_matrix.csv"),
            "serial_deltas": str(outdir / "gse163634_bulk_serial_deltas.csv"),
            "calibration": str(outdir / "gse235923_inferred_axis_calibration.csv"),
            "model_summary": str(outdir / "gse235063_rebuilt_transfer_model_summary.csv"),
            "coefficients_long": str(outdir / "gse235063_rebuilt_transfer_coefficients_long.csv"),
        },
    }
    with open(outdir / "gse163634_transfer_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("[OK] GSE163634 transfer rebuild/apply complete.")
    print(f"Output: {outdir}")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
