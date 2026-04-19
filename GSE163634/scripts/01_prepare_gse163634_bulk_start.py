from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ALL_PATTERN = re.compile(r"^ALL_(?P<patient>\d+)(?P<suffix>r[12])?$")
CONTROL_PATTERN = re.compile(r"^(?P<donor>\d+)_(?P<ctype>B|T)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare GSE163634 bulk RNA-seq for serial validation.")
    parser.add_argument(
        "--input",
        default="/GSE163634/GSE163634_FPKM_count_matrix.txt",
        help="Path to GSE163634 FPKM matrix (tab-delimited).",
    )
    parser.add_argument(
        "--outdir",
        default=None,
        help="Output directory. Default: <input_dir>/derived_bulk_start",
    )
    parser.add_argument(
        "--gene-column",
        default="Gene_symbol",
        help="Name of gene symbol column in the input file.",
    )
    parser.add_argument(
        "--frozen-genes",
        default=None,
        help="Optional gene list (txt/csv/tsv) for frozen projection intersection.",
    )
    parser.add_argument(
        "--top-var-genes",
        type=int,
        default=2000,
        help="Number of most variable genes to use for starter PCA.",
    )
    return parser.parse_args()


def parse_sample_name(sample_id: str) -> Dict[str, object]:
    m_all = ALL_PATTERN.match(sample_id)
    if m_all:
        patient_id = m_all.group("patient")
        suffix = m_all.group("suffix")
        if suffix is None:
            stage = "dx"
            stage_order = 0
        elif suffix == "r1":
            stage = "r1"
            stage_order = 1
        elif suffix == "r2":
            stage = "r2"
            stage_order = 2
        else:
            raise ValueError(f"Unexpected ALL suffix in sample ID: {sample_id}")

        return {
            "sample_id": sample_id,
            "patient_id": patient_id,
            "stage": stage,
            "stage_order": stage_order,
            "is_control": False,
            "control_type": "",
            "donor_id": "",
            "is_leukemia": True,
        }

    m_ctrl = CONTROL_PATTERN.match(sample_id)
    if m_ctrl:
        donor_id = m_ctrl.group("donor")
        control_type = m_ctrl.group("ctype")
        stage = f"control_{control_type}"
        return {
            "sample_id": sample_id,
            "patient_id": f"CTRL_{donor_id}",
            "stage": stage,
            "stage_order": np.nan,
            "is_control": True,
            "control_type": control_type,
            "donor_id": donor_id,
            "is_leukemia": False,
        }

    raise ValueError(f"Unrecognized sample ID format: {sample_id}")


def build_sample_metadata(sample_cols: List[str]) -> pd.DataFrame:
    meta = pd.DataFrame([parse_sample_name(s) for s in sample_cols])

    pair_group_map: Dict[str, str] = {}
    leukemia = meta.loc[~meta["is_control"]].copy()
    for patient_id, g in leukemia.groupby("patient_id"):
        stages = tuple(sorted(g["stage"].tolist(), key=lambda x: {"dx": 0, "r1": 1, "r2": 2}[x]))
        stage_set = set(stages)
        if stage_set == {"dx", "r1", "r2"}:
            pair_group = "dx_r1_r2"
        elif stage_set == {"dx", "r1"}:
            pair_group = "dx_r1"
        elif stage_set == {"r1", "r2"}:
            pair_group = "r1_r2"
        elif stage_set == {"dx", "r2"}:
            pair_group = "dx_r2"
        else:
            pair_group = "singleton"
        pair_group_map[patient_id] = pair_group

    meta["pair_group"] = np.where(
        meta["is_control"],
        "control",
        meta["patient_id"].map(pair_group_map).fillna("singleton"),
    )

    meta = meta.sort_values(
        by=["is_control", "patient_id", "stage_order", "control_type", "sample_id"],
        ascending=[True, True, True, True, True],
    ).reset_index(drop=True)
    return meta


def build_serial_transitions(sample_meta: pd.DataFrame) -> pd.DataFrame:
    transitions: List[Dict[str, object]] = []
    leukemia = sample_meta.loc[~sample_meta["is_control"]].copy()

    for patient_id, g in leukemia.groupby("patient_id"):
        stage_to_sample = dict(zip(g["stage"], g["sample_id"]))

        if "dx" in stage_to_sample and "r1" in stage_to_sample:
            transitions.append(
                {
                    "patient_id": patient_id,
                    "transition": "dx_to_r1",
                    "from_sample": stage_to_sample["dx"],
                    "to_sample": stage_to_sample["r1"],
                    "from_stage": "dx",
                    "to_stage": "r1",
                }
            )

        if "r1" in stage_to_sample and "r2" in stage_to_sample:
            transitions.append(
                {
                    "patient_id": patient_id,
                    "transition": "r1_to_r2",
                    "from_sample": stage_to_sample["r1"],
                    "to_sample": stage_to_sample["r2"],
                    "from_stage": "r1",
                    "to_stage": "r2",
                }
            )

    out = pd.DataFrame(transitions)
    if out.empty:
        return pd.DataFrame(columns=["patient_id", "transition", "from_sample", "to_sample", "from_stage", "to_stage"])
    return out.sort_values(["transition", "patient_id"]).reset_index(drop=True)


def build_gene_table(expr_raw: pd.DataFrame, gene_col: str, sample_cols: List[str]) -> pd.DataFrame:
    gene_table = pd.DataFrame({"gene_symbol_raw": expr_raw[gene_col].astype(str)})
    gene_table["gene_symbol_primary"] = gene_table["gene_symbol_raw"].str.split(",").str[0].str.strip()
    gene_table["n_aliases"] = gene_table["gene_symbol_raw"].str.split(",").str.len()
    gene_table["is_multi_symbol"] = gene_table["n_aliases"] > 1
    gene_table["row_mean_fpkm"] = expr_raw[sample_cols].mean(axis=1).astype(float)
    gene_table["row_nonzero_fraction"] = (expr_raw[sample_cols] > 0).mean(axis=1).astype(float)
    return gene_table


def make_unique_log2_matrix(
    log2_expr: pd.DataFrame,
    gene_table: pd.DataFrame,
    sample_cols: List[str],
) -> pd.DataFrame:
    keep_mask = (~gene_table["is_multi_symbol"]) & gene_table["gene_symbol_primary"].ne("")
    working = gene_table.loc[keep_mask, ["gene_symbol_primary", "row_mean_fpkm"]].copy()
    working["row_ix"] = working.index
    working = working.sort_values(["gene_symbol_primary", "row_mean_fpkm"], ascending=[True, False])
    working = working.drop_duplicates(subset="gene_symbol_primary", keep="first")
    keep_idx = working["row_ix"].tolist()

    unique_log = log2_expr.loc[keep_idx, sample_cols].copy()
    unique_log.insert(0, "Gene_symbol", gene_table.loc[keep_idx, "gene_symbol_primary"].values)
    unique_log = unique_log.sort_values("Gene_symbol").reset_index(drop=True)
    return unique_log


def read_gene_list(gene_list_path: Path) -> List[str]:
    suffix = gene_list_path.suffix.lower()
    if suffix in {".txt", ".list"}:
        genes = [line.strip() for line in gene_list_path.read_text().splitlines() if line.strip()]
        return genes

    sep = "\t" if suffix in {".tsv", ".tab"} else ","
    df = pd.read_csv(gene_list_path, sep=sep)
    candidate_cols = [c for c in df.columns if c.lower() in {"gene", "gene_symbol", "symbol", "genes"}]
    col = candidate_cols[0] if candidate_cols else df.columns[0]
    genes = df[col].astype(str).str.strip().tolist()
    return [g for g in genes if g]


def save_stage_counts_plot(sample_meta: pd.DataFrame, outdir: Path) -> None:
    order = ["dx", "r1", "r2", "control_B", "control_T"]
    counts = sample_meta["stage"].value_counts().reindex(order, fill_value=0)

    plt.figure(figsize=(7, 4.5))
    counts.plot(kind="bar")
    plt.ylabel("Number of samples")
    plt.title("GSE163634 stage/control counts")
    plt.tight_layout()
    plt.savefig(outdir / "gse163634_stage_counts.png", dpi=300)
    plt.savefig(outdir / "gse163634_stage_counts.pdf")
    plt.close()


def save_starter_pca(unique_log2: pd.DataFrame, sample_meta: pd.DataFrame, outdir: Path, top_var_genes: int) -> None:
    try:
        from sklearn.decomposition import PCA
    except Exception as exc:
        print(f"[WARN] sklearn not available; skipping PCA plot. ({exc})")
        return

    sample_cols = [c for c in unique_log2.columns if c != "Gene_symbol"]
    X = unique_log2.set_index("Gene_symbol")[sample_cols].T

    if X.shape[1] < 2:
        print("[WARN] Fewer than 2 genes in unique matrix; skipping PCA.")
        return

    gene_var = X.var(axis=0)
    keep_n = min(top_var_genes, X.shape[1])
    top_genes = gene_var.sort_values(ascending=False).head(keep_n).index
    X_use = X.loc[:, top_genes]

    pca = PCA(n_components=2)
    pcs = pca.fit_transform(X_use)

    pca_df = sample_meta.set_index("sample_id").loc[X_use.index].reset_index().copy()
    pca_df["PC1"] = pcs[:, 0]
    pca_df["PC2"] = pcs[:, 1]
    pca_df.to_csv(outdir / "gse163634_starter_pca_scores.csv", index=False)

    color_map = {
        "dx": "tab:blue",
        "r1": "tab:orange",
        "r2": "tab:red",
        "control_B": "tab:green",
        "control_T": "tab:purple",
    }
    marker_map = {
        "dx": "o",
        "r1": "s",
        "r2": "^",
        "control_B": "D",
        "control_T": "P",
    }

    plt.figure(figsize=(7, 5.5))
    for stage, g in pca_df.groupby("stage"):
        plt.scatter(
            g["PC1"],
            g["PC2"],
            label=stage,
            c=color_map.get(stage, "gray"),
            marker=marker_map.get(stage, "o"),
            s=55,
            alpha=0.9,
            edgecolor="black",
            linewidth=0.3,
        )

    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% var)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% var)")
    plt.title("GSE163634 starter PCA (top variable unique genes)")
    plt.legend(frameon=False, fontsize=9)
    plt.tight_layout()
    plt.savefig(outdir / "gse163634_starter_pca.png", dpi=300)
    plt.savefig(outdir / "gse163634_starter_pca.pdf")
    plt.close()


def main() -> None:
    args = parse_args()

    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    outdir = Path(args.outdir).expanduser().resolve() if args.outdir else input_path.parent / "derived_bulk_start"
    outdir.mkdir(parents=True, exist_ok=True)

    header = pd.read_csv(input_path, sep="\t", nrows=0)
    if args.gene_column not in header.columns:
        raise ValueError(f"Gene column '{args.gene_column}' not found. Available columns start with: {list(header.columns[:5])}")

    sample_cols = [c for c in header.columns if c != args.gene_column]
    dtypes = {c: np.float32 for c in sample_cols}
    expr_raw = pd.read_csv(input_path, sep="\t", dtype=dtypes)
    expr_raw[args.gene_column] = expr_raw[args.gene_column].astype(str)

    sample_meta = build_sample_metadata(sample_cols)
    transitions = build_serial_transitions(sample_meta)
    gene_table = build_gene_table(expr_raw, args.gene_column, sample_cols)

    log2_expr = expr_raw.copy()
    log2_expr[sample_cols] = np.log2(log2_expr[sample_cols].astype(np.float32) + 1.0)

    full_log2_out = log2_expr[[args.gene_column] + sample_cols].copy()
    full_log2_out.to_csv(outdir / "gse163634_log2fpkm_full_matrix.tsv.gz", sep="\t", index=False)

    unique_log2 = make_unique_log2_matrix(log2_expr, gene_table, sample_cols)
    unique_log2.to_csv(outdir / "gse163634_log2fpkm_unique_genes.tsv.gz", sep="\t", index=False)

    sample_meta.to_csv(outdir / "gse163634_sample_metadata.csv", index=False)
    transitions.to_csv(outdir / "gse163634_serial_transitions.csv", index=False)
    gene_table.to_csv(outdir / "gse163634_gene_annotation_table.csv", index=False)

    save_stage_counts_plot(sample_meta, outdir)
    save_starter_pca(unique_log2, sample_meta, outdir, args.top_var_genes)

    summary = {
        "input_file": str(input_path),
        "n_rows_raw": int(expr_raw.shape[0]),
        "n_sample_columns": int(len(sample_cols)),
        "n_unique_primary_genes": int(unique_log2.shape[0]),
        "n_ambiguous_multi_symbol_rows": int(gene_table["is_multi_symbol"].sum()),
        "stage_counts": sample_meta["stage"].value_counts().sort_index().to_dict(),
        "n_patients_leukemia": int(sample_meta.loc[~sample_meta["is_control"], "patient_id"].nunique()),
        "n_dx_to_r1_pairs": int((transitions["transition"] == "dx_to_r1").sum()) if not transitions.empty else 0,
        "n_r1_to_r2_pairs": int((transitions["transition"] == "r1_to_r2").sum()) if not transitions.empty else 0,
    }

    frozen_summary: Optional[Dict[str, object]] = None
    if args.frozen_genes:
        gene_list_path = Path(args.frozen_genes).expanduser().resolve()
        if not gene_list_path.exists():
            raise FileNotFoundError(f"Frozen gene list not found: {gene_list_path}")

        frozen_genes = read_gene_list(gene_list_path)
        frozen_gene_set = set(frozen_genes)

        unique_gene_df = unique_log2.copy()
        unique_gene_df["_order"] = np.arange(unique_gene_df.shape[0])
        unique_gene_df = unique_gene_df.set_index("Gene_symbol")

        intersected = [g for g in frozen_genes if g in unique_gene_df.index]
        frozen_gene_table = pd.DataFrame(
            {
                "gene_symbol": frozen_genes,
                "present_in_gse163634": [g in frozen_gene_set and g in unique_gene_df.index for g in frozen_genes],
            }
        )
        frozen_gene_table.to_csv(outdir / "gse163634_frozen_gene_intersection.csv", index=False)

        frozen_matrix_gene_by_sample = unique_gene_df.loc[intersected, sample_cols].copy()
        frozen_matrix_gene_by_sample.insert(0, "Gene_symbol", intersected)
        frozen_matrix_gene_by_sample.to_csv(
            outdir / "gse163634_log2fpkm_frozen_intersection_genes_by_samples.tsv.gz",
            sep="\t",
            index=False,
        )

        frozen_matrix_sample_by_gene = unique_gene_df.loc[intersected, sample_cols].T
        frozen_matrix_sample_by_gene.index.name = "sample_id"
        frozen_matrix_sample_by_gene.to_csv(
            outdir / "gse163634_log2fpkm_frozen_intersection_samples_by_genes.tsv.gz",
            sep="\t",
        )

        frozen_summary = {
            "frozen_gene_list": str(gene_list_path),
            "n_frozen_genes_requested": int(len(frozen_genes)),
            "n_frozen_genes_matched": int(len(intersected)),
        }
        summary["frozen_intersection"] = frozen_summary

    with open(outdir / "gse163634_starter_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n[OK] GSE163634 starter preparation complete.")
    print(f"Input:  {input_path}")
    print(f"Output: {outdir}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
