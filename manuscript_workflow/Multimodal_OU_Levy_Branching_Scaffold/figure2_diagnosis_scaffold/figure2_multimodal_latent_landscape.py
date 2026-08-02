from __future__ import annotations

from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import matplotlib.pyplot as plt
import textwrap
from matplotlib.colors import to_hex
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# ============================================================
# 1. PATHS
# ============================================================
BASE = Path("/Multimodal_OU_Lévy_Branching_Scaffold/Figure_2")
OUTDIR = BASE / "figure2_output"
OUTDIR.mkdir(parents=True, exist_ok=True)

MALIGNANT_H5AD_CANDIDATES = [
    BASE / "dx_primary_training_malignant_frozen_transfer.h5ad",
    BASE / "dx_primary_training_malignant.h5ad",
]

ALLCELLS_H5AD_CANDIDATES = [
    BASE / "dx_primary_training_allcells_frozen_transfer.h5ad",
    BASE / "dx_primary_training_allcells.h5ad",
]

CELL_METADATA_CSV = BASE / "dx_ou_malignant_cell_metadata.csv.gz"
DESIGN_MATRIX_CSV = BASE / "dx_ou_training_design_matrix_core4.csv"
COARSE_STATE_CSV = BASE / "dx_ou_malignant_coarse_states_with_aux.csv"
ALLCELLS_FRACTIONS_CSV = BASE / "allcells_celltype_fractions_by_sample.csv"
MALIGNANT_STATE_FRACTIONS_CSV = BASE / "dx_ou_malignant_state_fractions_by_sample_full.csv"

# Optional cell-level regulatory annotations. If absent, the script will try to infer
# a continuous regulatory proxy from AnnData obs columns, otherwise it will fall back
# to PC1 from the design matrix mapped by sample_id.
REGULATORY_CELL_CSV = BASE / "dx_regulatory_state_by_cell.csv"

RANDOM_STATE = 42
FIG_BASENAME = "Figure2_multimodal_latent_landscape_full"


# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================
def first_existing_path(paths):
    for p in paths:
        if p.exists():
            return p
    raise FileNotFoundError(f"None of the candidate paths exist: {paths}")


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def normalize_colmap(columns):
    return {str(c).lower(): c for c in columns}


def find_first_column(columns, candidates):
    colmap = normalize_colmap(columns)
    for c in candidates:
        if c.lower() in colmap:
            return colmap[c.lower()]
    return None


def infer_timepoint_from_sample_id(sample_id: str) -> str:
    if pd.isna(sample_id):
        return "Unknown"
    s = str(sample_id).upper()
    if s.endswith("_DX") or "_DX_" in s or s == "DX":
        return "Diagnosis"
    if "EOI" in s or "REM" in s or "CR" in s:
        return "EOI / remission"
    if "REL" in s or "RELAPSE" in s:
        return "Relapse"
    if "TX" in s or "THER" in s or "ON" in s:
        return "On-therapy"
    return "Unknown"


def infer_patient_from_sample_id(sample_id: str) -> str:
    if pd.isna(sample_id):
        return "Unknown"
    return str(sample_id).split("_")[0]


def safe_series_to_numeric(x):
    return pd.to_numeric(pd.Series(x), errors="coerce")


def entropy_from_rowprob(row):
    p = np.asarray(row, dtype=float)
    p = np.clip(p, 1e-12, None)
    p = p / p.sum()
    return -(p * np.log(p)).sum()


def choose_embedding(adata: ad.AnnData) -> str:
    """
    Freeze one common 2D coordinate matrix for all panels.
    Preference order is explicit and stable.
    """
    two_d_keys = [
        "X_fig2",
        "X_scaffold_umap",
        "X_integrated_umap",
        "X_frozen_umap",
        "X_umap",
    ]
    for key in two_d_keys:
        if key in adata.obsm and adata.obsm[key].shape[1] >= 2:
            adata.obsm["X_fig2"] = np.asarray(adata.obsm[key])[:, :2]
            return "X_fig2"

    # Build UMAP from a higher-dimensional representation if needed
    rep_keys = [
        "X_scaffold",
        "X_scVI",
        "X_integrated",
        "X_latent",
        "X_pca",
    ]
    rep_key = None
    for key in rep_keys:
        if key in adata.obsm:
            rep_key = key
            break

    if rep_key is None:
        # final fallback: try existing X_pca or compute PCA from X
        if adata.X is None:
            raise KeyError("No usable representation found in adata.obsm and adata.X is empty.")
        sc.pp.pca(adata, n_comps=30)
        rep_key = "X_pca"

    sc.pp.neighbors(adata, use_rep=rep_key, random_state=RANDOM_STATE)
    sc.tl.umap(adata, random_state=RANDOM_STATE)
    adata.obsm["X_fig2"] = np.asarray(adata.obsm["X_umap"])[:, :2]
    return "X_fig2"


def attach_base_obs(adata: ad.AnnData) -> ad.AnnData:
    """
    Standardize cell_id / sample_id / patient_id / clinical_timepoint
    using exact file-aware matching where possible.
    """
    adata.obs = adata.obs.copy()
    adata.obs["cell_id"] = adata.obs_names.astype(str)

    # Merge dx_ou_malignant_cell_metadata.csv.gz if available
    if CELL_METADATA_CSV.exists():
        meta = read_csv(CELL_METADATA_CSV).copy()
        cell_key = find_first_column(meta.columns, ["cell_id", "barcode", "cell_barcode"])
        if cell_key is None:
            raise KeyError("dx_ou_malignant_cell_metadata.csv.gz exists but no cell key was found.")
        meta = meta.rename(columns={cell_key: "cell_id"})
        meta["cell_id"] = meta["cell_id"].astype(str)

        add_cols = [c for c in meta.columns if c != "cell_id" and c not in adata.obs.columns]
        adata.obs = (
            adata.obs.reset_index(drop=False)
            .rename(columns={"index": "orig_obs_name"})
            .merge(meta[["cell_id"] + add_cols], on="cell_id", how="left")
            .set_index("cell_id")
        )

    # sample_id
    sample_col = find_first_column(
        adata.obs.columns,
        ["sample_id", "sample", "orig.ident", "library_id", "sample_name"]
    )
    if sample_col is None:
        raise KeyError(
            "Could not find sample_id-like column in malignant h5ad obs. "
            "Expected one of: sample_id, sample, orig.ident, library_id, sample_name."
        )
    if sample_col != "sample_id":
        adata.obs["sample_id"] = adata.obs[sample_col].astype(str)
    else:
        adata.obs["sample_id"] = adata.obs["sample_id"].astype(str)

    # patient_id
    patient_col = find_first_column(
        adata.obs.columns,
        ["patient_id", "Patient_ID", "patient", "participant_id", "case_id"]
    )
    if patient_col is not None:
        adata.obs["patient_id"] = adata.obs[patient_col].astype(str)
    else:
        adata.obs["patient_id"] = adata.obs["sample_id"].map(infer_patient_from_sample_id)

    # clinical_timepoint
    time_col = find_first_column(
        adata.obs.columns,
        ["clinical_timepoint", "timepoint", "phase", "clinical_phase", "state"]
    )
    if time_col is not None:
        adata.obs["clinical_timepoint"] = adata.obs[time_col].astype(str)
    else:
        adata.obs["clinical_timepoint"] = adata.obs["sample_id"].map(infer_timepoint_from_sample_id)

    # disease subgroup
    subgroup_col = find_first_column(
        adata.obs.columns,
        ["disease_subgroup", "Subgroup", "subgroup", "diagnosis", "subtype"]
    )
    if subgroup_col is not None:
        adata.obs["disease_subgroup"] = adata.obs[subgroup_col].astype(str)
    else:
        adata.obs["disease_subgroup"] = "Unknown"

    return adata


def attach_ecology_exact(adata: ad.AnnData) -> ad.AnnData:
    """
    Exact mapping using actual columns from dx_ou_training_design_matrix_core4.csv:
        sample_id
        Patient_ID
        Subgroup
        PC1
        PC2
        ecotype_cluster
        ecotype_label
        dominant_malignant_state
        dominant_malignant_state_frac
        state_HSC, state_Prog, state_GMP, state_MonoDC
        aux_EryBaso, aux_CLP
    """
    design = read_csv(DESIGN_MATRIX_CSV).copy()

    keep_cols = [
        "sample_id",
        "Patient_ID",
        "Subgroup",
        "PC1",
        "PC2",
        "ecotype_cluster",
        "ecotype_label",
        "dominant_malignant_state",
        "dominant_malignant_state_frac",
        "state_HSC",
        "state_Prog",
        "state_GMP",
        "state_MonoDC",
        "aux_EryBaso",
        "aux_CLP",
    ]
    missing = [c for c in keep_cols if c not in design.columns]
    if missing:
        raise KeyError(f"Missing required columns from design matrix: {missing}")

    eco = design[keep_cols].drop_duplicates("sample_id").copy()

    # exact field mapping
    eco["patient_id_design"] = eco["Patient_ID"].astype(str)
    eco["disease_subgroup_design"] = eco["Subgroup"].astype(str)
    eco["ecotype"] = eco["ecotype_label"].astype(str)
    eco["ecotype_cluster"] = eco["ecotype_cluster"].astype(str)
    eco["tme_axis_1"] = eco["PC1"]
    eco["tme_axis_2"] = eco["PC2"]

    adata.obs = (
        adata.obs.reset_index(drop=False)
        .rename(columns={"index": "cell_id"})
        .merge(
            eco[
                [
                    "sample_id",
                    "patient_id_design",
                    "disease_subgroup_design",
                    "ecotype",
                    "ecotype_cluster",
                    "tme_axis_1",
                    "tme_axis_2",
                    "dominant_malignant_state",
                    "dominant_malignant_state_frac",
                    "state_HSC",
                    "state_Prog",
                    "state_GMP",
                    "state_MonoDC",
                    "aux_EryBaso",
                    "aux_CLP",
                ]
            ],
            on="sample_id",
            how="left",
        )
        .set_index("cell_id")
    )

    # Fill subgroup from design matrix if not already present
    replace_mask = adata.obs["disease_subgroup"].isna() | (adata.obs["disease_subgroup"] == "Unknown")
    adata.obs.loc[replace_mask, "disease_subgroup"] = adata.obs.loc[replace_mask, "disease_subgroup_design"]

    return adata


def attach_branch_exact(adata: ad.AnnData) -> ad.AnnData:
    """
    Exact mapping using actual columns from dx_ou_malignant_coarse_states_with_aux.csv:
        sample_id
        state_HSC
        state_Prog
        state_GMP
        state_MonoDC
        aux_EryBaso
        aux_CLP

    This is sample-level, so branch assignment is mapped to each cell by sample_id.
    """
    coarse = read_csv(COARSE_STATE_CSV).copy()
    required = ["sample_id", "state_HSC", "state_Prog", "state_GMP", "state_MonoDC", "aux_EryBaso", "aux_CLP"]
    missing = [c for c in required if c not in coarse.columns]
    if missing:
        raise KeyError(f"Missing required columns from coarse-state table: {missing}")

    prob_cols = ["state_HSC", "state_Prog", "state_GMP", "state_MonoDC"]
    coarse["branch_id"] = coarse[prob_cols].idxmax(axis=1).str.replace("state_", "", regex=False)
    coarse["branch_maxprob"] = coarse[prob_cols].max(axis=1)
    coarse["branch_entropy"] = coarse[prob_cols].apply(entropy_from_rowprob, axis=1)

    coarse["branch_id"] = coarse["branch_id"].map({
        "HSC": "HSC-like basin",
        "Prog": "Progenitor-like basin",
        "GMP": "GMP-like basin",
        "MonoDC": "Mono/DC-like basin",
    }).fillna(coarse["branch_id"])

    adata.obs = (
        adata.obs.reset_index(drop=False)
        .rename(columns={"index": "cell_id"})
        .merge(
            coarse[
                [
                    "sample_id",
                    "state_HSC",
                    "state_Prog",
                    "state_GMP",
                    "state_MonoDC",
                    "aux_EryBaso",
                    "aux_CLP",
                    "branch_id",
                    "branch_maxprob",
                    "branch_entropy",
                ]
            ],
            on="sample_id",
            how="left",
        )
        .set_index("cell_id")
    )

    return adata


def attach_regulatory_exact_or_proxy(adata: ad.AnnData) -> ad.AnnData:
    """
    Priority:
    1) exact per-cell regulatory CSV if present
    2) existing obs columns in AnnData (chromatin/reg/tf/gene_activity)
    3) sample-level proxy from design matrix PC1
    """
    # 1) per-cell regulatory CSV
    if REGULATORY_CELL_CSV.exists():
        reg = read_csv(REGULATORY_CELL_CSV).copy()
        cell_key = find_first_column(reg.columns, ["cell_id", "barcode", "cell_barcode"])
        if cell_key is None:
            raise KeyError("dx_regulatory_state_by_cell.csv exists but no cell key was found.")
        reg = reg.rename(columns={cell_key: "cell_id"})
        reg["cell_id"] = reg["cell_id"].astype(str)

        score_col = find_first_column(reg.columns, ["reg_program_score", "gene_activity_score", "tf_program_score"])
        state_col = find_first_column(reg.columns, ["reg_state", "regulatory_state", "chromatin_state"])

        keep = ["cell_id"]
        if score_col is not None:
            keep.append(score_col)
        if state_col is not None and state_col not in keep:
            keep.append(state_col)

        adata.obs = (
            adata.obs.reset_index(drop=False)
            .rename(columns={"index": "cell_id"})
            .merge(reg[keep], on="cell_id", how="left")
            .set_index("cell_id")
        )

        if score_col is not None:
            adata.obs["reg_program_score"] = pd.to_numeric(adata.obs[score_col], errors="coerce")
        if state_col is not None:
            adata.obs["reg_state"] = adata.obs[state_col].astype(str)

    # 2) infer from existing AnnData obs columns if still missing
    if "reg_program_score" not in adata.obs.columns:
        numeric_candidates = []
        for c in adata.obs.columns:
            lc = c.lower()
            if any(tok in lc for tok in ["reg", "chrom", "tf", "access", "activity", "gene_activity"]):
                if pd.api.types.is_numeric_dtype(adata.obs[c]):
                    numeric_candidates.append(c)
        if numeric_candidates:
            adata.obs["reg_program_score"] = pd.to_numeric(adata.obs[numeric_candidates[0]], errors="coerce")

    if "reg_state" not in adata.obs.columns:
        cat_candidates = []
        for c in adata.obs.columns:
            lc = c.lower()
            if any(tok in lc for tok in ["reg", "chrom", "tf"]):
                if not pd.api.types.is_numeric_dtype(adata.obs[c]):
                    cat_candidates.append(c)
        if cat_candidates:
            adata.obs["reg_state"] = adata.obs[cat_candidates[0]].astype(str)

    # 3) fallback sample-level proxy from design matrix PC1
    if "reg_program_score" not in adata.obs.columns:
        design = read_csv(DESIGN_MATRIX_CSV).copy()
        if "sample_id" in design.columns and "PC1" in design.columns:
            proxy = design[["sample_id", "PC1"]].drop_duplicates("sample_id").rename(columns={"PC1": "reg_program_score"})
            adata.obs = (
                adata.obs.reset_index(drop=False)
                .rename(columns={"index": "cell_id"})
                .merge(proxy, on="sample_id", how="left")
                .set_index("cell_id")
            )
        else:
            adata.obs["reg_program_score"] = np.nan

    if "reg_state" not in adata.obs.columns:
        # discretize the proxy for a categorical backup
        score = pd.to_numeric(adata.obs["reg_program_score"], errors="coerce")
        if score.notna().any():
            q1, q2 = score.quantile([0.33, 0.66])
            adata.obs["reg_state"] = np.where(
                score <= q1, "Low regulatory activity",
                np.where(score <= q2, "Intermediate regulatory activity", "High regulatory activity")
            )
        else:
            adata.obs["reg_state"] = "Unknown"

    return adata


def export_obs_table(adata: ad.AnnData, out_csv: Path):
    desired = [
        "sample_id",
        "patient_id",
        "clinical_timepoint",
        "disease_subgroup",
        "ecotype",
        "ecotype_cluster",
        "tme_axis_1",
        "tme_axis_2",
        "dominant_malignant_state",
        "dominant_malignant_state_frac",
        "state_HSC",
        "state_Prog",
        "state_GMP",
        "state_MonoDC",
        "aux_EryBaso",
        "aux_CLP",
        "reg_state",
        "reg_program_score",
        "branch_id",
        "branch_maxprob",
        "branch_entropy",
    ]
    present = [c for c in desired if c in adata.obs.columns]
    adata.obs[present].to_csv(out_csv, compression="gzip")


# ============================================================
# 3. PLOTTING HELPERS
# ============================================================
def style_axis(ax, panel_label, title):
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

    # clear any default title
    ax.set_title("")

    # panel letter
    ax.text(
        0.00, 1.03, panel_label,
        transform=ax.transAxes,
        fontsize=18, fontweight="bold",
        ha="left", va="bottom"
    )

    # panel title
    ax.text(
        0.11, 1.025, title,
        transform=ax.transAxes,
        fontsize=12, fontweight="bold",
        ha="left", va="bottom"
    )


def palette_timepoint():
    return {
        "DX": "#A9A9A9",
        "Diagnosis": "#A9A9A9",
        "On-therapy": "#E3C567",
        "EOI": "#7FBF7B",
        "EOI / remission": "#7FBF7B",
        "Remission": "#7FBF7B",
        "REL": "#D97C7C",
        "Relapse": "#D97C7C",
        "Unknown": "#BDBDBD",
    }


def palette_ecotype(values):
    unique = sorted(pd.Series(values).dropna().astype(str).unique().tolist())
    defaults = {
        "E1_Bcell_HSPC": "#4E79A7",
        "E2_Mono_Inflammatory": "#F28E2B",
        "E3_NK_CD8Memory": "#59A14F",
        "E4_CD4Naive_CD8Naive": "#E15759",
    }
    out = {}
    fallback = plt.get_cmap("Set2")
    for i, v in enumerate(unique):
        out[v] = defaults.get(v, to_hex(fallback(i % fallback.N)))
    return out


def palette_branch():
    return {
        "HSC-like basin": "#4E79A7",
        "Progenitor-like basin": "#F28E2B",
        "GMP-like basin": "#59A14F",
        "Mono/DC-like basin": "#9C755F",
        "Unknown": "#BDBDBD",
    }


def palette_generic(values, cmap_name="tab20"):
    vals = sorted(pd.Series(values).dropna().astype(str).unique().tolist())
    cmap = plt.get_cmap(cmap_name)
    return {v: to_hex(cmap(i % cmap.N)) for i, v in enumerate(vals)}


def plot_categorical(ax, xy, values, title, panel_label, palette=None, legend=True, s=5, cats=None):
    vals = pd.Series(values).astype("category")

    if cats is None:
        cats = [c for c in vals.cat.categories if pd.notna(c)]
    else:
        cats = [c for c in cats if c in set(vals.astype(str)) or c in set(vals.cat.categories)]

    for cat in cats:
        idx = (vals == cat).to_numpy()
        ax.scatter(
            xy[idx, 0], xy[idx, 1],
            s=s,
            c=[palette.get(cat, "#BDBDBD")] if palette is not None else None,
            linewidths=0,
            alpha=0.9,
            rasterized=True,
            label=str(cat),
        )

    style_axis(ax, panel_label, title)

    if legend:
        ax.legend(frameon=False, loc="best", fontsize=9, markerscale=2)


def plot_continuous(ax, xy, values, title, panel_label, cmap="viridis", s=5, alpha=0.9):
    v = safe_series_to_numeric(values).to_numpy()
    sca = ax.scatter(
        xy[:, 0], xy[:, 1], c=v,
        s=s, cmap=cmap, linewidths=0, alpha=alpha, rasterized=True
    )
    style_axis(ax, panel_label, title)
    cbar = plt.colorbar(sca, ax=ax, fraction=0.046, pad=0.02)
    cbar.ax.tick_params(labelsize=8)

def add_panel_f_box(ax, x, y, w, h, text, fc="#ffffff", ec="#333333",
                    lw=1.2, fontsize=9, weight="normal", text_width=28):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.015,rounding_size=0.018",
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=2
    )
    ax.add_patch(patch)

    wrapped = "\n".join(textwrap.wrap(text, width=text_width))
    ax.text(
        x + w / 2, y + h / 2,
        wrapped,
        ha="center", va="center",
        fontsize=fontsize, fontweight=weight,
        color="black", zorder=5
    )
    return patch


def add_panel_f_arrow(ax, x1, y1, x2, y2, color="#555555", lw=1.5):
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="-|>",
        mutation_scale=14,
        linewidth=lw,
        color=color,
        connectionstyle="arc3,rad=0.0",
        zorder=4
    )
    ax.add_patch(arrow)
    return arrow


def plot_panel_f_interpretation(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Panel label and title
    ax.text(
        0.00, 1.09, "F",
        transform=ax.transAxes,
        fontsize=18, fontweight="bold",
        ha="left", va="top"
    )

    ax.text(
        0.50, 1.05,
        "Integrated interpretation of the diagnosis scaffold",
        fontsize=12.5, fontweight="bold",
        ha="center", va="top"
    )

    # -----------------------------
    # Frame geometry
    # -----------------------------
    frame_x = 0.05
    frame_y = 0.18
    frame_w = 0.90
    frame_h = 0.74
    frame_top = frame_y + frame_h

    add_panel_f_box(
        ax, frame_x, frame_y, frame_w, frame_h,
        "",
        fc="#ffffff", ec="#d8d8d8", lw=1.0,
        fontsize=1
    )

    # -----------------------------
    # Layer geometry inside frame
    # -----------------------------
    # Move top row upward, keep central box stable, move bottom row downward
    input_y = frame_y + frame_h * 0.72
    input_w = 0.21
    input_h = 0.15
    
    landscape_x = 0.31
    landscape_y = frame_y + frame_h * 0.39
    landscape_w = 0.38
    landscape_h = 0.15
    
    outcome_y = frame_y + frame_h * 0.06
    outcome_w = 0.22
    outcome_h = 0.14

    # -----------------------------
    # Input layer
    # -----------------------------
    input_specs = [
        (0.115, "Malignant-state\nstructure", "#e8f4fd", "#4E79A7"),
        (0.395, "Ecological /\nmicroenvironmental\ncontext", "#fdebd0", "#F28E2B"),
        (0.675, "Regulatory program\nvariation", "#eafaf1", "#59A14F"),
    ]

    for x, label, fc, ec in input_specs:
        add_panel_f_box(
            ax, x, input_y, input_w, input_h,
            label,
            fc=fc, ec=ec, lw=1.4,
            fontsize=9.2, weight="bold", text_width=24
        )

    # -----------------------------
    # Central landscape box
    # -----------------------------
    add_panel_f_box(
        ax, landscape_x, landscape_y, landscape_w, landscape_h,
        "Interpretable pretreatment\ndisease landscape",
        fc="#fff2cc", ec="#b8860b", lw=1.6,
        fontsize=10.5, weight="bold", text_width=34
    )

    # Arrows from inputs to landscape
    arrow_gap = 0.006

    add_panel_f_arrow(
        ax,
        0.115 + input_w / 2, input_y - arrow_gap,
        landscape_x + landscape_w * 0.25, landscape_y + landscape_h + arrow_gap
    )

    add_panel_f_arrow(
        ax,
        0.395 + input_w / 2, input_y - arrow_gap,
        landscape_x + landscape_w * 0.50, landscape_y + landscape_h + arrow_gap
    )

    add_panel_f_arrow(
        ax,
        0.675 + input_w / 2, input_y - arrow_gap,
        landscape_x + landscape_w * 0.75, landscape_y + landscape_h + arrow_gap
    )

    # -----------------------------
    # Outcome layer
    # -----------------------------
    outcome_specs = [
        (0.115, "Response-like\ncontraction", "#d9edf7", "#2e86c1"),
        (0.390, "Residual\npersistence", "#d5f5e3", "#239b56"),
        (0.665, "Relapse-associated\nescape", "#f5c6cb", "#c0392b"),
    ]

    for x, label, fc, ec in outcome_specs:
        add_panel_f_box(
            ax, x, outcome_y, outcome_w, outcome_h,
            label,
            fc=fc, ec=ec, lw=1.4,
            fontsize=9.2, weight="bold", text_width=24
        )

    # Arrows from landscape to outcomes
    add_panel_f_arrow(
        ax,
        landscape_x + landscape_w * 0.25, landscape_y - arrow_gap,
        0.115 + outcome_w / 2, outcome_y + outcome_h + arrow_gap
    )

    add_panel_f_arrow(
        ax,
        landscape_x + landscape_w * 0.50, landscape_y - arrow_gap,
        0.390 + outcome_w / 2, outcome_y + outcome_h + arrow_gap
    )

    add_panel_f_arrow(
        ax,
        landscape_x + landscape_w * 0.75, landscape_y - arrow_gap,
        0.665 + outcome_w / 2, outcome_y + outcome_h + arrow_gap
    )

    # -----------------------------
    # Take-home sentence below frame
    # -----------------------------
    ax.text(
        0.50, 0.035,
        "Malignant-state structure, ecological context and regulatory variation jointly define the pretreatment landscape\n"
        "used to quantify response-like contraction, residual persistence and relapse-associated escape.",
        fontsize=12.0,
        ha="center", va="bottom"
    )

# ============================================================
# 4. MAIN
# ============================================================
def main():
    main_h5ad = first_existing_path(MALIGNANT_H5AD_CANDIDATES)
    print(f"[INFO] Loading malignant object: {main_h5ad}")
    adata = sc.read_h5ad(main_h5ad)

    # Base obs schema
    adata = attach_base_obs(adata)

    # ------------------------------------------------------------
    # Restrict Figure 2 to samples present in both ecology and branch tables
    # ------------------------------------------------------------
    dm = pd.read_csv(DESIGN_MATRIX_CSV).copy()
    br = pd.read_csv(COARSE_STATE_CSV).copy()

    dm["sample_id"] = dm["sample_id"].astype(str).str.strip()
    br["sample_id"] = br["sample_id"].astype(str).str.strip()
    adata.obs["sample_id"] = adata.obs["sample_id"].astype(str).str.strip()

    dm_samples = set(dm["sample_id"])
    br_samples = set(br["sample_id"])
    matched_samples = sorted(dm_samples & br_samples)

    obs_samples = set(adata.obs["sample_id"])
    excluded_samples = sorted(obs_samples - set(matched_samples))

    print(f"[INFO] Matched samples retained for Figure 2: {len(matched_samples)}")
    print(f"[INFO] Excluded samples not present in ecology/branch tables: {excluded_samples}")

    adata = adata[adata.obs["sample_id"].isin(matched_samples)].copy()

    print(f"[INFO] Cells retained after sample intersection filter: {adata.n_obs}")
    print(f"[INFO] Remaining unique samples: {adata.obs['sample_id'].nunique()}")

    # Freeze a single 2D coordinate system for all panels
    emb_key = choose_embedding(adata)
    print(f"[INFO] Using embedding: {emb_key}")

    # Attach exact ecology and branch/sample-level context
    adata = attach_ecology_exact(adata)
    adata = attach_branch_exact(adata)

    # Regulatory layer
    adata = attach_regulatory_exact_or_proxy(adata)

    # Export final obs table for debugging and manuscript reproducibility
    export_obs_table(adata, OUTDIR / "figure2_obs_export.csv.gz")

    # Prepare coordinates
    xy = np.asarray(adata.obsm["X_fig2"])[:, :2]

    # Figure layout: 3 panels on top, 2 below
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.titlesize": 12,
        "figure.titlesize": 16,
    })

    fig = plt.figure(figsize=(15, 13.8))
    
    gs = fig.add_gridspec(
        3, 3,
        wspace=0.14,
        hspace=0.14,
        height_ratios=[1.0, 1.0, 0.72]
    )
    
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[0, 2])
    axD = fig.add_subplot(gs[1, 0:2])
    axE = fig.add_subplot(gs[1, 2])
    axF = fig.add_subplot(gs[2, :])

    # A. Clinical timepoint
    plot_categorical(
        axA, xy, adata.obs["clinical_timepoint"],
        title="Clinical timepoint",
        panel_label="A",
        palette=palette_timepoint(),
        legend=True,
        s=5,
    )
    
    # B. Patient or disease subgroup
    plot_categorical(
        axB,
        xy,
        adata.obs["patient_id"],
        title="Patient / disease subgroup",
        panel_label="B",
        legend=False,
        s=5,
    )
    
    # C. Ecological / microenvironmental context
    plot_categorical(
        axC, xy, adata.obs["ecotype"],
        title="Ecological /\nmicroenvironmental context",
        panel_label="C",
        palette=palette_ecotype(adata.obs["ecotype"]),
        legend=True,
        s=5,
    )
    
    # D. Regulatory program activity or state
    reg_score = safe_series_to_numeric(adata.obs["reg_program_score"])
    if reg_score.notna().any():
        plot_continuous(
            axD, xy, reg_score,
            title="Regulatory program activity",
            panel_label="D",
            cmap="viridis",
            s=5,
        )
    else:
        plot_categorical(
            axD, xy, adata.obs["reg_state"],
            title="Regulatory / chromatin-derived state",
            panel_label="D",
            palette=palette_generic(adata.obs["reg_state"], cmap_name="Set3"),
            legend=True,
            s=5,
        )
    
    # E. Branch assignments or branch probabilities
    plot_categorical(
        axE, xy, adata.obs["branch_id"],
        title="Branch assignments /\ndivergence structure",
        panel_label="E",
        palette=palette_branch(),
        legend=True,
        s=5,
        cats=[
            "Mono/DC-like basin",
            "GMP-like basin",
            "HSC-like basin",
            "Progenitor-like basin",
        ],
    )
    
    # Very light uncertainty overlay for only the most uncertain cells
    mp = pd.to_numeric(adata.obs["branch_maxprob"], errors="coerce").to_numpy()
    sel = np.isfinite(mp) & (mp < 0.35)
    
    idx = np.where(sel)[0]
    rng = np.random.default_rng(42)
    if idx.size > 5000:
        idx = rng.choice(idx, size=5000, replace=False)
    
    if idx.size > 0:
        axE.scatter(
            xy[idx, 0], xy[idx, 1],
            s=3,
            c="#666666",
            alpha=0.08,
            linewidths=0,
            rasterized=True,
            zorder=3
        )

    # F. Integrated interpretation
    plot_panel_f_interpretation(axF)

    fig.subplots_adjust(top=0.96, bottom=0.04)
    png_path = OUTDIR / f"{FIG_BASENAME}.png"
    pdf_path = OUTDIR / f"{FIG_BASENAME}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"[DONE] Saved:\n  {png_path}\n  {pdf_path}")


if __name__ == "__main__":
    main()
