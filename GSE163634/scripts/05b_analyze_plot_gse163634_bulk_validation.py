from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from scipy.stats import wilcoxon, mannwhitneyu, rankdata
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False


PRIMARY_TRANSITION = "dx_to_r1"
EXPLORATORY_TRANSITION = "r1_to_r2"
DEFAULT_TOP_TRAJECTORY_AXES = [
    "ilr_stem_vs_committed",
    "log_aux_clp",
    "log_aux_erybaso",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze GSE163634 bulk validation outputs.")
    p.add_argument("--score-matrix", required=True, help="Path to gse163634_bulk_score_matrix.csv")
    p.add_argument("--serial-deltas", required=True, help="Path to gse163634_bulk_serial_deltas.csv")
    p.add_argument("--outdir", required=True, help="Output directory")
    p.add_argument("--bootstrap-iters", type=int, default=5000, help="Bootstrap iterations for CIs")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    return p.parse_args()


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def bh_fdr(pvals: List[float]) -> List[float]:
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    if n == 0:
        return []
    order = np.argsort(p)
    ranked = p[order]
    q = np.empty(n, dtype=float)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        val = ranked[i] * n / rank
        prev = min(prev, val)
        q[i] = prev
    out = np.empty(n, dtype=float)
    out[order] = np.clip(q, 0, 1)
    return out.tolist()


def detect_axes_from_serial(deltas: pd.DataFrame) -> List[str]:
    axes = []
    for col in deltas.columns:
        if col.startswith("delta_") and col.endswith("_cal"):
            axes.append(col[len("delta_"):-len("_cal")])
    return axes


def available_score_columns(scores: pd.DataFrame, axes: List[str]) -> Dict[str, str]:
    out = {}
    for axis in axes:
        cal = f"{axis}_cal"
        raw = f"{axis}_raw"
        if cal in scores.columns:
            out[axis] = cal
        elif raw in scores.columns:
            out[axis] = raw
    return out


def safe_auc_binary(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores).astype(float)
    if labels.min() == labels.max():
        return float("nan")
    ranks = rankdata(scores)
    n_pos = labels.sum()
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    rank_sum_pos = ranks[labels == 1].sum()
    auc = (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def bootstrap_ci(values: np.ndarray, stat_fn, n_boot: int, seed: int) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    vals = np.asarray(values, dtype=float)
    if len(vals) == 0:
        return (float("nan"), float("nan"))
    boots = []
    for _ in range(n_boot):
        sample = rng.choice(vals, size=len(vals), replace=True)
        boots.append(stat_fn(sample))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(lo), float(hi)


def hodges_lehmann_paired(diffs: np.ndarray) -> float:
    """One-sample/paired HL estimate using Walsh averages."""
    diffs = np.asarray(diffs, dtype=float)
    n = len(diffs)
    if n == 0:
        return float("nan")
    walsh = []
    for i in range(n):
        for j in range(i, n):
            walsh.append((diffs[i] + diffs[j]) / 2.0)
    return float(np.median(np.asarray(walsh)))


def paired_stats_table(
    deltas: pd.DataFrame,
    transition: str,
    axes: List[str],
    n_boot: int,
    seed: int,
) -> pd.DataFrame:
    sub = deltas.loc[deltas["transition"] == transition].copy()
    rows = []
    for i, axis in enumerate(axes):
        col = f"delta_{axis}_cal"
        if col not in sub.columns:
            continue
        vals = sub[col].dropna().to_numpy(dtype=float)
        if len(vals) == 0:
            continue
        median_delta = float(np.median(vals))
        hl = hodges_lehmann_paired(vals)
        ci_lo, ci_hi = bootstrap_ci(vals, np.median, n_boot=n_boot, seed=seed + i)

        if SCIPY_AVAILABLE:
            nz = vals[vals != 0]
            if len(nz) >= 1:
                try:
                    w = wilcoxon(vals, alternative="two-sided", zero_method="wilcox", mode="auto")
                    pval = float(w.pvalue)
                except Exception:
                    pval = float("nan")
            else:
                pval = float("nan")
        else:
            pval = float("nan")

        sign_ref = 1 if median_delta >= 0 else -1
        directional_consistency = float(np.mean(np.sign(vals) == sign_ref))
        rows.append(
            {
                "axis": axis,
                "transition": transition,
                "n_pairs": int(len(vals)),
                "median_delta": median_delta,
                "hodges_lehmann_delta": hl,
                "ci_lo": ci_lo,
                "ci_hi": ci_hi,
                "wilcoxon_p": pval,
                "directional_consistency": directional_consistency,
            }
        )

    out = pd.DataFrame(rows)
    if not out.empty:
        out["fdr_q"] = bh_fdr(out["wilcoxon_p"].fillna(1.0).tolist())
        out = out.sort_values(["transition", "median_delta"], ascending=[True, False], kind="stable")
    return out


def leukemia_vs_control_table(scores: pd.DataFrame, score_cols: Dict[str, str]) -> pd.DataFrame:
    if "is_leukemia" not in scores.columns:
        return pd.DataFrame()
    rows = []
    for axis, col in score_cols.items():
        sub = scores.loc[scores[col].notna() & scores["is_leukemia"].notna(), [col, "is_leukemia"]].copy()
        if sub.empty:
            continue
        vals = sub[col].to_numpy(dtype=float)
        labels = sub["is_leukemia"].astype(int).to_numpy()
        auc = safe_auc_binary(labels, vals)
        med_leuk = float(np.median(vals[labels == 1])) if np.any(labels == 1) else float("nan")
        med_ctrl = float(np.median(vals[labels == 0])) if np.any(labels == 0) else float("nan")
        delta_med = med_leuk - med_ctrl
        if SCIPY_AVAILABLE and np.any(labels == 1) and np.any(labels == 0):
            try:
                mwu = mannwhitneyu(vals[labels == 1], vals[labels == 0], alternative="two-sided")
                pval = float(mwu.pvalue)
            except Exception:
                pval = float("nan")
        else:
            pval = float("nan")
        rows.append(
            {
                "axis": axis,
                "score_column": col,
                "n_leukemia": int(np.sum(labels == 1)),
                "n_control": int(np.sum(labels == 0)),
                "median_leukemia": med_leuk,
                "median_control": med_ctrl,
                "median_difference": delta_med,
                "auroc_leukemia_vs_control": auc,
                "mannwhitney_p": pval,
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["fdr_q"] = bh_fdr(out["mannwhitney_p"].fillna(1.0).tolist())
        out = out.sort_values("auroc_leukemia_vs_control", ascending=False, kind="stable")
    return out


def axis_rankings(dx_stats: pd.DataFrame, lvsc: pd.DataFrame) -> pd.DataFrame:
    if dx_stats.empty:
        return pd.DataFrame()
    dx = dx_stats.copy()
    dx = dx.rename(columns={
        "median_delta": "dx_to_r1_median_delta",
        "fdr_q": "dx_to_r1_fdr_q",
        "directional_consistency": "dx_to_r1_directional_consistency",
    })
    keep_dx = [
        "axis",
        "n_pairs",
        "dx_to_r1_median_delta",
        "hodges_lehmann_delta",
        "ci_lo",
        "ci_hi",
        "wilcoxon_p",
        "dx_to_r1_fdr_q",
        "dx_to_r1_directional_consistency",
    ]
    out = dx[keep_dx].copy()
    if not lvsc.empty:
        out = out.merge(
            lvsc[["axis", "auroc_leukemia_vs_control", "median_difference"]],
            on="axis",
            how="left",
        )
    out["abs_dx_to_r1_median_delta"] = out["dx_to_r1_median_delta"].abs()
    out = out.sort_values(
        ["abs_dx_to_r1_median_delta", "dx_to_r1_directional_consistency"],
        ascending=[False, False],
        kind="stable",
    )
    return out


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------

def plot_cohort_map(ax, scores: pd.DataFrame) -> None:
    leuk = scores.loc[scores["is_leukemia"] == True].copy()
    stage_order = {"dx": 0, "r1": 1, "r2": 2}
    patients = sorted(leuk["patient_id"].dropna().astype(str).unique(), key=lambda x: (len(x), x))
    mat = np.full((len(patients), 3), np.nan)
    p_to_i = {p: i for i, p in enumerate(patients)}
    for _, row in leuk.iterrows():
        p = str(row["patient_id"])
        st = row["stage"]
        if st in stage_order:
            mat[p_to_i[p], stage_order[st]] = 1.0
    show_patients = patients
    if len(patients) > 40:
        # Show the first 40 in panel label space but keep matrix correct.
        show_patients = patients[:40]
        mat = mat[:40, :]
    from matplotlib.colors import ListedColormap
    
    cmap = ListedColormap(["#d9ecff"])   # light blue
    cmap.set_bad(color="white")
    ax.imshow(mat, aspect="auto", interpolation="nearest", cmap=cmap, vmin=1, vmax=1)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["Dx", "R1", "R2"])
    ax.set_yticks(range(len(show_patients)))
    ax.set_yticklabels(show_patients, fontsize=6)
    ax.set_title("A. Cohort map")
    ax.set_xlabel("Stage")
    ax.set_ylabel("Patient")


def plot_heatmap(ax, scores: pd.DataFrame, score_cols: Dict[str, str]) -> None:
    keep_axes = list(score_cols.keys())
    order_stage = {"dx": 0, "r1": 1, "r2": 2, "control_B": 3, "control_T": 4}
    plot_df = scores.copy()
    plot_df["_stage_ord"] = plot_df["stage"].map(order_stage).fillna(99)
    plot_df["_patient_sort"] = plot_df["patient_id"].fillna(plot_df["sample_id"]).astype(str)
    plot_df = plot_df.sort_values(["is_control", "_patient_sort", "_stage_ord", "sample_id"], kind="stable")

    mat = plot_df[[score_cols[a] for a in keep_axes]].to_numpy(dtype=float)
    # Column-wise z-score for visual comparability.
    mu = np.nanmean(mat, axis=0)
    sd = np.nanstd(mat, axis=0)
    sd[sd == 0] = 1.0
    z = (mat - mu) / sd

    im = ax.imshow(z.T, aspect="auto", interpolation="nearest", cmap="coolwarm")
    ax.set_yticks(range(len(keep_axes)))
    ax.set_yticklabels(keep_axes, fontsize=8)
    ax.set_xticks([])
    ax.set_title("B. Transferred score heatmap (z-scored)")
    ax.set_xlabel("Samples ordered by patient / stage")
    plt.colorbar(im, ax=ax, fraction=0.025, pad=0.01)


def plot_trajectories(ax, deltas: pd.DataFrame, axes: List[str]) -> None:
    sub = deltas.loc[deltas["transition"] == PRIMARY_TRANSITION].copy()
    if sub.empty:
        ax.text(0.5, 0.5, "No Dx→R1 pairs", ha="center", va="center")
        ax.set_axis_off()
        return

    chosen = [a for a in axes if a in sub.columns or f"delta_{a}_cal" in sub.columns]
    if not chosen:
        ax.text(0.5, 0.5, "No primary axes available", ha="center", va="center")
        ax.set_axis_off()
        return

    chosen = chosen[:3]

    label_map = {
        "ilr_stem_vs_committed": "Stem/Comm",
        "ilr_prog_vs_mature": "Prog/Mature",
        "ilr_gmp_vs_monodc": "GMP/MonoDC",
        "log_aux_clp": "Aux CLP",
        "log_aux_erybaso": "Aux Ery/Baso",
        "pc1": "PC1",
        "pc2": "PC2",
    }

    group_gap = 2.2
    pair_gap = 1.0
    x_offsets = np.arange(len(chosen)) * group_gap

    xticks = []
    xtlabs = []

    for i, axis in enumerate(chosen):
        x_dx = x_offsets[i]
        x_r1 = x_dx + pair_gap

        fcol = f"from_{axis}_cal"
        tcol = f"to_{axis}_cal"

        for _, row in sub.iterrows():
            ax.plot([x_dx, x_r1], [row[fcol], row[tcol]], linewidth=0.8, alpha=0.7)
            ax.scatter([x_dx, x_r1], [row[fcol], row[tcol]], s=12)

        med_from = sub[fcol].median()
        med_to = sub[tcol].median()
        ax.plot([x_dx, x_r1], [med_from, med_to], linewidth=2.5)

        xticks.extend([x_dx, x_r1])
        xtlabs.extend(["Dx", "R1"])

        short = label_map.get(axis, axis.replace("_", " "))
        ax.text(
            x_dx + pair_gap / 2,
            -0.08,
            short,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=11.5,
            clip_on=False,
        )

    ax.set_xticks(xticks)
    ax.set_xticklabels(xtlabs, fontsize=8)
    ax.tick_params(axis="x", pad=2)
    ax.margins(x=0.10)

    ax.set_title("C. Paired Dx→R1 trajectories")
    ax.set_xlabel("")   # remove Stage
    ax.set_ylabel("Transferred score")


def plot_forest(ax, dx_stats: pd.DataFrame) -> None:
    if dx_stats.empty:
        ax.text(0.5, 0.5, "No Dx→R1 statistics", ha="center", va="center")
        ax.set_axis_off()
        return
    df = dx_stats.copy().sort_values("median_delta", ascending=True, kind="stable")
    y = np.arange(len(df))
    x = df["median_delta"].to_numpy(dtype=float)
    lo = df["ci_lo"].to_numpy(dtype=float)
    hi = df["ci_hi"].to_numpy(dtype=float)
    ax.errorbar(x, y, xerr=[x - lo, hi - x], fmt='o', capsize=3)
    ax.axvline(0, linestyle='--', linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(df["axis"].tolist(), fontsize=8)
    ax.set_title("D. Dx→R1 paired effect sizes")
    ax.set_xlabel("Median delta (bootstrap 95% CI)")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    ensure_dir(outdir)

    scores = pd.read_csv(args.score_matrix)
    deltas = pd.read_csv(args.serial_deltas)

    axes = detect_axes_from_serial(deltas)
    if not axes:
        raise ValueError("Could not detect transfer axes from serial delta table.")

    score_cols = available_score_columns(scores, axes)
    dx_stats = paired_stats_table(deltas, PRIMARY_TRANSITION, axes, args.bootstrap_iters, args.seed)
    r2_stats = paired_stats_table(deltas, EXPLORATORY_TRANSITION, axes, max(1000, args.bootstrap_iters // 2), args.seed + 1000)
    lvsc = leukemia_vs_control_table(scores, score_cols)
    rankings = axis_rankings(dx_stats, lvsc)

    # Figure-ready table aliases
    heatmap_table = scores.copy()
    paired_plot_table = deltas.loc[deltas["transition"] == PRIMARY_TRANSITION].copy()
    forest_plot_table = dx_stats.copy()

    # Write tables
    score_cols_path = outdir / "gse163634_bulk_leukemia_vs_control_stats.csv"
    dx_stats_path = outdir / "gse163634_bulk_dx_to_r1_paired_stats.csv"
    r2_stats_path = outdir / "gse163634_bulk_r1_to_r2_paired_stats.csv"
    rankings_path = outdir / "gse163634_bulk_axis_transfer_rankings.csv"
    heatmap_path = outdir / "gse163634_bulk_heatmap_matrix.csv"
    paired_plot_path = outdir / "gse163634_bulk_paired_plot_table.csv"
    forest_path = outdir / "gse163634_bulk_forest_plot_table.csv"

    lvsc.to_csv(score_cols_path, index=False)
    dx_stats.to_csv(dx_stats_path, index=False)
    r2_stats.to_csv(r2_stats_path, index=False)
    rankings.to_csv(rankings_path, index=False)
    heatmap_table.to_csv(heatmap_path, index=False)
    paired_plot_table.to_csv(paired_plot_path, index=False)
    forest_plot_table.to_csv(forest_path, index=False)

    # Plot main figure
    fig = plt.figure(figsize=(17.5, 12), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.1], width_ratios=[1.0, 1.5])
    axA = fig.add_subplot(gs[:, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[1, 1])

    # forest as inset-like sub-axes replacing a separate panel in same bottom area? better use 2x2.
    fig2 = plt.figure(figsize=(17.5, 12), constrained_layout=True)
    gs2 = fig2.add_gridspec(2, 2, height_ratios=[1, 1.0], width_ratios=[1.0, 1.4])
    axA2 = fig2.add_subplot(gs2[:, 0])
    axB2 = fig2.add_subplot(gs2[0, 1])
    axC2 = fig2.add_subplot(gs2[1, 1])
   
    # final figure with four panels
    fig = plt.figure(figsize=(17.5, 12), constrained_layout=True)
    gs = fig.add_gridspec(
        2, 2,
        height_ratios=[1.0, 1.0],
        width_ratios=[1.0, 1.4]
    )
    
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[1, 0])
    axD = fig.add_subplot(gs[1, 1])

    plot_cohort_map(axA, scores)
    plot_heatmap(axB, scores, score_cols)
    trajectory_axes = [a for a in DEFAULT_TOP_TRAJECTORY_AXES if a in axes] + [a for a in axes if a not in DEFAULT_TOP_TRAJECTORY_AXES]
    plot_trajectories(axC, deltas, trajectory_axes)
    plot_forest(axD, dx_stats)
#    fig.suptitle("External serial bulk validation in GSE163634 (available transfer axes)")

    fig_png = outdir / "Figure_GSE163634_bulk_validation_main_7axis.png"
    fig_pdf = outdir / "Figure_GSE163634_bulk_validation_main_7axis.pdf"
    fig.savefig(fig_png, dpi=300)
    fig.savefig(fig_pdf)
    plt.close(fig)

    # Supplemental figures
    # Leukemia vs control distributions for available axes
    if not lvsc.empty and not scores.loc[scores["is_leukemia"] == False].empty:
        n_axes = len(score_cols)
        ncols = 2
        nrows = int(math.ceil(n_axes / ncols))
        fig = plt.figure(figsize=(12, max(4, 3.0 * nrows)), constrained_layout=True)
        gs = fig.add_gridspec(nrows, ncols)
        for i, axis in enumerate(score_cols.keys()):
            ax = fig.add_subplot(gs[i // ncols, i % ncols])
            col = score_cols[axis]
            leuk = scores.loc[scores["is_leukemia"] == True, col].dropna().to_numpy()
            ctrl = scores.loc[scores["is_leukemia"] == False, col].dropna().to_numpy()
            ax.boxplot([leuk, ctrl], tick_labels=["Leukemia", "Control"])
            ax.set_title(axis)
            ax.set_ylabel(col)
        fig.suptitle("Leukemia vs control distributions")
        sup1_png = outdir / "Figure_SX_GSE163634_leukemia_vs_control_7axis.png"
        sup1_pdf = outdir / "Figure_SX_GSE163634_leukemia_vs_control_7axis.pdf"
        fig.savefig(sup1_png, dpi=300)
        fig.savefig(sup1_pdf)
        plt.close(fig)

    if not r2_stats.empty:
        fig = plt.figure(figsize=(10, 6), constrained_layout=True)
        ax = fig.add_subplot(111)
        plot_forest(ax, r2_stats.rename(columns={"median_delta": "median_delta"}))
        ax.set_title("Exploratory R1→R2 paired effect sizes")
        sup2_png = outdir / "Figure_SX_GSE163634_r1_to_r2_7axis.png"
        sup2_pdf = outdir / "Figure_SX_GSE163634_r1_to_r2_7axis.pdf"
        fig.savefig(sup2_png, dpi=300)
        fig.savefig(sup2_pdf)
        plt.close(fig)

    manifest = {
        "score_matrix": str(Path(args.score_matrix).resolve()),
        "serial_deltas": str(Path(args.serial_deltas).resolve()),
        "available_axes": axes,
        "primary_transition": PRIMARY_TRANSITION,
        "exploratory_transition": EXPLORATORY_TRANSITION,
        "n_scores": int(len(scores)),
        "n_dx_to_r1_pairs": int((deltas["transition"] == PRIMARY_TRANSITION).sum()),
        "n_r1_to_r2_pairs": int((deltas["transition"] == EXPLORATORY_TRANSITION).sum()),
        "outputs": {
            "leukemia_vs_control_stats": str(score_cols_path.resolve()),
            "dx_to_r1_stats": str(dx_stats_path.resolve()),
            "r1_to_r2_stats": str(r2_stats_path.resolve()),
            "axis_rankings": str(rankings_path.resolve()),
            "heatmap_table": str(heatmap_path.resolve()),
            "paired_plot_table": str(paired_plot_path.resolve()),
            "forest_plot_table": str(forest_path.resolve()),
            "main_figure_png": str(fig_png.resolve()),
            "main_figure_pdf": str(fig_pdf.resolve()),
        },
    }
    with open(outdir / "gse163634_bulk_validation_manifest.json", "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    print("[OK] GSE163634 bulk validation analysis complete.")
    print(f"Output: {outdir}")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
