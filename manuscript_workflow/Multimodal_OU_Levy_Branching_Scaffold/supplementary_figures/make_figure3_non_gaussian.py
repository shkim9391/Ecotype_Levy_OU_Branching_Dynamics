from __future__ import annotations

import argparse
import math
from pathlib import Path
from statistics import NormalDist
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ----------------------------- configuration -------------------------------- #

COLOR_STABLE = "#4C9ED9"
COLOR_SWITCHING = "#F28E2B"
COLOR_TOTAL = "#2F2F2F"
COLOR_EFFECT = "#B22222"
COLOR_REF = "#7A7A7A"

GROUP_DISPLAY = {
    "Stable": "Branch-continuous",
    "Switching": "Branch-switching",
}

PANEL_FONT_SIZE = 18
TITLE_FONT_SIZE = 15
LABEL_FONT_SIZE = 12
TICK_FONT_SIZE = 11
ANNOT_FONT_SIZE = 10

DEFAULT_FIGSIZE = (15.5, 10.0)
DEFAULT_DPI = 600
DEFAULT_BOOTSTRAPS = 10000
DEFAULT_TOP_ANNOTATE = 5
DEFAULT_TOP_JUMP_CANDIDATES = 12
EPS = 1e-12


ALIASES: Dict[str, List[str]] = {
    "sample": [
        "sample", "sampleid", "sample_id",
        "patient", "patientid", "patient_id", "Patient_ID",
        "case", "caseid", "case_id",
        "participant", "participantid", "participant_id",
        "pair", "pair_id", "patient_pair", "sample_pair",
    ],
    "dx_branch": [
        "DX_branch_ge50",
        "dxbranch", "dx_branch", "diagnosisbranch", "diagnosis_branch",
        "branchdx", "branch_dx",
    ],
    "rel_branch": [
        "REL_branch_ge50",
        "relbranch", "rel_branch", "relapsebranch", "relapse_branch",
        "branchrel", "branch_rel",
    ],
    "total_disp": [
        "disp_total_6d",
        "disptotal6d", "totaldisplacement", "total_displacement",
        "dxreldisplacement", "dx_rel_displacement",
        "disp6dtotal", "disp_total",
        "delta_total", "deltatotal", "totaldisp", "total_disp",
    ],
    "malignant_disp": [
        "disp_malignant_3d",
        "disp_malignant_6d",
        "dispmalignant3d", "dispmalignant6d",
        "malignantdisplacement", "malignant_displacement",
        "dxrelmalignantdisplacement", "dx_rel_malignant_displacement",
        "disp_malignant", "malignantdisp",
        "delta_malignant", "deltamalignant",
    ],
    "tme_disp": [
        "disp_tme_3d",
        "disp_tme_6d",
        "disptme3d", "disptme6d",
        "tmedisplacement", "tme_displacement",
        "dxreltmedisplacement", "dx_rel_tme_displacement",
        "disp_tme", "tmedisp",
        "delta_tme", "deltatme",
    ],
    "stability": [
        "dx_to_rel_switch",
        "group",
        "branchstability", "branch_stability", "stability",
        "stable_switching", "switchingstatus", "switching_status",
        "is_switching", "switching",
    ],
    "jump_tier": [
        "jumpcandidatetier", "jump_candidate_tier", "candidate_tier", "tier",
    ],
}


# -------------------------------- utilities --------------------------------- #

def canonicalize(name: str) -> str:
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def load_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(path, sep=None, engine="python")
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t")
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported input extension: {suffix}")


def resolve_column(df: pd.DataFrame, key: str, required: bool = True) -> Optional[str]:
    canon_to_original = {canonicalize(col): col for col in df.columns}
    candidates = [canonicalize(x) for x in ALIASES.get(key, [])]

    for candidate in candidates:
        if candidate in canon_to_original:
            return canon_to_original[candidate]

    # fallback: substring search
    for candidate in candidates:
        for canon_col, original_col in canon_to_original.items():
            if candidate in canon_col:
                return original_col

    if required:
        raise KeyError(
            f"Could not resolve required column for '{key}'. "
            f"Available columns: {list(df.columns)}"
        )
    return None


def standardize_stability(value: object) -> Optional[str]:
    if pd.isna(value):
        return np.nan

    s = str(value).strip().lower()
    if s in {"", "nan", "none", "null"}:
        return np.nan

    # textual matches
    if "stable" in s or "same" in s or "no_switch" in s:
        return "Stable"
    if "switch" in s or "chang" in s or "different" in s:
        return "Switching"

    # exact simple values
    if s in {"0", "false", "no"}:
        return "Stable"
    if s in {"1", "true", "yes"}:
        return "Switching"

    # numeric fallback
    try:
        x = float(s)
        return "Switching" if x != 0 else "Stable"
    except ValueError:
        return np.nan


def bootstrap_median_diff(
    switching: np.ndarray,
    stable: np.ndarray,
    n_boot: int,
    rng: np.random.Generator,
) -> Tuple[float, float, float]:
    """Return median(switching) - median(stable) and percentile CI."""
    switching = np.asarray(switching, dtype=float)
    stable = np.asarray(stable, dtype=float)

    point_est = float(np.median(switching) - np.median(stable))

    boot = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        s1 = rng.choice(switching, size=len(switching), replace=True)
        s0 = rng.choice(stable, size=len(stable), replace=True)
        boot[i] = np.median(s1) - np.median(s0)

    lo, hi = np.percentile(boot, [2.5, 97.5])
    return point_est, float(lo), float(hi)


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """
    Cliff's delta for x vs y.
    Positive => x tends to be larger than y.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    gt = 0
    lt = 0
    for xi in x:
        gt += np.sum(xi > y)
        lt += np.sum(xi < y)
    n = len(x) * len(y)
    if n == 0:
        return np.nan
    return (gt - lt) / n


def robust_z_scores(values: np.ndarray, baseline: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """
    Robust z-scores relative to baseline group:
        z = (x - median(baseline)) / (1.4826 * MAD(baseline))
    Falls back to sd if MAD is ~0.
    """
    values = np.asarray(values, dtype=float)
    baseline = np.asarray(baseline, dtype=float)

    med = float(np.median(baseline))
    mad = float(np.median(np.abs(baseline - med)))
    scale = 1.4826 * mad

    if not np.isfinite(scale) or scale < EPS:
        sd = float(np.std(baseline, ddof=1)) if len(baseline) > 1 else 1.0
        scale = sd if sd > EPS else 1.0

    z = (values - med) / scale
    return z, med, scale


def gaussian_quantiles(n: int) -> np.ndarray:
    nd = NormalDist()
    probs = (np.arange(1, n + 1) - 0.5) / n
    return np.array([nd.inv_cdf(float(p)) for p in probs], dtype=float)


def gaussian_surprisal_from_z(z: np.ndarray) -> np.ndarray:
    """-log10(one-sided Gaussian survival probability)."""
    z = np.asarray(z, dtype=float)
    sf = 0.5 * np.array([math.erfc(float(v) / math.sqrt(2.0)) for v in z], dtype=float)
    sf = np.maximum(sf, 1e-300)
    return -np.log10(sf)


def annotate_points(
    ax: plt.Axes,
    x: Sequence[float],
    y: Sequence[float],
    labels: Sequence[str],
    dx: float = 5.0,
    dy: float = 5.0,
) -> None:
    for xi, yi, label in zip(x, y, labels):
        ax.annotate(
            label,
            xy=(xi, yi),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=ANNOT_FONT_SIZE,
            ha="left",
            va="bottom",
        )


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.12, 1.06, label,
        transform=ax.transAxes,
        fontsize=PANEL_FONT_SIZE,
        fontweight="bold",
        va="top",
        ha="left",
    )


def prepare_dataframe(df: pd.DataFrame, subset_query: Optional[str]) -> pd.DataFrame:
    sample_col = resolve_column(df, "sample", required=True)
    dx_col = resolve_column(df, "dx_branch", required=True)
    rel_col = resolve_column(df, "rel_branch", required=True)
    total_col = resolve_column(df, "total_disp", required=True)
    malignant_col = resolve_column(df, "malignant_disp", required=True)
    tme_col = resolve_column(df, "tme_disp", required=True)
    stability_col = resolve_column(df, "stability", required=False)
    tier_col = resolve_column(df, "jump_tier", required=False)

    out = df.copy()
    out["sample_std"] = out[sample_col].astype(str).str.strip()
    out["dx_branch_std"] = out[dx_col].astype(str).str.strip()
    out["rel_branch_std"] = out[rel_col].astype(str).str.strip()
    out["total_disp_std"] = pd.to_numeric(out[total_col], errors="coerce")
    out["malignant_disp_std"] = pd.to_numeric(out[malignant_col], errors="coerce")
    out["tme_disp_std"] = pd.to_numeric(out[tme_col], errors="coerce")
    out["transition_std"] = out["dx_branch_std"] + "→" + out["rel_branch_std"]

    if stability_col is not None:
        out["stability_std"] = out[stability_col].map(standardize_stability)
    else:
        out["stability_std"] = np.nan

    # fallback from explicit branch comparison for any unresolved rows
    branch_fallback = np.where(
        out["dx_branch_std"] == out["rel_branch_std"],
        "Stable",
        "Switching",
    )
    out["stability_std"] = out["stability_std"].fillna(pd.Series(branch_fallback, index=out.index))

    if tier_col is not None:
        out["jump_tier_std"] = out[tier_col].astype(str).fillna("").str.strip()
    else:
        out["jump_tier_std"] = ""

    if subset_query:
        out = out.query(subset_query).copy()

    required_cols = [
        "sample_std",
        "dx_branch_std",
        "rel_branch_std",
        "total_disp_std",
        "malignant_disp_std",
        "tme_disp_std",
        "stability_std",
    ]
    out = out.dropna(subset=required_cols).copy()

    # normalize capitalization one more time
    out["stability_std"] = out["stability_std"].astype(str).str.strip().str.title()

    # keep only valid groups
    out = out[out["stability_std"].isin(["Stable", "Switching"])].copy()

    print("Resolved columns:")
    print({
        "sample": sample_col,
        "dx_branch": dx_col,
        "rel_branch": rel_col,
        "total_disp": total_col,
        "malignant_disp": malignant_col,
        "tme_disp": tme_col,
        "stability": stability_col,
        "jump_tier": tier_col,
    })
    print("Rows after cleaning:", len(out))
    print("Stability counts:")
    print(out["stability_std"].value_counts(dropna=False).to_dict())

    if out.empty:
        raise ValueError("No rows remain after filtering / column resolution.")

    n_stable = (out["stability_std"] == "Stable").sum()
    n_switch = (out["stability_std"] == "Switching").sum()
    if n_stable < 2 or n_switch < 2:
        raise ValueError(
            f"Need at least 2 Stable and 2 Switching rows. "
            f"Found Stable={n_stable}, Switching={n_switch}."
        )

    return out


# -------------------------------- plotting ---------------------------------- #

def plot_panel_a_ranked_displacement(
    ax: plt.Axes,
    df: pd.DataFrame,
    top_annotate: int,
) -> None:
    ranked = df.sort_values("total_disp_std", ascending=True).reset_index(drop=True)
    ranked["rank"] = np.arange(1, len(ranked) + 1)

    # background line
    ax.plot(
        ranked["rank"],
        ranked["total_disp_std"],
        color=COLOR_REF,
        linewidth=1.2,
        alpha=0.7,
        zorder=1,
    )

    for stability, color in [("Stable", COLOR_STABLE), ("Switching", COLOR_SWITCHING)]:
        sub = ranked[ranked["stability_std"] == stability]
        ax.scatter(
            sub["rank"],
            sub["total_disp_std"],
            s=50,
            color=color,
            edgecolor="white",
            linewidth=0.6,
            label=GROUP_DISPLAY.get(stability, stability),
            zorder=3,
        )

    stable_values = df.loc[df["stability_std"] == "Stable", "total_disp_std"].to_numpy(dtype=float)
    stable_q95 = float(np.quantile(stable_values, 0.95))
    ax.axhline(
        stable_q95,
        color=COLOR_REF,
        linestyle="--",
        linewidth=1.5,
        alpha=0.9,
        label="Branch-continuous 95th percentile",
    )

    top = ranked.nlargest(top_annotate, "total_disp_std").sort_values("total_disp_std", ascending=False)
    annotate_points(
        ax,
        top["rank"].tolist(),
        top["total_disp_std"].tolist(),
        top["sample_std"].tolist(),
        dx=6,
        dy=6,
    )

    ax.set_title("Ranked total displacement reveals an upper-tail excess", fontsize=TITLE_FONT_SIZE)
    ax.set_xlabel("Sample rank", fontsize=LABEL_FONT_SIZE)
    ax.set_ylabel("DX→REL total displacement (6D)", fontsize=LABEL_FONT_SIZE)
    ax.tick_params(labelsize=TICK_FONT_SIZE)
    ax.legend(frameon=False, fontsize=10, loc="upper left")


def plot_panel_b_qq(
    ax: plt.Axes,
    df: pd.DataFrame,
    top_annotate: int,
) -> None:
    ordered = df.sort_values("z_total_std", ascending=True).reset_index(drop=True)
    ordered["theoretical_q"] = gaussian_quantiles(len(ordered))

    theo = ordered["theoretical_q"].to_numpy(dtype=float)
    obs = ordered["z_total_std"].to_numpy(dtype=float)

    x_min = float(theo.min()) - 0.3
    x_max = float(theo.max()) + 0.3
    y_min = min(float(obs.min()), x_min) - 0.3
    y_max = max(float(obs.max()), x_max) + 0.3

    colors = ordered["stability_std"].map(
        {"Stable": COLOR_STABLE, "Switching": COLOR_SWITCHING}
    ).tolist()

    ax.scatter(
        theo,
        obs,
        s=52,
        c=colors,
        edgecolor="white",
        linewidth=0.6,
        zorder=3,
    )

    ax.plot(
        [x_min, x_max],
        [x_min, x_max],
        linestyle="--",
        color=COLOR_REF,
        linewidth=1.5,
        zorder=1,
    )

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    top = ordered.nlargest(top_annotate, "z_total_std").sort_values(
        "z_total_std", ascending=False
    )
    annotate_points(
        ax,
        top["theoretical_q"].tolist(),
        top["z_total_std"].tolist(),
        top["sample_std"].tolist(),
        dx=6,
        dy=6,
    )

    ax.set_title("Upper-tail departure from Gaussian expectation", fontsize=TITLE_FONT_SIZE)
    ax.set_xlabel("Theoretical normal quantile", fontsize=LABEL_FONT_SIZE)
    ax.set_ylabel("Observed robust z-score\n(vs branch-continuous baseline)", fontsize=LABEL_FONT_SIZE)
    ax.tick_params(labelsize=TICK_FONT_SIZE)

    legend_elems = [
        plt.Line2D(
            [0], [0], marker="o", color="none",
            markerfacecolor=COLOR_STABLE, markeredgecolor="white",
            markersize=8, label="Branch-continuous"
        ),
        plt.Line2D(
            [0], [0], marker="o", color="none",
            markerfacecolor=COLOR_SWITCHING, markeredgecolor="white",
            markersize=8, label="Branch-switching"
        ),
        plt.Line2D(
            [0], [0], linestyle="--", color=COLOR_REF,
            label="Gaussian reference"
        ),
    ]
    ax.legend(handles=legend_elems, frameon=False, fontsize=10, loc="upper left")


def plot_panel_c_effect_sizes(
    ax: plt.Axes,
    df: pd.DataFrame,
    n_boot: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    metrics = [
        ("Total", "total_disp_std"),
        ("Malignant", "malignant_disp_std"),
        ("TME", "tme_disp_std"),
    ]

    stable_df = df[df["stability_std"] == "Stable"]
    switch_df = df[df["stability_std"] == "Switching"]

    rows = []
    y_positions = np.arange(len(metrics))[::-1]  # 2,1,0

    for ypos, (label, col) in zip(y_positions, metrics):
        stable_vals = stable_df[col].to_numpy(dtype=float)
        switch_vals = switch_df[col].to_numpy(dtype=float)

        effect, lo, hi = bootstrap_median_diff(switch_vals, stable_vals, n_boot=n_boot, rng=rng)
        delta = cliffs_delta(switch_vals, stable_vals)

        rows.append(
            {
                "metric": label,
                "column": col,
                "median_diff_switching_minus_stable": effect,
                "ci_lo": lo,
                "ci_hi": hi,
                "cliffs_delta": delta,
                "n_stable": len(stable_vals),
                "n_switching": len(switch_vals),
            }
        )

        ax.hlines(y=ypos, xmin=lo, xmax=hi, color=COLOR_REF, linewidth=2.4, zorder=1)
        ax.scatter(effect, ypos, s=85, color=COLOR_EFFECT, edgecolor="white", linewidth=0.8, zorder=3)

    effect_df = pd.DataFrame(rows)

    x_min = min(0.0, float(effect_df["ci_lo"].min())) - 0.2
    x_max = float(effect_df["ci_hi"].max()) + 0.8

    ax.axvline(0.0, color=COLOR_REF, linestyle="--", linewidth=1.4, zorder=0)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(-0.6, len(metrics) - 0.4)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([m[0] for m in metrics], fontsize=TICK_FONT_SIZE)
    ax.tick_params(axis="x", labelsize=TICK_FONT_SIZE)
    ax.set_xlabel("Median difference (Branch-switching − Branch-continuous)", fontsize=LABEL_FONT_SIZE)
    ax.set_title("Branch-switching intervals show larger displacement\nby robust effect size", fontsize=TITLE_FONT_SIZE)

    # annotate Cliff's delta on the right
    x_text = x_max - 0.01 * (x_max - x_min)
    for ypos, (_, row) in zip(y_positions, effect_df.iterrows()):
        ax.text(
            x_text,
            ypos,
            f"Cliff's δ={row['cliffs_delta']:.2f}",
            fontsize=9,
            ha="right",
            va="center",
        )

    ax.text(
        0.98, 0.03,
        f"Branch-continuous n={int(effect_df['n_stable'].iloc[0])} | Branch-switching n={int(effect_df['n_switching'].iloc[0])}",
        transform=ax.transAxes,
        fontsize=10,
        ha="right",
        va="bottom",
    )

    return effect_df


def plot_panel_d_jump_candidates(
    ax: plt.Axes,
    df: pd.DataFrame,
    top_k: int,
) -> pd.DataFrame:
    ranked = df.sort_values("jump_score_std", ascending=False).reset_index(drop=True)
    ranked = ranked.head(min(top_k, len(ranked))).copy()
    ranked = ranked.iloc[::-1].reset_index(drop=True)  # reverse for top at top in barh-like plot

    y = np.arange(len(ranked))
    colors = ranked["stability_std"].map(
        {"Stable": COLOR_STABLE, "Switching": COLOR_SWITCHING}
    ).tolist()

    for yi, score, color in zip(y, ranked["jump_score_std"], colors):
        ax.hlines(yi, 0.0, score, color=COLOR_REF, linewidth=1.6, zorder=1)
        ax.scatter(score, yi, s=70, color=color, edgecolor="white", linewidth=0.7, zorder=3)

    labels = ranked["sample_std"].tolist()
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=TICK_FONT_SIZE)
    ax.tick_params(axis="x", labelsize=TICK_FONT_SIZE)

    # annotate transition and tier at right
    x_max = float(ranked["jump_score_std"].max()) if len(ranked) else 1.0
    right_pad = max(3.0, 0.1 * x_max)   # extra room for right-side labels
    x_text = x_max + 0.6
    
    ax.set_xlim(0.0, x_max + right_pad)
    
    for yi, (_, row) in zip(y, ranked.iterrows()):
        meta = row["transition_std"]
        tier = str(row["jump_tier_std"]).strip()
        if tier and tier.lower() != "nan":
            meta = f"{meta} | {tier}"
        ax.text(
            x_text,
            yi,
            meta,
            fontsize=9,
            ha="left",
            va="center",
            clip_on=False,
        )

    ax.set_title(
        "Jump-candidate ranking identifies a small extreme subset",
        fontsize=TITLE_FONT_SIZE
    )
    ax.set_xlabel(
        r"Jump score  $-\log_{10}\{1-\Phi(z)\}$",
        fontsize=LABEL_FONT_SIZE
    )
    
    legend_elems = [
        plt.Line2D(
            [0], [0],
            marker="o",
            color="none",
            markerfacecolor=COLOR_STABLE,
            markeredgecolor="white",
            markersize=8,
            label="Branch-continuous",
        ),
        plt.Line2D(
            [0], [0],
            marker="o",
            color="none",
            markerfacecolor=COLOR_SWITCHING,
            markeredgecolor="white",
            markersize=8,
            label="Branch-switching",
        ),
    ]
    
    ax.legend(
        handles=legend_elems,
        frameon=False,
        fontsize=10,
        loc="lower center"
    )
    
    return ranked.iloc[::-1].reset_index(drop=True)  # return descending


def build_figure(
    df: pd.DataFrame,
    output_prefix: Path,
    top_annotate: int,
    top_jump_candidates: int,
    n_boot: int,
    seed: int,
    figsize: Tuple[float, float],
    dpi: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # robust z relative to stable baseline
    stable_total = df.loc[df["stability_std"] == "Stable", "total_disp_std"].to_numpy(dtype=float)
    z_total, baseline_median, baseline_scale = robust_z_scores(
        df["total_disp_std"].to_numpy(dtype=float),
        stable_total,
    )
    df = df.copy()
    df["z_total_std"] = z_total
    df["jump_score_std"] = gaussian_surprisal_from_z(df["z_total_std"].to_numpy(dtype=float))

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    ax_a, ax_b, ax_c, ax_d = axes.flatten()

    plot_panel_a_ranked_displacement(ax_a, df, top_annotate=top_annotate)
    plot_panel_b_qq(ax_b, df, top_annotate=top_annotate)
    effect_df = plot_panel_c_effect_sizes(ax_c, df, n_boot=n_boot, seed=seed)
    jump_df = plot_panel_d_jump_candidates(ax_d, df, top_k=top_jump_candidates)

    for label, ax in zip(["A", "B", "C", "D"], [ax_a, ax_b, ax_c, ax_d]):
        panel_label(ax, label)

#    fig.suptitle(
#        "Evidence for non-Gaussian / jump-like relapse displacement",
#        fontsize=17,
#        y=0.98,
#    )

    fig.text(
        0.01,
        0.01,
        (
            f"Robust baseline for z-scores: median(branch-continuous total)={baseline_median:.3f}, "
            f"scale={baseline_scale:.3f} (MAD-based when available)"
        ),
        fontsize=9,
        ha="left",
        va="bottom",
    )

    fig.tight_layout(rect=[0, 0.03, 1, 0.96])

    png_path = output_prefix.with_suffix(".png")
    pdf_path = output_prefix.with_suffix(".pdf")

    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    return effect_df, jump_df


# ---------------------------------- main ------------------------------------ #

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Figure 3: evidence for non-Gaussian / jump-like behavior."
    )
    parser.add_argument("--input", required=True, help="Path to per-sample summary table.")
    parser.add_argument(
        "--output-prefix",
        required=True,
        help="Output prefix for figure and summary tables (without extension).",
    )
    parser.add_argument(
        "--subset-query",
        default=None,
        help=(
            "Optional pandas query string for subsetting before plotting, e.g. "
            "'min_cells >= 50' or 'pass_threshold_50 == 1'."
        ),
    )
    parser.add_argument("--top-annotate", type=int, default=DEFAULT_TOP_ANNOTATE)
    parser.add_argument("--top-jump-candidates", type=int, default=DEFAULT_TOP_JUMP_CANDIDATES)
    parser.add_argument("--bootstraps", type=int, default=DEFAULT_BOOTSTRAPS)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    parser.add_argument("--width", type=float, default=DEFAULT_FIGSIZE[0])
    parser.add_argument("--height", type=float, default=DEFAULT_FIGSIZE[1])
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    df_raw = load_table(input_path)
    df = prepare_dataframe(df_raw, subset_query=args.subset_query)

    effect_df, jump_df = build_figure(
        df=df,
        output_prefix=output_prefix,
        top_annotate=args.top_annotate,
        top_jump_candidates=args.top_jump_candidates,
        n_boot=args.bootstraps,
        seed=args.seed,
        figsize=(args.width, args.height),
        dpi=args.dpi,
    )

    effect_path = output_prefix.parent / f"{output_prefix.name}_effect_sizes.csv"
    jump_path = output_prefix.parent / f"{output_prefix.name}_jump_candidates.csv"
    effect_df.to_csv(effect_path, index=False)
    jump_df.to_csv(jump_path, index=False)

    print(f"saved: {output_prefix.with_suffix('.png')}")
    print(f"saved: {output_prefix.with_suffix('.pdf')}")
    print(f"saved: {effect_path}")
    print(f"saved: {jump_path}")


if __name__ == "__main__":
    main()
