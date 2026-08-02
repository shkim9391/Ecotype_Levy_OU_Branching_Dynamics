from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu


# ============================================================
# 1. CONFIG
# ============================================================
PROJECT_DIR = Path("/Multimodal_OU_Lévy_Branching_Scaffold/Figure_3")
DERIVED_DIR = PROJECT_DIR / "derived"
PANELS_DIR = PROJECT_DIR / "panels"

IN_MAIN = DERIVED_DIR / "patient_interval_metrics_main.csv"

OUT_PNG = PANELS_DIR / "Figure3C_displacement_distributions.png"
OUT_PDF = PANELS_DIR / "Figure3C_displacement_distributions.pdf"
OUT_TSV = DERIVED_DIR / "figure3C_dx_rel_stats.tsv"

GROUP_ORDER = ["Branch-continuous", "Branch-switching"]
GROUP_COLORS = {
    "Branch-continuous": "#7A8DA8",
    "Branch-switching": "#C97979",
}
TAIL_COLOR = "#2F2F2F"


# ============================================================
# 2. HELPERS
# ============================================================
def ensure_dirs() -> None:
    PANELS_DIR.mkdir(parents=True, exist_ok=True)


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    gt = 0
    lt = 0
    for xi in x:
        gt += np.sum(xi > y)
        lt += np.sum(xi < y)
    n = len(x) * len(y)
    return (gt - lt) / n if n > 0 else np.nan


def style_axis(ax, panel_label: str, title: str,
               panel_fontsize: int = 18,
               title_fontsize: int = 12,
               title_x: float = 0.10,
               title_ha: str = "left") -> None:
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)

    ax.text(
        0.00, 1.03, panel_label,
        transform=ax.transAxes,
        fontsize=panel_fontsize,
        fontweight="bold",
        ha="left", va="bottom"
    )
    ax.text(
        title_x, 1.03, title,
        transform=ax.transAxes,
        fontsize=title_fontsize,
        fontweight="bold",
        ha=title_ha, va="bottom"
    )


def add_summary_stats(df_plot: pd.DataFrame, tail_threshold: float) -> pd.DataFrame:
    rows = []

    for g in GROUP_ORDER:
        sub = df_plot[df_plot["branch_group"] == g].copy()
        vals = pd.to_numeric(sub["displacement_hd"], errors="coerce").dropna().to_numpy()

        rows.append({
            "group": g,
            "n": int(len(vals)),
            "median": float(np.nanmedian(vals)) if len(vals) else np.nan,
            "q25": float(np.nanquantile(vals, 0.25)) if len(vals) else np.nan,
            "q75": float(np.nanquantile(vals, 0.75)) if len(vals) else np.nan,
            "mean": float(np.nanmean(vals)) if len(vals) else np.nan,
            "std": float(np.nanstd(vals, ddof=1)) if len(vals) > 1 else np.nan,
            "tail_fraction_q90_all_dx_rel": float(np.mean(vals > tail_threshold)) if len(vals) else np.nan,
        })

    # between-group comparison if possible
    g0 = pd.to_numeric(
        df_plot.loc[df_plot["branch_group"] == "Branch-continuous", "displacement_hd"],
        errors="coerce"
    ).dropna().to_numpy()

    g1 = pd.to_numeric(
        df_plot.loc[df_plot["branch_group"] == "Branch-switching", "displacement_hd"],
        errors="coerce"
    ).dropna().to_numpy()

    if len(g0) >= 2 and len(g1) >= 2:
        u, p = mannwhitneyu(g0, g1, alternative="two-sided")
        cd = cliffs_delta(g1, g0)
        compare = pd.DataFrame([{
            "group": "comparison_switching_vs_continuous",
            "n": np.nan,
            "median": np.nan,
            "q25": np.nan,
            "q75": np.nan,
            "mean": np.nan,
            "std": np.nan,
            "tail_fraction_q90_all_dx_rel": np.nan,
            "mannwhitney_u": float(u),
            "mannwhitney_p": float(p),
            "cliffs_delta_switching_vs_continuous": float(cd),
            "tail_threshold_q90_all_dx_rel": float(tail_threshold),
        }])
    else:
        compare = pd.DataFrame([{
            "group": "comparison_switching_vs_continuous",
            "n": np.nan,
            "median": np.nan,
            "q25": np.nan,
            "q75": np.nan,
            "mean": np.nan,
            "std": np.nan,
            "tail_fraction_q90_all_dx_rel": np.nan,
            "mannwhitney_u": np.nan,
            "mannwhitney_p": np.nan,
            "cliffs_delta_switching_vs_continuous": np.nan,
            "tail_threshold_q90_all_dx_rel": float(tail_threshold),
        }])

    out = pd.concat([pd.DataFrame(rows), compare], ignore_index=True)
    return out


# ============================================================
# 3. MAIN
# ============================================================
def main() -> None:
    ensure_dirs()

    df = pd.read_csv(IN_MAIN)
    df = df[df["interval_class"] == "DX_to_REL"].copy()

    if df.empty:
        raise ValueError("No DX_to_REL intervals found in patient_interval_metrics_main.csv")

    df["displacement_hd"] = pd.to_numeric(df["displacement_hd"], errors="coerce")
    df = df[df["displacement_hd"].notna()].copy()

    df["branch_switch"] = pd.to_numeric(df["branch_switch"], errors="coerce").fillna(0).astype(int)
    df["branch_group"] = np.where(
        df["branch_switch"] == 1,
        "Branch-switching",
        "Branch-continuous"
    )

    tail_threshold = float(np.nanquantile(df["displacement_hd"], 0.90))
    df["tail_flag"] = df["displacement_hd"] >= tail_threshold

    # Save stats table
    stats = add_summary_stats(df, tail_threshold)
    stats.to_csv(OUT_TSV, sep="\t", index=False)

    # ---------------------------------------------------------
    # Plot
    # ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.8, 7.8))

    rng = np.random.default_rng(42)

    available_groups = [g for g in GROUP_ORDER if g in set(df["branch_group"])]
    pos_map = {g: i + 1 for i, g in enumerate(available_groups)}

    # Violin layer
    violin_data = [
        df.loc[df["branch_group"] == g, "displacement_hd"].to_numpy(dtype=float)
        for g in available_groups
    ]
    if all(len(v) > 0 for v in violin_data):
        vp = ax.violinplot(
            violin_data,
            positions=[pos_map[g] for g in available_groups],
            showmeans=False,
            showmedians=False,
            showextrema=False,
            widths=0.85,
        )
        for body, g in zip(vp["bodies"], available_groups):
            body.set_facecolor(GROUP_COLORS[g])
            body.set_edgecolor("none")
            body.set_alpha(0.18)

    # Box layer
    box_data = [
        df.loc[df["branch_group"] == g, "displacement_hd"].to_numpy(dtype=float)
        for g in available_groups
    ]
    bp = ax.boxplot(
        box_data,
        positions=[pos_map[g] for g in available_groups],
        widths=0.28,
        patch_artist=True,
        showfliers=False,
    )
    for patch, g in zip(bp["boxes"], available_groups):
        patch.set_facecolor(GROUP_COLORS[g])
        patch.set_alpha(0.45)
        patch.set_edgecolor("#333333")
    for key in ["whiskers", "caps", "medians"]:
        for line in bp[key]:
            line.set_color("#333333")

    # Jitter layer
    for g in available_groups:
        sub = df[df["branch_group"] == g].copy()
        x = rng.normal(loc=pos_map[g], scale=0.06, size=len(sub))

        non_tail = sub[~sub["tail_flag"]]
        tail = sub[sub["tail_flag"]]

        # non-tail points
        if len(non_tail) > 0:
            x_nt = x[~sub["tail_flag"].to_numpy()]
            ax.scatter(
                x_nt,
                non_tail["displacement_hd"],
                s=34,
                c=GROUP_COLORS[g],
                alpha=0.80,
                linewidths=0,
                zorder=3,
            )

        # tail points highlighted
        if len(tail) > 0:
            x_t = x[sub["tail_flag"].to_numpy()]
            ax.scatter(
                x_t,
                tail["displacement_hd"],
                s=42,
                c=TAIL_COLOR,
                alpha=0.90,
                edgecolors="white",
                linewidths=0.4,
                zorder=4,
            )

    # Tail threshold line
    ax.axhline(
        tail_threshold,
        color="#555555",
        linestyle="--",
        linewidth=1.2,
        alpha=0.9,
        zorder=1,
    )
    ax.text(
        0.98, tail_threshold + 0.01,
        "DX→REL q90",
        fontsize=9.0,
        color="#555555",
        ha="right", va="bottom",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=1.2)
    )

    # Axis formatting
    counts = df["branch_group"].value_counts()
    ax.set_xticklabels([
        f"Branch-continuous\n(n={counts.get('Branch-continuous', 0)})",
        f"Branch-switching\n(n={counts.get('Branch-switching', 0)})",
    ], rotation=0, ha="center")
    ax.set_ylabel("High-dimensional displacement")

    style_axis(
        ax,
        "C",
        "DX→REL displacement and upper-tail departures",
        panel_fontsize=20,
        title_fontsize=14,
        title_x=0.09,
        title_ha="left",
    )

    fig.savefig(OUT_PNG, dpi=600, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    plt.close(fig)

    print(f"[DONE] Saved {OUT_PNG}")
    print(f"[DONE] Saved {OUT_PDF}")
    print(f"[DONE] Saved {OUT_TSV}")

    print("\n[SUMMARY]")
    print(df["branch_group"].value_counts(dropna=False))
    print(f"\nTail threshold (DX->REL q90): {tail_threshold:.4f}")
    print("\nTop intervals:")
    print(
        df.sort_values("displacement_hd", ascending=False)[
            ["patient_id", "sample_start", "sample_end", "displacement_hd", "branch_switch", "branch_group", "tail_flag"]
        ].head(10).to_string(index=False)
    )


if __name__ == "__main__":
    main()
