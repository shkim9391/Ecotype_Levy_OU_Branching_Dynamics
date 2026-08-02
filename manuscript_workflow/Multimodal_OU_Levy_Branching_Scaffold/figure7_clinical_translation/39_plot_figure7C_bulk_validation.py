from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu


# ============================================================
# 1. CONFIG
# ============================================================
FIG7_DIR = Path("/Multimodal_OU_Lévy_Branching_Scaffold/Figure_7")
FIG7_DERIVED = FIG7_DIR / "derived"
FIG7_PANELS = FIG7_DIR / "panels"

IN_SUMMARY = FIG7_DERIVED / "gse163634_bulk_validation_summary.csv"
IN_REPL = FIG7_DERIVED / "gse163634_calibration_summary.tsv"

OUT_PNG = FIG7_PANELS / "Figure7C_bulk_validation.png"
OUT_PDF = FIG7_PANELS / "Figure7C_bulk_validation.pdf"
OUT_TSV = FIG7_DERIVED / "figure7C_bulk_validation_stats.tsv"

GROUP_ORDER = ["control", "dx_leukemia", "response_r1", "response_r2"]
GROUP_LABELS = {
    "control": "Control",
    "dx_leukemia": "DX leukemia",
    "response_r1": "Response r1",
    "response_r2": "Response r2",
}
GROUP_COLORS = {
    "control": "#BDBDBD",
    "dx_leukemia": "#7A8DA8",
    "response_r1": "#6BAF92",
    "response_r2": "#88C4A8",
}


# ============================================================
# 2. HELPERS
# ============================================================
def ensure_dirs() -> None:
    FIG7_PANELS.mkdir(parents=True, exist_ok=True)


def style_axis(ax, panel_label: str, title: str,
               panel_fontsize: int = 18,
               title_fontsize: int = 12,
               title_x: float = 0.10) -> None:
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
        ha="left", va="bottom"
    )


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    gt = 0
    lt = 0
    for xi in x:
        gt += np.sum(xi > y)
        lt += np.sum(xi < y)
    n = len(x) * len(y)
    return (gt - lt) / n if n > 0 else np.nan


def add_violin_box_jitter(ax, df: pd.DataFrame, group_order: list[str], value_col: str):
    rng = np.random.default_rng(42)

    arrays = [
        pd.to_numeric(df.loc[df["clinical_group"] == g, value_col], errors="coerce").dropna().to_numpy()
        for g in group_order
    ]

    vp = ax.violinplot(
        arrays,
        positions=np.arange(1, len(group_order) + 1),
        widths=0.82,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )

    for body, g in zip(vp["bodies"], group_order):
        body.set_facecolor(GROUP_COLORS[g])
        body.set_edgecolor("none")
        body.set_alpha(0.20)

    bp = ax.boxplot(
        arrays,
        positions=np.arange(1, len(group_order) + 1),
        widths=0.28,
        patch_artist=True,
        showfliers=False,
    )

    for patch, g in zip(bp["boxes"], group_order):
        patch.set_facecolor(GROUP_COLORS[g])
        patch.set_alpha(0.48)
        patch.set_edgecolor("#333333")
        patch.set_linewidth(1.0)

    for key in ["whiskers", "caps", "medians"]:
        for line in bp[key]:
            line.set_color("#333333")
            line.set_linewidth(1.0)

    for i, g in enumerate(group_order, start=1):
        sub = df[df["clinical_group"] == g].copy()
        x = rng.normal(loc=i, scale=0.055, size=len(sub))

        ax.scatter(
            x,
            sub[value_col],
            s=32,
            c=GROUP_COLORS[g],
            alpha=0.84,
            edgecolors="white",
            linewidths=0.35,
            zorder=4,
        )


# ============================================================
# 3. MAIN
# ============================================================
def main() -> None:
    ensure_dirs()

    df = pd.read_csv(IN_SUMMARY)
    repl = pd.read_csv(IN_REPL, sep="\t")

    df["clinical_group"] = df["clinical_group"].astype(str)
    df["mu_shift_from_dx"] = pd.to_numeric(df["mu_shift_from_dx"], errors="coerce")
    df["stage_order"] = pd.to_numeric(df["stage_order"], errors="coerce")

    plot_df = df[df["clinical_group"].isin(GROUP_ORDER)].copy()

    # stats
    ctrl = plot_df.loc[plot_df["clinical_group"] == "control", "mu_shift_from_dx"].dropna().to_numpy()
    dx = plot_df.loc[plot_df["clinical_group"] == "dx_leukemia", "mu_shift_from_dx"].dropna().to_numpy()
    r1 = plot_df.loc[plot_df["clinical_group"] == "response_r1", "mu_shift_from_dx"].dropna().to_numpy()
    r2 = plot_df.loc[plot_df["clinical_group"] == "response_r2", "mu_shift_from_dx"].dropna().to_numpy()

    stats_rows = []

    comparisons = [
        ("control_vs_dx", ctrl, dx),
        ("dx_vs_r1", dx, r1),
        ("r1_vs_r2", r1, r2),
    ]

    for name, a, b in comparisons:
        if len(a) >= 2 and len(b) >= 2:
            u, p = mannwhitneyu(a, b, alternative="two-sided")
            cd = cliffs_delta(b, a)
        else:
            u, p, cd = np.nan, np.nan, np.nan

        stats_rows.append({
            "comparison": name,
            "n_group1": len(a),
            "n_group2": len(b),
            "median_group1": float(np.nanmedian(a)) if len(a) else np.nan,
            "median_group2": float(np.nanmedian(b)) if len(b) else np.nan,
            "mannwhitney_u": float(u) if pd.notna(u) else np.nan,
            "mannwhitney_p": float(p) if pd.notna(p) else np.nan,
            "cliffs_delta_group2_vs_group1": float(cd) if pd.notna(cd) else np.nan,
        })

    stats = pd.DataFrame(stats_rows)
    stats.to_csv(OUT_TSV, sep="\t", index=False)

    # ---------------------------------------------------------
    # Plot
    # ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9.3, 6.9))

    add_violin_box_jitter(ax, plot_df, GROUP_ORDER, "mu_shift_from_dx")

    # serial overlays for leukemia patients
    serial = plot_df[plot_df["clinical_group"].isin(["dx_leukemia", "response_r1", "response_r2"])].copy()
    for patient_id, sub in serial.groupby("patient_id", sort=False):
        sub = sub.sort_values("stage_order")
        if sub.shape[0] < 2:
            continue

        xs = []
        ys = []
        for _, r in sub.iterrows():
            grp = r["clinical_group"]
            xs.append(GROUP_ORDER.index(grp) + 1)
            ys.append(r["mu_shift_from_dx"])

        ax.plot(
            xs,
            ys,
            color="#A8A8A8",
            linewidth=0.8,
            alpha=0.28,
            zorder=2,
        )

    # x labels with n
    counts = plot_df["clinical_group"].value_counts()
    ax.set_xticks(np.arange(1, len(GROUP_ORDER) + 1))
    ax.set_xticklabels([
        f"{GROUP_LABELS[g]}\n(n={counts.get(g, 0)})" for g in GROUP_ORDER
    ], rotation=0, ha="center")

    ax.set_ylabel(r"Discovery-anchored displacement ($\mu$-shift)")

    style_axis(
    ax,
    "C",
    "Serial bulk validation preserves disease-state ordering",
    title_x=0.12,
    )

    # p-value annotations (two main comparisons)
    y_top = np.nanmax(plot_df["mu_shift_from_dx"])
    y_bar1 = y_top + 0.18
    y_bar2 = y_top + 0.38

    # control vs dx
    p1 = stats.loc[stats["comparison"] == "control_vs_dx", "mannwhitney_p"].iloc[0]
    ax.plot([1, 1, 2, 2], [y_bar1 - 0.03, y_bar1, y_bar1, y_bar1 - 0.03], color="#444444", linewidth=1.1)
    ax.text(
        1.5,
        y_bar1 + 0.03,
        f"Control vs DX, p = {p1:.3g}" if pd.notna(p1) else "Control vs DX, p = NA",
        ha="center",
        va="bottom",
        fontsize=8.5,
        color="#444444",
    )

    # dx vs r1
    p2 = stats.loc[stats["comparison"] == "dx_vs_r1", "mannwhitney_p"].iloc[0]
    ax.plot([2, 2, 3, 3], [y_bar2 - 0.03, y_bar2, y_bar2, y_bar2 - 0.03], color="#444444", linewidth=1.1)
    ax.text(
        2.5,
        y_bar2 + 0.03,
        f"DX vs response r1, p = {p2:.3g}" if pd.notna(p2) else "DX vs response r1, p = NA",
        ha="center",
        va="bottom",
        fontsize=8.5,
        color="#444444",
    )

    y_min = np.nanmin(plot_df["mu_shift_from_dx"])
    y_max = y_bar2 + 0.22
    ax.set_ylim(y_min - 0.15, y_max)
    
    ax.text(
        0.98,
        0.03,
        "Lower-resolution serial bulk cohort",
        transform=ax.transAxes,
        fontsize=8.5,
        ha="right",
        va="bottom",
        color="#555555",
    )

    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color="#E6E6E6", linewidth=0.7)
    ax.xaxis.grid(False)

    fig.savefig(OUT_PNG, dpi=600, bbox_inches="tight")
    fig.savefig(OUT_PDF, dpi=600, bbox_inches="tight")
    plt.close(fig)

    print(f"[DONE] Saved {OUT_PNG}")
    print(f"[DONE] Saved {OUT_PDF}")
    print(f"[DONE] Saved {OUT_TSV}")

    print("\n[SUMMARY]")
    print(
        plot_df.groupby("clinical_group")["mu_shift_from_dx"]
        .describe()
        .round(4)
        .to_string()
    )

    print("\n[Leukemia-vs-control replication rows]")
    print(repl[repl["comparison"] == "leukemia_vs_control"].to_string(index=False))


if __name__ == "__main__":
    main()
