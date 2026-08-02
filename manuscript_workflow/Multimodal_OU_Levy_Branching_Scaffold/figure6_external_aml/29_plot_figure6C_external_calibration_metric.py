from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu


# ============================================================
# 1. CONFIG
# ============================================================
FIG6_DIR = Path("/Multimodal_OU_Lévy_Branching_Scaffold/Figure_6")
FIG6_DERIVED = FIG6_DIR / "derived"
FIG6_PANELS = FIG6_DIR / "panels"

IN_CSV = FIG6_DERIVED / "gse235923_sample_dynamic_parameters.csv"

OUT_PNG = FIG6_PANELS / "Figure6C_external_calibration_metric.png"
OUT_PDF = FIG6_PANELS / "Figure6C_external_calibration_metric.pdf"
OUT_TSV = FIG6_DERIVED / "figure6C_calibration_metric_stats.tsv"

PHASE_ORDER = ["DX", "EOI_REM"]
PHASE_COLORS = {
    "DX": "#7A8DA8",
    "EOI_REM": "#6BAF92",
    "REL": "#C97979",
}

PHASE_LABELS = {
    "DX": "DX",
    "EOI_REM": "EOI/REM",
    "REL": "REL",
}

# ============================================================
# 2. HELPERS
# ============================================================
def ensure_dirs() -> None:
    FIG6_PANELS.mkdir(parents=True, exist_ok=True)


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


# ============================================================
# 3. MAIN
# ============================================================
def main() -> None:
    ensure_dirs()

    df = pd.read_csv(IN_CSV)
    df["mu_shift_from_dx"] = pd.to_numeric(df["mu_shift_from_dx"], errors="coerce")
    df["clinical_timepoint_coarse"] = df["clinical_timepoint_coarse"].astype(str)

    dx = df[df["clinical_timepoint_coarse"] == "DX"].copy()
    eoi = df[df["clinical_timepoint_coarse"] == "EOI_REM"].copy()
    rel = df[df["clinical_timepoint_coarse"] == "REL"].copy()

    x_dx = dx["mu_shift_from_dx"].dropna().to_numpy()
    x_eoi = eoi["mu_shift_from_dx"].dropna().to_numpy()

    if len(x_dx) == 0 or len(x_eoi) == 0:
        raise ValueError("Need both DX and EOI_REM samples for external calibration metric panel.")

    u_stat, u_p = mannwhitneyu(x_dx, x_eoi, alternative="two-sided")
    cd = cliffs_delta(x_eoi, x_dx)

    stats = pd.DataFrame([{
        "comparison": "EOI_REM_vs_DX",
        "n_DX": len(x_dx),
        "n_EOI_REM": len(x_eoi),
        "n_REL": int(rel.shape[0]),
        "median_DX": float(np.nanmedian(x_dx)),
        "median_EOI_REM": float(np.nanmedian(x_eoi)),
        "median_REL": float(np.nanmedian(rel["mu_shift_from_dx"])) if len(rel) else np.nan,
        "mannwhitney_u": float(u_stat),
        "mannwhitney_p": float(u_p),
        "cliffs_delta_EOI_vs_DX": float(cd),
    }])
    stats.to_csv(OUT_TSV, sep="\t", index=False)

    # ---------------------------------------------------------
    # Plot
    # ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.4, 6.6))
    rng = np.random.default_rng(42)

    positions = {"DX": 1, "EOI_REM": 2}
    rel_x = 3.05

    # Violin for DX and EOI/REM
    violin_data = [x_dx, x_eoi]
    vp = ax.violinplot(
        violin_data,
        positions=[positions["DX"], positions["EOI_REM"]],
        widths=0.82,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )

    for body, phase in zip(vp["bodies"], ["DX", "EOI_REM"]):
        body.set_facecolor(PHASE_COLORS[phase])
        body.set_edgecolor("none")
        body.set_alpha(0.20)

    # Box for DX and EOI/REM
    bp = ax.boxplot(
        violin_data,
        positions=[positions["DX"], positions["EOI_REM"]],
        widths=0.28,
        patch_artist=True,
        showfliers=False,
    )

    for patch, phase in zip(bp["boxes"], ["DX", "EOI_REM"]):
        patch.set_facecolor(PHASE_COLORS[phase])
        patch.set_alpha(0.48)
        patch.set_edgecolor("#333333")
        patch.set_linewidth(1.0)

    for key in ["whiskers", "caps", "medians"]:
        for line in bp[key]:
            line.set_color("#333333")
            line.set_linewidth(1.0)

    # Jittered DX and EOI/REM points
    for phase in ["DX", "EOI_REM"]:
        sub = df[df["clinical_timepoint_coarse"] == phase].copy()
        x = rng.normal(loc=positions[phase], scale=0.055, size=len(sub))

        ax.scatter(
            x,
            sub["mu_shift_from_dx"],
            s=34,
            c=PHASE_COLORS[phase],
            alpha=0.82,
            edgecolors="white",
            linewidths=0.35,
            zorder=3,
        )

    # REL points shown individually because external REL is sparse
    if not rel.empty:
        rel_sorted = rel.sort_values("mu_shift_from_dx").copy()

        for i, (_, r) in enumerate(rel_sorted.iterrows()):
            y = float(r["mu_shift_from_dx"])
            label = f"{r['patient_id']} REL"

            x_point = rel_x + (i - (len(rel_sorted) - 1) / 2) * 0.035

            ax.scatter(
                [x_point],
                [y],
                s=86,
                c=PHASE_COLORS["REL"],
                edgecolors="#2F2F2F",
                linewidths=1.0,
                marker="D",
                zorder=4,
            )

            ax.text(
                x_point + 0.10,
                y,
                label,
                fontsize=9.0,
                fontweight="bold",
                ha="left",
                va="center",
                color="#2F2F2F",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.0),
                zorder=5,
            )

    # Axes / labels
    ax.set_xlim(0.55, 3.65)
    ax.set_xticks([1, 2, rel_x])
    ax.set_xticklabels([
        f"DX\n(n={len(x_dx)})",
        f"EOI/REM\n(n={len(x_eoi)})",
        f"REL\n(exploratory n={len(rel)})",
    ])

    ax.set_ylabel(r"Attractor displacement from discovery DX baseline ($\mu$-shift)")

    style_axis(
        ax,
        "C",
        "External calibration of attractor displacement",
        title_x=0.12,
    )

    # p-value annotation for DX vs EOI/REM comparison
    y_candidates = [np.nanmax(x_dx), np.nanmax(x_eoi)]
    if len(rel):
        y_candidates.append(np.nanmax(rel["mu_shift_from_dx"]))

    y_top = max(y_candidates)
    y_bar = y_top + 0.055

    ax.plot(
        [1, 1, 2, 2],
        [y_bar - 0.012, y_bar, y_bar, y_bar - 0.012],
        color="#444444",
        linewidth=1.1,
    )

    ax.text(
        1.5,
        y_bar + 0.012,
        f"DX vs EOI/REM, p = {u_p:.3g}",
        ha="center",
        va="bottom",
        fontsize=9.0,
        color="#444444",
    )

    # Dynamic y-limits
    y_min_candidates = [np.nanmin(x_dx), np.nanmin(x_eoi)]
    if len(rel):
        y_min_candidates.append(np.nanmin(rel["mu_shift_from_dx"]))

    y_min = min(y_min_candidates)
    ax.set_ylim(max(0, y_min - 0.06), y_bar + 0.08)

    ax.text(
        0.98,
        0.03,
        "REL shown individually",
        transform=ax.transAxes,
        fontsize=9.0,
        ha="right",
        va="bottom",
        color="#555555",
    )

    fig.savefig(OUT_PNG, dpi=600, bbox_inches="tight")
    fig.savefig(OUT_PDF, dpi=600, bbox_inches="tight")
    plt.close(fig)

    print(f"[DONE] Saved {OUT_PNG}")
    print(f"[DONE] Saved {OUT_PDF}")
    print(f"[DONE] Saved {OUT_TSV}")

    print("\n[SUMMARY]")
    print("DX median:", float(np.nanmedian(x_dx)))
    print("EOI_REM median:", float(np.nanmedian(x_eoi)))
    if len(rel):
        print("REL values:")
        print(rel[["patient_id", "sample_id", "mu_shift_from_dx"]].to_string(index=False))


if __name__ == "__main__":
    main()
