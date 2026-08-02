from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D


# ============================================================
# 1. CONFIG
# ============================================================
FIG7_DIR = Path("/Multimodal_OU_Lévy_Branching_Scaffold/Figure_7")
FIG7_DERIVED = FIG7_DIR / "derived"
FIG7_PANELS = FIG7_DIR / "panels"

IN_CSV = FIG7_DERIVED / "figure7_clinical_risk_map.csv"
IN_SUMMARY = FIG7_DERIVED / "figure7_clinical_risk_map_summary.tsv"

OUT_PNG = FIG7_PANELS / "Figure7A_clinical_dynamic_map.png"
OUT_PDF = FIG7_PANELS / "Figure7A_clinical_dynamic_map.pdf"

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

COHORT_LABELS = {
    "discovery": "Discovery AML",
    "external_aml": "External AML",
}

COHORT_MARKERS = {
    "discovery": "o",
    "external_aml": "s",
}

ZONE_COLORS = {
    "Constrained response-like": "#E7F4EE",
    "Residual persistent": "#F6F0D9",
    "Escape-prone relapse-like": "#F8E8E8",
}


# ============================================================
# 2. HELPERS
# ============================================================
def ensure_dirs() -> None:
    FIG7_PANELS.mkdir(parents=True, exist_ok=True)


def assert_columns(df: pd.DataFrame, required: list[str], label: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{label} missing required columns: {missing}")


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
        ha="left",
        va="bottom"
    )
    ax.text(
        title_x, 1.03, title,
        transform=ax.transAxes,
        fontsize=title_fontsize,
        fontweight="bold",
        ha="left",
        va="bottom"
    )


def get_threshold(summary_df: pd.DataFrame, item: str) -> float:
    row = summary_df.loc[
        (summary_df["section"] == "thresholds") & (summary_df["item"] == item),
        "value"
    ]
    if row.empty:
        raise ValueError(f"Threshold '{item}' not found in summary table.")
    return float(row.iloc[0])


def add_legends(ax):
    handles = [
        Line2D(
            [0], [0],
            marker="o",
            linestyle="None",
            markersize=8,
            markerfacecolor=PHASE_COLORS["DX"],
            markeredgecolor="none",
            label="DX",
        ),
        Line2D(
            [0], [0],
            marker="o",
            linestyle="None",
            markersize=8,
            markerfacecolor=PHASE_COLORS["EOI_REM"],
            markeredgecolor="none",
            label="EOI/REM",
        ),
        Line2D(
            [0], [0],
            marker="o",
            linestyle="None",
            markersize=8,
            markerfacecolor=PHASE_COLORS["REL"],
            markeredgecolor="none",
            label="REL",
        ),
        Line2D(
            [0], [0],
            marker="o",
            linestyle="None",
            markersize=8,
            markerfacecolor="white",
            markeredgecolor="#333333",
            label="Discovery AML",
        ),
        Line2D(
            [0], [0],
            marker="s",
            linestyle="None",
            markersize=8,
            markerfacecolor="white",
            markeredgecolor="#333333",
            label="External AML",
        ),
    ]

    ax.legend(
        handles=handles,
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(0.02, 0.96),
        fontsize=9.0,
        handletextpad=0.5,
        labelspacing=0.45,
        borderaxespad=0.0,
    )


# ============================================================
# 3. MAIN
# ============================================================
def main() -> None:
    ensure_dirs()

    df = pd.read_csv(IN_CSV)
    summary = pd.read_csv(IN_SUMMARY, sep="\t")

    assert_columns(
        df,
        [
            "cohort",
            "source_group",
            "sample_id",
            "patient_id",
            "clinical_timepoint_coarse",
            "theta_eff",
            "sigma_eff",
            "mu_shift_from_dx",
            "clinical_risk_score",
            "risk_zone",
            "risk_tier",
            "n_cells",
        ],
        "figure7_clinical_risk_map.csv",
    )

    df["clinical_timepoint_coarse"] = df["clinical_timepoint_coarse"].astype(str)
    df["source_group"] = df["source_group"].astype(str)
    df["sample_id"] = df["sample_id"].astype(str)

    for c in ["theta_eff", "sigma_eff", "mu_shift_from_dx", "clinical_risk_score", "n_cells"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    mu_q75 = get_threshold(summary, "mu_shift_q75_ref")
    theta_med = get_threshold(summary, "theta_eff_median_ref")

    # axis limits
    x = df["mu_shift_from_dx"].to_numpy(dtype=float)
    y = df["theta_eff"].to_numpy(dtype=float)

    x_lo, x_hi = np.nanquantile(x, [0.02, 0.98])
    y_lo, y_hi = np.nanquantile(y, [0.02, 0.98])

    pad_x = 0.18 * (x_hi - x_lo if x_hi > x_lo else 1.0)
    pad_y = 0.18 * (y_hi - y_lo if y_hi > y_lo else 1.0)

    x_min = max(0.0, x_lo - pad_x)
    x_max = x_hi + pad_x
    y_min = max(0.0, y_lo - pad_y)
    y_max = min(1.05, y_hi + pad_y)

    fig, ax = plt.subplots(figsize=(9.0, 7.2))

    # --------------------------------------------------------
    # Background clinical-interpretation zones
    # Simplified 2D interpretation using discovery thresholds
    # --------------------------------------------------------
    # residual zone (left / lower)
    ax.add_patch(Rectangle(
        (x_min, y_min),
        max(mu_q75 - x_min, 0),
        max(theta_med - y_min, 0),
        facecolor=ZONE_COLORS["Residual persistent"],
        edgecolor="none",
        alpha=0.55,
        zorder=0,
    ))
    # constrained zone (left / upper)
    ax.add_patch(Rectangle(
        (x_min, theta_med),
        max(mu_q75 - x_min, 0),
        max(y_max - theta_med, 0),
        facecolor=ZONE_COLORS["Constrained response-like"],
        edgecolor="none",
        alpha=0.55,
        zorder=0,
    ))
    # escape zone (right)
    ax.add_patch(Rectangle(
        (mu_q75, y_min),
        max(x_max - mu_q75, 0),
        max(y_max - y_min, 0),
        facecolor=ZONE_COLORS["Escape-prone relapse-like"],
        edgecolor="none",
        alpha=0.55,
        zorder=0,
    ))

    # threshold lines
    ax.axvline(mu_q75, color="#666666", linestyle="--", linewidth=1.0, alpha=0.8, zorder=1)
    ax.axhline(theta_med, color="#666666", linestyle="--", linewidth=1.0, alpha=0.8, zorder=1)

    ax.text(mu_q75 + 0.01, y_max - 0.01, "μ-shift q75", fontsize=8.5, color="#555555", ha="left", va="top")
    ax.text(x_min + 0.01, theta_med + 0.01, "θ median", fontsize=8.5, color="#555555", ha="left", va="bottom")

    # --------------------------------------------------------
    # Scatter by cohort and phase
    # --------------------------------------------------------
    for source_group in ["discovery", "external_aml"]:
        marker = COHORT_MARKERS[source_group]
        sub_src = df[df["source_group"] == source_group].copy()

        for phase in ["DX", "EOI_REM", "REL"]:
            sub = sub_src[sub_src["clinical_timepoint_coarse"] == phase].copy()
            if sub.empty:
                continue

            ax.scatter(
                sub["mu_shift_from_dx"],
                sub["theta_eff"],
                s=np.clip(np.sqrt(sub["n_cells"]) * 0.70, 32, 90),
                c=PHASE_COLORS.get(phase, "#999999"),
                marker=marker,
                edgecolors="#333333",
                linewidths=0.7,
                alpha=0.92,
                zorder=3,
            )

    # --------------------------------------------------------
    # Representative labels
    # --------------------------------------------------------
    labels = []

    # top-risk sample per cohort
    for sg in ["discovery", "external_aml"]:
        sub = df[df["source_group"] == sg].sort_values("clinical_risk_score", ascending=False)
        if not sub.empty:
            labels.append(sub.iloc[0]["sample_id"])

    # all external REL
    labels.extend(
        df.loc[
            (df["source_group"] == "external_aml") &
            (df["clinical_timepoint_coarse"] == "REL"),
            "sample_id"
        ].tolist()
    )

    # discovery residual example if present
    if "AML21_REM" in set(df["sample_id"]):
        labels.append("AML21_REM")

    labels = list(dict.fromkeys(labels))

    for sid in labels:
        sub = df[df["sample_id"] == sid]
        if sub.empty:
            continue
        r = sub.iloc[0]
        ax.text(
            float(r["mu_shift_from_dx"]) + 0.012,
            float(r["theta_eff"]) + 0.008,
            sid,
            fontsize=8.2,
            fontweight="bold",
            ha="left",
            va="center",
            color="#222222",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.72, pad=1.0),
            zorder=4,
        )

    # zone labels
    ax.text(
        x_min + 0.23 * (x_max - x_min),
        theta_med + 0.26 * (y_max - theta_med),
        "Constrained\nresponse-like",
        fontsize=9.2,
        fontweight="bold",
        color="#355F4B",
        ha="left",
        va="center",
        zorder=2,
    )
    # residual zone label -> move to right-bottom of lower-left quadrant
    ax.text(
        x_min + 0.85 * (mu_q75 - x_min),
        y_min + 0.25 * (theta_med - y_min),
        "Residual persistent\nunstable",
        fontsize=9.2,
        fontweight="bold",
        color="#7A611F",
        ha="right",
        va="bottom",
        zorder=2,
    )
    
    # escape zone label -> move to right-center, just above theta median line
    ax.text(
        mu_q75 + 0.14 * (x_max - mu_q75),
        theta_med + 0.12 * (y_max - theta_med),
        "Escape-prone\nrelapse-like",
        fontsize=9.2,
        fontweight="bold",
        color="#7A3B3B",
        ha="left",
        va="bottom",
        zorder=2,
    )

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    ax.set_xlabel(r"Attractor displacement from DX baseline ($\mu$-shift)")
    ax.set_ylabel(r"Effective restoring strength ($\theta_{\mathrm{eff}}$)")

    style_axis(ax, "A", "Dynamic clinical map")
    add_legends(ax)
    
    ax.text(
    0.98,
    0.01,
    "Higher μ-shift = more displaced; higher θeff = stronger restoration",
    transform=ax.transAxes,
    fontsize=8.5,
    ha="right",
    va="bottom",
    color="#555555",
    )

    fig.savefig(OUT_PNG, dpi=600, bbox_inches="tight")
    fig.savefig(OUT_PDF, dpi=600, bbox_inches="tight")
    plt.close(fig)

    print(f"[DONE] Saved {OUT_PNG}")
    print(f"[DONE] Saved {OUT_PDF}")

    print("\n[SUMMARY: points by cohort / phase / risk zone]")
    out = (
        df.groupby(["cohort", "clinical_timepoint_coarse", "risk_zone"])
          .size()
          .reset_index(name="n_samples")
    )
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
