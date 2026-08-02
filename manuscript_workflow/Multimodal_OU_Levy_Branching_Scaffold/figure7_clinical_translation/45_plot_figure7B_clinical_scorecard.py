from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
from matplotlib import cm, colors


# ============================================================
# 1. CONFIG
# ============================================================
FIG7_DIR = Path("/Multimodal_OU_Lévy_Branching_Scaffold/Figure_7")
FIG7_DERIVED = FIG7_DIR / "derived"
FIG7_PANELS = FIG7_DIR / "panels"

IN_CSV = FIG7_DERIVED / "figure7_clinical_scorecard.csv"

OUT_PNG = FIG7_PANELS / "Figure7B_clinical_scorecard.png"
OUT_PDF = FIG7_PANELS / "Figure7B_clinical_scorecard.pdf"

METRIC_COLS = ["theta_eff", "sigma_eff", "mu_shift_from_dx"]
METRIC_LABELS = {
    "theta_eff": r"$\theta_{\mathrm{eff}}$",
    "sigma_eff": r"$\sigma_{\mathrm{eff}}$",
    "mu_shift_from_dx": r"$\mu$-shift",
}

STATE_TIER_COLORS = {
    "Constrained": "#DDEFE4",
    "Residual": "#F6F0D9",
    "Escape-prone": "#F8E0E0",
}

STATE_TIER_LABELS = {
    "Low-risk constrained": "Constrained",
    "Intermediate residual": "Residual",
    "High-risk escape-prone": "Escape-prone",
    "Constrained": "Constrained",
    "Residual": "Residual",
    "Escape-prone": "Escape-prone",
}


# ============================================================
# 2. HELPERS
# ============================================================
def ensure_dirs() -> None:
    FIG7_PANELS.mkdir(parents=True, exist_ok=True)


def assert_columns(df: pd.DataFrame, required: list[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"figure7_clinical_scorecard.csv missing required columns: {missing}")


def style_axis(ax, panel_label: str, title: str,
               panel_fontsize: int = 18,
               title_fontsize: int = 12,
               title_x: float = 0.10) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

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


def clean_branch_label(x: str) -> str:
    return str(x).replace("-like basin", "").replace(" basin", "")


def zscore_columns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for c in cols:
        vals = pd.to_numeric(df[c], errors="coerce")
        mu = np.nanmean(vals)
        sd = np.nanstd(vals, ddof=0)
        if not np.isfinite(sd) or sd == 0:
            out[c] = 0.0
        else:
            out[c] = (vals - mu) / sd
    return out


def fmt_num(x, nd=3):
    x = pd.to_numeric(pd.Series([x]), errors="coerce").iloc[0]
    if pd.isna(x):
        return "—"
    return f"{float(x):.{nd}f}"


def clean_timepoint_label(x: str) -> str:
    return str(x).replace("EOI_REM", "EOI/REM")


# ============================================================
# 3. MAIN
# ============================================================
def main() -> None:
    ensure_dirs()

    df = pd.read_csv(IN_CSV)
    assert_columns(
        df,
        [
            "display_order",
            "row_group",
            "selection_reason",
            "cohort",
            "source_group",
            "sample_id",
            "patient_id",
            "clinical_timepoint_coarse",
            "theta_eff",
            "sigma_eff",
            "mu_shift_from_dx",
            "risk_tier",
            "branch_id_dominant",
            "jump_score_display",
            "branch_switch_display",
        ],
    )

    df = df.sort_values("display_order").reset_index(drop=True)

    for c in ["theta_eff", "sigma_eff", "mu_shift_from_dx", "jump_score_display", "branch_switch_display"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    z = zscore_columns(df, METRIC_COLS)
    z = z.clip(-2.0, 2.0)

    n = df.shape[0]
    cmap = cm.get_cmap("coolwarm")
    norm = colors.Normalize(vmin=-2, vmax=2)

    fig, ax = plt.subplots(figsize=(12.4, 5.8))
    style_axis(ax, "B", "Representative dynamic-state scorecard")

    # layout coordinates
    CELL_W = 0.90
    CELL_H = 0.84
    CELL_GAP = 0.06
    
    HEADER_Y = n + 0.06
    ROW_Y_SHIFT = 0.24

    x_row = -2.35
    x_heat0 = -0.15
    metric_x = [x_heat0 + i * (CELL_W + CELL_GAP) for i in range(len(METRIC_COLS))]
    
    x_branch = metric_x[-1] + CELL_W + 0.35
    x_tier = x_branch + 1.55
    x_jump = x_tier + 2.25

    ax.set_xlim(-2.8, 8.8)
    ax.set_ylim(-0.8, n + 0.62)

    # alternating faint row background
    for i in range(n):
        y = n - 1 - i + ROW_Y_SHIFT
        if i % 2 == 0:
            ax.add_patch(Rectangle(
                (-2.75, y - 0.48),
                11.4,
                0.96,
                facecolor="#FAFAFA",
                edgecolor="none",
                alpha=0.50,
                zorder=0,
            ))

    # headers
    ax.text(x_row, HEADER_Y, "Representative sample", fontsize=10.2, fontweight="bold", ha="left", va="bottom")

    for j, c in enumerate(METRIC_COLS):
        ax.text(
            metric_x[j] + CELL_W / 2,
            HEADER_Y,
            METRIC_LABELS[c],
            fontsize=10.2,
            fontweight="bold",
            ha="center",
            va="bottom",
        )

    ax.text(x_branch, HEADER_Y, "Branch", fontsize=10.2, fontweight="bold", ha="left", va="bottom")
    ax.text(x_tier, HEADER_Y, "State tier", fontsize=10.2, fontweight="bold", ha="left", va="bottom")
    ax.text(x_jump, HEADER_Y, "Jump / switch", fontsize=10.2, fontweight="bold", ha="left", va="bottom")

    # rows
    for i, row in df.iterrows():
        y = n - 1 - i + ROW_Y_SHIFT

        # row label
        ax.text(
            x_row,
            y + 0.12,
            str(row["sample_id"]),
            fontsize=10.0,
            fontweight="bold",
            ha="left",
            va="center",
            color="#222222",
        )
        ax.text(
            x_row,
            y - 0.16,
            f"{row['row_group']} | {clean_timepoint_label(row['clinical_timepoint_coarse'])}",
            fontsize=7.2,
            ha="left",
            va="center",
            color="#555555",
        )

        # heatmap-style metric cells
        for j, c in enumerate(METRIC_COLS):
            val_z = float(z.loc[i, c])
            face = cmap(norm(val_z))
        
            ax.add_patch(Rectangle(
                (metric_x[j], y - CELL_H / 2),
                CELL_W,
                CELL_H,
                facecolor=face,
                edgecolor="white",
                linewidth=0.9,
                zorder=2,
            ))
        
            raw_txt = fmt_num(row[c], nd=3)
            txt_color = "white" if abs(val_z) > 1.0 else "#222222"
            ax.text(
                metric_x[j] + CELL_W / 2,
                y,
                raw_txt,
                fontsize=8.7,
                fontweight="bold",
                ha="center",
                va="center",
                color=txt_color,
                zorder=3,
            )

        # branch text
        ax.text(
            x_branch,
            y,
            clean_branch_label(row["branch_id_dominant"]),
            fontsize=9.4,
            ha="left",
            va="center",
            color="#222222",
        )

        # risk tier badge
        tier_raw = str(row.get("risk_tier_short", row["risk_tier"]))
        tier = STATE_TIER_LABELS.get(tier_raw, tier_raw)
        badge_color = STATE_TIER_COLORS.get(tier, "#EEEEEE")
        ax.add_patch(FancyBboxPatch(
            (x_tier, y - 0.24),
            1.30,
            0.48,
            boxstyle="round,pad=0.02,rounding_size=0.07",
            facecolor=badge_color,
            edgecolor="#888888",
            linewidth=0.7,
            zorder=2,
        ))
        ax.text(
            x_tier + 0.65,
            y,
            tier,
            fontsize=8.3,
            fontweight="bold",
            ha="center",
            va="center",
            color="#333333",
            zorder=3,
        )

        # jump / switch text
        jump = row["jump_score_display"]
        sw = row["branch_switch_display"]
        if pd.notna(jump) and pd.notna(sw):
            jt = f"{jump:.2f} / {int(sw)}"
        else:
            jt = "—"
        ax.text(
            x_jump,
            y,
            jt,
            fontsize=9.0,
            ha="left",
            va="center",
            color="#333333",
        )

    # small scale note
    ax.text(
        x_heat0,
        -0.28,
        "Metric colors show column-wise standardized relative values;\nnumbers show raw values.",
        fontsize=12.0,
        ha="left",
        va="top",
        color="#555555",
    )

    fig.savefig(OUT_PNG, dpi=600, bbox_inches="tight")
    fig.savefig(OUT_PDF, dpi=600, bbox_inches="tight")
    plt.close(fig)

    print(f"[DONE] Saved {OUT_PNG}")
    print(f"[DONE] Saved {OUT_PDF}")

    print("\n[SUMMARY]")
    print(
        df[
            [
                "display_order",
                "sample_id",
                "row_group",
                "theta_eff",
                "sigma_eff",
                "mu_shift_from_dx",
                "risk_tier",
                "branch_id_dominant",
                "jump_score_display",
                "branch_switch_display",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
