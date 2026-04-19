from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path("/GSE235063/derived_dx_primary_training/final_small_model")
OUTDIR = ROOT
ANALYSIS = "full19"

RESPONSES = [
    "ilr_stem_vs_committed",
    "log_aux_erybaso",
]

PANEL_TITLES = {
    "ilr_stem_vs_committed": "A. Stem-versus-committed balance",
    "log_aux_erybaso": "B. Erythroid/basophil auxiliary program",
}

OBS_LABELS = {
    "ilr_stem_vs_committed": "Observed stem-committed ILR",
    "log_aux_erybaso": "Observed log Ery/Baso",
}

PRED_LABELS = {
    "ilr_stem_vs_committed": "LOO predicted stem-committed ILR",
    "log_aux_erybaso": "LOO predicted log Ery/Baso",
}

TERM_ORDER = [
    "PC1",
    "PC2",
    "is_blood",
    "Subgroup_RUNX",
    "Subgroup_CBFB",
    "Subgroup_FLT",
    "Subgroup_Other",
]

TERM_LABELS = {
    "PC1": "PC1",
    "PC2": "PC2",
    "is_blood": "is_blood",
    "Subgroup_RUNX": "Subgroup RUNX",
    "Subgroup_CBFB": "Subgroup CBFB",
    "Subgroup_FLT": "Subgroup FLT",
    "Subgroup_Other": "Subgroup Other",
}

def load_inputs(root: Path, analysis: str):
    perf = pd.read_csv(root / f"small_model_performance__{analysis}.csv")
    coef = pd.read_csv(root / f"small_model_coefficients__{analysis}.csv")
    pred = pd.read_csv(root / f"small_model_predictions__{analysis}.csv")
    sigma = pd.read_csv(root / f"small_model_sigmahat__{analysis}.csv", index_col=0)
    return perf, coef, pred, sigma

def build_compact_table(root: Path):
    rows = []
    for analysis in ["full19", "no_AML23"]:
        perf = pd.read_csv(root / f"small_model_performance__{analysis}.csv")
        coef = pd.read_csv(root / f"small_model_coefficients__{analysis}.csv")
        sigma = pd.read_csv(root / f"small_model_sigmahat__{analysis}.csv", index_col=0)

        for response in RESPONSES:
            prow = perf.loc[perf["response"] == response].iloc[0]
            csub = coef.loc[coef["response"] == response].copy()
            cmap = dict(zip(csub["term"], csub["coefficient"]))

            rows.append({
                "analysis": analysis,
                "response": response,
                "n_samples": int(prow["n_samples"]),
                "loo_rmse": float(prow["loo_rmse"]),
                "loo_r2": float(prow["loo_r2"]),
                "loo_corr": float(prow["loo_corr"]),
                "alpha_full_fit": float(csub["alpha_full_fit"].iloc[0]),
                "PC1": cmap.get("PC1", np.nan),
                "PC2": cmap.get("PC2", np.nan),
                "is_blood": cmap.get("is_blood", np.nan),
                "RUNX_vs_KMT2A": cmap.get("Subgroup_RUNX", np.nan),
                "CBFB_vs_KMT2A": cmap.get("Subgroup_CBFB", np.nan),
                "FLT_vs_KMT2A": cmap.get("Subgroup_FLT", np.nan),
                "Other_vs_KMT2A": cmap.get("Subgroup_Other", np.nan),
                "sigma_var_response": float(sigma.loc[response, response]),
            })

    table = pd.DataFrame(rows)
    table_csv = root / "small_model_results_compact.csv"
    table_txt = root / "small_model_results_compact.txt"
    table.to_csv(table_csv, index=False)

    with open(table_txt, "w", encoding="utf-8") as f:
        f.write("Compact results table\n\n")
        f.write(table.to_string(index=False))
        f.write("\n")

    return table

def coefficient_panel(ax, coef_df: pd.DataFrame, response: str):
    csub = coef_df[(coef_df["response"] == response) & (coef_df["term"] != "intercept")].copy()
    csub["term"] = pd.Categorical(csub["term"], categories=TERM_ORDER, ordered=True)
    csub = csub.sort_values("term")

    x = np.arange(len(csub))
    ax.bar(x, csub["coefficient"].to_numpy())
    ax.axhline(0, linestyle="--", linewidth=1)

    ax.set_xticks(x)
    ax.set_xticklabels([TERM_LABELS[t] for t in csub["term"]], rotation=40, ha="right")
    ax.set_ylabel("Coefficient")
    ax.set_title(PANEL_TITLES[response], fontsize=15, pad=6)

    # move this INSIDE the axes so it doesn't collide with the figure title
    ax.text(
        0.01, 0.98,
        "Reference subgroup: KMT2A",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
    )

def observed_predicted_panel(ax, pred_df: pd.DataFrame, perf_df: pd.DataFrame, response: str):
    x = pred_df[f"true__{response}"].to_numpy()
    y = pred_df[f"predloo__{response}"].to_numpy()

    ax.scatter(x, y)

    lo = min(np.min(x), np.min(y))
    hi = max(np.max(x), np.max(y))
    pad = 0.06 * (hi - lo if hi > lo else 1.0)
    lo -= pad
    hi += pad

    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.5)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)

    ax.set_xlabel(OBS_LABELS[response])
    ax.set_ylabel(PRED_LABELS[response])

    prow = perf_df.loc[perf_df["response"] == response].iloc[0]
    metrics_txt = (
        f"LOO RMSE = {prow['loo_rmse']:.2f}\n"
        f"R² = {prow['loo_r2']:.2f}\n"
        f"r = {prow['loo_corr']:.2f}"
    )
    ax.text(
        0.03, 0.97,
        metrics_txt,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="black", linewidth=0.8),
    )

    # smaller labels for readability
    dx = 0.01 * (hi - lo)
    dy = 0.01 * (hi - lo)
    for _, r in pred_df.iterrows():
        ax.text(
            r[f"true__{response}"] + dx,
            r[f"predloo__{response}"] + dy,
            str(r["sample"]),
            fontsize=7,
            ha="left",
            va="bottom",
        )

def make_figure(root: Path, analysis: str):
    perf_df, coef_df, pred_df, sigma_df = load_inputs(root, analysis)

    fig, axes = plt.subplots(
        2, 2,
        figsize=(13.5, 10.5),
        gridspec_kw={"height_ratios": [1.0, 2.1]},
        constrained_layout=False,
    )

    # reserve space at the top for title + subtitle
    fig.subplots_adjust(
        left=0.07,
        right=0.99,
        bottom=0.07,
        top=0.90,
        wspace=0.14,
        hspace=0.24,
    )

    #fig.suptitle("Reduced equilibrium model summary", fontsize=22, y=0.975)
    #fig.text(0.5, 0.94, "Primary fit (full19)", ha="center", va="center", fontsize=14)

    coefficient_panel(axes[0, 0], coef_df, "ilr_stem_vs_committed")
    coefficient_panel(axes[0, 1], coef_df, "log_aux_erybaso")

    observed_predicted_panel(axes[1, 0], pred_df, perf_df, "ilr_stem_vs_committed")
    observed_predicted_panel(axes[1, 1], pred_df, perf_df, "log_aux_erybaso")

    png_path = root / f"small_model_2panel_figure__{analysis}__journal.png"
    pdf_path = root / f"small_model_2panel_figure__{analysis}__journal.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    return png_path, pdf_path

def main():
    table = build_compact_table(ROOT)
    png_path, pdf_path = make_figure(ROOT, ANALYSIS)

    print("\nWrote:")
    print(ROOT / "small_model_results_compact.csv")
    print(ROOT / "small_model_results_compact.txt")
    print(png_path)
    print(pdf_path)

if __name__ == "__main__":
    main()
