from pathlib import Path
import pandas as pd
from PIL import Image, ImageDraw, ImageOps

ROOT = Path("/GSE235063/derived_dx_primary_training/final_small_model")
OUTDIR = ROOT

def load_summary_table(root: Path) -> pd.DataFrame:
    analyses = ["full19", "no_AML23"]
    responses = ["ilr_stem_vs_committed", "log_aux_erybaso"]
    rows = []

    for analysis in analyses:
        perf = pd.read_csv(root / f"small_model_performance__{analysis}.csv")
        coef = pd.read_csv(root / f"small_model_coefficients__{analysis}.csv")
        sigma = pd.read_csv(root / f"small_model_sigmahat__{analysis}.csv", index_col=0)

        for response in responses:
            prow = perf.loc[perf["response"] == response].iloc[0]
            csub = coef.loc[coef["response"] == response].copy()
            cmap = dict(zip(csub["term"], csub["coefficient"]))
            alpha = float(csub["alpha_full_fit"].iloc[0])

            rows.append({
                "analysis": analysis,
                "response": response,
                "n_samples": int(prow["n_samples"]),
                "loo_rmse": float(prow["loo_rmse"]),
                "loo_r2": float(prow["loo_r2"]),
                "loo_corr": float(prow["loo_corr"]),
                "alpha_full_fit": alpha,
                "intercept": cmap.get("intercept", float("nan")),
                "PC1": cmap.get("PC1", float("nan")),
                "PC2": cmap.get("PC2", float("nan")),
                "is_blood": cmap.get("is_blood", float("nan")),
                "RUNX_vs_KMT2A": cmap.get("Subgroup_RUNX", float("nan")),
                "CBFB_vs_KMT2A": cmap.get("Subgroup_CBFB", float("nan")),
                "FLT_vs_KMT2A": cmap.get("Subgroup_FLT", float("nan")),
                "Other_vs_KMT2A": cmap.get("Subgroup_Other", float("nan")),
                "sigma_var_response": float(sigma.loc[response, response]),
            })

    return pd.DataFrame(rows)

def check_exists(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

def make_panel_image(root: Path, analysis: str, response: str, title: str, panel_label: str) -> Image.Image:
    coef_path = root / f"small_model_coefficients__{analysis}__{response}.png"
    pred_path = root / f"small_model_obs_vs_pred__{analysis}__{response}.png"
    check_exists(coef_path)
    check_exists(pred_path)

    coef_img = Image.open(coef_path).convert("RGB")
    pred_img = Image.open(pred_path).convert("RGB")

    target_width = max(coef_img.width, pred_img.width)
    if coef_img.width != target_width:
        coef_img = coef_img.resize((target_width, int(coef_img.height * target_width / coef_img.width)))
    if pred_img.width != target_width:
        pred_img = pred_img.resize((target_width, int(pred_img.height * target_width / pred_img.width)))

    title_h = 90
    gap = 20
    pad = 24
    canvas_w = target_width + 2 * pad
    canvas_h = title_h + coef_img.height + gap + pred_img.height + 2 * pad

    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((pad, 18), f"{panel_label}. {title}", fill="black")
    draw.text((pad, 48), f"{analysis}: coefficients (top) and observed vs predicted (bottom)", fill="black")

    canvas.paste(coef_img, (pad, title_h))
    canvas.paste(pred_img, (pad, title_h + coef_img.height + gap))

    return ImageOps.expand(canvas, border=1, fill="black")

def combine_two_panels(left: Image.Image, right: Image.Image, title: str) -> Image.Image:
    top_h = 80
    gap = 30
    pad = 30
    total_w = left.width + right.width + gap + 2 * pad
    total_h = max(left.height, right.height) + top_h + 2 * pad

    canvas = Image.new("RGB", (total_w, total_h), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((pad, 18), title, fill="black")
    draw.text((pad, 46), "Reduced equilibrium model: primary fit (full19)", fill="black")

    y0 = top_h + pad
    canvas.paste(left, (pad, y0))
    canvas.paste(right, (pad + left.width + gap, y0))
    return canvas

def main():
    table = load_summary_table(ROOT)

    table_csv = OUTDIR / "small_model_results_compact.csv"
    table_txt = OUTDIR / "small_model_results_compact.txt"
    
    table.to_csv(table_csv, index=False)
    
    with open(table_txt, "w", encoding="utf-8") as f:
        f.write("Compact results table\n\n")
        f.write(table.to_string(index=False))
        f.write("\n")

    left = make_panel_image(
        ROOT,
        analysis="full19",
        response="ilr_stem_vs_committed",
        title="Stem-versus-committed balance",
        panel_label="A",
    )
    right = make_panel_image(
        ROOT,
        analysis="full19",
        response="log_aux_erybaso",
        title="Erythroid/basophil auxiliary program",
        panel_label="B",
    )

    fig = combine_two_panels(left, right, "Two-panel summary figure")
    fig_png = OUTDIR / "small_model_2panel_figure__full19.png"
    fig_pdf = OUTDIR / "small_model_2panel_figure__full19.pdf"
    fig.save(fig_png)
    fig.save(fig_pdf)

    print("Wrote:")
    print(table_csv)
    print(table_txt)
    print(fig_png)
    print(fig_pdf)

if __name__ == "__main__":
    main()
