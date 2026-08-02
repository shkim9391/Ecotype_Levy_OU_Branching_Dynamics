from __future__ import annotations

from pathlib import Path
from PIL import Image, ImageChops, ImageOps
import numpy as np
import matplotlib.pyplot as plt


PROJECT_DIR = Path("/Multimodal_OU_Lévy_Branching_Scaffold/Figure_4")
PANELS_DIR = PROJECT_DIR / "panels"
FINAL_DIR = PROJECT_DIR / "final"

A_PNG = PANELS_DIR / "Figure4A_theta_by_phase.png"
B_PNG = PANELS_DIR / "Figure4B_mu_shift_from_dx.png"
C_PNG = PANELS_DIR / "Figure4C_sigma_eff_by_phase.png"
D_PNG = PANELS_DIR / "Figure4D_jump_score_by_branch_transition.png"
E_PNG = PANELS_DIR / "Figure4E_regime_schematic.png"

OUT_PNG = FINAL_DIR / "Figure4_treatment_aware_dynamic_parameters_full.png"
OUT_PDF = FINAL_DIR / "Figure4_treatment_aware_dynamic_parameters_full.pdf"

# Final canvas/layout
CANVAS_W = 4800
MARGIN = 70
GAP_X = 70
GAP_Y = 50

TOP_H = 1450
MID_H = 1450
E_H = 1040

PANEL_W = (CANVAS_W - 2 * MARGIN - GAP_X) // 2
E_W = CANVAS_W - 2 * MARGIN


def ensure_dirs() -> None:
    FINAL_DIR.mkdir(parents=True, exist_ok=True)


def trim_white(img: Image.Image, border: int = 8) -> Image.Image:
    rgb = img.convert("RGB")
    bg = Image.new("RGB", rgb.size, "white")
    diff = ImageChops.difference(rgb, bg)
    bbox = diff.getbbox()

    if bbox is None:
        return rgb

    left, upper, right, lower = bbox
    left = max(0, left - border)
    upper = max(0, upper - border)
    right = min(rgb.size[0], right + border)
    lower = min(rgb.size[1], lower + border)

    return rgb.crop((left, upper, right, lower))


def fit_panel(img: Image.Image, target_w: int, target_h: int) -> Image.Image:
    return ImageOps.contain(
        img.convert("RGB"),
        (target_w, target_h),
        method=Image.Resampling.LANCZOS,
    )


def paste_centered(
    canvas: Image.Image,
    img: Image.Image,
    x0: int,
    y0: int,
    box_w: int,
    box_h: int,
) -> None:
    x = x0 + (box_w - img.size[0]) // 2
    y = y0 + (box_h - img.size[1]) // 2
    canvas.paste(img, (x, y))


def main() -> None:
    ensure_dirs()

    required = [A_PNG, B_PNG, C_PNG, D_PNG, E_PNG]
    for fp in required:
        if not fp.exists():
            raise FileNotFoundError(fp)

    A = trim_white(Image.open(A_PNG))
    B = trim_white(Image.open(B_PNG))
    C = trim_white(Image.open(C_PNG))
    D = trim_white(Image.open(D_PNG))
    E = trim_white(Image.open(E_PNG), border=4)

    A2 = fit_panel(A, PANEL_W, TOP_H)
    B2 = fit_panel(B, PANEL_W, TOP_H)
    C2 = fit_panel(C, PANEL_W, MID_H)
    D2 = fit_panel(D, PANEL_W, MID_H)
    E2 = fit_panel(E, E_W, E_H)

    canvas_h = (
        2 * MARGIN
        + TOP_H
        + GAP_Y
        + MID_H
        + GAP_Y
        + E_H
    )

    canvas = Image.new("RGB", (CANVAS_W, canvas_h), "white")

    xL = MARGIN
    xR = MARGIN + PANEL_W + GAP_X

    yTop = MARGIN
    yMid = yTop + TOP_H + GAP_Y
    yE = yMid + MID_H + GAP_Y

    paste_centered(canvas, A2, xL, yTop, PANEL_W, TOP_H)
    paste_centered(canvas, B2, xR, yTop, PANEL_W, TOP_H)

    paste_centered(canvas, C2, xL, yMid, PANEL_W, MID_H)
    paste_centered(canvas, D2, xR, yMid, PANEL_W, MID_H)

    paste_centered(canvas, E2, MARGIN, yE, E_W, E_H)

    canvas.save(OUT_PNG)

    arr = np.asarray(canvas)
    fig = plt.figure(figsize=(CANVAS_W / 300, canvas_h / 300), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(arr)
    ax.axis("off")
    fig.savefig(OUT_PDF, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

    print(f"[DONE] Saved {OUT_PNG}")
    print(f"[DONE] Saved {OUT_PDF}")


if __name__ == "__main__":
    main()
