from __future__ import annotations

from pathlib import Path
from PIL import Image, ImageChops, ImageOps
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. CONFIG
# ============================================================
PROJECT_DIR = Path("/Multimodal_OU_Lévy_Branching_Scaffold/Figure_7")
PANELS_DIR = PROJECT_DIR / "panels"
FINAL_DIR = PROJECT_DIR / "final"

A_PNG = PANELS_DIR / "Figure7A_clinical_dynamic_map.png"
B_PNG = PANELS_DIR / "Figure7B_clinical_scorecard.png"
C_PNG = PANELS_DIR / "Figure7C_bulk_validation.png"
D_PNG = PANELS_DIR / "Figure7D_clinical_translation_summary.png"

OUT_PNG = FINAL_DIR / "Figure7_clinical_translation.png"
OUT_PDF = FINAL_DIR / "Figure7_clinical_translation.pdf"

CANVAS_W = 3600
MARGIN = 70
GAP_X = 60
GAP_Y = 70

TOP_H = 1425
BOT_H = 1425

TOP_LEFT_W = (CANVAS_W - 2 * MARGIN - GAP_X) // 2
TOP_RIGHT_W = TOP_LEFT_W

BOT_LEFT_W = int((CANVAS_W - 2 * MARGIN - GAP_X) * 0.47)
BOT_RIGHT_W = (CANVAS_W - 2 * MARGIN - GAP_X) - BOT_LEFT_W


# ============================================================
# 2. HELPERS
# ============================================================
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


def paste_top_aligned(canvas: Image.Image, img: Image.Image, x0: int, y0: int, box_w: int) -> None:
    x = x0 + (box_w - img.size[0]) // 2
    y = y0
    canvas.paste(img, (x, y))


# ============================================================
# 3. MAIN
# ============================================================
def main() -> None:
    ensure_dirs()

    for fp in [A_PNG, B_PNG, C_PNG, D_PNG]:
        if not fp.exists():
            raise FileNotFoundError(fp)

    A = trim_white(Image.open(A_PNG))
    B = trim_white(Image.open(B_PNG))
    C = trim_white(Image.open(C_PNG))
    D = trim_white(Image.open(D_PNG))

    A2 = fit_panel(A, TOP_LEFT_W, TOP_H)
    B2 = fit_panel(B, TOP_RIGHT_W, TOP_H)
    C2 = fit_panel(C, BOT_LEFT_W, BOT_H)
    D2 = fit_panel(D, BOT_RIGHT_W, BOT_H)

    canvas_h = 2 * MARGIN + TOP_H + GAP_Y + BOT_H
    canvas = Image.new("RGB", (CANVAS_W, canvas_h), "white")

    # Top row
    xL = MARGIN
    xR = MARGIN + TOP_LEFT_W + GAP_X
    yT = MARGIN

    paste_top_aligned(canvas, A2, xL, yT, TOP_LEFT_W)
    paste_top_aligned(canvas, B2, xR, yT, TOP_RIGHT_W)

    # Bottom row
    yB = MARGIN + TOP_H + GAP_Y
    xL_bot = MARGIN
    xR_bot = MARGIN + BOT_LEFT_W + GAP_X

    paste_top_aligned(canvas, C2, xL_bot, yB, BOT_LEFT_W)
    paste_top_aligned(canvas, D2, xR_bot, yB, BOT_RIGHT_W)

    canvas.save(OUT_PNG)

    arr = np.asarray(canvas)
    fig = plt.figure(figsize=(CANVAS_W / 300, canvas_h / 300), dpi=600)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(arr)
    ax.axis("off")
    fig.savefig(OUT_PDF, dpi=600, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

    print(f"[DONE] Saved {OUT_PNG}")
    print(f"[DONE] Saved {OUT_PDF}")


if __name__ == "__main__":
    main()
