from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


# ============================================================
# 1. CONFIG
# ============================================================
PROJECT_DIR = Path("/Multimodal_OU_Lévy_Branching_Scaffold/Figure_3")
DERIVED_DIR = PROJECT_DIR / "derived"

IN_ALL = DERIVED_DIR / "patient_timepoint_centroids_all.csv"
IN_MAIN = DERIVED_DIR / "patient_timepoint_centroids_main.csv"

OUT_ALL = DERIVED_DIR / "patient_interval_metrics_all.csv"
OUT_MAIN = DERIVED_DIR / "patient_interval_metrics_main.csv"

TIME_ORDER = {"DX": 0, "EOI_REM": 1, "REL": 2}


# ============================================================
# 2. HELPERS
# ============================================================
def euclidean(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.sum((a - b) ** 2)))


def directional_discontinuity(v1: np.ndarray, v2: np.ndarray) -> float:
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return np.nan
    cos_sim = float(np.dot(v1, v2) / (n1 * n2))
    cos_sim = np.clip(cos_sim, -1.0, 1.0)
    return 1.0 - cos_sim


def build_interval_record(
    row_start: pd.Series,
    row_end: pd.Series,
    hd_cols: list[str],
    *,
    prior_row: pd.Series | None = None,
    interval_kind: str | None = None,
) -> dict:
    start_xy = np.array([row_start["x2d"], row_start["y2d"]], dtype=float)
    end_xy = np.array([row_end["x2d"], row_end["y2d"]], dtype=float)

    start_hd = row_start[hd_cols].to_numpy(dtype=float)
    end_hd = row_end[hd_cols].to_numpy(dtype=float)

    interval_class = interval_kind if interval_kind is not None else f"{row_start['clinical_timepoint_coarse']}_to_{row_end['clinical_timepoint_coarse']}"

    rec = {
        "patient_id": row_start["patient_id"],
        "interval_class": interval_class,
        "t_start": row_start["clinical_timepoint_coarse"],
        "t_end": row_end["clinical_timepoint_coarse"],
        "sample_start": row_start["sample_id"],
        "sample_end": row_end["sample_id"],
        "n_cells_start": int(row_start["n_cells"]),
        "n_cells_end": int(row_end["n_cells"]),
        "displacement_2d": euclidean(start_xy, end_xy),
        "displacement_hd": euclidean(start_hd, end_hd),
        "direction_x": float(end_xy[0] - start_xy[0]),
        "direction_y": float(end_xy[1] - start_xy[1]),
        "branch_start": row_start.get("branch_id_dominant", "Unknown"),
        "branch_end": row_end.get("branch_id_dominant", "Unknown"),
        "branch_switch": int(str(row_start.get("branch_id_dominant", "Unknown")) != str(row_end.get("branch_id_dominant", "Unknown"))),
        "ecotype_start": row_start.get("ecotype_dominant", "Unknown"),
        "ecotype_end": row_end.get("ecotype_dominant", "Unknown"),
        "reg_score_start": row_start.get("reg_program_score_median", np.nan),
        "reg_score_end": row_end.get("reg_program_score_median", np.nan),
        "disease_subgroup": row_start.get("disease_subgroup", "Unknown"),
    }

    if prior_row is not None:
        p0 = np.array([prior_row["x2d"], prior_row["y2d"]], dtype=float)
        p1 = np.array([row_start["x2d"], row_start["y2d"]], dtype=float)
        p2 = np.array([row_end["x2d"], row_end["y2d"]], dtype=float)
        v1 = p1 - p0
        v2 = p2 - p1
        rec["directional_discontinuity"] = directional_discontinuity(v1, v2)
    else:
        rec["directional_discontinuity"] = np.nan

    return rec


def compute_intervals(cent: pd.DataFrame) -> pd.DataFrame:
    req = ["patient_id", "sample_id", "clinical_timepoint_coarse", "x2d", "y2d", "n_cells"]
    missing = [c for c in req if c not in cent.columns]
    if missing:
        raise ValueError(f"Missing required centroid columns: {missing}")

    hd_cols = [c for c in cent.columns if c.startswith("hd_")]
    if not hd_cols:
        raise ValueError("No high-dimensional centroid columns found (expected hd_1 ... hd_k).")

    cent = cent.copy()
    cent["time_order"] = cent["clinical_timepoint_coarse"].map(TIME_ORDER)
    cent = cent.sort_values(["patient_id", "time_order", "sample_id"]).reset_index(drop=True)

    rows = []

    for patient_id, sub in cent.groupby("patient_id", sort=False):
        recs = {r["clinical_timepoint_coarse"]: r for _, r in sub.iterrows()}

        # Direct DX -> REL interval for all patients with both endpoints
        if "DX" in recs and "REL" in recs:
            rows.append(
                build_interval_record(
                    recs["DX"],
                    recs["REL"],
                    hd_cols,
                    interval_kind="DX_to_REL",
                )
            )

        # Explicit remission-linked intervals when present
        if "DX" in recs and "EOI_REM" in recs:
            rows.append(
                build_interval_record(
                    recs["DX"],
                    recs["EOI_REM"],
                    hd_cols,
                    interval_kind="DX_to_EOI_REM",
                )
            )

        if "EOI_REM" in recs and "REL" in recs:
            prior_row = recs["DX"] if "DX" in recs else None
            rows.append(
                build_interval_record(
                    recs["EOI_REM"],
                    recs["REL"],
                    hd_cols,
                    prior_row=prior_row,
                    interval_kind="EOI_REM_to_REL",
                )
            )

    out = pd.DataFrame(rows)

    if out.empty:
        raise ValueError("No interval records were generated.")

    # Tail flag relative to the DX -> REL distribution for this table
    dx_rel = pd.to_numeric(
        out.loc[out["interval_class"] == "DX_to_REL", "displacement_hd"],
        errors="coerce"
    ).dropna()

    if len(dx_rel) > 0:
        tail_threshold = float(np.nanquantile(dx_rel, 0.90))
        out["tail_threshold_dx_rel_q90"] = tail_threshold
        out["tail_flag_dx_rel_q90"] = pd.to_numeric(out["displacement_hd"], errors="coerce") > tail_threshold
    else:
        out["tail_threshold_dx_rel_q90"] = np.nan
        out["tail_flag_dx_rel_q90"] = False

    out = out.sort_values(["patient_id", "interval_class", "sample_start", "sample_end"]).reset_index(drop=True)
    return out


# ============================================================
# 3. MAIN
# ============================================================
def main() -> None:
    cent_all = pd.read_csv(IN_ALL)
    cent_main = pd.read_csv(IN_MAIN)

    out_all = compute_intervals(cent_all)
    out_main = compute_intervals(cent_main)

    out_all.to_csv(OUT_ALL, index=False)
    out_main.to_csv(OUT_MAIN, index=False)

    print(f"[DONE] Saved all intervals:  {OUT_ALL}")
    print(f"[DONE] Saved main intervals: {OUT_MAIN}")

    print("\n[SUMMARY: all interval counts]")
    print(out_all["interval_class"].value_counts(dropna=False).sort_index())

    print("\n[SUMMARY: main interval counts]")
    print(out_main["interval_class"].value_counts(dropna=False).sort_index())

    print("\n[SUMMARY: main DX_to_REL displacement]")
    dxrel = pd.to_numeric(out_main.loc[out_main["interval_class"] == "DX_to_REL", "displacement_hd"], errors="coerce")
    if dxrel.notna().any():
        print(dxrel.describe())

    if "AML21" in set(out_main["patient_id"].astype(str)):
        print("\n[INFO] AML21 interval records:")
        print(
            out_main[out_main["patient_id"] == "AML21"][
                [
                    "patient_id", "interval_class",
                    "sample_start", "sample_end",
                    "displacement_2d", "displacement_hd",
                    "branch_start", "branch_end",
                    "branch_switch", "directional_discontinuity"
                ]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()
