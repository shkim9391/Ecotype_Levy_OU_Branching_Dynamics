from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


# ============================================================
# 1. CONFIG
# ============================================================
PROJECT_DIR = Path("/Multimodal_OU_Lévy_Branching_Scaffold/Figure_3")
DERIVED_DIR = PROJECT_DIR / "derived"

IN_CSV = DERIVED_DIR / "patient_interval_metrics_main.csv"
OUT_CSV = DERIVED_DIR / "relapse_jump_candidates.csv"


# ============================================================
# 2. HELPERS
# ============================================================
def robust_z(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    if not np.isfinite(mad) or mad == 0:
        return pd.Series(np.zeros(len(x)), index=s.index, dtype=float)
    return 0.6745 * (x - med) / mad


# ============================================================
# 3. MAIN
# ============================================================
def main() -> None:
    df = pd.read_csv(IN_CSV)
    df = df[df["interval_class"] == "DX_to_REL"].copy()

    if df.empty:
        raise ValueError("No DX_to_REL intervals found in patient_interval_metrics_main.csv")

    df["displacement_hd"] = pd.to_numeric(df["displacement_hd"], errors="coerce")
    df["displacement_2d"] = pd.to_numeric(df["displacement_2d"], errors="coerce")
    df["branch_switch"] = pd.to_numeric(df["branch_switch"], errors="coerce").fillna(0).astype(int)

    df["z_displacement_hd"] = robust_z(df["displacement_hd"])
    df["z_displacement_2d"] = robust_z(df["displacement_2d"])

    df["jump_score"] = (
        df["z_displacement_hd"].fillna(0.0)
        + 0.50 * df["branch_switch"].astype(float)
    )

    df["jump_class"] = np.where(
        df["branch_switch"] == 1,
        "Branch-switching",
        "Branch-continuous"
    )

    df = df.sort_values(["jump_score", "displacement_hd"], ascending=[False, False]).reset_index(drop=True)
    df["jump_rank"] = np.arange(1, df.shape[0] + 1)

    cols_front = [
        "jump_rank",
        "patient_id",
        "interval_class",
        "sample_start",
        "sample_end",
        "displacement_hd",
        "displacement_2d",
        "branch_start",
        "branch_end",
        "branch_switch",
        "jump_class",
        "z_displacement_hd",
        "jump_score",
    ]
    rest = [c for c in df.columns if c not in cols_front]
    df = df[cols_front + rest]

    df.to_csv(OUT_CSV, index=False)

    print(f"[DONE] Saved {OUT_CSV}")
    print("\n[SUMMARY]")
    print(df[["patient_id", "sample_start", "sample_end", "displacement_hd", "branch_switch", "jump_score"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
