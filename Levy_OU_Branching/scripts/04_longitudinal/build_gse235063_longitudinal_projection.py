from pathlib import Path
import pandas as pd
import numpy as np
import re

# --------------------------------------------------
# Paths
# --------------------------------------------------
raw_dir = Path("/GSE235063/GSE235063_RAW")
base = Path("/Lévy_OU_Branching")
base.mkdir(parents=True, exist_ok=True)

manifest_file = base / "gse235063_longitudinal_manifest_fixed.csv"
broad_file = base / "gse235063_broad_cellgroup_fractions_by_sample.csv"
dx_ref_file = base / "dx_diagnosis_baseline_matrix_full.csv"
centroids_file = base / "dx_branch_centroids_k3.csv"

out_counts = base / "gse235063_malignant_coarse_counts_by_sample.csv"
out_calib = base / "gse235063_ilr_pseudocount_calibration.csv"
out_state = base / "gse235063_longitudinal_state_table.csv"
out_proj = base / "gse235063_longitudinal_branch_projection.csv"
out_traj = base / "gse235063_patient_branch_trajectories.csv"

pattern = re.compile(r"^(GSM\d+)_(AML\d+)_(DX|REM|REL)_processed_metadata\.tsv\.gz$")

# --------------------------------------------------
# Read helper tables
# --------------------------------------------------
manifest = pd.read_csv(manifest_file)
broad = pd.read_csv(broad_file)
dx_ref = pd.read_csv(dx_ref_file)
centroids = pd.read_csv(centroids_file)

for df in [manifest, broad, dx_ref, centroids]:
    if "sample_id" in df.columns:
        df["sample_id"] = df["sample_id"].astype(str).str.strip()

# Rename TME projection columns to match DX scaffold names
rename_tme = {
    "T_NK_given_known_z_dxtrain": "T_NK_given_known_z",
    "Myeloid_APC_given_known_z_dxtrain": "Myeloid_APC_given_known_z",
    "B_Plasma_given_known_z_dxtrain": "B_Plasma_given_known_z",
}
broad = broad.rename(columns=rename_tme)

# --------------------------------------------------
# Step 1. Build malignant coarse counts from raw metadata
# --------------------------------------------------
rows = []

for f in sorted(raw_dir.glob("*_processed_metadata.tsv.gz")):
    m = pattern.match(f.name)
    if not m:
        continue

    gsm, sample, timepoint = m.groups()
    df = pd.read_csv(f, sep="\t", compression="gzip")

    # robust sample_id
    library_ids = df["Library_ID"].dropna().astype(str).str.strip().unique() if "Library_ID" in df.columns else []
    sample_id = library_ids[0] if len(library_ids) > 0 else f"{sample}_{timepoint}"

    mal_df = df.loc[df["Malignant"].astype(str).str.strip().eq("Malignant")].copy()
    vc = mal_df["Classified_Celltype"].astype(str).value_counts() if len(mal_df) > 0 else pd.Series(dtype=int)

    row = {
        "gsm": gsm,
        "sample": sample,
        "timepoint": timepoint,
        "sample_id": sample_id,
        "malignant_cells": int(len(mal_df)),
        "HSC": int(vc.get("HSC", 0)),
        "Progenitor": int(vc.get("Progenitor", 0)),
        "GMP": int(vc.get("GMP", 0)),
        "Monocytes": int(vc.get("Monocytes", 0)),
        "cDC": int(vc.get("cDC", 0)),
        "CLP": int(vc.get("CLP", 0)),
        "Early.Basophil": int(vc.get("Early.Basophil", 0)),
        "Early.Erythrocyte": int(vc.get("Early.Erythrocyte", 0)),
    }
    rows.append(row)

counts = pd.DataFrame(rows).sort_values(["sample", "timepoint", "gsm"]).reset_index(drop=True)
counts["MonoDC"] = counts["Monocytes"] + counts["cDC"]
counts["EryBaso"] = counts["Early.Basophil"] + counts["Early.Erythrocyte"]
counts.to_csv(out_counts, index=False)

# --------------------------------------------------
# Step 2. Derive longitudinal ILR state table
# --------------------------------------------------
def derive_states(df_counts: pd.DataFrame, pseudocount: float) -> pd.DataFrame:
    out = df_counts.copy()

    core_parts = ["HSC", "Progenitor", "GMP", "MonoDC"]
    aux_parts = ["EryBaso", "CLP"]
    six_parts = core_parts + aux_parts

    # add pseudocount to the 6-part composition
    for c in six_parts:
        out[f"{c}_pc"] = out[c].astype(float) + pseudocount

    out["core_total_pc"] = out[[f"{c}_pc" for c in core_parts]].sum(axis=1)
    out["six_total_pc"] = out[[f"{c}_pc" for c in six_parts]].sum(axis=1)

    # core state fractions
    out["state_HSC"] = out["HSC_pc"] / out["core_total_pc"]
    out["state_Prog"] = out["Progenitor_pc"] / out["core_total_pc"]
    out["state_GMP"] = out["GMP_pc"] / out["core_total_pc"]
    out["state_MonoDC"] = out["MonoDC_pc"] / out["core_total_pc"]

    # aux fractions on 6-part composition
    out["aux_EryBaso"] = out["EryBaso_pc"] / out["six_total_pc"]
    out["aux_CLP"] = out["CLP_pc"] / out["six_total_pc"]

    # ILR / balance coordinates on the 4-part core composition
    out["ilr_stem_vs_committed"] = np.sqrt(3.0/4.0) * np.log(
        out["state_HSC"] / (out["state_Prog"] * out["state_GMP"] * out["state_MonoDC"]) ** (1.0/3.0)
    )
    out["ilr_prog_vs_mature"] = np.sqrt(2.0/3.0) * np.log(
        out["state_Prog"] / np.sqrt(out["state_GMP"] * out["state_MonoDC"])
    )
    out["ilr_gmp_vs_monodc"] = np.sqrt(1.0/2.0) * np.log(
        out["state_GMP"] / out["state_MonoDC"]
    )

    out["log_aux_erybaso"] = np.log(out["aux_EryBaso"])
    out["log_aux_clp"] = np.log(out["aux_CLP"])

    out["pseudocount"] = pseudocount
    return out

# --------------------------------------------------
# Step 3. Calibrate pseudocount on DX against frozen DX values
# --------------------------------------------------
candidate_pcs = [0.01, 0.05, 0.10, 0.25, 0.50, 1.00]

frozen_cols = [
    "state_HSC", "state_Prog", "state_GMP", "state_MonoDC",
    "aux_EryBaso", "aux_CLP",
    "ilr_stem_vs_committed", "ilr_prog_vs_mature", "ilr_gmp_vs_monodc",
    "log_aux_erybaso", "log_aux_clp",
]

dx_ref_sub = dx_ref[["sample_id"] + [c for c in frozen_cols if c in dx_ref.columns]].copy()
dx_ref_sub = dx_ref_sub.dropna(subset=["ilr_stem_vs_committed", "ilr_prog_vs_mature", "ilr_gmp_vs_monodc"])

calib_rows = []

for pc in candidate_pcs:
    d = derive_states(counts, pseudocount=pc)
    d_dx = d.loc[d["timepoint"] == "DX"].merge(dx_ref_sub, on="sample_id", how="inner", suffixes=("_rebuild", "_frozen"))

    row = {"pseudocount": pc, "n_dx_compared": len(d_dx)}
    total_sq = 0.0
    total_n = 0

    for c in frozen_cols:
        rc = f"{c}_rebuild"
        fc = f"{c}_frozen"
        if rc in d_dx.columns and fc in d_dx.columns:
            diff = d_dx[rc] - d_dx[fc]
            rmse = np.sqrt(np.mean(diff ** 2))
            row[f"rmse_{c}"] = rmse
            if c.startswith("ilr_"):
                total_sq += np.sum(diff ** 2)
                total_n += len(diff)

    row["rmse_ilr_total"] = np.sqrt(total_sq / total_n) if total_n > 0 else np.nan
    calib_rows.append(row)

calib = pd.DataFrame(calib_rows).sort_values("rmse_ilr_total").reset_index(drop=True)
calib.to_csv(out_calib, index=False)

best_pc = float(calib.iloc[0]["pseudocount"])
print("Best pseudocount by ILR RMSE:", best_pc)

# --------------------------------------------------
# Step 4. Build final state table with best pseudocount
# --------------------------------------------------
state = derive_states(counts, pseudocount=best_pc)

# merge manifest
state = manifest.merge(
    state[[
        "sample_id", "HSC", "Progenitor", "GMP", "Monocytes", "cDC", "CLP", "Early.Basophil",
        "Early.Erythrocyte", "MonoDC", "EryBaso", "malignant_cells",
        "state_HSC", "state_Prog", "state_GMP", "state_MonoDC",
        "aux_EryBaso", "aux_CLP",
        "ilr_stem_vs_committed", "ilr_prog_vs_mature", "ilr_gmp_vs_monodc",
        "log_aux_erybaso", "log_aux_clp", "pseudocount"
    ]],
    on=["sample_id", "malignant_cells"],
    how="left",
)

print("\nState table shape after manifest merge:", state.shape)
print("Rows missing rebuilt ILR block:",
      state[["ilr_stem_vs_committed", "ilr_prog_vs_mature", "ilr_gmp_vs_monodc"]].isna().all(axis=1).sum())

# merge TME block
keep_broad_cols = [
    "sample_id",
    "T_NK_given_known_z",
    "Myeloid_APC_given_known_z",
    "B_Plasma_given_known_z",
    "HSPC_Prog_given_known_z_dxtrain",
    "Erythroid_Baso_given_known_z_dxtrain",
    "T_NK", "Myeloid_APC", "B_Plasma", "HSPC_Prog", "Erythroid_Baso", "Unknown"
]
keep_broad_cols = [c for c in keep_broad_cols if c in broad.columns]

state = state.merge(broad[keep_broad_cols], on="sample_id", how="left")

print("Rows missing TME block:",
      state[["T_NK_given_known_z", "Myeloid_APC_given_known_z", "B_Plasma_given_known_z"]].isna().all(axis=1).sum())

# branch eligibility
state["projection_eligible"] = state["malignant_cells"].fillna(0) >= 20

state_cols = [
    "ilr_stem_vs_committed",
    "ilr_prog_vs_mature",
    "ilr_gmp_vs_monodc",
    "T_NK_given_known_z",
    "Myeloid_APC_given_known_z",
    "B_Plasma_given_known_z",
]

for c in state_cols:
    if c not in state.columns:
        raise ValueError(f"Missing required projection column: {c}")

state["projection_eligible"] = state["projection_eligible"] & (~state[state_cols].isna().any(axis=1))

# --------------------------------------------------
# Step 5. Project to frozen DX branch centroids
# --------------------------------------------------
cent = centroids[["branch_label"] + state_cols].copy()

# force numeric after merges
for c in state_cols:
    state[c] = pd.to_numeric(state[c], errors="coerce")
    cent[c] = pd.to_numeric(cent[c], errors="coerce")

print("\nState dtypes for projection:")
print(state[state_cols].dtypes)

print("\nCentroid dtypes for projection:")
print(cent[state_cols].dtypes)

state_mat = state[state_cols].to_numpy(dtype=float)

for _, row in cent.iterrows():
    b = row["branch_label"]
    centroid_vec = row[state_cols].to_numpy(dtype=float)
    d = np.sqrt(np.sum((state_mat - centroid_vec) ** 2, axis=1))
    state[f"dist_{b}"] = d

dist_cols = [c for c in state.columns if c.startswith("dist_B")]
state["projected_branch"] = pd.NA
state.loc[state["projection_eligible"], "projected_branch"] = (
    state.loc[state["projection_eligible"], dist_cols]
    .idxmin(axis=1)
    .str.replace("dist_", "", regex=False)
)

# --------------------------------------------------
# Step 6. Patient trajectory summary
# --------------------------------------------------
traj = state[[
    "sample", "Patient_ID", "sample_id", "timepoint", "time_index",
    "Biopsy_Origin", "Subgroup", "Treatment_Outcome",
    "n_cells", "malignant_cells", "normal_cells",
    "malignant_frac", "normal_frac",
    "projection_eligible", "projected_branch"
] + dist_cols].copy()

traj = traj.sort_values(["sample", "time_index", "sample_id"]).reset_index(drop=True)

# save
state.to_csv(out_state, index=False)
state.to_csv(out_proj, index=False)
traj.to_csv(out_traj, index=False)

print("\nCalibration table:")
print(calib.to_string(index=False))

print("\nProjection eligibility by timepoint:")
print(state.groupby("timepoint")["projection_eligible"].sum().to_string())

print("\nProjected branch counts by timepoint:")
print(pd.crosstab(state["timepoint"], state["projected_branch"], dropna=False).to_string())

print("\nFirst 20 projected rows:")
print(
    traj.head(20).to_string(index=False)
)

print(f"\nsaved: {out_counts}")
print(f"saved: {out_calib}")
print(f"saved: {out_state}")
print(f"saved: {out_proj}")
print(f"saved: {out_traj}")
