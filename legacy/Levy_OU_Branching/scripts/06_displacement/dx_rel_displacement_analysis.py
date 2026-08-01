from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

base = Path("/Lévy_OU_Branching")

pairs_file = base / "gse235063_dx_rel_patient_pairs_threshold50.csv"
long_file = base / "gse235063_longitudinal_branch_projection.csv"

out_disp = base / "gse235063_dx_rel_displacement_table_threshold50.csv"
out_summary = base / "gse235063_dx_rel_displacement_summary_threshold50.csv"
out_perm = base / "gse235063_dx_rel_displacement_permutation_tests_threshold50.csv"
out_text = base / "gse235063_dx_rel_displacement_summary_threshold50.txt"

fig_total = base / "gse235063_dx_rel_total_displacement_stable_vs_switch_threshold50.png"
fig_blocks = base / "gse235063_dx_rel_block_displacement_stable_vs_switch_threshold50.png"
fig_ranked = base / "gse235063_dx_rel_ranked_total_displacement_threshold50.png"

threshold = 50
rng = np.random.default_rng(42)

state_cols = [
    "ilr_stem_vs_committed",
    "ilr_prog_vs_mature",
    "ilr_gmp_vs_monodc",
    "T_NK_given_known_z",
    "Myeloid_APC_given_known_z",
    "B_Plasma_given_known_z",
]
malignant_cols = state_cols[:3]
tme_cols = state_cols[3:]
dist_cols = ["dist_B1", "dist_B2", "dist_B3"]

# --------------------------------------------------
# Helpers
# --------------------------------------------------
def euclidean(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.sqrt(np.sum((b - a) ** 2)))

def permutation_test_mean(x, y, n_perm=20000, seed=42):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]

    if len(x) == 0 or len(y) == 0:
        return np.nan

    rng_local = np.random.default_rng(seed)
    obs = np.mean(y) - np.mean(x)

    pooled = np.concatenate([x, y])
    n_x = len(x)

    perm_stats = np.empty(n_perm, dtype=float)
    for i in range(n_perm):
        perm = rng_local.permutation(pooled)
        perm_x = perm[:n_x]
        perm_y = perm[n_x:]
        perm_stats[i] = np.mean(perm_y) - np.mean(perm_x)

    p = np.mean(np.abs(perm_stats) >= abs(obs))
    return float(obs), float(p)

# --------------------------------------------------
# Load
# --------------------------------------------------
pairs = pd.read_csv(pairs_file)
long = pd.read_csv(long_file)

required_pairs = [
    "sample", "Patient_ID", "DX_branch_ge50", "REL_branch_ge50", "dx_to_rel_switch"
]
missing_pairs = [c for c in required_pairs if c not in pairs.columns]
if missing_pairs:
    raise ValueError(f"Missing columns in pairs file: {missing_pairs}")

required_long = [
    "sample", "sample_id", "Patient_ID", "timepoint", "malignant_cells"
] + state_cols + dist_cols
missing_long = [c for c in required_long if c not in long.columns]
if missing_long:
    raise ValueError(f"Missing columns in longitudinal file: {missing_long}")

for c in ["malignant_cells"] + state_cols + dist_cols:
    long[c] = pd.to_numeric(long[c], errors="coerce")

pairs = pairs.copy()
pairs["dx_to_rel_switch"] = pairs["dx_to_rel_switch"].astype(bool)

# only keep actual evaluable pairs
pairs = pairs[
    pairs["DX_branch_ge50"].notna() &
    pairs["REL_branch_ge50"].notna()
].copy()

# --------------------------------------------------
# Build one-row-per-patient displacement table
# --------------------------------------------------
rows = []

for _, pr in pairs.iterrows():
    sample = pr["sample"]

    dx = long[(long["sample"] == sample) & (long["timepoint"] == "DX")].copy()
    rel = long[(long["sample"] == sample) & (long["timepoint"] == "REL")].copy()

    if len(dx) == 0 or len(rel) == 0:
        continue

    dx = dx.iloc[0]
    rel = rel.iloc[0]

    row = {
        "sample": sample,
        "Patient_ID": pr["Patient_ID"],
        "Treatment_Outcome": pr["Treatment_Outcome"] if "Treatment_Outcome" in pr.index else pd.NA,
        "Subgroup": pr["Subgroup"] if "Subgroup" in pr.index else pd.NA,
        "DX_branch_ge50": pr["DX_branch_ge50"],
        "REL_branch_ge50": pr["REL_branch_ge50"],
        "dx_to_rel_switch": bool(pr["dx_to_rel_switch"]),
        "dx_malignant_cells": dx["malignant_cells"],
        "rel_malignant_cells": rel["malignant_cells"],
    }

    # total and block displacements
    row["disp_total_6d"] = euclidean(dx[state_cols], rel[state_cols])
    row["disp_malignant_3d"] = euclidean(dx[malignant_cols], rel[malignant_cols])
    row["disp_tme_3d"] = euclidean(dx[tme_cols], rel[tme_cols])

    # per-coordinate deltas
    for c in state_cols:
        row[f"dx_{c}"] = dx[c]
        row[f"rel_{c}"] = rel[c]
        row[f"delta_{c}"] = rel[c] - dx[c]

    # centroid-distance deltas
    for c in dist_cols:
        row[f"dx_{c}"] = dx[c]
        row[f"rel_{c}"] = rel[c]
        row[f"delta_{c}"] = rel[c] - dx[c]

    # distance to own DX and REL branches
    dx_branch = pr["DX_branch_ge50"]
    rel_branch = pr["REL_branch_ge50"]

    row["dx_dist_to_dx_branch"] = dx[f"dist_{dx_branch}"]
    row["rel_dist_to_dx_branch"] = rel[f"dist_{dx_branch}"]
    row["dx_dist_to_rel_branch"] = dx[f"dist_{rel_branch}"]
    row["rel_dist_to_rel_branch"] = rel[f"dist_{rel_branch}"]

    rows.append(row)

disp = pd.DataFrame(rows).sort_values(["dx_to_rel_switch", "disp_total_6d", "sample"], ascending=[False, False, True]).reset_index(drop=True)
disp.to_csv(out_disp, index=False)

# --------------------------------------------------
# Summary by stable vs switching
# --------------------------------------------------
summary_rows = []
for flag, label in [(False, "stable"), (True, "switching")]:
    sub = disp[disp["dx_to_rel_switch"] == flag].copy()

    for metric in ["disp_total_6d", "disp_malignant_3d", "disp_tme_3d"]:
        vals = sub[metric].dropna().to_numpy(dtype=float)
        summary_rows.append({
            "group": label,
            "metric": metric,
            "n": len(vals),
            "mean": float(np.mean(vals)) if len(vals) > 0 else np.nan,
            "median": float(np.median(vals)) if len(vals) > 0 else np.nan,
            "sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan,
            "min": float(np.min(vals)) if len(vals) > 0 else np.nan,
            "max": float(np.max(vals)) if len(vals) > 0 else np.nan,
        })

summary = pd.DataFrame(summary_rows)
summary.to_csv(out_summary, index=False)

# permutation tests
perm_rows = []
stable = disp.loc[~disp["dx_to_rel_switch"]].copy()
switch = disp.loc[disp["dx_to_rel_switch"]].copy()

for metric in ["disp_total_6d", "disp_malignant_3d", "disp_tme_3d"]:
    obs, p = permutation_test_mean(
        stable[metric].to_numpy(dtype=float),
        switch[metric].to_numpy(dtype=float),
        n_perm=20000,
        seed=42,
    )
    perm_rows.append({
        "metric": metric,
        "mean_switch_minus_stable": obs,
        "permutation_pvalue": p,
        "mean_stable": float(stable[metric].mean()),
        "mean_switching": float(switch[metric].mean()),
        "median_stable": float(stable[metric].median()),
        "median_switching": float(switch[metric].median()),
    })

perm = pd.DataFrame(perm_rows)
perm.to_csv(out_perm, index=False)

# --------------------------------------------------
# Text summary
# --------------------------------------------------
lines = []
lines.append("GSE235063 DX→REL displacement summary (threshold >= 50)")
lines.append("=" * 70)
lines.append("")
lines.append(f"n pairs = {len(disp)}")
lines.append(f"n stable = {int((~disp['dx_to_rel_switch']).sum())}")
lines.append(f"n switching = {int(disp['dx_to_rel_switch'].sum())}")
lines.append("")

lines.append("Stable vs switching summary:")
lines.append(summary.to_string(index=False))
lines.append("")

lines.append("Permutation tests (switching minus stable mean difference):")
lines.append(perm.to_string(index=False))
lines.append("")

lines.append("Per-patient displacement table:")
keep_cols = [
    "sample", "Patient_ID", "Subgroup",
    "DX_branch_ge50", "REL_branch_ge50", "dx_to_rel_switch",
    "dx_malignant_cells", "rel_malignant_cells",
    "disp_total_6d", "disp_malignant_3d", "disp_tme_3d"
]
lines.append(disp[keep_cols].to_string(index=False))

with open(out_text, "w") as f:
    f.write("\n".join(lines))

# --------------------------------------------------
# Figures
# --------------------------------------------------
# 1) Total displacement stable vs switching
plt.figure(figsize=(6, 4))
plot_data = [
    disp.loc[~disp["dx_to_rel_switch"], "disp_total_6d"].to_numpy(dtype=float),
    disp.loc[disp["dx_to_rel_switch"], "disp_total_6d"].to_numpy(dtype=float),
]
plt.boxplot(plot_data, tick_labels=["Stable", "Switching"])
plt.ylabel("DX→REL total displacement (6D)")
plt.title("DX→REL total displacement by branch stability")
plt.tight_layout()
plt.savefig(fig_total, dpi=300, bbox_inches="tight")
plt.close()

# 2) Malignant vs TME block displacement
fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=False)

mal_data = [
    disp.loc[~disp["dx_to_rel_switch"], "disp_malignant_3d"].to_numpy(dtype=float),
    disp.loc[disp["dx_to_rel_switch"], "disp_malignant_3d"].to_numpy(dtype=float),
]
axes[0].boxplot(mal_data, tick_labels=["Stable", "Switching"])
axes[0].set_title("Malignant block")
axes[0].set_ylabel("DX→REL displacement")

tme_data = [
    disp.loc[~disp["dx_to_rel_switch"], "disp_tme_3d"].to_numpy(dtype=float),
    disp.loc[disp["dx_to_rel_switch"], "disp_tme_3d"].to_numpy(dtype=float),
]
axes[1].boxplot(tme_data, tick_labels=["Stable", "Switching"])
axes[1].set_title("TME block")
axes[1].set_ylabel("DX→REL displacement")

fig.suptitle("DX→REL displacement by branch stability")
fig.tight_layout()
fig.savefig(fig_blocks, dpi=300, bbox_inches="tight")
plt.close(fig)

# 3) Ranked total displacement
ranked = disp.sort_values("disp_total_6d", ascending=False).copy()
plt.figure(figsize=(10, 4))
x = np.arange(len(ranked))
plt.bar(x, ranked["disp_total_6d"])
plt.xticks(x, ranked["sample"], rotation=45, ha="right")
plt.ylabel("DX→REL total displacement (6D)")
plt.title("Ranked DX→REL total displacement")
plt.tight_layout()
plt.savefig(fig_ranked, dpi=300, bbox_inches="tight")
plt.close()

# --------------------------------------------------
# Console
# --------------------------------------------------
print(f"saved: {out_disp}")
print(f"saved: {out_summary}")
print(f"saved: {out_perm}")
print(f"saved: {out_text}")
print(f"saved: {fig_total}")
print(f"saved: {fig_blocks}")
print(f"saved: {fig_ranked}")

print("\nStable vs switching summary:")
print(summary.to_string(index=False))

print("\nPermutation tests:")
print(perm.to_string(index=False))

print("\nPer-patient displacement table:")
print(disp[keep_cols].to_string(index=False))
