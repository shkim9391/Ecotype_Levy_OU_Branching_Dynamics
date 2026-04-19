python - <<'PY'
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path("/GSE235063/derived_dx_primary_training")

frac = pd.read_csv(ROOT / "dx_ou_malignant_state_fractions_by_sample_filtered.csv", index_col=0)
design = pd.read_csv(ROOT / "dx_ou_training_design_matrix.csv")

# ensure missing columns are present
for c in ["HSC", "Progenitor", "GMP", "Monocytes", "cDC", "Early.Erythrocyte", "Early.Basophil", "CLP"]:
    if c not in frac.columns:
        frac[c] = 0.0

coarse = pd.DataFrame(index=frac.index)
coarse["state_HSC"] = frac["HSC"]
coarse["state_Prog"] = frac["Progenitor"]
coarse["state_GMP"] = frac["GMP"]
coarse["state_MonoDC"] = frac["Monocytes"] + frac["cDC"]

# auxiliary side programs
coarse["aux_EryBaso"] = frac["Early.Erythrocyte"] + frac["Early.Basophil"]
coarse["aux_CLP"] = frac["CLP"]

# renormalized 4-state core simplex
core_cols = ["state_HSC", "state_Prog", "state_GMP", "state_MonoDC"]
core = coarse[core_cols].copy()
core = core.div(core.sum(axis=1), axis=0).fillna(0)

# CLR transform for compositional modeling
eps = 1e-4
core_clr = np.log(core + eps)
core_clr = core_clr.sub(core_clr.mean(axis=1), axis=0)

coarse.to_csv(ROOT / "dx_ou_malignant_coarse_states_with_aux.csv")
core.to_csv(ROOT / "dx_ou_malignant_core4_fractions.csv")
core_clr.to_csv(ROOT / "dx_ou_malignant_core4_clr.csv")

# attach to design matrix
design2 = design.set_index("sample_id").join(core).join(core_clr.add_prefix("clr_")).join(
    coarse[["aux_EryBaso", "aux_CLP"]]
)
design2.to_csv(ROOT / "dx_ou_training_design_matrix_core4.csv")

# edge list for your OU-branching graph
edges = pd.DataFrame({
    "parent": ["state_HSC", "state_Prog", "state_GMP", "state_Prog", "state_HSC"],
    "child":  ["state_Prog", "state_GMP", "state_MonoDC", "aux_EryBaso", "aux_CLP"]
})
edges.to_csv(ROOT / "dx_ou_core4_branching_edges.csv", index=False)

print("\n=== CORE 4-STATE FRACTIONS ===")
print(core.round(4).to_string())

print("\n=== AUXILIARY PROGRAMS ===")
print(coarse[["aux_EryBaso", "aux_CLP"]].round(4).to_string())

print("\n=== FILES WRITTEN ===")
for fn in [
    "dx_ou_malignant_coarse_states_with_aux.csv",
    "dx_ou_malignant_core4_fractions.csv",
    "dx_ou_malignant_core4_clr.csv",
    "dx_ou_training_design_matrix_core4.csv",
    "dx_ou_core4_branching_edges.csv",
]:
    print(ROOT / fn)
PY
