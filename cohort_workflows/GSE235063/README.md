# GSE235063: Primary Discovery Cohort and Diagnosis-Anchored Scaffold

This directory contains the scripts, derived tables, and model outputs used to process the **GSE235063 longitudinal pediatric AML cohort**, construct the diagnosis-anchored disease-state scaffold, and generate the primary discovery analyses within the **Ecotype–Lévy–OU–Branching** framework.

GSE235063 serves as the **primary discovery cohort** for the project. It provides the reference disease-state representation used throughout the repository, including longitudinal projection, therapy-aware dynamic summaries, branch-conditioned analyses, and external cohort calibration.

The repository is organized to maximize **transparency, reproducibility, and modularity** while remaining lightweight. Large intermediate objects and rendered figure files are not distributed directly because they can be regenerated from the provided scripts and publicly available source data.

---

## Analytical role

Within the overall workflow, GSE235063 is responsible for:

- construction of the diagnosis-anchored disease-state scaffold;
- ecological-context inference;
- branch assignment;
- preparation of Ornstein–Uhlenbeck (OU) model inputs;
- longitudinal trajectory reconstruction;
- diagnosis-to-relapse displacement analysis;
- estimation of therapy-aware dynamic summaries;
- generation of the primary discovery results presented in the manuscript.

The frozen diagnosis scaffold generated from this cohort is subsequently used for:

- projection of the external pediatric AML cohort (GSE235923);
- supplementary cross-lineage transfer (GSE227122);
- conservative serial bulk validation (GSE163634).

---

## Workflow overview

The workflow proceeds through six major stages:

1. diagnosis-stage cohort construction and preprocessing;
2. treatment and clinical subgroup definition;
3. ecological-context assignment and refinement;
4. preparation of malignant-state and OU modeling inputs;
5. fitting and comparison of compact equilibrium-style models;
6. generation of manuscript summary figures and derived outputs.

The repository includes:

- executable analysis scripts;
- diagnosis-stage training tables;
- ecological summaries;
- design matrices;
- branch-ready latent-state variables;
- model coefficients;
- prediction summaries;
- performance metrics;
- variance estimates.

Large intermediate AnnData objects and rendered figure files are intentionally excluded from this directory and are regenerated during the manuscript workflow.

---

## Directory structure

```text
GSE235063/
├── README.md
├── scripts/
│   ├── 01_build/
│   │   └── build_diagnosis_only_cohort.py
│   ├── 02_subgroup/
│   │   └── subgroup_treatment_outcome.py
│   ├── 03_ecotype/
│   │   ├── assign_first_pass.py
│   │   └── refined_ecotype_pass.py
│   ├── 04_ou_inputs/
│   │   ├── build_malignant_state_input.py
│   │   ├── build_coarse_state_space.py
│   │   └── make_ilr_branch_ready.py
│   ├── 05_models/
│   │   ├── ou_equilibrium.py
│   │   ├── ou_equilibrium_subgroup.py
│   │   └── small_model.py
│   └── 06_figures/
│       ├── make_summary_figure.py
│       └── make_summary_figure_journal.py
│
├── derived_dx_primary_training/
│   ├── dx_primary_training_sample_summary.csv
│   ├── dx_primary_training_sample_level_summary.csv
│   ├── dx_allcells_celltype_counts_by_sample.csv
│   ├── dx_allcells_celltype_fractions_by_sample.csv
│   ├── dx_normal_broad_cellgroup_fractions_by_sample.csv
│   ├── dx_ecotype_firstpass_assignments.csv
│   ├── dx_ecotype_refined_fine_assignments.csv
│   ├── dx_ecotype_refined_fine_cluster_means.csv
│   ├── dx_ecotype_continuous_covariates_for_ou.csv
│   ├── dx_ou_malignant_state_fractions_by_sample_filtered.csv
│   ├── dx_ou_malignant_core4_fractions.csv
│   ├── dx_ou_training_design_matrix.csv
│   ├── dx_ou_training_design_matrix_core4.csv
│   ├── dx_ou_core4_branching_edges.csv
│   └── dx_ou_ilr_branch_ready.csv
│
└── model_outputs/
    ├── small_model/
    ├── ou_equilibrium/
    └── ou_equilibrium_plus_subgroup/
```

---

## Key outputs

The most important derived files include:

**Diagnosis scaffold**

- `dx_ou_training_design_matrix_core4.csv`
- `dx_ou_ilr_branch_ready.csv`

These files provide the diagnosis-stage latent variables used to construct the frozen disease-state scaffold.

**Ecological summaries**

- `dx_ecotype_firstpass_assignments.csv`
- `dx_ecotype_refined_fine_assignments.csv`
- `dx_ecotype_continuous_covariates_for_ou.csv`

These outputs define the ecological-context variables integrated into the diagnosis-anchored scaffold.

**Model outputs**

The `model_outputs/` directory contains:

- model coefficients;
- prediction tables;
- model-performance summaries;
- estimated variance parameters.

These files document the compact equilibrium-style OU models used during scaffold development.

---

## Interpretation

This directory contains the **discovery-stage analytical workflow**.

The outputs generated here form the reference disease-state representation used throughout the remainder of the repository.

Subsequent analyses—including longitudinal projection, therapy-aware dynamic summaries, external AML calibration, and conservative bulk validation—operate within this frozen diagnosis-anchored reference rather than reconstructing a new latent representation.

---

## Reproducibility notes

Scripts assume that the required GSE235063 input files have been downloaded from the Gene Expression Omnibus (GEO) and placed in the expected directory structure.

Users should update local file paths before execution.

The recommended execution order follows the numbered script directories:

```text
01_build/
02_subgroup/
03_ecotype/
04_ou_inputs/
05_models/
06_figures/
```

The outputs generated by this workflow are subsequently consumed by the manuscript-specific workflow under:

```text
manuscript_workflow/
Multimodal_OU_Levy_Branching_Scaffold/
```
