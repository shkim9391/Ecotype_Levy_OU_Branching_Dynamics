# GSE235063: Ecotype–Lévy–OU Dynamics

This directory contains the code, derived tables, and model outputs used to process the **GSE235063** pediatric leukemia cohort and to build the Ecotype–Lévy–OU analysis workflow.

The current release is organized to support **transparency and reproducibility of the computational pipeline** while keeping the repository lightweight during manuscript submission and revision.  
**Figure image files are intentionally omitted at this stage** and will be added in a later update after manuscript submission and project progression.

---

## Overview

The workflow in this directory proceeds from:

1. preprocessing and cohort construction,
2. treatment/outcome subgroup definition,
3. ecotype assignment and refinement,
4. preparation of Ornstein–Uhlenbeck (OU) model inputs,
5. fitting and comparing compact equilibrium-style models,
6. generation of manuscript summary figures.

The repository currently includes:

- analysis scripts,
- derived diagnosis-stage training tables,
- model coefficient, prediction, performance, and variance summaries.

It currently does **not** include:

- rendered figure image files,
- raw large intermediate objects not needed for the present release.

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
    │   ├── small_model_coefficients__full19.csv
    │   ├── small_model_predictions__full19.csv
    │   ├── small_model_performance__full19.csv
    │   ├── small_model_sigmahat__full19.csv
    │   └── small_model_results_compact.csv
    ├── ou_equilibrium/
    │   ├── ou_equilibrium_coefficients__full19.csv
    │   ├── ou_equilibrium_performance__full19.csv
    │   ├── ou_equilibrium_predictions__full19.csv
    │   └── ou_equilibrium_sigmahat__full19.csv
    └── ou_equilibrium_plus_subgroup/
        ├── ou_equilibrium_plus_subgroup_coefficients__full19.csv
        ├── ou_equilibrium_plus_subgroup_performance__full19.csv
        ├── ou_equilibrium_plus_subgroup_predictions__full19.csv
        └── ou_equilibrium_plus_subgroup_sigmahat__full19.csv
