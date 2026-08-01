# Levy_OU_Branching

This repository contains the code, derived tables, and selected output files used to build and evaluate a **Lévy–OU–Branching** framework for pediatric leukemia evolution.

The workflow is centered on **diagnosis-stage ecological state construction**, **branch scaffold definition**, **longitudinal diagnosis-to-relapse projection**, and **displacement-based analysis of punctuated change**. In the current repository layout, the emphasis is on making the computational pipeline transparent and reusable through organized scripts and derived outputs.

## Overview

The repository is organized around seven main analysis stages:

1. building the diagnosis baseline matrix and model-ready state tables,
2. performing state-space quality control,
3. constructing a diagnosis branch scaffold,
4. projecting longitudinal samples onto the learned scaffold,
5. evaluating transition behavior across branch-switch thresholds,
6. quantifying diagnosis-to-relapse displacement and jump-like candidates,
7. generating summary figures for manuscript presentation.

The repository includes:

- analysis scripts,
- derived diagnosis-stage and longitudinal summary tables,
- branch scaffold outputs,
- threshold-sensitivity summaries,
- displacement and jump-candidate tables,
- selected intermediate and visualization outputs.

## Directory structure

```text
Levy_OU_Branching/
├── README.md
├── scripts/
│   ├── 01_baseline/
│   │   ├── diagnosis_baseline_matrix.py
│   │   ├── merge_diagnosis_baseline_matrix.py
│   │   ├── build_dx_diagnosis_baseline_matrix_full.py
│   │   ├── build_dx_state_ready_matrix.py
│   │   └── build_dx_cohort_provenance.py
│   ├── 02_qc/
│   │   ├── inspect_dx_state_missingness.py
│   │   └── plot_dx_state_space_qc.py
│   ├── 03_branch_scaffold/
│   │   ├── build_dx_branch_scaffold.py
│   │   └── characterize_dx_branches_k3.py
│   ├── 04_longitudinal/
│   │   ├── build_gse235063_longitudinal_manifest.py
│   │   ├── inspect_gse235063_raw_metadata_semantics.py
│   │   ├── build_gse235063_longitudinal_sample_tables.py
│   │   └── build_gse235063_longitudinal_projection.py
│   ├── 05_transition/
│   │   ├── transition_summary_threshold_sensitivity.py
│   │   └── make_dx_rel_transition_figures.py
│   ├── 06_displacement/
│   │   ├── dx_rel_displacement_analysis.py
│   │   └── build_dx_rel_jump_candidate_table.py
│   └── 07_figures/
│       ├── make_levy_summary_figure2_threshold50.py
│       ├── make_figure3_non_gaussian.py
│       ├── make_figure4_model_comparison.py
│       └── make_figure1_conceptual_hybrid_ou_levy_branching_v3.py
│
├── derived_dx_baseline/
│   ├── dx_broad_cellgroup_fractions_by_sample.csv
│   ├── dx_broad_cellgroup_fractions_summary.csv
│   ├── dx_diagnosis_baseline_matrix_minimal.csv
│   ├── dx_missing_samples_report.txt
│   ├── dx_diagnosis_baseline_matrix_full.csv
│   ├── dx_diagnosis_baseline_matrix_model_ready.csv
│   ├── dx_diagnosis_state_ready_matrix.csv
│   ├── dx_diagnosis_state_ready_excluded_samples.csv
│   ├── dx_cohort_provenance_table.csv
│   └── dx_transfer_missing_after_eligibility.csv
│
├── derived_dx_qc/
│   ├── dx_state_missingness_report.csv
│   ├── dx_state_correlation_matrix.csv
│   ├── dx_state_pca_scores.csv
│   ├── dx_state_centroid_distance_ranking.csv
│
├── derived_dx_branch_scaffold/
│   ├── dx_branch_kmeans_model_selection.csv
│   ├── dx_branch_assignments_k3.csv
│   ├── dx_branch_centroids_k3.csv
│   ├── dx_branch_membership_summary_k3.txt
│   ├── dx_state_pca_by_branch_k3.png
│   ├── dx_branch_sizes_k3.png
│   ├── dx_branch_state_means_k3.csv
│   ├── dx_branch_state_sds_k3.csv
│   ├── dx_branch_characterization_k3.txt
│
├── derived_longitudinal/
│   ├── gse235063_longitudinal_manifest_raw.csv
│   ├── gse235063_longitudinal_metadata_columns_summary.txt
│   ├── gse235063_longitudinal_manifest_fixed.csv
│   ├── gse235063_allcells_celltype_counts_by_sample.csv
│   ├── gse235063_allcells_celltype_fractions_by_sample.csv
│   ├── gse235063_broad_cellgroup_fractions_by_sample.csv
│   ├── gse235063_normal_celltype_counts_by_sample.csv
│   ├── gse235063_normal_celltype_fractions_by_sample.csv
│   ├── gse235063_normal_broad_cellgroup_fractions_by_sample.csv
│   ├── gse235063_malignant_coarse_counts_by_sample.csv
│   ├── gse235063_ilr_pseudocount_calibration.csv
│   ├── gse235063_longitudinal_state_table.csv
│   ├── gse235063_longitudinal_branch_projection.csv
│   └── gse235063_patient_branch_trajectories.csv
│
├── derived_transition_sensitivity/
│   ├── gse235063_branch_counts_by_timepoint_thresholds.csv
│   ├── gse235063_dx_to_rel_transition_summary_by_threshold.csv
│   ├── gse235063_patient_branch_table_by_threshold.csv
│   ├── gse235063_dx_to_rel_switch_rates_by_threshold.csv
│   ├── gse235063_transition_summary_threshold_sensitivity.txt
│
└── derived_dx_rel_threshold50/
    ├── gse235063_dx_rel_transition_matrix_threshold50.csv
    ├── gse235063_dx_rel_patient_pairs_threshold50.csv
    ├── gse235063_dx_rel_pca_scores_threshold50.csv
    ├── gse235063_dx_rel_displacement_table_threshold50.csv
    ├── gse235063_dx_rel_displacement_summary_threshold50.csv
    ├── gse235063_dx_rel_displacement_permutation_tests_threshold50.csv
    ├── gse235063_dx_rel_displacement_summary_threshold50.txt
    ├── gse235063_dx_rel_jump_candidate_table_threshold50.csv
    └── gse235063_dx_rel_jump_candidate_summary_threshold50.txt
