# Ecotype Lévy–OU–Branching Dynamics

This repository accompanies the manuscript:

## A therapy-aware multimodal dynamical scaffold reveals residual persistence and relapse-associated escape in pediatric leukemia

The repository contains the code, derived tables, workflow documentation, and selected outputs used to construct and evaluate a diagnosis-anchored, therapy-aware disease-state scaffold for pediatric leukemia.

The framework integrates complementary transcriptomic, inferred regulatory, ecological-composition, and clinical-phase feature blocks. Longitudinal disease-state movement is interpreted using an Ornstein–Uhlenbeck–Lévy–branching formulation that combines:

- constrained or mean-reverting within-state dynamics;
- branch-structured disease-state organization;
- therapy-associated attractor displacement;
- residual persistence;
- punctuated or jump-sensitive relapse-associated reorganization.

The repository distinguishes the final manuscript workflow from cohort-specific preprocessing, supplementary analyses, legacy exploratory workflows, and manuscript-linked data products.

---

## Scientific scope

The analyses were developed to investigate how pediatric leukemia samples occupy and move through a shared disease-state landscape across diagnosis, treatment, remission-related states, and relapse.

The computational workflow emphasizes:

- construction of diagnosis-anchored ecological and malignant-state representations;
- integration of transcriptomic, inferred regulatory, ecological, and clinical information;
- frozen-space projection of longitudinal and external samples;
- comparison of branch-continuous and branch-switching trajectories;
- diagnosis-to-relapse displacement analysis;
- identification of upper-tail and jump-sensitive departures;
- estimation of descriptive therapy-aware dynamic summaries;
- external pediatric AML calibration;
- cross-lineage transfer as a supplementary robustness analysis;
- conservative serial bulk validation.

The OU–Lévy–branching formulation is used as a conceptual statistical scaffold. The reported dynamic quantities are interpretable summaries of attraction, displacement, instability, branch behavior, and jump-sensitive movement rather than fully identified continuous-time stochastic-process parameters.

---

## Repository organization

```text
Ecotype_Levy_OU_Branching_Dynamics/
├── README.md
├── LICENSE
├── CITATION.cff
├── environment.yml
├── requirements.txt
│
├── manuscript_workflow/
│   └── Multimodal_OU_Levy_Branching_Scaffold/
│       ├── README.md
│       ├── workflow_manifest.tsv
│       │
│       ├── figure1_framework/
│       │   └── figure1_multimodal_therapy_aware_ou_levy_branching_full.py
│       │
│       ├── figure2_diagnosis_scaffold/
│       │   └── figure2_multimodal_latent_landscape.py
│       │
│       ├── figure3_longitudinal_projection/
│       │   ├── longitudinal_cohort_manifest.py
│       │   ├── 01_build_longitudinal_malignant_object.py
│       │   ├── 02_project_longitudinal_cells_into_frozen_scaffold.py
│       │   ├── 03_compute_patient_timepoint_centroids.py
│       │   ├── 04_compute_patient_interval_metrics.py
│       │   ├── 05_plot_figure3A_patient_trajectories.py
│       │   ├── 06_plot_figure3B_group_average_trajectories.py
│       │   ├── 07_plot_figure3C_displacement_distributions.py
│       │   ├── 08_compute_jump_candidates.py
│       │   ├── 09_plot_figure3D_jump_candidates.py
│       │   ├── 09b_plot_figure3E_integrated_interpretation.py
│       │   └── 10_assemble_figure3_composite.py
│       │
│       ├── figure4_dynamic_parameters/
│       │   ├── 11_compute_sample_dynamic_parameters.py
│       │   ├── 12_plot_figure4A_theta_by_phase.py
│       │   ├── 13_plot_figure4B_mu_shift.py
│       │   ├── 14a_plot_figure4c_sigma_eff_by_phase.py
│       │   ├── 14b_plot_figure4d_jump_score_by_branch_transition.py
│       │   ├── 15_plot_figure4E_regime_schematic.py
│       │   └── 16a_assemble_figure4_composite_full.py
│       │
│       ├── figure5_branch_ecology/
│       │   ├── 17_compute_branch_transition_tables.py
│       │   ├── 18_plot_figure5A_branch_transition_alluvial.py
│       │   ├── 19_plot_figure5B_branch_ecology_context.py
│       │   ├── 20_plot_figure5C_branch_scaffold_programs.py
│       │   ├── 21_plot_figure5D_branch_escape_risk.py
│       │   └── 22_assemble_figure5_composite.py
│       │
│       ├── figure6_external_aml/
│       │   ├── 23_build_gse235923_longitudinal_malignant_object.py
│       │   ├── 24_project_gse235923_into_frozen_scaffold.py
│       │   ├── 25_compute_gse235923_sample_centroids.py
│       │   ├── 26_compute_gse235923_sample_dynamic_parameters.py
│       │   ├── 27_plot_figure6A_external_projection.py
│       │   ├── 28_plot_figure6B_external_timepoint_organization.py
│       │   ├── 29_plot_figure6C_external_calibration_metric.py
│       │   ├── 30_plot_figure6D_calibration_summary.py
│       │   └── 31_assemble_figure6_composite.py
│       │
│       ├── figure7_clinical_translation/
│       │   ├── 36_prepare_gse163634_bulk_validation.py
│       │   ├── 39_plot_figure7C_bulk_validation.py
│       │   ├── 42_build_figure7_clinical_risk_map.py
│       │   ├── 43_plot_figure7A_clinical_risk_map.py
│       │   ├── 44_build_figure7_clinical_scorecard.py
│       │   ├── 45_plot_figure7B_clinical_scorecard.py
│       │   ├── 46_plot_figure7D_clinical_translation_summary.py
│       │   └── 47_assemble_figure7_redesigned_composite.py
│       │
│       └── supplementary_figures/
│           ├── make_figure_s1_workflow.py
│           ├── make_figure_s2_leukemia_control.py
│           ├── plot_dx_state_space_qc.py
│           ├── make_summary_figure_journal.py
│           ├── cross_cohort_comparison.py
│           ├── make_figure3_non_gaussian.py
│           ├── make_figure4_model_comparison.py
│           └── 05_plot_gse227122_strict_transfer_compact.py
│
├── outputs/
│   └── figures/
│       ├── main/
│       │   ├── Figure1_multimodal_therapy_aware_scaffold_full.png
│       │   ├── Figure2_multimodal_latent_landscape_full.png
│       │   ├── Figure3_therapy_induces_contraction_persistence_escape_full.png
│       │   ├── Figure4_treatment_aware_dynamic_parameters_full.png
│       │   ├── Figure5_multimodal_branch_context.png
│       │   ├── Figure6_external_longitudinal_calibration.png
│       │   └── Figure7_clinical_translation.png
│       │
│       └── supplementary/
│           ├── Figure_S1_workflow.png
│           ├── Figure_S2_leukemia_control.png
│           ├── Figure_S3A_centroid_distance.png
│           ├── Figure_S3B_correlation_matrix.png
│           ├── Figure_S4_equilibrium_diagnostics.png
│           ├── Figure_S5_cross_cohort_calibration.png
│           ├── Figure_S6A-D_upper_tail.png
│           ├── Figure_S6E-H_model_comparison.png
│           └── Figure_S7_cross_lineage_transfer.png
│
├── cohort_workflows/
│   ├── GSE235063/
│   ├── GSE235923/
│   ├── GSE227122/
│   └── GSE163634/
│
├── legacy/
│   └── Levy_OU_Branching/
│
├── supplementary_data/
│   ├── Supplementary_Data_1.xlsx
│   ├── Supplementary_Data_2.xlsx
│   ├── Supplementary_Data_3.xlsx
│   ├── Supplementary_Data_4.xlsx
│   ├── Supplementary_Data_5.xlsx
│   ├── Supplementary_Data_6.xlsx
│   └── Supplementary_Data_Guide.xlsx
│
└── docs/
    ├── workflow_overview.md
    ├── cohort_roles.md
    ├── reproducibility_notes.md
    └── manuscript_to_code_map.tsv
```text
Ecotype_Levy_OU_Branching_Dynamics/
├── README.md
├── LICENSE
├── CITATION.cff
├── environment.yml
├── requirements.txt
│
├── manuscript_workflow/
│   └── Multimodal_OU_Levy_Branching_Scaffold/
│       ├── README.md
│       ├── workflow_manifest.tsv
│       ├── Figure_1/
│       │   ├── scripts/
│       │   ├── panels/
│       │   └── final/
│       ├── Figure_2/
│       │   ├── scripts/
│       │   ├── inputs/
│       │   ├── derived/
│       │   └── final/
│       ├── Figure_3/
│       │   ├── scripts/
│       │   ├── inputs/
│       │   ├── derived/
│       │   ├── panels/
│       │   └── final/
│       ├── Figure_4/
│       │   ├── scripts/
│       │   ├── derived/
│       │   ├── panels/
│       │   └── final/
│       ├── Figure_5/
│       │   ├── scripts/
│       │   ├── derived/
│       │   ├── panels/
│       │   └── final/
│       ├── Figure_6/
│       │   ├── scripts/
│       │   ├── inputs/
│       │   ├── derived/
│       │   ├── panels/
│       │   └── final/
│       └── Figure_7/
│           ├── scripts/
│           ├── derived/
│           ├── panels/
│           └── final/
│
├── cohort_workflows/
│   ├── GSE235063/
│   ├── GSE235923/
│   ├── GSE227122/
│   └── GSE163634/
│
├── legacy/
│   └── Levy_OU_Branching/
│
├── supplementary_data/
│   ├── Supplementary_Data_1.xlsx
│   ├── Supplementary_Data_2.xlsx
│   ├── Supplementary_Data_3.xlsx
│   ├── Supplementary_Data_4.xlsx
│   ├── Supplementary_Data_5.xlsx
│   ├── Supplementary_Data_6.xlsx
│   └── Supplementary_Data_Guide.xlsx
│
└── docs/
    ├── workflow_overview.md
    ├── cohort_roles.md
    ├── reproducibility_notes.md
    └── manuscript_to_code_map.tsv
