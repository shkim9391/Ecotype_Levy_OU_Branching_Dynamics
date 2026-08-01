# GSE163634: Conservative Serial Bulk Validation of Disease-State Transfer

This directory contains the scripts and derived tables used to process the **GSE163634 bulk RNA-seq cohort** as the conservative cross-modality validation dataset within the **Ecotype–Lévy–OU–Branching** framework.

Unlike the discovery and external AML cohorts, GSE163634 is **not** used to construct the diagnosis-anchored disease-state scaffold. Instead, it serves as a stringent validation cohort that evaluates whether disease-state summaries learned from the single-cell reference remain interpretable after transfer to a lower-resolution longitudinal bulk RNA-seq dataset.

The goal of this workflow is not to reproduce the complete single-cell latent geometry or branch-level ecological organization, but rather to determine whether clinically meaningful disease-state ordering is preserved following transfer.

---

## Analytical role

Within the manuscript workflow, GSE163634 provides a **conservative serial bulk validation** of the diagnosis-anchored disease-state scaffold.

Specifically, the workflow evaluates whether transferred disease-state summaries preserve the expected directional organization among:

- healthy controls,
- diagnosis leukemia,
- early treatment response,
- later treatment response.

The resulting transferred variables are used for the bulk-validation analyses presented in **Figure 7**.

---

## Workflow overview

The workflow proceeds through five major stages:

1. preparation of the GSE163634 bulk-expression input space;
2. identification of transfer artifacts and axis-specific transfer limitations;
3. reconstruction and application of the frozen transfer model;
4. recovery of additional projected ecological axes from all-cell pseudobulk references;
5. generation of serial bulk-validation summaries and plotting tables.

These analyses support:

- transparent reporting of the bulk-transfer workflow;
- reproducible generation of transferred score matrices;
- serial diagnosis-to-response comparisons;
- evaluation of disease-state ordering under lower-resolution measurement.

---

## Directory structure

```text
GSE163634/
├── README.md
├── scripts/
│   ├── 01_prepare_gse163634_bulk_start.py
│   ├── 02_find_transfer_artifacts_fixed.py
│   ├── 03_find_axis_transfer_artifacts.py
│   ├── 04_rebuild_apply_gse163634_transfer.py
│   ├── 05_analyze_plot_gse163634_bulk_validation.py
│   ├── 05b_analyze_plot_gse163634_bulk_validation.py
│   └── 06_recover_pc12_from_allcells_pseudobulk.py
│
├── derived_bulk_start/
│   ├── gse163634_frozen_gene_intersection.csv
│   ├── gse163634_log2fpkm_frozen_intersection_genes_by_samples.tsv.gz
│   ├── gse163634_log2fpkm_frozen_intersection_samples_by_genes.tsv.gz
│   └── targeted_finder/
│       ├── gse235063_targeted_model_candidates.csv
│       ├── gse235923_targeted_calibration_candidates.csv
│       ├── gse235063_targeted_model_report.txt
│       ├── gse235923_targeted_calibration_report.txt
│       └── targeted_axis_artifact_summary.json
│
├── derived_transfer_projection/
│   ├── gse163634_bulk_score_matrix.csv
│   ├── gse163634_bulk_serial_deltas.csv
│   ├── gse235923_inferred_axis_calibration.csv
│   ├── gse235063_rebuilt_transfer_model_summary.csv
│   ├── gse235063_rebuilt_transfer_coefficients_long.csv
│   └── gse163634_transfer_manifest.json
│
├── derived_pc12_recovery/
│   ├── gse235063_pc12_model_summary.csv
│   ├── gse235063_pc12_coefficients_long.csv
│   ├── gse235923_pc12_calibration.csv
│   ├── gse235923_pc12_pred_vs_obs.csv
│   ├── gse163634_pc12_score_matrix.csv
│   ├── gse163634_bulk_score_matrix_with_pc12.csv
│   ├── gse163634_bulk_serial_deltas_with_pc12.csv
│   └── gse235063_pc12_fit_vs_reference_qc.csv
│
└── derived_bulk_validation_with_pc12/
    ├── gse163634_bulk_leukemia_vs_control_stats.csv
    ├── gse163634_bulk_dx_to_r1_paired_stats.csv
    ├── gse163634_bulk_r1_to_r2_paired_stats.csv
    ├── gse163634_bulk_axis_transfer_rankings.csv
    ├── gse163634_bulk_heatmap_matrix.csv
    ├── gse163634_bulk_paired_plot_table.csv
    └── gse163634_bulk_forest_plot_table.csv
```

---

## Key outputs

The primary outputs used by the manuscript workflow include:

- `gse163634_bulk_score_matrix_with_pc12.csv`
- `gse163634_bulk_serial_deltas_with_pc12.csv`
- `gse163634_bulk_validation_summary.csv`
- `gse163634_bulk_leukemia_vs_control_stats.csv`
- `gse163634_bulk_paired_plot_table.csv`

These files provide the transferred disease-state variables and statistical summaries used for the serial bulk validation presented in **Figure 7**.

---

## Notes

- The transferred disease-state summaries are derived from the diagnosis-anchored single-cell reference and should not be interpreted as independently learned bulk disease states.
- Bulk RNA-seq does not preserve the full ecological and branch-level resolution of the single-cell scaffold.
- Consequently, this workflow emphasizes **directional consistency** and **clinical interpretability** rather than exact reconstruction of the original latent space.
