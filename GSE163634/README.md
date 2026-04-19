# GSE163634: Conservative Serial Bulk Validation of Ecotype Transfer

This directory contains the code and derived tables used to process the **GSE163634** bulk RNA-seq cohort as a **serial validation dataset** within the Ecotype–Lévy–OU workflow.

In the broader project, this dataset is used as a **conservative external bulk validation setting**. Rather than redefining the ecological structure from scratch, the workflow tests whether signals learned from the single-cell reference framework remain informative when transferred into a longitudinal bulk-expression context.

As with the current repository release more broadly, **rendered figure image files are not included at this stage**. Figure-generation scripts are provided, and figure assets can be added later as the manuscript progresses.

## Overview

The workflow in this directory proceeds from:

1. preparation of the GSE163634 bulk-expression input space,
2. identification of transfer artifacts and axis-specific transfer issues,
3. rebuilding and applying the transfer model,
4. recovery of additional projected axes from all-cell pseudobulk references,
5. serial bulk validation and summary-statistic generation.

This directory is intended to support:

- transparent reporting of the **bulk transfer and validation workflow**,
- reuse of score matrices and serial-delta summaries,
- evaluation of whether the transferred ecological structure remains interpretable in a conservative bulk setting.

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
