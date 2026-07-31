# GSE235923: Secondary Calibration and Cross-Cohort Projection

This directory contains the code, reference projection files, and derived tables used to process the **GSE235923** pediatric leukemia cohort as a **secondary calibration dataset** within the Ecotype–Lévy–OU workflow.

The main purpose of this release is to document how the GSE235923 diagnosis cohort was prepared, aligned to the primary GSE235063 ecotype framework, projected into the primary ecotype backbone, and summarized for cross-cohort comparison.

As with the current repository release more broadly, **rendered figure image files are not included at this stage**. Figure-generation scripts are provided and figure assets can be added in a later update as the manuscript progresses.

## Overview

The workflow in this directory proceeds from:

1. construction and correction of the diagnosis-stage cohort manifest,
2. building a diagnosis-only secondary calibration cohort,
3. transfer of labels or reference annotations from **GSE235063**,
4. projection of the secondary cohort into the **primary ecotype backbone**,
5. derivation of sample-level calibration tables and cross-cohort summaries.

This directory is intended to support:

- transparent reporting of the **secondary calibration workflow**,
- reuse of derived projection outputs,
- comparison between the primary and secondary cohorts within a shared ecotype reference frame.


## Directory structure

```text
GSE235923/
├── README.md
├── scripts/
│   ├── 01_manifest/
│   │   ├── diagnosis_cohort_manifest.py
│   │   └── corrected_manifest_builder.py
│   ├── 02_build/
│   │   └── build_dx_secondary_calibration_gexonly.py
│   ├── 03_transfer/
│   │   └── transfer_labels_from_gse235063.py
│   ├── 04_projection/
│   │   └── project_to_primary_ecotype_backbone.py
│   └── 05_figures/
│       └── cross_cohort_comparison.py
│
├── derived_secondary_calibration/
│   ├── gse235923_dx_secondary_calibration_sample_summary_gexonly.csv
│   ├── gse235923_dx_feature_name_choice_summary.csv
│   ├── gse235923_dx_predicted_sample_summary.csv
│   ├── gse235923_dx_projected_ecotype_pcs.csv
│   ├── gse235923_dx_pred_broad_fractions_by_sample_restricted.csv
│   ├── gse235923_dx_pred_malignant_coarse_fractions_by_sample.csv
│   ├── gse235923_dx_pred_core4_fractions_by_sample.csv
│   ├── gse235923_dx_secondary_outcomes.csv
│   └── gse235923_dx_secondary_calibration_table.csv
│
└── reference_projection/
    ├── primary_ecotype_pca_reference_scores.csv
    └── primary_ecotype_pca_loadings.csv
