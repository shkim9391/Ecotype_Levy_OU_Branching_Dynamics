# GSE227122: Cross-Lineage Ecotype Transfer and Projection

This directory contains the code and result tables used to process the **GSE227122** pediatric leukemia cohort for **strict cross-cohort transfer** within the Ecotype–Lévy–OU workflow.

In the broader project, this dataset is used to test whether the ecological structure learned in the primary reference cohort remains interpretable when transferred into an external longitudinal setting. The emphasis here is on **annotation, projection, and comparison under a fixed transferred framework**, rather than on redefining the ecotype structure de novo.

As with the current repository release more broadly, **rendered figure image files are not included at this stage**. Figure-generation scripts are provided, and figure assets can be added later as the manuscript progresses.

## Overview

The workflow in this directory proceeds from:

1. ingestion of the GSE227122 dataset,
2. cell annotation and marker-based summary generation,
3. projection into the transferred ecotype framework,
4. comparison of projected structure across samples and timepoints,
5. compact plotting for strict-transfer presentation.

This directory is intended to support:

- transparent reporting of the **strict transfer workflow**,
- reuse of annotation and projection outputs,
- comparison of projected ecological organization across longitudinal states.

## Directory structure

```text
GSE227122/
├── README.md
├── scripts/
│   ├── 01_ingest_gse227122.py
│   ├── 02a_annotate_gse227122_cells.py
│   ├── 03_project_gse227122_ecotype_transfer.py
│   ├── 04_compare_gse227122_transfer.py
│   └── 05_plot_gse227122_strict_transfer_compact.py
│
└── results/
    └── gse227122_transfer/
        ├── gse227122_cell_annotations.csv
        ├── gse227122_cluster_annotation_summary.csv
        ├── gse227122_marker_sets_used.csv
        ├── gse227122_annotation_parameters.json
        ├── gse227122_cluster_top_markers.csv
        ├── gse227122_normal_broad_cellgroup_counts_by_sample.csv
        ├── gse227122_normal_broad_cellgroup_fractions_by_sample.csv
        ├── gse227122_sample_summary_for_transfer.csv
        ├── gse227122_projected_sample_scores.csv
        ├── cross_cohort_reference_table.csv
        ├── gse227122_dx_projected_samples.csv
        ├── gse227122_eoi_projected_samples.csv
        ├── gse227122_rel_projected_samples.csv
        ├── gse227122_dx_eoi_paired_pc_deltas.csv
        ├── cross_cohort_ecotype_summary_stats.csv
        └── gse227122_projected_samples_with_optional_programs.csv
