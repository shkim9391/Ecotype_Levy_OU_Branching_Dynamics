# Levy_OU_Branching: Legacy Development Workflow

This directory contains the original scripts, derived tables, and intermediate outputs developed during the early implementation of the **Lévy–OU–Branching** framework for pediatric leukemia evolution.

These analyses formed an important foundation for the final therapy-aware multimodal disease-state scaffold but **do not represent the authoritative workflow used for the accompanying manuscript**.

The current manuscript workflow has been reorganized under:

```text
manuscript_workflow/
└── Multimodal_OU_Levy_Branching_Scaffold/
```

This legacy directory is retained for transparency, provenance, and historical reproducibility.

---

## Historical role

The original workflow focused on:

- construction of diagnosis-stage ecological state matrices;
- diagnosis branch-scaffold definition;
- longitudinal diagnosis-to-relapse projection;
- branch-transition analysis;
- displacement-based identification of putative jump-like events;
- exploratory OU–Lévy–Branching model development.

Many concepts introduced here were subsequently refined and incorporated into the manuscript workflow.

---

## Relationship to the manuscript

The submitted manuscript reorganizes these analyses into a more modular and therapy-aware framework.

Compared with the legacy workflow, the manuscript implementation introduces:

- diagnosis-anchored multimodal disease-state construction;
- therapy-aware dynamic summaries;
- frozen-space projection;
- external pediatric AML calibration;
- clinical translation;
- standardized workflow organization.

Accordingly, the scripts in this directory should be viewed as developmental analyses rather than the primary implementation supporting the manuscript.

---

## Workflow overview

The legacy workflow proceeds through seven major stages:

1. construction of diagnosis baseline matrices;
2. diagnosis-state quality control;
3. diagnosis branch-scaffold construction;
4. longitudinal projection;
5. branch-transition sensitivity analysis;
6. diagnosis-to-relapse displacement analysis;
7. exploratory figure generation.

---

## Directory structure

```text
Levy_OU_Branching/
├── README.md
├── scripts/
│   ├── 01_baseline/
│   ├── 02_qc/
│   ├── 03_branch_scaffold/
│   ├── 04_longitudinal/
│   ├── 05_transition/
│   ├── 06_displacement/
│   └── 07_figures/
│
├── derived_dx_baseline/
├── derived_dx_qc/
├── derived_dx_branch_scaffold/
├── derived_longitudinal/
├── derived_transition_sensitivity/
└── derived_dx_rel_threshold50/
```

---

## Principal outputs

The directory contains:

- diagnosis-stage ecological state matrices;
- diagnosis branch assignments;
- branch centroids;
- longitudinal projection tables;
- branch-transition summaries;
- threshold-sensitivity analyses;
- diagnosis-to-relapse displacement summaries;
- exploratory jump-candidate rankings.

These outputs document the evolution of the framework and provide provenance for later methodological developments.

---

## Legacy status

Several scripts in this directory have been superseded by the manuscript workflow.

Examples include:

- diagnosis-scaffold construction;
- longitudinal projection;
- dynamic-summary estimation;
- branch-conditioned analyses;
- manuscript figure generation.

Readers interested in reproducing the submitted manuscript should use the scripts located under:

```text
manuscript_workflow/
└── Multimodal_OU_Levy_Branching_Scaffold/
```

rather than the legacy workflow documented here.

---

## Reproducibility notes

This directory is retained for archival purposes.

The scripts may rely on earlier directory layouts and intermediate files that differ from the finalized repository organization.

Consequently, this workflow should be considered a historical implementation rather than the recommended starting point for new analyses.
