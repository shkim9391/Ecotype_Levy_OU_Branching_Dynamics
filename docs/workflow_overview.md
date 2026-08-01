# Workflow Overview

## Overview

This repository implements a reproducible computational framework for studying therapy-associated disease-state dynamics in pediatric leukemia using a diagnosis-anchored ecological scaffold.

The complete workflow integrates complementary transcriptomic, inferred regulatory, ecological, and clinical information into a shared disease-state representation. Longitudinal movement through this space is interpreted using a therapy-aware Ornstein–Uhlenbeck–Lévy–Branching (OU–Lévy–Branching) framework that emphasizes constrained dynamics, branch-structured reorganization, residual persistence, and relapse-associated escape.

The workflow is organized into two complementary layers:

1. **Cohort-specific preprocessing workflows**, which prepare and harmonize each dataset.
2. **The manuscript workflow**, which integrates processed datasets into the final analyses and figures.

The repository is designed to maximize transparency, reproducibility, and traceability from raw public datasets to the final manuscript figures.

---

# Workflow architecture

```text
Public datasets
        │
        ▼
Cohort-specific preprocessing
        │
        ▼
Diagnosis-anchored scaffold construction
        │
        ▼
Frozen-space projection
        │
        ▼
Longitudinal trajectory reconstruction
        │
        ▼
Therapy-aware dynamic summaries
        │
        ▼
Branch and ecological analyses
        │
        ▼
External AML calibration
        │
        ▼
Conservative bulk validation
        │
        ▼
Clinical interpretation
```

---

# Repository organization

The repository is divided into four major components.

## 1. Cohort workflows

Located in

```text
cohort_workflows/
```

These directories contain preprocessing pipelines specific to each public dataset.

### GSE235063

Primary discovery cohort.

Responsible for:

- diagnosis-stage preprocessing;
- malignant and all-cell object construction;
- ecological feature generation;
- branch assignment;
- frozen diagnosis-scaffold construction;
- longitudinal projection.

---

### GSE235923

Independent pediatric AML calibration cohort.

Responsible for:

- cohort harmonization;
- projection into the frozen diagnosis scaffold;
- external calibration;
- longitudinal trajectory reconstruction.

---

### GSE227122

Supplementary cross-lineage transfer cohort.

Responsible for:

- strict ecological transfer;
- longitudinal transfer evaluation;
- supplementary robustness analyses.

Because this cohort contains only one relapse sample, it is used only in supplementary analyses.

---

### GSE163634

Conservative serial bulk validation cohort.

Responsible for:

- reconstruction of transferable disease-state variables;
- transfer-model rebuilding;
- ecological PC recovery;
- serial diagnosis-to-response validation.

---

# 2. Manuscript workflow

Located in

```text
manuscript_workflow/
```

This directory contains the complete workflow used to generate the submitted manuscript.

The workflow is organized by manuscript figure.

```text
Figure_1
Figure_2
Figure_3
Figure_4
Figure_5
Figure_6
Figure_7
```

Each figure directory contains

- scripts
- derived tables
- intermediate outputs
- panel figures
- final composite figures

---

# 3. Supplementary data

Located in

```text
supplementary_data/
```

This directory contains the manuscript supplementary workbooks corresponding to Supplementary Data 1–6 together with the Supplementary Data Guide.

---

# 4. Documentation

Located in

```text
docs/
```

This directory provides documentation describing the computational workflow, cohort roles, reproducibility considerations, and mappings between manuscript figures and repository scripts.

---

# Computational workflow

The complete analysis proceeds through the following stages.

---

## Step 1

### Discovery-cohort preparation

The GSE235063 workflow imports processed single-cell data and constructs sample-level malignant and ecological representations.

Outputs include

- diagnosis-stage sample summaries
- ecological variables
- branch assignments
- latent scaffold coordinates

---

## Step 2

### Diagnosis-anchored scaffold construction

Only diagnosis samples are used to construct the latent disease-state scaffold.

This strategy ensures that subsequent therapy-associated movement is interpreted relative to a fixed pretreatment reference rather than a jointly learned embedding.

The scaffold integrates

- transcriptomic features
- inferred regulatory programs
- ecological composition
- clinical context

---

## Step 3

### Frozen-space projection

Longitudinal samples are projected into the fixed diagnosis scaffold.

Projection is performed without relearning the latent representation.

This enables direct comparison among

- diagnosis
- treatment
- remission
- relapse

within a common disease-state coordinate system.

---

## Step 4

### Longitudinal trajectory reconstruction

Patient trajectories are reconstructed by connecting projected samples across clinical time.

Primary summaries include

- patient centroids
- interval displacements
- diagnosis-to-relapse movement
- branch continuity
- branch switching

The principal discovery analysis contains

- 21 matched DX→REL intervals
- 9 branch-continuous trajectories
- 12 branch-switching trajectories

---

## Step 5

### Therapy-aware dynamic summaries

Projected trajectories are summarized using interpretable quantities representing

- effective restoring strength
- attractor displacement
- effective instability
- jump-sensitive behavior

These quantities are descriptive summaries of latent disease-state dynamics and are not interpreted as fully identified continuous-time stochastic-process parameters.

---

## Step 6

### Branch and ecological analyses

The workflow evaluates

- branch transitions
- ecological composition
- scaffold-program organization
- branch-conditioned escape propensity

These analyses identify non-equivalent routes through the diagnosis-defined disease-state landscape.

---

## Step 7

### External AML calibration

The GSE235923 cohort is projected into the frozen diagnosis scaffold.

No cohort-specific latent embedding is learned.

The calibration workflow evaluates

- diagnosis
- EOI/REM
- relapse

within the shared disease-state space.

---

## Step 8

### Conservative serial bulk validation

Transferred disease-state summaries are reconstructed in GSE163634 bulk RNA-seq data.

Rather than reproducing the full single-cell geometry, the analysis evaluates whether

- diagnosis
- treatment response
- disease ordering

remain detectable under lower-resolution measurement.

---

## Step 9

### Clinical interpretation

The manuscript concludes with clinically interpretable summaries including

- dynamic disease-state maps
- representative patient scorecards
- serial validation
- translational summaries

These outputs illustrate how scaffold-derived dynamic summaries can support interpretation of therapy response, residual persistence, and relapse-associated escape.

---

# Statistical framework

The principal statistical analyses include

- two-sided Mann–Whitney U tests for independent-group comparisons;
- two-sided Wilcoxon signed-rank tests for matched diagnosis-to-relapse comparisons;
- Cliff's delta for non-parametric effect-size estimation.

Formal statistical testing was restricted to prespecified phase-level and transition-class contrasts.

Small exploratory subgroups were displayed individually and interpreted cautiously.

---

# Reproducibility

The repository includes

- executable Python scripts;
- workflow manifests;
- statistical summary tables;
- derived intermediate files;
- manuscript-linked supplementary data;
- figure-generation scripts.

Large raw datasets and controlled-access resources are not redistributed.

Users should obtain the original public datasets through the NCBI Gene Expression Omnibus (GEO) and any applicable controlled-access repositories before executing the workflow.

---

# Recommended execution order

```text
1. Cohort preprocessing
      GSE235063
      GSE235923
      GSE227122
      GSE163634

2. Diagnosis scaffold construction

3. Frozen-space projection

4. Longitudinal trajectory reconstruction

5. Therapy-aware dynamic summaries

6. Branch analyses

7. External AML calibration

8. Conservative bulk validation

9. Clinical interpretation

10. Figure assembly

11. Supplementary-data generation
```

The complete script-level execution order is documented in

```text
manuscript_workflow/Multimodal_OU_Levy_Branching_Scaffold/workflow_manifest.tsv
```

This manifest provides a direct mapping between scripts, inputs, outputs, manuscript figures, and supplementary-data files.
