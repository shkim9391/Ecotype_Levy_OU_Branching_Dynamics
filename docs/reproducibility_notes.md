# Reproducibility Notes

## Overview

This repository is intended to provide a transparent and reproducible implementation of the computational workflow described in the accompanying manuscript:

> **A therapy-aware multimodal dynamical scaffold reveals residual persistence and relapse-associated escape in pediatric leukemia**

The repository contains executable Python scripts, workflow manifests, derived analysis tables, statistical summaries, and figure-generation code. Large raw datasets, controlled-access resources, and selected intermediate objects are not redistributed but can be regenerated from the original public data sources.

---

# Computational reproducibility

The workflow was developed and tested using **Python 3.12**.

A recommended software environment is provided through:

```text
environment.yml
```

Alternatively, required Python packages can be installed using:

```text
requirements.txt
```

The recommended installation is:

```bash
conda env create -f environment.yml
conda activate ecotype_ou_levy_branching
```

---

# Data sources

The analyses rely on publicly available datasets.

| Dataset | Role |
|---------|------|
| GSE235063 | Primary discovery cohort |
| GSE235923 | External AML calibration cohort |
| GSE227122 | Supplementary cross-lineage transfer cohort |
| GSE163634 | Conservative serial bulk validation cohort |

Controlled-access raw sequencing data associated with GSE235063 are available through the European Genome-phenome Archive under accession:

```text
EGAS00001007323
```

Users are responsible for obtaining all datasets through the appropriate repositories and complying with all applicable data-access policies.

---

# Repository philosophy

The repository separates four distinct layers of analysis:

1. cohort-specific preprocessing;
2. manuscript-specific analyses;
3. supplementary analyses;
4. legacy exploratory workflows.

Only the workflow contained in

```text
manuscript_workflow/
```

should be considered the authoritative implementation corresponding to the submitted manuscript.

Earlier exploratory scripts are retained under

```text
legacy/
```

for transparency and provenance.

---

# Workflow documentation

The complete execution order is documented in:

```text
manuscript_workflow/
└── Multimodal_OU_Levy_Branching_Scaffold/
    └── workflow_manifest.tsv
```

A direct mapping between manuscript figures and repository scripts is provided in:

```text
docs/manuscript_to_code_map.tsv
```

---

# Directory independence

Each cohort directory contains its own local README describing:

- required inputs;
- script execution order;
- generated outputs;
- cohort-specific interpretation.

Individual workflows can therefore be executed independently before integration into the manuscript pipeline.

---

# Frozen reference framework

A central design principle of the repository is the use of a **diagnosis-anchored frozen disease-state scaffold**.

The latent representation is learned only from diagnosis samples in the discovery cohort.

Subsequent analyses—including longitudinal projection, external AML calibration, and supplementary transfer—are performed by projecting samples into this frozen reference space without relearning the latent representation.

This strategy minimizes information leakage between discovery and validation analyses.

---

# Therapy-aware dynamic summaries

The manuscript reports several interpretable dynamic quantities, including

- effective restoring strength;
- attractor displacement;
- effective instability;
- jump-sensitive score;
- relapse-escape score.

These quantities are **descriptive summaries of latent disease-state dynamics**.

They should **not** be interpreted as fully identified continuous-time stochastic-process parameters.

The Ornstein–Uhlenbeck–Lévy–Branching formulation is used as a conceptual statistical scaffold that links observable disease-state movement to interpretable dynamic summaries.

---

# Statistical analyses

Formal statistical testing was intentionally restricted to prespecified comparisons.

Independent-group analyses used:

- two-sided Mann–Whitney U tests.

Matched diagnosis-to-relapse analyses used:

- two-sided Wilcoxon signed-rank tests.

Independent-group effect sizes were summarized using:

- Cliff's delta.

The nominal significance threshold was

```text
α = 0.05
```

Displayed P values are unadjusted.

Small exploratory groups, particularly EOI/REM and sparse relapse subsets, are shown individually where appropriate and interpreted cautiously.

---

# Supplementary analyses

The repository contains analyses that are included only in the Supplementary Information.

Examples include:

- strict cross-lineage transfer;
- additional robustness analyses;
- heavy-tail model comparisons;
- supplementary quality-control summaries.

These analyses are retained to document methodological development and robustness but should not be interpreted as primary discovery analyses.

---

# Figure generation

Each manuscript figure is organized within its own directory.

Typical structure:

```text
Figure_X/
├── scripts/
├── inputs/
├── derived/
├── panels/
└── final/
```

The repository provides:

- figure-generation scripts;
- intermediate plotting tables;
- derived statistics;
- final composite figures (where applicable).

Rendered figures can be regenerated directly from the corresponding scripts.

---

# Randomness and reproducibility

Where randomization is used for visualization (for example, jittered point placement in violin plots), scripts use fixed random seeds to ensure reproducible output.

These graphical randomizations do **not** affect the underlying statistical analyses.

---

# Large files

Large intermediate objects (for example, AnnData files, processed matrices, and selected temporary outputs) may not be included in the GitHub repository because of file-size limitations.

Whenever possible, scripts regenerate these objects from public inputs.

If regeneration requires controlled-access resources, this is documented in the corresponding workflow README.

---

# Platform considerations

The workflow was originally developed under macOS using Conda-managed Python environments.

Most scripts rely only on cross-platform Python libraries and should execute on Linux, macOS, or Windows after updating local file paths.

Absolute paths appearing in configuration blocks should be replaced with user-specific paths before execution.

---

# Versioning

Each archived software release corresponds to a specific repository snapshot.

The manuscript should always be reproduced using the archived Zenodo release associated with the manuscript version rather than the continuously evolving GitHub repository.

The archived DOI provides a permanent reference for the exact code used in the study.

---

# Citation

If this repository contributes to published work, please cite:

1. the accompanying manuscript; and
2. the archived Zenodo software release.

Citation metadata are provided in:

```text
CITATION.cff
```

---

# Contact

Questions regarding the workflow, implementation, or reproducibility may be directed to the corresponding author through the contact information provided in the accompanying manuscript.
