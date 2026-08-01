# Cohort Roles

## Overview

The analyses in this repository use four publicly available pediatric leukemia datasets, each assigned a predefined analytical role within the therapy-aware multimodal ecological framework.

Rather than combining all datasets into a single pooled analysis, each cohort contributes a distinct component of the overall workflow. This design separates model development, external calibration, supplementary transfer evaluation, and conservative cross-modality validation, reducing overfitting while improving biological interpretability and reproducibility.

The roles of the cohorts are summarized below.

---

# Cohort overview

| Cohort | Primary role | Used in main manuscript | Purpose |
|---------|--------------|------------------------|---------|
| **GSE235063** | Discovery cohort | Yes | Diagnosis-anchored scaffold construction and primary longitudinal analyses |
| **GSE235923** | External calibration cohort | Yes | Independent pediatric AML projection and treatment-aware calibration |
| **GSE227122** | Cross-lineage transfer cohort | Supplementary only | Supplementary evaluation of scaffold transferability |
| **GSE163634** | Conservative bulk validation cohort | Yes | Cross-modality validation using longitudinal bulk RNA-seq |

---

# GSE235063

## Role

**Primary discovery cohort**

GSE235063 provides the principal dataset used to develop the diagnosis-anchored disease-state scaffold and to perform the primary longitudinal analyses presented in the manuscript.

This cohort defines the reference disease-state landscape used throughout the remaining analyses.

---

## Principal analyses

The workflow includes:

- preprocessing and quality control;
- malignant-cell object construction;
- ecological feature generation;
- inferred regulatory summaries;
- diagnosis-stage scaffold construction;
- frozen-space projection;
- patient-level trajectory reconstruction;
- diagnosis-to-relapse displacement analysis;
- branch assignment;
- branch-continuous versus branch-switching classification;
- therapy-aware dynamic-summary estimation;
- jump-sensitive scoring;
- branch-conditioned escape analysis.

---

## Manuscript contributions

GSE235063 supports:

- Figure 2
- Figure 3
- Figure 4
- Figure 5

It also provides the frozen diagnosis reference used in external cohort projection.

---

## Primary analysis cohort

The principal longitudinal analysis contains:

- 21 matched DX→REL intervals;
- 9 branch-continuous trajectories;
- 12 branch-switching trajectories.

The therapy-aware dynamic analyses include:

- 21 diagnosis samples;
- 21 relapse samples;
- 2 exploratory EOI/REM samples displayed individually.

---

# GSE235923

## Role

**Independent pediatric AML calibration cohort**

GSE235923 provides an external validation of the diagnosis-anchored scaffold.

Unlike the discovery cohort, no new latent representation is learned from this dataset.

Instead, all samples are projected directly into the frozen GSE235063 disease-state scaffold.

---

## Principal analyses

The workflow includes:

- cohort harmonization;
- label transfer;
- frozen-space projection;
- sample-centroid calculation;
- dynamic-summary estimation;
- external calibration;
- visualization of longitudinal trajectories.

---

## Manuscript contributions

GSE235923 supports:

- Figure 6
- Figure 7

---

## Cohort summary

The calibration cohort includes:

- 19 diagnosis samples;
- 10 EOI/REM samples;
- 2 relapse samples.

Two patients contain complete:

DX → EOI/REM → REL

longitudinal trajectories.

---

# GSE227122

## Role

**Supplementary cross-lineage transfer cohort**

GSE227122 evaluates whether the ecological scaffold remains interpretable outside the primary pediatric AML discovery cohort.

Because this cohort contains only one relapse sample, it is not used in the primary relapse-dynamics analyses.

---

## Principal analyses

The workflow includes:

- cohort ingestion;
- ecological transfer;
- frozen-space projection;
- longitudinal organization;
- paired trajectory visualization;
- transfer-support summaries.

---

## Manuscript contributions

This cohort contributes only to the Supplementary Information.

It supports:

- Supplementary Figure S7
- Supplementary Data 5

No conclusions in the primary manuscript depend on this cohort.

---

# GSE163634

## Role

**Conservative serial bulk RNA-seq validation cohort**

GSE163634 provides a deliberately lower-resolution validation of the diagnosis-anchored scaffold.

Rather than reconstructing ecological structure directly from bulk RNA-seq, the workflow evaluates whether disease-state summaries learned from the single-cell reference remain interpretable after transfer.

---

## Principal analyses

The workflow includes:

- frozen-gene intersection;
- transfer-model rebuilding;
- ecological PC recovery;
- bulk score reconstruction;
- serial diagnosis-to-response comparisons;
- lower-resolution validation.

---

## Manuscript contributions

GSE163634 supports:

- Figure 7

---

## Validation objective

The bulk workflow evaluates whether transferred disease-state summaries preserve the expected ordering among:

- control samples;
- diagnosis leukemia;
- first response;
- later response.

The analysis is intentionally conservative and does **not** attempt to reproduce:

- full single-cell latent geometry;
- branch assignments;
- fine-grained ecological structure.

---

# Relationship among cohorts

The complete analytical design is hierarchical.

```text
                     GSE235063
                 (Discovery cohort)
                         │
                         ▼
        Diagnosis-anchored disease-state scaffold
                         │
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
     GSE235923      GSE227122      GSE163634
   External AML     Supplementary     Bulk
    calibration      transfer        validation
```

The discovery cohort defines the shared disease-state reference.

The remaining cohorts evaluate different aspects of generalizability:

- external pediatric AML calibration;
- supplementary cross-lineage transfer;
- conservative cross-modality validation.

---

# Design philosophy

Each cohort addresses a distinct scientific question.

| Cohort | Scientific question |
|---------|--------------------|
| **GSE235063** | Can therapy-aware disease-state dynamics be learned from longitudinal pediatric AML single-cell data? |
| **GSE235923** | Does the diagnosis-anchored scaffold generalize to an independent pediatric AML cohort without relearning the latent space? |
| **GSE227122** | Does the ecological scaffold remain interpretable outside the discovery lineage? |
| **GSE163634** | Can reduced disease-state summaries remain informative after transfer to longitudinal bulk RNA-seq? |

Assigning predefined roles to each cohort avoids information leakage between discovery and validation analyses while providing complementary evidence for robustness, transferability, and clinical interpretability.
