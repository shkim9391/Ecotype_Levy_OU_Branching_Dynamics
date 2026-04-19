# Ecotype_Levy_OU_Branching_Dynamics

This repository accompanies the manuscript:

**Ecological constraint and punctuated reorganization in pediatric leukemia evolution**

It contains the code, derived tables, and selected outputs used to analyze pediatric leukemia evolution through an ecological-state framework that combines **constraint**, **branch-structured organization**, and **punctuated longitudinal reorganization**.

The repository is organized to support transparency and reproducibility of the computational workflow across the main development cohort, secondary calibration cohort, cross-lineage transfer cohort, conservative bulk validation cohort, and the dedicated Lévy–OU–Branching analysis scaffold.

## Repository overview

The project is organized around five major analysis components:

- **`GSE235063/`**  
  Primary training and development cohort.  
  Includes diagnosis-stage cohort construction, ecotype assignment, OU input generation, compact model fitting, and figure-generation scripts.

- **`GSE235923/`**  
  Secondary calibration cohort.  
  Includes manifest construction, cohort building, transfer from the primary reference, and projection into the primary ecotype backbone.

- **`GSE227122/`**  
  Cross-lineage transfer cohort.  
  Includes ingestion, annotation, strict ecotype transfer, longitudinal comparison, and compact transfer plotting outputs.

- **`GSE163634/`**  
  Conservative serial bulk validation cohort.  
  Includes bulk preparation, transfer artifact analysis, transfer-model rebuilding, PC1/PC2 recovery, and serial validation summaries.

- **`Levy_OU_Branching/`**  
  Dedicated Lévy–OU–Branching workflow.  
  Includes diagnosis baseline construction, state-space QC, branch scaffold definition, longitudinal projection, transition sensitivity analysis, displacement analysis, jump-candidate ranking, and manuscript figure scripts.

Each directory contains its own local `README.md` with more detailed information about scripts, derived outputs, and workflow logic.

## Scientific scope

The analyses in this repository were developed to study how pediatric leukemia samples occupy and move through a constrained ecological state space across diagnosis, treatment, remission-related states, and relapse.

The computational logic emphasizes:

- construction of ecological state representations from single-cell and bulk data,
- projection of external cohorts into a shared reference framework,
- comparison of stable versus switching longitudinal trajectories,
- diagnosis-to-relapse displacement analysis,
- identification of candidate punctuated or jump-like reorganizations,
- evaluation of constrained dynamics through OU-style and Lévy-extended perspectives.

## Repository structure

```text
Ecotype_Levy_OU_Branching_Dynamics/
├── GSE235063/
├── GSE235923/
├── GSE227122/
├── GSE163634/
└── Levy_OU_Branching/
