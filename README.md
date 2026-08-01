# Ecotype Lévy–OU–Branching Dynamics

This repository accompanies the manuscript:

## A therapy-aware multimodal dynamical scaffold reveals residual persistence and relapse-associated escape in pediatric leukemia

The repository contains the code, derived tables, workflow documentation, and selected outputs used to construct and evaluate a diagnosis-anchored, therapy-aware disease-state scaffold for pediatric leukemia.

The framework integrates complementary transcriptomic, inferred regulatory, ecological-composition, and clinical-phase feature blocks. Longitudinal disease-state movement is interpreted using an Ornstein–Uhlenbeck–Lévy–branching formulation that combines:

- constrained or mean-reverting within-state dynamics;
- branch-structured disease-state organization;
- therapy-associated attractor displacement;
- residual persistence;
- and punctuated or jump-sensitive relapse-associated reorganization.

The repository is organized to distinguish the final manuscript workflow from cohort-specific preprocessing, supplementary analyses, legacy exploratory workflows, and manuscript-linked data products.

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
- and conservative serial bulk validation.

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
