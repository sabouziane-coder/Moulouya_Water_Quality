# Moulouya River Water Quality — Pollution Index (PI) Prediction

This repository provides the data and Python scripts associated with the study  
**“An Ensemble Learning Approach for Water Quality Assessment and Prediction Using the Pollution Index: A Case Study of the Moulouya River Basin, Morocco.”**

The repository includes preprocessing routines, Pollution Index (PI) calculation, baseline and ensemble model training, model evaluation and robustness analysis, and the figures used in the manuscript.

---

## Repository Structure

```text
.
├── data/
│   ├── raw/
│   │   ├── README.md
│   │   └── standards.csv
│   └── processed/
│       └── Final_Pollution_Index_Results.csv
├── src/
│   ├── preprocess.py
│   ├── calc_pi.py
│   ├── training_models_pi.py
│   └── plot_ensemble.py
├── requirements.txt
├── README.md
└── .gitignore

```text
## Directory Description
data/raw/
Contains reference tables and metadata used for Pollution Index (PI) computation.
Due to data ownership restrictions, original raw measurement files are not publicly shared.
The standards.csv file reconstructs the thresholds and weights required for PI calculation.

data/processed/
Contains the processed and analysis-ready dataset used for model training and evaluation.
Final_Pollution_Index_Results.csv includes all physicochemical parameters and the computed PI values.

src/
Python scripts implementing the full modeling pipeline:

preprocess.py: data cleaning, normalization, and preparation

calc_pi.py: Pollution Index (PI) computation

training_models_pi.py: model training, cross-validation, evaluation, and uncertainty analysis

plot_ensemble.py: generation of performance, diagnostic, and comparison figures

requirements.txt
Lists all Python dependencies with fixed versions to ensure full reproducibility.

## Reproducibility
All results presented in the manuscript can be reproduced by executing the scripts in the src/ directory using the processed dataset provided.
