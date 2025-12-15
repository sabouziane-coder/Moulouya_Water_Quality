# Moulouya River Water Quality — PI prediction (Ensemble models)

This repository contains data and scripts for "AI-Driven Real-Time Prediction and Monitoring of Water Quality in the Moulouya River Basin, Morocco" (Y. [Your Name], et al.). It includes preprocessing code, Pollution Index (PI) calculation, ensemble model training (RF, GB, XGBoost), evaluation scripts, and the figures used in the manuscript.

## Repository structure
.
├── data/
│   ├── raw/
│   │   ├── README.md
│   │   └── standards.csv
│   └── processed/
│       └── Final_Pollution_Index_Results.csv
├── src/
│   ├── calc_pi.py
│   ├── preprocess.py
│   ├── training_models_pi.py
│   └── plot_ensemble.py
├── requirements.txt
├── README.md
└── .gitignore

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
