# Financial Risk Analytics: Synthetic Data Simulation & Fraud Detection

**Project Status**: Work in Progress > This project is currently under active development. The data generation phase is complete, while EDA, advanced Feature Engineering and Model Training are upcoming.

A comprehensive Data Science project simulating a banking transaction environment to train and validate fraud detection models. This repository contains a data generation engine, exploratory data analysis (EDA), and documentation on the underlying business logic.

## Project Overview

This project engineers a synthetic dataset from scratch. This approach serves a dual purpose: to refine advanced Python programming skills and to gain a basic understanding of the data structures and behavioral patterns in a banking ecosystem.

The simulation attempts to implement two critical aspects of Risk Advisory:

- Behavioral Consistency: Modeling human spending habits (seasonality, profiles, fixed vs. discretionary expenses).

- Fraud Typologies: Injecting two specific Account Takeover (ATO) attack patterns, Velocity and Magnitude.

## 📂 Repository Structure

```
├── data/               # Generated datasets (transactions_simulated.csv)
├── notebooks/          # Jupyter Notebooks for analysis and storytelling
│   ├── 01_simulation_logic.ipynb  # Logic behind the code
│   └── 02_eda_analysis.ipynb      # Visual validation of patterns WIP
├── src/                # Source code
│   └── generate_data.py           # The simulation engine
├── .gitignore          # Files and folders excluded from version control
├── README.md
└── requirements.txt    # Python dependencies for reproducibility
```

## Tech Stack and Methodology

- Python: Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn.

- Statistical Logic:

    - Log-Normal Distribution: Used for transaction amounts to reflect the heavy tail of spending.

    - Gaussian Mixture Models: Used to simulate bimodal daily activity peaks (morning commute/evening leisure).

    - Behavioral Profiling: Customers are segmented into archetypes (Thrifty, Standard, Well-off, Techie).

## Fraud Scenarios Implemented

The engine injects controlled anomalies to simulate real-world attacks:

| Attack Type | Mechanism |
| :--- | :--- |
| Velocity Fraud | High-frequency bot attacks (5-15 transactions in minutes). | 
| Magnitude Fraud | Account Takeover (ATO) attempting a "cash-out". |

**Author:** Daniel Expósito Viana