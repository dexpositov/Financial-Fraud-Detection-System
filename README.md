# 🛡️ Risk Analytics: Transaction Fraud Detection System

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-EB4223?style=for-the-badge&logo=xgboost&logoColor=white)
![Status](https://img.shields.io/badge/Status-Completed-success?style=for-the-badge)

> **A Hybrid Fraud Detection Pipeline combining Unsupervised Profiling (K-Means) with Supervised Learning (XGBoost) to detect Account Takeover (ATO) attacks.**

## Executive Summary

This project implements an end-to-end Machine Learning pipeline to detect fraudulent credit card transactions. Unlike standard implementations, this project features a **custom-built Scikit-Learn Transformer** to engineer behavioral features (customer profiling) from scratch.

**Key Achievement:**
The model successfully identified the injected attack patterns, shifting detection logic from simple contextual proxies (Time of Day) to robust behavioral indicators, achieving a **ROC-AUC of 1.0** on the test set.

## Project Overview

This project simulates a **Risk Advisory engagement**, aiming to build a defense system against financial fraud. The objective was not just to train a model, but to architect a complete **Data Science solution** that mirrors a real-world production environment.

The system addresses the challenge of distinguishing between legitimate high-spending behavior and actual criminal activity by implementing a **Hybrid Strategy**:
1.  **Behavioral Profiling (Unsupervised):** Establishing "normal" customer baselines using K-Means clustering.
2.  **Anomaly Detection (Supervised):** Training classifiers to identify deviations from these profiles.

While the data environment is synthetic (simulated to ensure specific fraud typologies), the pipeline architecture is production-ready, handling chronological data splitting, custom transformations, and preventig data leakage.

* **Behavioral Consistency:** Modeled human spending habits using **Gaussian Mixture Models** to simulate bimodal daily activity peaks (morning commute/evening leisure) and seasonal patterns.
* **Customer Profiling:** Segmented the user base into distinct archetypes (*Thrifty, Standard, Well-off, Techie*) using statistical thresholds.
* **Fraud Injection:** Imposed specific Account Takeover (ATO) attack patterns to serve as ground truth for the models.

## 🏴‍☠️ Fraud Scenarios Implemented

The engine injected controlled anomalies to simulate real-world attacks. These scenarios were the primary targets for our detection models:

| Attack Type | Mechanism | Detection Outcome |
| :--- | :--- | :--- |
| **Velocity Fraud** | High-frequency bot attacks (5-15 transactions in minutes). | **Fully Detected** by XGBoost via `Transactions_Last_Hour`. |
| **Magnitude Fraud** | Account Takeover (ATO) attempting a "cash-out". | **Fully Detected** via outlier analysis on `Amount`. |

## 🛠️ Tech Stack and Methodology

* **Core Logic:** Python, Pandas, NumPy.
* **Machine Learning:** Scikit-learn (Pipelines, Custom Transformers), XGBoost, Joblib.
* **Visualization:** Matplotlib, Seaborn.
* **Key Engineering Concepts:**
    * **Custom Transformers:** Inheriting from `BaseEstimator` to build domain-specific logic.
    * **Strict Validation:** Chronological Train/Test splitting to simulate real-time deployment.
    * **Unsupervised Features:** Using Cluster Distance as a predictive feature.

## 📊 Model Performance & Insights

We compared a baseline linear model against a tree-based ensemble to understand the "Why" behind the fraud.

| Model | ROC-AUC Score | Key Driver (Feature) |
| :--- | :---: | :--- |
| **Logistic Regression** | 0.9997 | `Is_Night` (Context) |
| **XGBoost (Final)** | **1.0000** | `Transactions_Last_Hour` (Behavior) |

### The "Context vs. Behavior" Discovery
* **The Linear Model** "memorized" the schedule of attacks (Night time).
* **The XGBoost Model** discovered the root cause: **Velocity**. It identified that 76% of the fraud signal came from the engineered feature `Transactions_Last_Hour`, proving that the attack vector was a bot/script execution.

## Architecture: The Custom Transformer

A key engineering highlight is the **`FraudPreprocessor`** class, built from scratch to inherit from Scikit-Learn's `BaseEstimator`.

* **Goal:** Create a production-ready pipeline component.
* **Innovation:** Implements a custom `.fit()` method that stores client history (spending centroids) without peeking at future test data (Data Leakage prevention).
* **Function:** Calculates dynamic features like *distance to spending centroid* and *transaction velocity* in real-time.

## Pipeline Workflow
```mermaid
A[Raw Data] --> B(FraudPreprocessor (K-Means ))
B --> C{Column Transformer (OHE)}
C --> D[XGBoost Classifier]
```

## 📂 Repository Structure

```
├── data/               # Generated datasets (transactions_simulated.csv)
├── images/             # Useful visualizations
├── notebooks/          # Jupyter Notebooks for analysis and storytelling
│   ├── 01_simulation_logic.ipynb       # Logic behind the code
│   ├── 02_eda_analysis.ipynb           # Visual validation of patterns WIP
│   ├── 03_customer_segmentation.ipynb  # Customer segmentation K-means model
│   ├── 04_fraud_detection_model.ipynb  # Pipeline creation and XGBoost model
│   └── 05_project_retrospective.ipynb  # Final Insights on qualitative aspects
├── scripts/                # Source code
│   ├── generate_data.py                # The simulation engine
│   └── transformers.py                 # Custom transformer class  
├── .gitignore          # Files and folders excluded from version control
├── README.md
└── requirements.txt    # Python dependencies for reproducibility
```

## Critical Retrospective (Self-Assessment)
* **Synthetic Data Limitations:** The perfect AUC score (1.0) indicates the synthetic dataset was strictly deterministic. While this simplified the classification task, it allowed for a transparent analysis of feature importance.

* **Engineering Focus:** Given the data simplicity, the project's primary value lies in the Software Engineering aspects (Custom Transformers, Pipelines, Modular Code) rather than purely in model tuning.

* (More in the 5th notebook)

### **Author:** Daniel Expósito Viana

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](www.linkedin.com/in/daniel-expósito-viana)  