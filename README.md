
# Adaptive Fraud Detection System with Data Drift Monitoring

An end-to-end machine learning system for fraud detection that monitors incoming transaction data for drift and automatically retrains the model when significant distribution shifts are detected.

---

## Overview

In real-world fraud detection systems, transaction behavior changes over time as attackers evolve their strategies. Models trained only on historical data can degrade when the input distribution shifts.

This project builds an adaptive fraud detection pipeline that:

* Trains an initial fraud detection model
* Evaluates performance on time-based future batches
* Detects feature-level data drift using KS test and PSI
* Identifies the most unstable drifting features
* Triggers retraining when drift exceeds a threshold
* Compares old and retrained model performance
* Retains the better-performing model

---

## Dataset

This project uses the IEEE-CIS Credit Card Fraud Detection dataset from Kaggle.

Files used:

* train_transaction.csv
* train_identity.csv (optional)

Place them inside:
data/raw/

---

## Project Structure

adaptive_fraud_detection/
│
├── data/
│   └── raw/
├── src/
│   ├── config.py
│   ├── data_loader.py
│   ├── preprocess.py
│   ├── train.py
│   ├── evaluate.py
│   ├── simulator.py
│   ├── drift.py
│   ├── drift_analysis.py
│   ├── retrain.py
│   └── plots.py
│
├── models/
├── results/
├── notebooks/
├── main.py
├── requirements.txt
├── .gitignore
└── README.md

---

## Pipeline Stages

Phase 1: Baseline Fraud Detection

* Load and clean data
* Sort chronologically
* Train baseline model
* Evaluate batch-wise performance

Phase 2: Drift Detection

* Detect drift using:

  * Kolmogorov–Smirnov (KS) Test
  * Population Stability Index (PSI)

Phase 2.5: Drift Analysis

* Identify top drifting features
* Compute global drift ranking

Phase 3: Adaptive Retraining

* Trigger retraining when drift exceeds threshold
* Retrain on recent data
* Compare old vs new model using PR-AUC
* Replace model only if performance improves

Phase 4: Visualization

* Plot drift and performance trends

---

## Model

The final implementation uses XGBoost for fraud detection.

Why XGBoost:

* Handles tabular data effectively
* Captures nonlinear relationships
* Works well with imbalanced datasets using scale_pos_weight

---

## Drift Detection Methods

Kolmogorov–Smirnov (KS) Test
Used to compare distribution between training and incoming data.

Population Stability Index (PSI)
Used to measure magnitude of distribution shift.

Thresholds:

* avg_psi > 0.1 → retraining triggered
* psi > 0.25 → strong drift

---

## Key Results

* High drift detected in features:
  D9, V160, V151, V159, V143, V144

* Model performance improved significantly using XGBoost:
  PR-AUC improved from ~0.06 to ~0.46

* Adaptive retraining improved performance across drifting batches:

Batch 3: 0.2439 → 0.7860
Batch 4: 0.5556 → 0.8564
Batch 7: 0.5764 → 0.8341
Batch 9: 0.3982 → 0.7358

These results show that retraining on recent data allows the model to adapt to evolving fraud patterns.

---

## Visual Outputs

Generated plots:

* phase4_adaptive_pr_auc.png
* phase4_avg_psi.png
* phase4_drift_vs_performance.png

These plots illustrate:

* When drift occurs
* When retraining is triggered
* How performance improves

---

## Setup Instructions

1. Clone the repository
   git clone https://github.com/rishikaranjan/adaptive-fraud-detection.git
   cd adaptive-fraud-detection

2. Create virtual environment

Windows:
python -m venv .venv
.venv\Scripts\activate

macOS/Linux:
python3 -m venv .venv
source .venv/bin/activate

3. Install dependencies
   pip install -r requirements.txt

4. Add dataset

Place:
train_transaction.csv
train_identity.csv

inside:
data/raw/

5. Run the project
   python main.py

---

## Outputs

The system generates:

* Batch evaluation metrics
* Drift summary CSV files
* Feature-level drift details
* Top drifting feature rankings
* Adaptive retraining results
* Visualization plots
* Trained model pipeline

---

## Future Improvements

* Sliding window retraining
* Real-time drift detection (ADWIN)
* Streamlit dashboard
* MLflow tracking
* Real-time streaming pipeline
* Threshold tuning

---

## Author

Rishika Ranjan
