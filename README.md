# ✈️ Predictive Maintenance MLOps Pipeline

This repository contains an end-to-end Machine Learning Operations (MLOps) system designed to predict the **Remaining Useful Life (RUL)** of jet engines using the NASA C-MAPSS dataset.

## 🏗️ System Architecture

- **Data Layer:** PostgreSQL (Stores structured sensor telemetry).
- **Orchestration:** Prefect (Automated ETL, Data Validation, and RUL labeling).
- **Experiment Tracking:** MLflow (Logs XGBoost hyperparameters and performance metrics).
- **Deployment:** FastAPI (A REST API serving real-time maintenance predictions).

## 📊 Performance Results

- **Model:** XGBoost Regressor
- **Metric:** Root Mean Squared Error (RMSE): **35.9**
- **Traceability:** Every training run is versioned and logged in the MLflow Model Registry.

## 🚀 Tech Stack

`Python` | `SQL` | `PostgreSQL` | `Prefect` | `MLflow` | `XGBoost` | `FastAPI`
