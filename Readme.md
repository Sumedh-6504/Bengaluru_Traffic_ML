# 🚦 Bengaluru Traffic Predictor AI

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://bengaluru-traffic.streamlit.app/)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Framework](https://img.shields.io/badge/Streamlit-1.31-red)
![ML](https://img.shields.io/badge/XGBoost-Regression-orange)
![Status](https://img.shields.io/badge/Maintained-Yes-green)

### 🚀 **[Click Here to Open Live App](https://bengaluru-traffic.streamlit.app/)**

A comprehensive Machine Learning solution to predict **Traffic Volume** and **Congestion Levels** across major Bengaluru junctions. This project moves beyond simple analysis to provide an **End-to-End deployment** using XGBoost time-series forecasting and a Streamlit interactive dashboard.

---

### 🚀 Live Traffic Intelligence Dashboard
A comprehensive Machine Learning solution to predict **Traffic Volume** and **Congestion Levels** across major Bengaluru junctions. This project moves beyond simple analysis to provide an **End-to-End deployment** using XGBoost time-series forecasting and a Streamlit interactive dashboard.

---

## 🧠 Project Overview
Bangalore is known for its unpredictable traffic. This project aims to solve the "When should I leave?" problem by predicting future traffic conditions.

**Key Features:**
*   **Time-Series Forecasting:** Uses 7-day Lag features (Autoregression) to capture weekly traffic cycles.
*   **Smart Backtracking:** Handles future dates (e.g., 2025/2026) by intelligently looking up historical baselines from previous years.
*   **Conservative Modeling:** Tuned XGBoost model ($R^2 \approx 0.31$) that prioritizes trend accuracy over overfitting.
*   **Geospatial Heatmap:** Live interactive map showing congestion zones across the city.
*   **Event Impact:** Simulates the impact of "Roadwork" or "Incidents" on travel time.

---

## 🛠️ Tech Stack
*   **Core:** Python 3.10
*   **Data Processing:** Pandas, NumPy
*   **Machine Learning:** XGBoost, Scikit-Learn (RandomizedSearchCV)
*   **Visualization:** Matplotlib, Seaborn, Folium
*   **Deployment:** Streamlit (Web App)
*   **Persistence:** Joblib (Model Serialization)

---

## 📂 Project Structure
```text
Bengaluru_Traffic_ML/                   
├── data/
│   ├── raw/                 # Original Dataset
│   └── processed/           # Feature Engineered Data 
├── eda_and_model/
│   ├── models/saved_models/ # Serialized Artifacts (.joblib)
│   └── bengaluru-traffic.ipynb # Training & Tuning Notebook
├── app.py                   # Main Streamlit Dashboard Application
├── requirements.txt         # Dependencies
└── README.md                # Documentation
