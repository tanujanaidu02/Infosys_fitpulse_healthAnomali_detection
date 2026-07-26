# FitPulse - Health Anomaly Detection System (Infosys Springboard)

An AI-powered health monitoring system designed to detect anomalies in heart-rate time-series data using statistical forecasting and machine learning. Developed as part of the **Infosys Springboard Internship (Batch 7)**.

## Project Overview

**FitPulse** is a data science solution aimed at early diagnosis and preventative healthcare. By analyzing heart-rate, activity, and sleep data, the system identifies irregular patterns ("anomalies") that may indicate potential health risks, and presents them through an interactive dashboard.

This project was developed in **4 Strategic Milestones**, simulating a real-world software development lifecycle (SDLC).

## Project Structure & Milestones

- **Fitpulse_Milestone1 (Data Collection & Preprocessing)**
  - Gathering raw health datasets and handling missing values.
  - Implementing data cleaning pipelines (`fitpulse_preprocessing.py`) to ensure data quality.

- **Fitpulse_Milestone2 (Forecasting, Anomaly Detection & Clustering)**
  - Time-series forecasting of heart-rate patterns using **Prophet**.
  - Residual-based anomaly detection, flagging points that deviate significantly (>3σ) from the forecasted trend.
  - Unsupervised clustering of daily heart-rate patterns using **KMeans**, with **PCA** for dimensionality reduction and visualization.

- **Fitpulse_Milestone3 (Statistical Feature Pipeline & Dashboard)**
  - Rolling-window feature extraction (mean, standard deviation) over heart-rate and activity data.
  - Threshold-based real-time anomaly flagging, with an adjustable sensitivity setting.

- **Fitpulse_Milestone4 (Final Dashboard & Deployment)**
  - Interactive, production-style dashboard built with **Streamlit**.
  - Live metrics, anomaly timeline visualization, and automated **PDF health report generation** for sharing with a doctor.

## Tech Stack

- **Language:** Python
- **Data Processing:** Pandas, NumPy
- **Forecasting & ML:** Prophet (time-series forecasting), Scikit-Learn (KMeans clustering, PCA, StandardScaler)
- **Anomaly Detection:** Residual-based statistical detection (Milestone 2), rolling-statistics threshold detection (Milestone 3–4)
- **Web Framework:** Streamlit
- **Visualization:** Plotly, Matplotlib
- **Reporting:** FPDF (automated PDF report generation)

## How to Run

1. **Clone the Repository**

```bash
git clone https://github.com/tanujanaidu02/Infosys_fitpulse_healthAnomaly_detection.git
cd Infosys_fitpulse_healthAnomaly_detection
```

2. **Install Dependencies**

```bash
pip install pandas numpy scikit-learn matplotlib plotly streamlit prophet fpdf2 yfinance
```

3. **Run the Final Dashboard (Milestone 4)**

```bash
cd Fitpulse_Milestone4
streamlit run app4.py
```

*Developed by Chennamsetti Tanuja*
