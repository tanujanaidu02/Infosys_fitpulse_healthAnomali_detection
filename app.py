# filename: app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
import os

# Import pipeline
from data_pipeline import run_pipeline

# --- PAGE CONFIG ---
st.set_page_config(page_title="FitPulse Analysis", layout="wide")
st.title("🏥 FitPulse: Comprehensive Health Analysis")
st.markdown("""
* **Milestone 1:** Data Collection & Cleaning
* **Milestone 2:** Feature Extraction & Modeling (Prophet, KMeans)
* **Milestone 3:** Anomaly Detection (Rules, Residuals, Clusters)
""")

# --- DATA LOADER ---
@st.cache_data
def get_data():
    if not os.path.exists('fitpulse_data.csv'):
        return run_pipeline()
    return pd.read_csv('fitpulse_data.csv', parse_dates=['timestamp'])

df = get_data()

# --- TABS FOR TASKS ---
tab1, tab2, tab3 = st.tabs(["📊 Data & Features", "🧠 Models (Prophet/KMeans)", "🚨 Anomaly Detection"])

# =========================================================
# TAB 1: DATA & FEATURE EXTRACTION
# =========================================================
with tab1:
    st.header("Milestone 1 & 2: Preprocessing & Features")
    st.write("Cleaned data with Statistical Features extracted (Rolling Means, Std Dev).")
    
    # Show Raw Data vs Extracted Features
    col1, col2 = st.columns(2)
    with col1: 
        st.dataframe(df.head())
    with col2:
        st.info(f"**Total Records:** {len(df)}")
        st.info("**Features Extracted:** Rolling Mean, Rolling Std, Rolling Max")

    # Visualize Feature Extraction
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['heart_rate'], name='Raw HR', line=dict(color='blue', width=1)))
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['heart_rate_rolling_mean'], name='Trend (Rolling Mean)', line=dict(color='orange', width=2)))
    st.plotly_chart(fig, use_container_width=True)

# =========================================================
# TAB 2: MODELING (PROPHET & CLUSTERING)
# =========================================================
with tab2:
    st.header("Milestone 2: Modeling")
    
    # --- 1. CLUSTERING (KMEANS) ---
    st.subheader("1. Clustering Behaviors (KMeans)")
    st.write("Grouping time points into behaviors: 'Resting', 'Active', 'Stressed'.")
    
    # Run KMeans
    features_for_clustering = df[['heart_rate', 'steps', 'sleep_minutes']].fillna(0)
    kmeans = KMeans(n_clusters=3, random_state=42)
    df['cluster'] = kmeans.fit_predict(features_for_clustering)
    
    # Visualize Clusters
    fig_cluster = px.scatter_3d(df, x='heart_rate', y='steps', z='sleep_minutes', color='cluster',
                                title="3D Cluster Analysis of Behaviors")
    st.plotly_chart(fig_cluster, use_container_width=True)
    
    # --- 2. FORECASTING (PROPHET) ---
    st.subheader("2. Seasonal Modeling (Facebook Prophet)")
    st.write("Modeling the daily seasonality of Heart Rate.")
    
    # Prepare for Prophet
    prophet_df = df[['timestamp', 'heart_rate']].rename(columns={'timestamp': 'ds', 'heart_rate': 'y'})
    
    # Train Model
    m = Prophet(daily_seasonality=True)
    m.fit(prophet_df)
    forecast = m.predict(m.make_future_dataframe(periods=24, freq='H'))
    
    # Plot
    fig_prophet = go.Figure()
    fig_prophet.add_trace(go.Scatter(x=df['timestamp'], y=df['heart_rate'], name='Actual'))
    fig_prophet.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='Prophet Model', line=dict(color='red')))
    st.plotly_chart(fig_prophet, use_container_width=True)

# =========================================================
# TAB 3: ANOMALY DETECTION
# =========================================================
with tab3:
    st.header("Milestone 3: Anomaly Detection")
    
    # --- METHOD 1: RULE-BASED ---
    st.subheader("1. Rule-Based (Thresholds)")
    threshold = st.slider("Max Heart Rate Threshold", 100, 160, 130)
    rule_anomalies = df[df['heart_rate'] > threshold]
    st.write(f"Found {len(rule_anomalies)} instances where HR > {threshold} BPM.")
    
    # --- METHOD 2: MODEL-BASED (RESIDUALS) ---
    st.subheader("2. Model-Based (Prophet Residuals)")
    # Merge forecast to get residuals
    df_merged = pd.merge(df, forecast[['ds', 'yhat', 'yhat_upper', 'yhat_lower']], 
                         left_on='timestamp', right_on='ds')
    
    # Anomaly = Actual value is outside the Prophet confidence interval
    df_merged['is_anomaly'] = (df_merged['heart_rate'] > df_merged['yhat_upper']) | \
                              (df_merged['heart_rate'] < df_merged['yhat_lower'])
    
    model_anomalies = df_merged[df_merged['is_anomaly'] == True]
    st.write(f"Found {len(model_anomalies)} anomalies where data deviated from the model predictions.")
    
    # --- VISUALIZATION ---
    st.subheader("Anomaly Visualization")
    fig_anom = go.Figure()
    
    # Normal Data
    fig_anom.add_trace(go.Scatter(x=df_merged['timestamp'], y=df_merged['heart_rate'], 
                                  mode='lines', name='Normal Data', line=dict(color='gray', width=1)))
    
    # Anomalies
    fig_anom.add_trace(go.Scatter(x=model_anomalies['timestamp'], y=model_anomalies['heart_rate'], 
                                  mode='markers', name='Detected Anomalies', 
                                  marker=dict(color='red', size=10, symbol='x')))
    
    st.plotly_chart(fig_anom, use_container_width=True)