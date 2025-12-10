import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

# -----------------------------
# 1. PAGE SETUP
# -----------------------------
st.set_page_config(
    page_title="Milestone 2 – Heart Rate Analyzer",
    page_icon="❤️",
    layout="wide"
)

st.title("❤️ Milestone 2: Heart Rate Feature Extraction & Modeling")
st.markdown(
    """
**Modules in this app**

- Data simulation & basic statistics  
- Prophet forecasting + anomaly detection  
- Simple clustering of daily patterns (KMeans + PCA)
"""
)

# -----------------------------
# 2. SIDEBAR CONTROLS
# -----------------------------
st.sidebar.header("⚙️ Configuration")

n_days = st.sidebar.slider(
    "Number of days of data", min_value=7, max_value=60, value=30, step=1
)

forecast_days = st.sidebar.slider(
    "Forecast horizon (days)", min_value=7, max_value=30, value=14, step=1
)

cluster_k = st.sidebar.slider(
    "Number of clusters (daily patterns)", min_value=2, max_value=5, value=3
)

st.sidebar.info(
    "This app uses **synthetic heart-rate data** (no real patients). "
    "Data is auto-generated each time for reproducibility."
)

# -----------------------------
# 3. SYNTHETIC HEART RATE DATA
# -----------------------------
@st.cache_data
def generate_heart_rate_data(n_days: int) -> pd.DataFrame:
    """
    Generate realistic synthetic heart-rate data for n_days.
    1-min resolution; includes circadian pattern + workout spikes + noise.
    """
    start = datetime(2024, 1, 1, 0, 0, 0)
    end = start + timedelta(days=n_days)
    timestamps = pd.date_range(start=start, end=end, freq="1min", inclusive="left")

    np.random.seed(42)
    hr_values = []

    for ts in timestamps:
        hour = ts.hour + ts.minute / 60

        # Base resting heart rate
        base_hr = 65

        # Circadian effect: lower at night, higher during day
        circadian = 8 * np.sin((hour - 3) / 24 * 2 * np.pi)

        # Simple "workout" pattern: 7–8 AM and 6–7 PM
        workout_boost = 0
        if 7 <= hour < 8 or 18 <= hour < 19:
            workout_boost = 25

        noise = np.random.normal(0, 3)

        hr = base_hr + circadian + workout_boost + noise
        hr = np.clip(hr, 50, 160)  # keep realistic bounds
        hr_values.append(hr)

    df = pd.DataFrame({"timestamp": timestamps, "heart_rate": hr_values})
    return df

df_hr = generate_heart_rate_data(n_days)

st.subheader("📊 Synthetic Heart Rate Data")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total points", len(df_hr))
with col2:
    st.metric("Average HR", f"{df_hr['heart_rate'].mean():.1f} bpm")
with col3:
    st.metric("Max HR", f"{df_hr['heart_rate'].max():.1f} bpm")

fig_raw = go.Figure()
fig_raw.add_trace(
    go.Scatter(
        x=df_hr["timestamp"],
        y=df_hr["heart_rate"],
        mode="lines",
        line=dict(color="royalblue", width=1),
        name="Heart rate"
    )
)
fig_raw.update_layout(
    title="Raw Heart Rate Time Series",
    xaxis_title="Time",
    yaxis_title="Heart rate (bpm)",
    height=350
)
st.plotly_chart(fig_raw, use_container_width=True)

# -----------------------------
# 4. PROPHET FORECASTING + ANOMALIES
# -----------------------------
st.subheader("🔮 Prophet Forecasting & Anomaly Detection")

# Prepare data for Prophet
prophet_df = df_hr.rename(columns={"timestamp": "ds", "heart_rate": "y"})

model = Prophet(
    daily_seasonality=True,
    weekly_seasonality=False,
    yearly_seasonality=False,
    interval_width=0.9
)
model.fit(prophet_df)

future = model.make_future_dataframe(periods=forecast_days * 24 * 60, freq="min")
forecast = model.predict(future)

# Merge to compute residuals on historical part
merged = prophet_df.merge(
    forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]],
    on="ds",
    how="left"
)
merged["residual"] = merged["y"] - merged["yhat"]
merged["abs_residual"] = merged["residual"].abs()

mae = merged["abs_residual"].mean()
st.write(f"**MAE (Mean Absolute Error)**: `{mae:.2f} bpm`")

# Anomaly detection: residual > 3 * std
res_mean = merged["residual"].mean()
res_std = merged["residual"].std()
threshold = 3 * res_std
anomalies = merged[
    (merged["residual"] > res_mean + threshold) |
    (merged["residual"] < res_mean - threshold)
]

# Plot forecast + CI + anomalies
fig_fc = go.Figure()

# Actual
fig_fc.add_trace(
    go.Scatter(
        x=prophet_df["ds"],
        y=prophet_df["y"],
        mode="markers",
        marker=dict(size=3, color="black"),
        name="Actual"
    )
)

# Forecast line
fig_fc.add_trace(
    go.Scatter(
        x=forecast["ds"],
        y=forecast["yhat"],
        mode="lines",
        line=dict(color="red", width=2),
        name="Forecast"
    )
)

# Confidence interval
fig_fc.add_trace(
    go.Scatter(
        x=forecast["ds"],
        y=forecast["yhat_upper"],
        mode="lines",
        line=dict(width=0),
        showlegend=False
    )
)
fig_fc.add_trace(
    go.Scatter(
        x=forecast["ds"],
        y=forecast["yhat_lower"],
        mode="lines",
        fill="tonexty",
        fillcolor="rgba(255,0,0,0.1)",
        line=dict(width=0),
        name="Confidence interval"
    )
)

# Anomalies
if not anomalies.empty:
    fig_fc.add_trace(
        go.Scatter(
            x=anomalies["ds"],
            y=anomalies["y"],
            mode="markers",
            marker=dict(size=7, color="orange", symbol="x"),
            name="Anomalies"
        )
    )

fig_fc.update_layout(
    title="Prophet Forecast with Confidence Interval & Anomalies",
    xaxis_title="Time",
    yaxis_title="Heart rate (bpm)",
    height=400
)
st.plotly_chart(fig_fc, use_container_width=True)

if not anomalies.empty:
    st.write(f"**Detected {len(anomalies)} potential anomaly points (>|3σ| residuals).**")
    st.dataframe(anomalies[["ds", "y", "yhat", "residual"]].head(10))
else:
    st.info("No strong anomalies detected with 3σ threshold.")

# -----------------------------
# 5. SIMPLE CLUSTERING OF DAILY PATTERNS
# -----------------------------
st.subheader("🧩 Clustering Daily Heart Rate Patterns (KMeans)")

# Aggregate to daily features: mean, max, min HR per day
df_hr["date"] = df_hr["timestamp"].dt.date
daily_features = df_hr.groupby("date").agg(
    mean_hr=("heart_rate", "mean"),
    max_hr=("heart_rate", "max"),
    min_hr=("heart_rate", "min"),
    std_hr=("heart_rate", "std")
).reset_index()

st.write("Daily feature table (used for clustering):")
st.dataframe(daily_features)

# Scale features and run KMeans
scaler = StandardScaler()
X_scaled = scaler.fit_transform(daily_features[["mean_hr", "max_hr", "min_hr", "std_hr"]])

kmeans = KMeans(n_clusters=cluster_k, random_state=42, n_init=10)
labels = kmeans.fit_predict(X_scaled)
daily_features["cluster"] = labels.astype(str)

# Reduce to 2D with PCA for visualization
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
daily_features["pc1"] = X_pca[:, 0]
daily_features["pc2"] = X_pca[:, 1]

fig_clusters = px.scatter(
    daily_features,
    x="pc1",
    y="pc2",
    color="cluster",
    hover_data=["date", "mean_hr", "max_hr", "min_hr"],
    title="Daily Heart Rate Patterns – PCA + KMeans Clusters",
    color_discrete_sequence=px.colors.qualitative.Set1
)
fig_clusters.update_traces(marker=dict(size=10, opacity=0.8))
fig_clusters.update_layout(height=450)
st.plotly_chart(fig_clusters, use_container_width=True)

# Cluster statistics
cluster_stats = daily_features.groupby("cluster").size().reset_index(name="days")
cluster_stats["percentage"] = 100 * cluster_stats["days"] / len(daily_features)

st.write("Cluster statistics:")
st.dataframe(cluster_stats)

st.success("✅ Custom Milestone 2 app completed: forecasting + anomalies + clustering on synthetic heart-rate data.")
