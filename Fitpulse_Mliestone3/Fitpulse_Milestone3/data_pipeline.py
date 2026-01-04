# filename: data_pipeline.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
import warnings

warnings.filterwarnings('ignore')

def generate_health_data():
    """Generates synthetic data to simulate CSV import."""
    np.random.seed(42)
    n_samples = 1000  # Approx 20 days of hourly data
    start_date = datetime(2025, 1, 1)
    data = []
    
    for i in range(n_samples):
        timestamp = start_date + timedelta(hours=i)
        
        # Base patterns
        is_day = 8 <= timestamp.hour <= 22
        
        # Simulating metrics
        hr = np.random.normal(80, 10) if is_day else np.random.normal(60, 5)
        steps = np.random.randint(0, 1000) if is_day else 0
        sleep = 0 if is_day else np.random.randint(0, 60)
        
        # Add random anomalies (5% chance)
        if np.random.random() < 0.05:
            hr += 40  # Anomaly spike
            steps += 2000 # Impossible steps
        
        data.append({
            'timestamp': timestamp,
            'heart_rate': max(40, hr),
            'steps': max(0, steps),
            'sleep_minutes': max(0, sleep)
        })
    
    return pd.DataFrame(data)

def clean_data(df):
    """Task: Clean timestamps, fix missing values, align intervals."""
    print("✓ Cleaning Data...")
    
    # 1. Clean Timestamps
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # 2. Fix Missing Values (Imputation)
    df = df.fillna(method='ffill').fillna(0)
    
    # 3. Align Time Intervals (Resample to hourly to ensure consistency)
    df = df.set_index('timestamp').resample('H').mean().reset_index()
    
    return df

def extract_statistical_features(df):
    """
    Task: Extract statistical features (TSFresh equivalent).
    We calculate Rolling Means, Standard Deviations, and Peaks.
    """
    print("✓ Extracting Features (TSFresh style)...")
    
    window = 24 # 24-hour window
    
    for col in ['heart_rate', 'steps']:
        # Mean (Trend)
        df[f'{col}_rolling_mean'] = df[col].rolling(window=window, min_periods=1).mean()
        # Std Dev (Volatility)
        df[f'{col}_rolling_std'] = df[col].rolling(window=window, min_periods=1).std()
        # Max (Peak)
        df[f'{col}_rolling_max'] = df[col].rolling(window=window, min_periods=1).max()
    
    return df.fillna(0)

def run_pipeline():
    # 1. Collection
    df_raw = generate_health_data()
    
    # 2. Preprocessing
    df_clean = clean_data(df_raw)
    
    # 3. Feature Extraction
    df_features = extract_statistical_features(df_clean)
    
    # Save
    df_features.to_csv('fitpulse_data.csv', index=False)
    print("✓ Pipeline Complete. Saved to 'fitpulse_data.csv'")
    return df_features

if __name__ == "__main__":
    run_pipeline()
