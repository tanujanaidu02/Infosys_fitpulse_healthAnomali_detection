# filename: data_pipeline.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

def generate_health_data():
    """Generates synthetic data for the 'Demo Mode'."""
    np.random.seed(42)
    n_samples = 1000  
    start_date = datetime(2025, 1, 1)
    data = []
    
    for i in range(n_samples):
        timestamp = start_date + timedelta(hours=i)
        is_day = 8 <= timestamp.hour <= 22
        
        hr = np.random.normal(80, 10) if is_day else np.random.normal(60, 5)
        steps = np.random.randint(0, 1000) if is_day else 0
        sleep = 0 if is_day else np.random.randint(0, 60)
        
        if np.random.random() < 0.05: # Random Anomalies
            hr += 40  
            steps += 2000 
        
        data.append({
            'timestamp': timestamp,
            'heart_rate': max(40, hr),
            'steps': max(0, steps),
            'sleep_minutes': max(0, sleep)
        })
    
    return pd.DataFrame(data)

def clean_data(df):
    """Standardizes column names and fixes missing values."""
    # Ensure column names are standard (useful for uploaded files)
    df.columns = [c.lower().replace(' ', '_') for c in df.columns]
    
    # Ensure timestamp is datetime
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Fix missing values
    df = df.fillna(method='ffill').fillna(0)
    return df

def extract_statistical_features(df):
    """Calculates Rolling Means and Trends."""
    window = 24 
    for col in ['heart_rate', 'steps']:
        if col in df.columns:
            df[f'{col}_rolling_mean'] = df[col].rolling(window=window, min_periods=1).mean()
            df[f'{col}_rolling_std'] = df[col].rolling(window=window, min_periods=1).std()
    return df.fillna(0)

def process_data(df):
    """Main function to process ANY data (Generated or Uploaded)."""
    df_clean = clean_data(df)
    df_features = extract_statistical_features(df_clean)
    return df_features

def run_pipeline():
    # Only for Demo Mode generation
    df_raw = generate_health_data()
    df_final = process_data(df_raw)
    df_final.to_csv('fitpulse_data.csv', index=False)
    return df_final

if __name__ == "_main_":
    run_pipeline()