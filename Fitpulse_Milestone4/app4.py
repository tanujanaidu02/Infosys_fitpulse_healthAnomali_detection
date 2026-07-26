# filename: app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from fpdf import FPDF
import os

# Import pipeline logic
from data_pipeline4 import run_pipeline, process_data

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Fitpulse", 
    layout="wide", 
    page_icon="⚡",
    initial_sidebar_state="expanded"
)

# --- 2. PROFESSIONAL DARK MODE CSS ---
st.markdown("""
    <style>
    /* Main Background */
    .stApp {
        background: rgb(15,23,42);
        background: linear-gradient(180deg, rgba(15,23,42,1) 0%, rgba(30,41,59,1) 100%);
        color: white;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #0F172A;
        border-right: 1px solid #334155;
    }
    
    /* Headers */
    .header-title {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(to right, #4facfe 0%, #00f2fe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 0 20px rgba(79, 172, 254, 0.5);
    }
    
    /* 4. Glassmorphism Cards (Dark) */
    .metric-card {
        background: linear-gradient(135deg, rgba(255, 99, 132, 0.15), rgba(255, 99, 132, 0.05));
        border: 1px solid rgba(255, 99, 132, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.5);
        transition: all 0.3s ease-in-out;
    }
    
    /* 🌟 HERO CARD (Red/Pink Glow for Heart Rate) */
    .metric-card-hero {
        background: linear-gradient(135deg, rgba(255, 99, 132, 0.15), rgba(255, 99, 132, 0.05));
        border: 1px solid rgba(255, 99, 132, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.5);
        transition: all 0.3s ease-in-out;
    }
    
    .metric-card:hover, .metric-card-hero:hover {
        transform: translateY(-5px);
        background: rgba(255, 255, 255, 0.15);
    }
    
    .metric-value { font-size: 2.5rem; font-weight: bold; color: #fff; text-shadow: 0 0 10px rgba(255,255,255,0.5); }
    .metric-label { font-size: 1rem; color: #A5F3FC; text-transform: uppercase; letter-spacing: 1px; font-weight: 600; }
    .metric-icon { font-size: 2rem; margin-bottom: 10px; }
    
    /* Status Banners */
    .status-safe { 
        background: rgba(16, 185, 129, 0.15); 
        color: #6EE7B7; 
        padding: 15px; 
        border-radius: 10px; 
        border: 1px solid #10B981; 
        box-shadow: 0 0 10px rgba(16, 185, 129, 0.2);
    }
    .status-danger { 
        background: rgba(239, 68, 68, 0.15); 
        color: #FCA5A5; 
        padding: 15px; 
        border-radius: 10px; 
        border: 1px solid #EF4444; 
        box-shadow: 0 0 15px rgba(239, 68, 68, 0.3);
        animation: pulse 2s infinite; 
    }
    
    /* Disclaimer Box */
    .disclaimer-box {
        background-color: rgba(255, 255, 255, 0.05); 
        padding: 15px; 
        border-radius: 10px; 
        font-size: 0.85rem; 
        color: #94A3B8;
        border-left: 3px solid #4facfe;
        margin-top: 20px;
    }
    
    @keyframes pulse { 0% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.4); } 70% { box-shadow: 0 0 0 10px rgba(239, 68, 68, 0); } 100% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0); } }
    
    h1, h2, h3 { color: white !important; }
    p, label { color: #CBD5E1 !important; }
    </style>
""", unsafe_allow_html=True)

# --- 3. DATA LOADING ---
@st.cache_data
def load_data(file):
    if file is not None:
        df_raw = pd.read_csv(file)
        return process_data(df_raw)
    else:
        if not os.path.exists('fitpulse_data.csv'):
            return run_pipeline()
        return pd.read_csv('fitpulse_data.csv', parse_dates=['timestamp'])

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://img.icons8.com/?size=100&id=47294&format=png&color=000000", width=120)
    
    st.markdown("### ⚙️ My Settings")
    
    # NEW: Better Help Text
    uploaded_file = st.file_uploader("Upload CSV", type=['csv'], help="File must contain 'timestamp', 'heart_rate', and 'steps'.")
    st.success("🔒 Your data is private & secure")
    
    # NEW: Time Filter
    st.markdown("### 📅 Time Filter")
    time_filter = st.selectbox("Select Range", ["All Time", "Last 24 Hours", "Last 7 Days"])
    
    st.markdown("""
    <div class="disclaimer-box">
        <b>💙 Note on Health:</b><br>
        This dashboard helps you track patterns, but it is not a medical diagnosis. 
        If you see repeated alerts, please share this report with a doctor.
    </div>
    """, unsafe_allow_html=True)

df = load_data(uploaded_file)

# Apply Time Filter Logic (Simple implementation for demo)
if time_filter == "Last 24 Hours":
    df = df.tail(24) 
elif time_filter == "Last 7 Days":
    df = df.tail(168) # Approx hours in a week

latest = df.iloc[-1]

# --- 4. HEADER ---
col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.markdown('<div class="header-title">💙 FitPulse Health Anomaly Detection</div>', unsafe_allow_html=True)
    st.markdown('<div class="header-subtitle">Simple. Smart. Secure.</div>', unsafe_allow_html=True)
with col_h2:
    st.markdown(f"*Hello, User*<br>System Active 🟢", unsafe_allow_html=True)
    # NEW: Last Updated Label
    last_ts = pd.to_datetime(latest['timestamp']).strftime('%H:%M %p')
    st.caption(f"Last updated: {last_ts}")

# Spacer for breathing room
st.markdown("<br>", unsafe_allow_html=True)

# --- 5. ALERTS (REWRITTEN COPY) ---
is_critical = latest['heart_rate'] > 130

if is_critical:
    st.markdown("""
    <div class="status-danger">
        <b>⚠️ Attention Needed:</b> Your heart rate is higher than your set limit. 
        Please check your activity context (e.g., are you exercising?).
    </div>
    """, unsafe_allow_html=True)
    
    with st.expander("🧘 What should I do now? (Click to open)"):
        st.markdown("""
        1.  *🛑 Stop & Sit:* Pause what you are doing.
        2.  *🌬️ Breathe:* Take slow, deep breaths for 2 minutes.
        3.  *💧 Hydrate:* Drink a glass of water.
        """)
else:
    st.markdown("""
    <div class="status-safe">
        <b>✅ All Good.</b> No anomalies detected right now.
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# --- 6. METRICS (HERO CARD + FRIENDLY DATA) ---
m_col1, m_col2, m_col3 = st.columns(3)

# Logic for friendly text
sleep_display = f"{round(latest['sleep_minutes']/60, 1)} hrs" if latest['sleep_minutes'] > 0 else "No data yet"

with m_col1:
    # HERO CARD (Different Class)
    st.markdown(f"""<div class="metric-card-hero"><div class="metric-icon">❤️</div><div class="metric-label">Heart Rate (BPM)</div><div class="metric-value">{int(latest['heart_rate'])}</div></div>""", unsafe_allow_html=True)
with m_col2:
    st.markdown(f"""<div class="metric-card"><div class="metric-icon">👣</div><div class="metric-label">Steps Today</div><div class="metric-value">{int(latest['steps'])}</div></div>""", unsafe_allow_html=True)
with m_col3:
    st.markdown(f"""<div class="metric-card"><div class="metric-icon">🌙</div><div class="metric-label">Sleep (Hrs)</div><div class="metric-value">{round(latest['sleep_minutes']/60, 1)}</div></div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# --- 7. TABS (IMPROVED CHARTS) ---
tab1, tab2, tab3 = st.tabs(["📈 My Trends", "🚨 Pattern Check", "📤 Share Report"])

with tab1:
    st.subheader("🫀 Heart Rate Trends")
    st.caption("View your heart rate activity over time. Spikes usually indicate exercise or stress.")
    
    fig = px.area(df, x='timestamp', y='heart_rate', title="Heart Rate History", color_discrete_sequence=['#4facfe'], template='plotly_dark')
    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#94A3B8'))
    # Better Axis Labels
    fig.update_xaxes(title_text="Time of Day")
    fig.update_yaxes(title_text="BPM")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    col_a1, col_a2 = st.columns([1, 2])
    with col_a1:
        st.markdown("### ⚙️ Alert Settings")
        threshold = st.slider("Alert Threshold (BPM)", 100, 200, 130)
        st.caption("Adjust this limit to see fewer or more alerts. Resting heart rate is usually 60-100 BPM.")
        
        df['is_anomaly'] = df['heart_rate'] > threshold
        anomalies = df[df['is_anomaly'] == True]
        
        st.metric("Total Anomalies Found", len(anomalies))

    with col_a2:
        st.subheader("Anomaly Timeline")
        # Scatter Plot with Reference Line
        fig_anom = px.scatter(df, x='timestamp', y='heart_rate', color='is_anomaly', 
                              color_discrete_map={False: '#334155', True: '#F87171'}, 
                              title="Detected Anomalies", template='plotly_dark')
        
        # Add Horizontal Line for Threshold
        fig_anom.add_hline(y=threshold, line_dash="dash", line_color="white", annotation_text="Limit")
        
        fig_anom.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', 
                               xaxis_title="Time of Day", yaxis_title="Heart Rate (BPM)")
        
        # Better Tooltips
        fig_anom.update_traces(hovertemplate='<b>Time:</b> %{x}<br><b>BPM:</b> %{y}')
        
        st.plotly_chart(fig_anom, use_container_width=True)

with tab3:
    st.header("📄 Share with my Doctor")
    st.write("This generates a professional PDF summary of your anomalies to email to your healthcare provider.")
    
    def create_pdf(df, anomalies, threshold):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(0, 10, "FitPulse Health Anomaly Report", ln=True, align='C')
        pdf.ln(10)
        
        pdf.set_font("Arial", size=12)
        pdf.cell(0, 10, txt=f"Total Records Analyzed: {len(df)}", ln=True)
        pdf.cell(0, 10, txt=f"Anomalies Detected: {len(anomalies)}", ln=True)
        pdf.cell(0, 10, txt=f"Safety Threshold Used: {threshold} BPM", ln=True)
        pdf.ln(10)
        
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, "Critical Events (Top 10):", ln=True)
        pdf.ln(2)
        
        pdf.set_fill_color(240, 240, 240)
        pdf.cell(65, 10, "Timestamp", border=1, fill=True)
        pdf.cell(40, 10, "Heart Rate", border=1, fill=True)
        pdf.cell(40, 10, "Status", border=1, ln=True, fill=True)
        
        pdf.set_font("Arial", size=11)
        top_anomalies = anomalies.head(10)
        
        if len(top_anomalies) > 0:
            for index, row in top_anomalies.iterrows():
                ts_str = str(row['timestamp'])
                hr_str = f"{int(row['heart_rate'])} BPM"
                pdf.cell(65, 10, ts_str, border=1)
                pdf.cell(40, 10, hr_str, border=1)
                pdf.cell(40, 10, "CRITICAL", border=1, ln=True)
        else:
            pdf.cell(145, 10, "No critical events found.", border=1, ln=True)
            
       return bytes(pdf.output(dest='S'))

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1: 
        st.download_button("📄 Download PDF Report", create_pdf(df, anomalies, threshold), "health_report.pdf", "application/pdf")
    with col_btn2: 
        st.download_button("📂 Download Raw CSV", df.to_csv(index=False).encode('utf-8'), "my_health_data.csv", "text/csv")
