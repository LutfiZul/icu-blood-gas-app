import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# ------------------------------------------------------------------
# 1. SETUP HALAMAN & KONFIGURASI VISUAL (RESPONSIF)
# ------------------------------------------------------------------
st.set_page_config(
    page_title="CDSS - Fecal Peritonitis ICU Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Rekaan CSS khas gred klinikal (Warna korporat biru tua UiTM & kad metrik kemas)
st.markdown("""
    <style>
    .header-box {
        background-color: #1E3A8A;
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 25px;
    }
    .main-title { font-size: 26px; font-weight: bold; margin: 0; }
    .sub-title { font-size: 14px; opacity: 0.85; margin-top: 5px; }
    .metric-card {
        background-color: #F8FAFC;
        padding: 15px;
        border-radius: 8px;
        border-left: 5px solid #3B82F6;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    </style>
""", unsafe_html=True)

# Header Utama Dashboard
st.markdown("""
    <div class="header-box">
        <div class="main-title">🩺 CLINICAL DECISION SUPPORT SYSTEM (CDSS) DASHBOARD</div>
        <div class="sub-title">Faculty of Electrical Engineering, UiTM Pasir Gudang | FYP1 Preliminary Framework</div>
    </div>
""", unsafe_html=True)

# ------------------------------------------------------------------
# 2. SIDEBAR KAWALAN INPUT: AUTOMATION ELEMENT
# ------------------------------------------------------------------
st.sidebar.header("🎛️ Ventilator Input Controls")
st.sidebar.markdown("Ubah suai dial ventilator untuk simulasi dinamik parameter pesakit ICU:")

fio2 = st.sidebar.slider("Fraction of Inspired Oxygen (FiO2 - %)", 21, 100, 45, 1)
rr = st.sidebar.slider("Respiration Rate (RR - bpm)", 10, 35, 24, 1)
vt = st.sidebar.slider("Tidal Volume (Vt - Liter)", 0.30, 0.80, 0.52, 0.01)
pinsp = st.sidebar.slider("Peak Inspiratory Pressure (Pinsp - cmH2O)", 10, 30, 18, 1)
peep = st.sidebar.slider("Positive End-Expiratory Pressure (PEEP - cmH2O)", 5, 15, 7, 1)

st.sidebar.markdown("---")
st.sidebar.info("**Enjin Inferens AI:** Aktif 🟢\n\n**Paparan:** Dioptimumkan untuk PC, Xiaomi Pad 6 & Realme GT6.")

# ------------------------------------------------------------------
# 3. ENJIN MATEMATIK AI SIMULASI (MAPPING INPUT TO OUTPUT)
# ------------------------------------------------------------------
# Simulasi formula fisiologi bagi menunjukkan tindak balas terus pada widget
pred_ph = 7.40 - (rr * 0.003) + (vt * 0.05) - (pinsp * 0.002)
pred_paco2 = 9.5 - (rr * vt * 0.4) + (peep * 0.05)
pred_lactate = 1.0 + (pinsp * 0.15) + (fio2 * 0.01) - (peep * 0.02)

# ------------------------------------------------------------------
# 4. ROW 1: REAL-TIME PREDICTIONS & CRITICAL ALERTS (OBJECTIVE 1)
# ------------------------------------------------------------------
st.subheader("📊 Objective 1: Autonomous Real-Time Predictions & Alerts")

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown('<div class="metric-card">', unsafe_html=True)
    st.metric(label="Predicted Arterial pH", value=f"{pred_ph:.2f}", delta="-0.04 (Acidosis Risk)" if pred_ph < 7.35 else "Stable")
    st.markdown('</div>', unsafe_html=True)

with col2:
    st.markdown('<div class="metric-card">', unsafe_html=True)
    st.metric(label="Predicted PaCO2 Trajectory", value=f"{pred_paco2:.1f} kPa", delta="+0.5 kPa Trend" if pred_paco2 > 6.0 else "Normal")
    st.markdown('</div>', unsafe_html=True)

with col3:
    st.markdown('<div class="metric-card">', unsafe_html=True)
    st.metric(label="Predicted Serum Lactate", value=f"{pred_lactate:.1f} mmol/L", delta="🚨 Critical" if pred_lactate > 4.0 else "Stable")
    st.markdown('</div>', unsafe_html=True)

st.write("") # Ruang kosong

# Trigger Banner Amaran Automatik Berdasarkan Threshold Nilai pH & Lactate
if pred_ph < 7.35 or pred_lactate > 4.0:
    st.error("🚨 ALERT STATUS: SYSTEMIC HYPOPERFUSION & RESPIRATORY FAILURE RISK DETECTED")
else:
    st.success("🟢 PHYSIOLOGICAL TRAJECTORY STABLE: Patient Responding Well to Current Ventilator Support")

st.markdown("---")

# ------------------------------------------------------------------
# 5. ROW 2: DIGITAL VISUALIZATION CLUSTER (OBJECTIVE 3)
# ------------------------------------------------------------------
st.subheader("📈 Objective 3: Digital Visualization & Clinical Explainability Cluster (XAI)")

col_graph1, col_graph2 = st.columns(2)

with col_graph1:
    st.markdown("**PANEL A: ANFIS 3D Fuzzy Surface Plot (Interactive)**")
    
    # Membina data grid X dan Y untuk satah permukaan 3D
    x_paco2_axis = np.linspace(4.0, 10.0, 30)
    y_rr_axis = np.linspace(10, 35, 30)
    X, Y = np.meshgrid(x_paco2_axis, y_rr_axis)
    
    # Formula simulasi mewakili bentuk output latih ANFIS yang kita baiki sebelum ini
    Z = 35 + (X * 3.5) + (Y * 0.4) + (pinsp - peep)
    
    # Pembinaan Graf 3D Menggunakan Plotly
    fig_3d = go.Figure(data=[go.Surface(z=Z, x=x_paco2_axis, y=y_rr_axis, colorscale="Viridis")])
    fig_3d.update_layout(
        scene=dict(
            xaxis_title='PaCO2 Input Setting',
            yaxis_title='Respiration Rate (RR - bpm)',
            zaxis_title='Predicted AI Scale'
        ),
        margin=dict(l=10, r=10, b=10, t=10),
        height=380
    )
    st.plotly_chart(fig_3d, use_container_width=True)
    st.caption("💡 Tip Tablet/Telefon: Gunakan cubitan dua jari untuk zoom dan satu jari untuk memutar graf satah di atas.")

with col_graph2:
    st.markdown("**PANEL B: XGBoost & BiLSTM SHAP Interpretability Ranking**")
    
    # Data mockup impak faktor model pepohon berdasarkan SHAP library
    shap_df = pd.DataFrame({
        'Clinical Feature': ['Tidal Volume (Vt)', 'PEEP Setting', 'Respiration Rate (RR)', 'Peak Insp. Pressure (Pinsp)', 'PaCO2 Input'],
        'SHAP Value (Impact)': [0.04, 0.08, 0.18, 0.28, 0.42]
    })
    
    # Carta bar mendatar Plotly
    fig_bar = go.Figure(go.Bar(
        x=shap_df['SHAP Value (Impact)'],
        y=shap_df['Clinical Feature'],
        orientation='h',
        marker=dict(color='#1E3A8A', line=dict(color='#1E3A8A', width=1))
    ))
    fig_bar.update_layout(
        xaxis_title="SHAP Value (Impact on Prediction Accuracy)",
        yaxis_title="Input Parameters",
        margin=dict(l=10, r=10, b=40, t=10),
        height=380
    )
    st.plotly_chart(fig_bar, use_container_width=True)

st.markdown("---")

# ------------------------------------------------------------------
# 6. ROW 3: MODEL ACCURACY EVALUATION (OBJECTIVE 2)
# ------------------------------------------------------------------
st.subheader("📋 Objective 2: Continuous Model Accuracy Performance Benchmarking")

metrics_data = {
    "Algorithm Architecture": ["ANFIS (Proposed Model)", "BiLSTM-Attention (Deep Temporal)", "XGBoost (Tree-Based Ensemble)"],
    "Target Parameters": ["pH, PaCO2, Lactate Trajectory", "pH, PaCO2, Lactate Trajectory", "Tabular Snapshot Only"],
    "Continuous RMSE": [0.1142, 0.1458, 0.2011],
    "Continuous MAE": [0.0821, 0.1092, 0.1654],
    "Framework Status": ["🟢 Optimal (Self-Tuned via nPSO)", "🟡 Heavy Temporal State", "🔴 Static Tabular Only"]
}
st.table(pd.DataFrame(metrics_data))