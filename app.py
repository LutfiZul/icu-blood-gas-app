import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import plotly.graph_objects as go

# ---------------------------------------------------------
# 1. PAGE CONFIGURATION & TITLE
# ---------------------------------------------------------
st.set_page_config(
    page_title="ICU AI Data Manager - Blood Gas Predictor",
    page_layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🩺 ICU AI Data Manager: Blood Gas (PaO2) Predictor")
st.markdown("---")

# ---------------------------------------------------------
# 2. LOAD TRAINED MODEL & SCALER
# ---------------------------------------------------------
@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model("bilstm_model.h5", compile=False)
    scaler = joblib.load("scaler.pkl")
    return model, scaler

try:
    model, scaler = load_assets()
    st.sidebar.success("✅ AI Model & Scaler Loaded Successfully!")
except Exception as e:
    st.sidebar.error(f"❌ Error loading assets: {e}")

# ---------------------------------------------------------
# 3. SIDEBAR: CLINICAL INPUT SLIDERS
# ---------------------------------------------------------
st.sidebar.header("🎛️ Patient Vital Signs (Current)")

hr = st.sidebar.slider("Heart Rate (BPM)", min_value=40, max_value=160, value=85)
spo2 = st.sidebar.slider("SpO2 (%)", min_value=70, max_value=100, value=96)
rr = st.sidebar.slider("Respiratory Rate (bpm)", min_value=8, max_value=40, value=18)
fio2 = st.sidebar.slider("FiO2 (Fraction of Inspired O2)", min_value=0.21, max_value=1.00, value=0.40, step=0.01)

# ---------------------------------------------------------
# 4. PREDICTION INFERENCE ENGINE
# ---------------------------------------------------------
# Prepare sequence data (6 time-steps using current inputs)
# Features order: [Heart_Rate, SpO2, Respiratory_Rate, FiO2, PaO2_Dummy]
raw_input = np.array([[hr, spo2, rr, fio2, 0.0]])

# Transform using fitted MinMaxScaler
dummy_df = pd.DataFrame(raw_input, columns=['Heart_Rate', 'SpO2', 'Respiratory_Rate', 'FiO2', 'PaO2_Target'])
scaled_input = scaler.transform(dummy_df)[:, :-1] # Take only feature columns

# Tile input to match 6 time-step requirement: Shape (1, 6, 4)
sequence_input = np.tile(scaled_input, (1, 6, 1))

# Run BiLSTM Inference
scaled_pred = model.predict(sequence_input, verbose=0)[0][0]

# Inverse transform prediction to original PaO2 scale (mmHg)
# Unscale formula: value * (max - min) + min
pao2_min = scaler.data_min_[-1]
pao2_max = scaler.data_max_[-1]
predicted_pao2 = scaled_pred * (pao2_max - pao2_min) + pao2_min

# ---------------------------------------------------------
# 5. DASHBOARD LAYOUT & METRICS
# ---------------------------------------------------------
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("🎯 Predicted Parameter")
    st.metric(
        label="Predicted PaO2 (Partial Pressure of O2)",
        value=f"{predicted_pao2:.2f} mmHg",
        delta=f"{'Normal' if predicted_pao2 >= 80 else 'Warning: Low PaO2'}"
    )
    
    # Clinical Status Alert
    if predicted_pao2 < 60:
        st.error("🚨 **CRITICAL ALERT:** Severe Hypoxemia Detected! Immediate Oxygen Therapy Adjustment Required.")
    elif 60 <= predicted_pao2 < 80:
        st.warning("⚠️ **WARNING:** Mild-to-Moderate Hypoxemia. Monitor PaO2/FiO2 Ratio.")
    else:
        st.success("🟢 **NORMAL:** Oxygenation level is adequate.")

with col2:
    st.subheader("📊 Oxygenation Gauge Indicator")
    
    # Gauge Chart
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = predicted_pao2,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "PaO2 Level (mmHg)"},
        gauge = {
            'axis': {'range': [0, 300]},
            'bar': {'color': "darkblue"},
            'steps' : [
                {'range': [0, 60], 'color': "red"},
                {'range': [60, 80], 'color': "orange"},
                {'range': [80, 300], 'color': "green"}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': predicted_pao2
            }
        }
    ))
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.info("💡 **FYP System Info:** Model powered by **BiLSTM-Attention Neural Network** trained on time-series ICU data.")
