"""
AI-Driven ICU Blood Gas Assistant (Deep Learning Edition)
System: Digitalization & Automation
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import io

# ─────────────────────────────────────────────
# 1. KONFIGURASI INTERFACE
# ─────────────────────────────────────────────
st.set_page_config(page_title="ICU AI Assistant", page_icon="🩺", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #F0F2F6; }
    .stMetric { background-color: #ffffff; padding: 20px; border-radius: 15px; border-left: 5px solid #0E9E8E; }
    </style>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# 2. ENGINE PEMPROSESAN (DATA CLEANING)
# ─────────────────────────────────────────────
def clean_data(df_input):
    df_temp = df_input.copy()
    
    # Deteksi Blood Pressure secara dinamik
    bp_col = next((c for c in df_temp.columns if 'blood' in c.lower() or 'bp' in c.lower()), None)
    if bp_col:
        new_bp = df_temp[bp_col].astype(str).str.split('/', expand=True)
        df_temp['Systolic'] = pd.to_numeric(new_bp[0], errors='coerce').fillna(120)
        df_temp['Diastolic'] = pd.to_numeric(new_bp[1], errors='coerce').fillna(80)
    
    # Mapping Kategori ke Digital (0/1)
    gen_col = next((c for c in df_temp.columns if 'gender' in c.lower()), None)
    if gen_col:
        df_temp['Gender'] = df_temp[gen_col].map(lambda x: 1 if str(x).lower() == 'male' else 0)
    
    vent_col = next((c for c in df_temp.columns if 'vent' in c.lower()), None)
    if vent_col:
        df_temp['Mechanical_Ventilation'] = df_temp[vent_col].map(lambda x: 1 if str(x).lower() == 'yes' else 0)
        
    return df_temp

# ─────────────────────────────────────────────
# 3. SIDEBAR: KAWALAN FAIL
# ─────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Control Panel")
    uploaded_file = st.file_uploader("Upload CSV (Min 1 KB)", type=["csv"])
    
    if uploaded_file:
        if uploaded_file.size < 1024:
            st.error("❌ Fail terlalu kecil. Sila upload data pesakit yang lengkap.")
            uploaded_file = None
        else:
            st.success("✅ Fail sedia untuk diproses.")

# ─────────────────────────────────────────────
# 4. DEEP LEARNING LOGIC (ANN)
# ─────────────────────────────────────────────
if uploaded_file:
    raw_df = pd.read_csv(uploaded_file)
    df = clean_data(raw_df)

    # 10 Parameter Input
    features = ['Age', 'Gender', 'Severity_Score', 'Heart_Rate', 'Temperature', 
                'WBC_Count', 'Lactate_Level', 'Mechanical_Ventilation', 'Systolic', 'Diastolic']
    # 5 Parameter Output
    targets = ['pH', 'PaCO2', 'PaO2', 'O2_Saturation', 'HCO3']

    @st.cache_resource
    def train_deep_learning(data):
        models, scalers = {}, {}
        for t in targets:
            # Cari baris yang ada data (Ground Truth) untuk fasa pembelajaran
            train_subset = data.dropna(subset=features + [t])
            
            if len(train_subset) < 2:
                st.error(f"❌ Kolum {t} tiada data rujukan langsung! AI tidak boleh bina formula.")
                return None, None

            X = train_subset[features]
            y = train_subset[t]

            scaler = StandardScaler()
            X_sc = scaler.fit_transform(X)
            
            # Arsitektur Deep Learning (Hidden Layers 12, 12)
            model = MLPRegressor(hidden_layer_sizes=(12, 12), activation='relu', 
                                 solver='adam', max_iter=3000, random_state=42)
            model.fit(X_sc, y)
            
            models[t] = model
            scalers[t] = scaler
        return models, scalers

    models, scalers = train_deep_learning(df)

    if models:
        # PREDISKI: Menggunakan Weights yang telah dipelajari
        X_all = df[features].fillna(df[features].mean())
        final_df = raw_df.copy()

        for t in targets:
            X_scaled = scalers[t].transform(X_all)
            preds = models[t].predict(X_scaled)
            
            # Had Fisiologi (Bukan anggaran, tapi kekangan klinikal)
            if t == 'O2_Saturation': preds = np.clip(preds, 0, 100)
            elif t == 'pH': preds = np.clip(preds, 6.5, 8.0)
            else: preds = np.clip(preds, 0, None)
            
            final_df[f'AI_{t}'] = preds

        # ─────────────────────────────────────────────
        # 5. DASHBOARD DOKTOR
        # ─────────────────────────────────────────────
        st.title("🩺 AI Personal Assistant Dashboard")
        
        tab1, tab2 = st.tabs(["Individual Analysis", "Full Dataset"])

        with tab1:
            id_col = next((c for c in final_df.columns if 'id' in c.lower()), final_df.index.name)
            p_id = st.selectbox("Select Patient ID", final_df[id_col] if id_col else final_df.index)
            
            p_data = final_df[final_df[id_col if id_col else final_df.index] == p_id].iloc[0]
            
            # Paparan Metrik
            cols = st.columns(5)
            metrics = [('pH', 'AI_pH', '.3f'), ('PaCO2', 'AI_PaCO2', '.1f'), 
                       ('PaO2', 'AI_PaO2', '.1f'), ('O2 Sat', 'AI_O2_Saturation', '.1f'), 
                       ('HCO3', 'AI_HCO3', '.1f')]
            
            for i, (label, col, fmt) in enumerate(metrics):
                cols[i].metric(label, f"{p_data[col]:{fmt}}")

            # Status Klinikal
            ph = p_data['AI_pH']
            status = "NORMAL" if 7.35 <= ph <= 7.45 else "CRITICAL"
            color = "#0E9E8E" if status == "NORMAL" else "#E84C4C"
            st.markdown(f"<div style='background:{color}; color:white; padding:10px; border-radius:10px; text-align:center; font-weight:bold;'>STATUS: {status}</div>", unsafe_allow_html=True)

        with tab2:
            st.subheader("Predicted Results Table")
            st.dataframe(final_df.style.background_gradient(subset=['AI_pH'], cmap='coolwarm'))
            
            csv = final_df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Export CSV", csv, "ai_blood_gas_results.csv", "text/csv")