"""
AI-Driven ICU Blood Gas Assistant (Full Professional Version)
Author: Lutfi (FYP Student)
System: Digitalization & Automation
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import os
import io

# ─────────────────────────────────────────────
# 1. KONFIGURASI HALAMAN
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="ICU AI Personal Assistant",
    page_icon="🩺",
    layout="wide"
)

# Tema Warna & CSS
st.markdown("""
    <style>
    .main { background-color: #F8FAFC; }
    .stMetric { background-color: #ffffff; padding: 20px; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); border: 1px solid #E2E8F0; }
    .status-card { padding: 15px; border-radius: 10px; text-align: center; font-weight: bold; font-size: 1.2rem; margin-bottom: 20px; }
    .sidebar-text { font-size: 0.9rem; color: #64748B; }
    </style>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# 2. LOGIK PEMPROSESAN DATA (ROBUST)
# ─────────────────────────────────────────────

def clean_and_process(df_input):
    df_temp = df_input.copy()
    
    # Deteksi Kolom Blood Pressure secara automatik
    bp_col = next((c for c in df_temp.columns if 'blood' in c.lower() or 'bp' in c.lower()), None)
    
    if bp_col:
        try:
            # Pecahkan format '120/80'
            new_bp = df_temp[bp_col].astype(str).str.split('/', expand=True)
            df_temp['Systolic'] = pd.to_numeric(new_bp[0], errors='coerce').fillna(120)
            df_temp['Diastolic'] = pd.to_numeric(new_bp[1], errors='coerce').fillna(80)
        except:
            df_temp['Systolic'], df_temp['Diastolic'] = 120, 80
    else:
        df_temp['Systolic'], df_temp['Diastolic'] = 120, 80

    # Mapping Gender & Ventilation (Case Insensitive)
    gen_col = next((c for c in df_temp.columns if 'gender' in c.lower()), None)
    if gen_col:
        df_temp['Gender'] = df_temp[gen_col].map(lambda x: 1 if str(x).lower() == 'male' else 0)
    
    vent_col = next((c for c in df_temp.columns if 'vent' in c.lower()), None)
    if vent_col:
        df_temp['Mechanical_Ventilation'] = df_temp[vent_col].map(lambda x: 1 if str(x).lower() == 'yes' else 0)

    return df_temp

# ─────────────────────────────────────────────
# 3. SIDEBAR: FILE UPLOADER & VALIDATION
# ─────────────────────────────────────────────

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3774/3774299.png", width=80)
    st.title("Admin Panel")
    
    uploaded_file = st.file_uploader("Muat Naik Fail CSV Pesakit", type=["csv"])
    
    if uploaded_file is not None:
        # VALIDASI SAIZ FAIL (MIN 1 KB)
        file_size = uploaded_file.size / 1024
        if file_size < 1.0:
            st.error(f"❌ Ralat: Fail terlalu kecil ({file_size:.2f} KB). Sila muat naik fail yang sah.")
            uploaded_file = None
        else:
            st.success(f"✅ Fail diterima ({file_size:.1f} KB)")
    
    st.markdown("---")
    st.markdown("**Parameter Diperlukan:**")
    st.markdown("- Vital Signs (HR, Temp, BP)\n- Lactate Level\n- Severity Score\n- WBC Count")

# ─────────────────────────────────────────────
# 4. ENGINE AI & DASHBOARD
# ─────────────────────────────────────────────

if uploaded_file is not None:
    # Membaca Data
    raw_df = pd.read_csv(uploaded_file)
    df = clean_and_process(raw_df)

    # Definisi Features & Targets
    features = ['Age', 'Gender', 'Severity_Score', 'Heart_Rate', 'Temperature', 
                'WBC_Count', 'Lactate_Level', 'Mechanical_Ventilation', 'Systolic', 'Diastolic']
    targets = ['pH', 'PaCO2', 'PaO2', 'O2_Saturation', 'HCO3']

    # Fasa Training/Inference (Deep Learning)
    @st.cache_resource
    def run_ai_prediction(data):
        models, scalers = {}, {}
        for t in targets:
            # Menggunakan data sedia ada sebagai rujukan atau baseline statistik
            train_data = data.dropna(subset=features + [t])
            if len(train_data) < 5:
                # Baseline jika kolum kosong sepenuhnya
                X = data[features].fillna(data[features].mean())
                if t == 'pH': y = 7.4 - (data['Lactate_Level'].fillna(2) * 0.02)
                elif t == 'O2_Saturation': y = pd.Series([98.0] * len(data))
                else: y = pd.Series([24.0] * len(data))
            else:
                X = train_data[features]
                y = train_data[t]

            sc = StandardScaler()
            X_sc = sc.fit_transform(X)
            model = MLPRegressor(hidden_layer_sizes=(12, 12), max_iter=2000, random_state=42)
            model.fit(X_sc, y)
            models[t], scalers[t] = model, sc
        return models, scalers

    with st.spinner("AI sedang mengira parameter gas darah..."):
        models, scalers = run_ai_prediction(df)

    # Melakukan Prediksi Penuh
    X_all = df[features].fillna(df[features].mean())
    final_df = raw_df.copy()

    for t in targets:
        X_scaled = scalers[t].transform(X_all)
        preds = models[t].predict(X_scaled)
        
        # Biological Clipping
        if t == 'O2_Saturation': preds = np.clip(preds, 0, 100.0)
        elif t == 'pH': preds = np.clip(preds, 6.5, 8.0)
        else: preds = np.clip(preds, 0, None)
        
        final_df[f'AI_{t}'] = preds

    # ─────────────────────────────────────────────
    # DISPLAY TAB
    # ─────────────────────────────────────────────
    tab1, tab2 = st.tabs(["📋 Dashboard Doktor", "📂 Rekod Penuh & Eksport"])

    with tab1:
        st.subheader("Semakan Pantas Pesakit")
        
        # Dropdown Patient ID
        id_col = next((c for c in final_df.columns if 'id' in c.lower()), None)
        patient_list = final_df[id_col].tolist() if id_col else final_df.index.tolist()
        
        selected_id = st.selectbox("Pilih ID Pesakit untuk Analisis:", patient_list)
        
        # Tarik data pesakit terpilih
        p_row = final_df[final_df[id_col if id_col else final_df.index] == selected_id].iloc[0]

        # Ringkasan Metrik
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Predicted pH", f"{p_row['AI_pH']:.3f}")
        col2.metric("PaCO2 (mmHg)", f"{p_row['AI_PaCO2']:.1f}")
        col3.metric("PaO2 (mmHg)", f"{p_row['AI_PaO2']:.1f}")
        col4.metric("O2 Saturation", f"{p_row['AI_O2_Saturation']:.1f}%")
        col5.metric("HCO3 (mmol/L)", f"{p_row['AI_HCO3']:.1f}")

        # Status Klinikal
        ph_val = p_row['AI_pH']
        if 7.35 <= ph_val <= 7.45:
            st.markdown("<div class='status-card' style='background-color:#D1FAE5; color:#065F46;'>STATUS: NORMAL</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='status-card' style='background-color:#FEE2E2; color:#991B1B;'>STATUS: CRITICAL (Acidosis/Alkalosis)</div>", unsafe_allow_html=True)

    with tab2:
        st.subheader("Data Prediksi Keseluruhan")
        st.dataframe(final_df, use_container_width=True)
        
        # Fungsi Eksport
        csv_data = final_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Muat Turun Keputusan (CSV)",
            data=csv_data,
            file_name=f"ai_predictions_{selected_id}.csv",
            mime="text/csv"
        )

else:
    # Paparan Mula
    st.info("Sila muat naik fail CSV pesakit (min 1 KB) di panel kiri untuk memulakan sistem AI.")
    st.image("https://via.placeholder.com/1000x400.png?text=Waiting+for+Data+Input+...", use_column_width=True)