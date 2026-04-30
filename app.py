"""
AI-Driven ICU Blood Gas Assistant (Full Integrated System)
Version: Professional FYP Standard
Author: Lutfi
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
# 1. PAGE SETUP & THEME
# ─────────────────────────────────────────────
st.set_page_config(page_title="ICU AI Research Dashboard", page_icon="🔬", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #F8FAFC; }
    .stMetric { background-color: #ffffff; padding: 20px; border-radius: 12px; border: 1px solid #E2E8F0; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    .report-card { background-color: #ffffff; padding: 25px; border-radius: 15px; border-top: 5px solid #0E9E8E; margin-bottom: 20px; }
    </style>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# 2. AUTOMATED DATA CLEANING ENGINE
# ─────────────────────────────────────────────
def robust_preprocessing(df_input):
    df_temp = df_input.copy()
    
    # Smart Detection for Blood Pressure (Prevents KeyError)
    bp_col = next((c for c in df_temp.columns if 'blood' in c.lower() or 'bp' in c.lower()), None)
    if bp_col:
        new_bp = df_temp[bp_col].astype(str).str.split('/', expand=True)
        df_temp['Systolic'] = pd.to_numeric(new_bp[0], errors='coerce').fillna(120)
        df_temp['Diastolic'] = pd.to_numeric(new_bp[1], errors='coerce').fillna(80)
    else:
        df_temp['Systolic'], df_temp['Diastolic'] = 120, 80

    # Smart Detection for Gender & Ventilation
    gen_col = next((c for c in df_temp.columns if 'gender' in c.lower()), None)
    if gen_col:
        df_temp['Gender'] = df_temp[gen_col].map(lambda x: 1 if str(x).lower() == 'male' else 0)
    
    vent_col = next((c for c in df_temp.columns if 'vent' in c.lower() or 'mech' in c.lower()), None)
    if vent_col:
        df_temp['Mechanical_Ventilation'] = df_temp[vent_col].map(lambda x: 1 if str(x).lower() == 'yes' else 0)
        
    return df_temp

# ─────────────────────────────────────────────
# 3. SIDEBAR CONTROLS
# ─────────────────────────────────────────────
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3774/3774299.png", width=70)
    st.title("System Controls")
    uploaded_file = st.file_uploader("Muat Naik Dataset Pesakit (CSV)", type=["csv"])
    
    if uploaded_file:
        if uploaded_file.size < 1024:
            st.error("❌ Fail < 1 KB dikesan. Sila muat naik dataset yang lengkap.")
            uploaded_file = None
        else:
            st.success(f"✅ Fail sedia diproses ({uploaded_file.size/1024:.1f} KB)")

    st.markdown("---")
    st.caption("Fasa Automation & Digitalization: Membantu pakar perubatan meramal gas darah secara saintifik.")

# ─────────────────────────────────────────────
# 4. DEEP LEARNING & STATISTICAL OUTPUT
# ─────────────────────────────────────────────
if uploaded_file:
    raw_df = pd.read_csv(uploaded_file)
    df_proc = robust_preprocessing(raw_df)

    # Parameters
    features = ['Age', 'Gender', 'Severity_Score', 'Heart_Rate', 'Temperature', 
                'WBC_Count', 'Lactate_Level', 'Mechanical_Ventilation', 'Systolic', 'Diastolic']
    targets = ['pH', 'PaCO2', 'PaO2', 'O2_Saturation', 'HCO3']

    # Only use features that exist in the dataframe
    available_features = [f for f in features if f in df_proc.columns]

    @st.cache_resource
    def train_deep_learning_system(data_json):
        data = pd.read_json(io.StringIO(data_json))
        models, scalers, results_meta = {}, {}, {}
        for t in targets:
            training_set = data.dropna(subset=available_features + [t])
            if len(training_set) < 5:
                continue
            X = training_set[available_features]
            y = training_set[t]
            scaler = StandardScaler()
            X_sc = scaler.fit_transform(X)
            model = MLPRegressor(hidden_layer_sizes=(12, 12), activation='relu', 
                                 solver='adam', max_iter=3000, random_state=42)
            model.fit(X_sc, y)
            y_pred = model.predict(X_sc)
            r2 = r2_score(y, y_pred)
            models[t] = model
            scalers[t] = scaler
            results_meta[t] = {'r2': r2, 'actual': y.values.tolist(), 'predicted': y_pred.tolist()}
        return models, scalers, results_meta

    with st.spinner("AI sedang menganalisis corak biologi..."):
        models, scalers, meta = train_deep_learning_system(df_proc.to_json())

    # ─────────────────────────────────────────────
    # 5. PREDICTION & OUTPUT ASSEMBLY
    # ─────────────────────────────────────────────
    X_full = df_proc[available_features].fillna(df_proc[available_features].mean())
    output_df = raw_df.copy()

    for t in targets:
        if t in models:
            X_scaled = scalers[t].transform(X_full)
            preds = models[t].predict(X_scaled)
            if t == 'O2_Saturation': preds = np.clip(preds, 0, 100)
            elif t == 'pH': preds = np.clip(preds, 6.5, 8.0)
            else: preds = np.clip(preds, 0, None)
            output_df[f'AI_{t}'] = preds

    # ─────────────────────────────────────────────
    # 6. DASHBOARD INTERFACE
    # ─────────────────────────────────────────────
    st.title("🩺 ICU Blood Gas AI Assistant")
    
    t1, t2, t3 = st.tabs(["🏥 Dashboard Doktor", "📊 Analisis Regresi", "📂 Rekod Penuh"])

    with t1:
        st.subheader("Semakan Keputusan Pesakit")
        id_col = next((c for c in output_df.columns if 'id' in c.lower()), None)
        id_list = output_df[id_col].tolist() if id_col else output_df.index.tolist()
        sel_id = st.selectbox("Pilih ID Pesakit:", id_list)

        if id_col:
            p_data = output_df[output_df[id_col] == sel_id].iloc[0]
        else:
            p_data = output_df.iloc[sel_id]

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Predicted pH",    f"{p_data.get('AI_pH', 0):.3f}")
        c2.metric("PaCO2 (mmHg)",    f"{p_data.get('AI_PaCO2', 0):.1f}")
        c3.metric("PaO2 (mmHg)",     f"{p_data.get('AI_PaO2', 0):.1f}")
        c4.metric("O2 Saturation",   f"{p_data.get('AI_O2_Saturation', 0):.1f}%")
        c5.metric("HCO3 (mmol/L)",   f"{p_data.get('AI_HCO3', 0):.1f}")

        ph_val = p_data.get('AI_pH', 7.4)
        if 7.35 <= ph_val <= 7.45:
            status, color = "NORMAL ✔", "#0E9E8E"
        elif ph_val < 7.2 or ph_val > 7.6:
            status, color = "KRITIKAL ✖", "#E84C4C"
        else:
            status, color = "AMARAN ⚠", "#F4A623"

        st.markdown(
            f"<div style='background:{color}; color:white; padding:15px; border-radius:10px;"
            f"text-align:center; font-weight:bold; font-size:1.1rem; margin-top:16px;'>"
            f"KLINIKAL STATUS: {status}</div>",
            unsafe_allow_html=True
        )

        # Vitals summary for selected patient
        st.markdown("---")
        st.markdown("#### 📟 Data Klinikal Pesakit")
        vital_display = {k: v for k, v in p_data.items() if k not in [f'AI_{t}' for t in targets]}
        st.dataframe(pd.DataFrame([vital_display]).round(2), use_container_width=True, hide_index=True)

    with t2:
        st.subheader("Pengesahan Model (Regression Line Analysis)")
        st.write("Graf ini membuktikan ketepatan AI dalam meramal parameter berbanding data rujukan.")

        if meta:
            sel_t = st.selectbox("Pilih Parameter untuk Lihat Regresi:", list(meta.keys()))
            m = meta[sel_t]
            actual    = np.array(m['actual'])
            predicted = np.array(m['predicted'])

            fig, ax = plt.subplots(figsize=(10, 5))
            sns.regplot(x=actual, y=predicted, ax=ax,
                        scatter_kws={'alpha': 0.6, 'color': '#0E9E8E'},
                        line_kws={'color': '#E84C4C', 'label': f"R² = {m['r2']:.4f}"})
            ax.set_xlabel(f"Actual {sel_t}")
            ax.set_ylabel(f"Predicted {sel_t}")
            ax.set_title(f"Regression Analysis: Actual vs Predicted — {sel_t}")
            ax.legend()
            st.pyplot(fig)
            plt.close(fig)

            col_r2, col_n = st.columns(2)
            col_r2.metric("R² Score", f"{m['r2']:.4f}")
            col_n.metric("Training Samples", len(actual))
            st.success(f"Model menunjukkan korelasi R² sebanyak {m['r2']:.4f} untuk {sel_t}.")
        else:
            st.warning("Tiada model berjaya dilatih. Pastikan data ground truth mencukupi (≥5 baris per target).")

    with t3:
        st.subheader("Data Lengkap & Eksport")
        st.dataframe(output_df, use_container_width=True)
        csv = output_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Muat Turun Fail Lengkap (CSV)", csv, "icu_ai_results.csv", "text/csv")

else:
    st.info("Sila muat naik fail CSV pesakit di sidebar untuk memulakan pengiraan AI.")
    st.markdown("""
    **Format Fail yang Disyorkan:**
    - Kolum Vital Signs mestilah lengkap.
    - Kolum Blood Gas (pH, PaO2, dll.) boleh kosong (NaN) untuk AI proses.
    """)
