"""
AI-Driven ICU Blood Gas Assistant
Streamlit Application — FYP Fecal Peritonitis
Author: Lutfi
Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="ICU Blood Gas Assistant",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS — Teal / Navy / White palette
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=Space+Mono:wght@400;700&display=swap');
:root {
    --navy:#0A1628; --navy2:#112240; --teal:#0E9E8E; --teal2:#12C2B0;
    --white:#F4F7FB; --muted:#8A9BB8; --alert:#F4A623; --crit:#E84C4C;
    --card:#132035; --border:rgba(14,158,142,0.25);
}
html,body,[class*="css"]{font-family:'DM Sans',sans-serif;background-color:var(--navy)!important;color:var(--white)!important;}
section[data-testid="stSidebar"]{background:var(--navy2)!important;border-right:1px solid var(--border);}
section[data-testid="stSidebar"] *{color:var(--white)!important;}
h1{font-family:'Space Mono',monospace;color:var(--teal2)!important;letter-spacing:-1px;}
h2,h3{color:var(--teal)!important;}
h4,h5,h6{color:var(--muted)!important;font-weight:500;}
[data-testid="metric-container"]{background:var(--card)!important;border:1px solid var(--border)!important;border-radius:12px!important;padding:18px 22px!important;}
[data-testid="stMetricLabel"]{color:var(--muted)!important;font-size:0.75rem;text-transform:uppercase;letter-spacing:1px;}
[data-testid="stMetricValue"]{color:var(--teal2)!important;font-family:'Space Mono',monospace;font-size:1.9rem!important;}
[data-testid="stMetricDelta"]{color:var(--muted)!important;}
.stButton>button{background:linear-gradient(135deg,var(--teal),#0A7A6D)!important;color:#fff!important;border:none!important;border-radius:8px!important;padding:0.55rem 1.4rem!important;font-weight:600!important;}
.stDownloadButton>button{background:var(--navy2)!important;color:var(--teal2)!important;border:1px solid var(--teal)!important;border-radius:8px!important;font-weight:600!important;}
.stSelectbox>div>div,.stTextInput>div>input{background:var(--card)!important;color:var(--white)!important;border:1px solid var(--border)!important;border-radius:8px!important;}
.stDataFrame{border:1px solid var(--border);border-radius:10px;overflow:hidden;}
hr{border-color:var(--border)!important;}
.badge{display:inline-block;padding:5px 14px;border-radius:20px;font-weight:700;font-size:0.85rem;letter-spacing:0.8px;font-family:'Space Mono',monospace;}
.badge-normal{background:rgba(14,158,142,0.18);color:#0E9E8E;border:1px solid #0E9E8E;}
.badge-alert{background:rgba(244,166,35,0.18);color:#F4A623;border:1px solid #F4A623;}
.badge-critical{background:rgba(232,76,76,0.18);color:#E84C4C;border:1px solid #E84C4C;}
.section-card{background:var(--card);border:1px solid var(--border);border-radius:14px;padding:24px;margin-bottom:20px;}
.top-banner{background:linear-gradient(135deg,#0E3D35 0%,#0A1628 60%);border:1px solid var(--border);border-radius:14px;padding:28px 32px;margin-bottom:24px;}
.top-banner h1{margin:0 0 6px 0;font-size:2rem;}
.top-banner p{color:var(--muted);margin:0;font-size:0.95rem;}
.formula-box{background:#0A1628;border:1px solid var(--border);border-radius:10px;padding:16px 20px;font-family:'Space Mono',monospace;font-size:0.78rem;color:#12C2B0;margin:8px 0;}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
TARGETS  = ["pH", "PaCO2", "PaO2", "O2_Saturation", "HCO3"]
FEATURES = ["Age","Gender","Severity_Score","Heart_Rate","Temperature",
            "WBC_Count","Lactate_Level","Mechanical_Ventilation","Systolic","Diastolic"]
NORMAL_RANGES = {
    "pH":            (7.35, 7.45),
    "PaCO2":         (35,   45),
    "PaO2":          (75,   100),
    "O2_Saturation": (95,   100),
    "HCO3":          (22,   26),
}
UNITS = {"pH":"","PaCO2":"mmHg","PaO2":"mmHg","O2_Saturation":"%","HCO3":"mEq/L"}
MIN_FILE_BYTES = 1229  # 1.2 KB

# ─────────────────────────────────────────────
# PHYSIOLOGICAL BLOOD GAS FORMULAS
# ─────────────────────────────────────────────
# Formula 1: Henderson–Hasselbalch  →  pH
#   pH = 6.1 + log10(HCO3 / (0.0307 × PaCO2))
#
# Formula 2: PaCO2 from pH + HCO3
#   PaCO2 = HCO3 / (0.0307 × 10^(pH − 6.1))
#
# Formula 3: HCO3 from pH + PaCO2
#   HCO3 = 0.0307 × PaCO2 × 10^(pH − 6.1)
#
# Formula 4: PaO2 — Alveolar Gas Equation (assumed FiO2 = 0.21 room air)
#   PAO2 = FiO2 × (760−47) − PaCO2 / 0.8
#   PaO2 ≈ PAO2 − A-a gradient (est. 10 mmHg young, rises with age)
#
# Formula 5: O2 Saturation — Hill / ODC equation
#   SaO2 = PaO2^2.7 / (PaO2^2.7 + 26.8^2.7) × 100
#
# These formulas are used as:
#   (a) Fallback when ANN training data < 5 rows
#   (b) Cross-check layer — blend 70% ANN + 30% formula
# ─────────────────────────────────────────────

def formula_pH(hco3, paco2):
    """Henderson–Hasselbalch"""
    try:
        val = 6.1 + np.log10(hco3 / (0.0307 * paco2))
        return np.clip(val, 6.5, 8.0)
    except Exception:
        return np.full_like(hco3, 7.40)

def formula_HCO3(pH, paco2):
    """HCO3 dari pH + PaCO2"""
    val = 0.0307 * paco2 * (10 ** (pH - 6.1))
    return np.clip(val, 0, None)

def formula_PaCO2(pH, hco3):
    """PaCO2 dari pH + HCO3"""
    denom = 0.0307 * (10 ** (pH - 6.1))
    val = hco3 / np.where(denom == 0, 1e-9, denom)
    return np.clip(val, 0, None)

def formula_PaO2(paco2, age, fio2=0.21):
    """Alveolar Gas Equation → PaO2 anggaran"""
    pa_o2 = fio2 * (760 - 47) - (paco2 / 0.8)
    aa_gradient = 2.5 + 0.21 * np.clip(age, 20, 90)   # A-a gradient meningkat dgn umur
    pa_o2_art = pa_o2 - aa_gradient
    return np.clip(pa_o2_art, 0, None)

def formula_O2Sat(pao2):
    """Hill / ODC equation — SaO2 dari PaO2"""
    n, p50 = 2.7, 26.8
    sat = (pao2**n / (pao2**n + p50**n)) * 100
    return np.clip(sat, 0, 100)

def apply_formula_fallback(df, target, available_features):
    """
    Guna formula fisiologi sebagai fallback jika data latihan < 5.
    Returns array of predicted values.
    """
    n = len(df)
    age     = df["Age"].fillna(50).values        if "Age"     in df.columns else np.full(n, 50)
    paco2   = df["PaCO2"].fillna(40).values      if "PaCO2"   in df.columns else np.full(n, 40.0)
    hco3    = df["HCO3"].fillna(24).values       if "HCO3"    in df.columns else np.full(n, 24.0)
    ph      = df["pH"].fillna(7.4).values        if "pH"      in df.columns else np.full(n, 7.40)
    pao2    = df["PaO2"].fillna(90).values       if "PaO2"    in df.columns else np.full(n, 90.0)
    lactate = df["Lactate_Level"].fillna(1).values if "Lactate_Level" in df.columns else np.ones(n)

    if target == "pH":
        # Henderson-Hasselbalch + Lactate koreksi
        base = formula_pH(hco3, paco2)
        corrected = base - (lactate * 0.02)   # Lactate acidosis correction
        return np.clip(corrected, 6.5, 8.0)

    elif target == "HCO3":
        base = formula_HCO3(ph, paco2)
        corrected = base - (lactate * 0.5)    # Lactate compensation
        return np.clip(corrected, 0, None)

    elif target == "PaCO2":
        return formula_PaCO2(ph, hco3)

    elif target == "PaO2":
        return formula_PaO2(paco2, age)

    elif target == "O2_Saturation":
        return formula_O2Sat(pao2)

    return np.full(n, np.nan)

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def dark_fig(w=10, h=5):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor("#132035")
    ax.set_facecolor("#0A1628")
    for sp in ax.spines.values(): sp.set_edgecolor("#1E3A5F")
    ax.tick_params(colors="#8A9BB8")
    ax.xaxis.label.set_color("#8A9BB8")
    ax.yaxis.label.set_color("#8A9BB8")
    ax.title.set_color("#0E9E8E")
    return fig, ax

def ph_status(ph_val):
    if pd.isna(ph_val): return "unknown"
    if ph_val < 7.2 or ph_val > 7.6: return "critical"
    if ph_val < 7.35 or ph_val > 7.45: return "alert"
    return "normal"

def status_badge_html(status):
    icons = {"normal":"✔","alert":"⚠","critical":"✖","unknown":"?"}
    return f'<span class="badge badge-{status}">{icons.get(status,"?")} {status.upper()}</span>'

# ─────────────────────────────────────────────
# PRE-PROCESSING
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_and_process(raw_bytes: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(raw_bytes))
    bp_col = next((c for c in df.columns if 'blood' in c.lower() or 'bp' in c.lower()), None)
    if bp_col:
        bp = df[bp_col].astype(str).str.split("/", expand=True)
        df["Systolic"]  = pd.to_numeric(bp[0], errors="coerce")
        df["Diastolic"] = pd.to_numeric(bp[1], errors="coerce")
        if bp_col not in ("Systolic","Diastolic"):
            df.drop(columns=[bp_col], inplace=True)
    gen_col = next((c for c in df.columns if 'gender' in c.lower()), None)
    if gen_col:
        df["Gender"] = df[gen_col].map(lambda x: 1 if str(x).strip().lower() in ("male","m") else 0)
    vent_col = next((c for c in df.columns if 'vent' in c.lower() or 'mech' in c.lower()), None)
    if vent_col:
        df["Mechanical_Ventilation"] = df[vent_col].map(lambda x: 1 if str(x).strip().lower() == "yes" else 0)
    for col in df.columns:
        if col != "Patient_ID":
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

# ─────────────────────────────────────────────
# DEEP LEARNING INFERENCE ENGINE
# ANN (12,12) + Formula Cross-Check
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def train_and_predict(raw_bytes: bytes):
    df = load_and_process(raw_bytes)
    available_features = [f for f in FEATURES if f in df.columns]
    df_complete = df.dropna(subset=available_features)
    importances, log_messages, r2_scores = {}, [], {}

    for target in TARGETS:
        if target not in df.columns:
            df[target] = np.nan

        df_train = df_complete.dropna(subset=[target])
        used_formula = False

        if len(df_train) < 5:
            # ── FALLBACK: Physiological Formula ──
            log_messages.append(f"⚠️ {target}: Data latihan terhad ({len(df_train)} baris) → Formula Fisiologi digunakan.")
            formula_preds = apply_formula_fallback(df, target, available_features)
            nan_mask = df[target].isna()
            if nan_mask.any():
                df.loc[nan_mask, target] = formula_preds[nan_mask]
            importances[target] = {f: 1/len(available_features) for f in available_features}
            r2_scores[target] = None
            used_formula = True
        else:
            log_messages.append(f"✅ {target}: {len(df_train)} rekod ground truth → ANN (12,12) dilatih.")

            X_train = df_train[available_features]
            y_train = df_train[target]

            scaler = StandardScaler()
            X_sc   = scaler.fit_transform(X_train)

            # ── ANN: 2 Hidden Layers (12,12) ──
            model = MLPRegressor(
                hidden_layer_sizes=(12, 12),
                activation="relu",
                solver="adam",
                max_iter=5000,
                random_state=42,
            )
            model.fit(X_sc, y_train)

            # R² score
            y_pred_train = model.predict(X_sc)
            ss_res = np.sum((y_train.values - y_pred_train)**2)
            ss_tot = np.sum((y_train.values - y_train.mean())**2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
            r2_scores[target] = round(r2, 4)

            # Predict NaN rows
            nan_mask = df[target].isna()
            if nan_mask.any():
                X_pred = df.loc[nan_mask, available_features].fillna(df[available_features].mean())
                X_pred_sc = scaler.transform(X_pred)
                ann_preds = model.predict(X_pred_sc)

                # ── Formula Cross-Check: 70% ANN + 30% Formula ──
                formula_preds = apply_formula_fallback(
                    df.loc[nan_mask].reset_index(drop=True), target, available_features
                )
                blended = 0.70 * ann_preds + 0.30 * formula_preds

                # Biological capping
                if target == "pH":            blended = np.clip(blended, 6.5, 8.0)
                elif target == "O2_Saturation": blended = np.clip(blended, 0, 100.0)
                else:                           blended = np.clip(blended, 0, None)

                df.loc[nan_mask, target] = blended

            # Permutation Feature Importance
            X_all = scaler.transform(df[available_features].fillna(df[available_features].mean()))
            y_all = df[target].values
            base_mse = np.mean((model.predict(X_all) - y_all)**2)
            imp = []
            for i in range(X_all.shape[1]):
                Xp = X_all.copy(); np.random.shuffle(Xp[:, i])
                imp.append(max(np.mean((model.predict(Xp) - y_all)**2) - base_mse, 0))
            importances[target] = dict(zip(available_features, imp))

    result_df = df.copy()
    for t in TARGETS:
        result_df[f"AI_{t}"] = df[t]

    return result_df, importances, log_messages, r2_scores

# ─────────────────────────────────────────────
# SAMPLE DATA GENERATOR  (≥ 1.2 KB)
# ─────────────────────────────────────────────
def generate_sample_csv() -> bytes:
    np.random.seed(42)
    n = 60  # 60 pesakit supaya fail > 1.2 KB
    data = {
        "Patient_ID":             [f"FP{2000+i}" for i in range(n)],
        "Age":                    np.random.randint(25, 85, n).astype(float),
        "Gender":                 np.random.choice(["Male","Female"], n),
        "Severity_Score":         np.round(np.random.uniform(5, 28, n), 1),
        "Heart_Rate":             np.random.randint(55, 135, n).astype(float),
        "Temperature":            np.round(np.random.uniform(36.0, 40.0, n), 1),
        "WBC_Count":              np.round(np.random.uniform(4.0, 22.0, n), 1),
        "Lactate_Level":          np.round(np.random.uniform(0.5, 9.0, n), 2),
        "Mechanical_Ventilation": np.random.choice(["Yes","No"], n),
        "Blood_Pressure":         [f"{np.random.randint(88,172)}/{np.random.randint(50,108)}" for _ in range(n)],
        # Blood gas targets — 35% sengaja dikosongkan untuk AI predict
        "pH":            np.where(np.random.rand(n)<0.35, np.nan, np.round(np.random.uniform(7.18,7.58,n),2)),
        "PaCO2":         np.where(np.random.rand(n)<0.35, np.nan, np.round(np.random.uniform(26,60,n),1)),
        "PaO2":          np.where(np.random.rand(n)<0.35, np.nan, np.round(np.random.uniform(52,118,n),1)),
        "O2_Saturation": np.where(np.random.rand(n)<0.35, np.nan, np.round(np.random.uniform(83,100,n),1)),
        "HCO3":          np.where(np.random.rand(n)<0.35, np.nan, np.round(np.random.uniform(13,33,n),1)),
    }
    df  = pd.DataFrame(data)
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    csv_bytes = buf.getvalue().encode()
    # Pastikan ≥ 1.2 KB
    while len(csv_bytes) < MIN_FILE_BYTES:
        df = pd.concat([df, df.sample(5, replace=True)], ignore_index=True)
        buf = io.StringIO(); df.to_csv(buf, index=False)
        csv_bytes = buf.getvalue().encode()
    return csv_bytes

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🩺 ICU Blood Gas\n**AI Assistant**")
    st.markdown("---")
    uploaded = st.file_uploader("Upload Patient CSV", type=["csv"],
        help="CSV mesti mengandungi vital signs. Blood_Pressure format: '120/80'.")

    if uploaded:
        if uploaded.size < MIN_FILE_BYTES:
            st.error(f"❌ Fail terlalu kecil ({uploaded.size} bytes). Minimum {MIN_FILE_BYTES/1024:.1f} KB diperlukan.")
            uploaded = None
        else:
            st.success(f"✅ Fail diterima ({uploaded.size/1024:.1f} KB)")

    st.markdown("---")
    st.markdown("### 🧮 Formula Yang Digunakan")
    st.markdown("""
<div class="formula-box">
pH = 6.1 + log₁₀(HCO₃ / 0.0307×PaCO₂)<br><br>
HCO₃ = 0.0307 × PaCO₂ × 10^(pH−6.1)<br><br>
PaO₂ = FiO₂×713 − PaCO₂/0.8 − A-a<br><br>
SaO₂ = PaO₂²·⁷ / (PaO₂²·⁷ + 26.8²·⁷)
</div>
""", unsafe_allow_html=True)
    st.markdown("---")
    st.caption("AI predictions: 70% ANN + 30% Formula Fisiologi. Decision-support sahaja.")

# ─────────────────────────────────────────────
# BANNER
# ─────────────────────────────────────────────
st.markdown("""
<div class="top-banner">
  <h1>🔬 ICU Blood Gas Assistant</h1>
  <p>AI-Driven Arterial Blood Gas Prediction · ANN (12,12) + Physiological Formulas · Fecal Peritonitis ICU</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# MAIN FLOW
# ─────────────────────────────────────────────
if uploaded is None:
    st.info("👈  Muat naik CSV di sidebar, atau cuba dengan data contoh di bawah.")
    col_demo, _ = st.columns([1, 3])
    with col_demo:
        if st.button("▶  Load Sample Dataset"):
            st.session_state["demo_bytes"] = generate_sample_csv()
    raw_bytes = st.session_state.get("demo_bytes")
else:
    raw_bytes = uploaded.read()
    st.session_state.pop("demo_bytes", None)

if raw_bytes is None:
    st.stop()

with st.spinner("🧠  Melatih ANN & mengira blood gas..."):
    result_df, importances, ai_logs, r2_scores = train_and_predict(raw_bytes)
    processed_df = load_and_process(raw_bytes)

st.success(f"✅  Selesai — **{len(result_df)} pesakit** diproses.")

with st.expander("🔬 AI Engine Log", expanded=False):
    for msg in ai_logs:
        color = "#F4A623" if "⚠️" in msg else "#0E9E8E"
        st.markdown(f"<span style='color:{color};font-family:monospace'>{msg}</span>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab_dash, tab_data, tab_viz, tab_formula, tab_export = st.tabs([
    "🏥 Patient Dashboard", "📋 All Patients", "📊 Analytics",
    "🧮 Formula Validation", "⬇️ Export"
])

# ══════════════════════════════════════════════
# TAB 1 — PATIENT DASHBOARD
# ══════════════════════════════════════════════
with tab_dash:
    st.markdown("### 🔍 Patient Lookup")
    patient_ids = result_df["Patient_ID"].astype(str).tolist() if "Patient_ID" in result_df.columns else result_df.index.astype(str).tolist()
    selected_id = st.selectbox("Pilih ID Pesakit", patient_ids)

    if "Patient_ID" in result_df.columns:
        row = result_df[result_df["Patient_ID"].astype(str) == selected_id].iloc[0]
    else:
        row = result_df.iloc[int(selected_id)]

    st.markdown("---")
    ph_pred = row.get("AI_pH", np.nan)
    status  = ph_status(ph_pred)
    st.markdown(
        f"**Klinikal Status:** {status_badge_html(status)}&nbsp;&nbsp;"
        f"<span style='color:#8A9BB8;font-size:0.85rem'>(berdasarkan pH yang diramal)</span>",
        unsafe_allow_html=True)
    st.markdown("")

    st.markdown("#### 🩸 Keputusan Blood Gas (AI Predicted)")
    cols = st.columns(5)
    metric_icons = {"pH":"⚗","PaCO2":"💨","PaO2":"🫁","O2_Saturation":"💉","HCO3":"⚡"}
    for i, target in enumerate(TARGETS):
        val = row.get(f"AI_{target}", np.nan)
        unit = UNITS[target]
        lo, hi = NORMAL_RANGES[target]
        with cols[i]:
            st.metric(
                label=f"{metric_icons[target]}  {target}",
                value=f"{val:.2f} {unit}".strip() if not pd.isna(val) else "N/A",
                delta=f"Normal {lo}–{hi} {unit}".strip(),
            )

    st.markdown("---")
    st.markdown("#### 📟 Vital Signs Pesakit")
    vitals_cols = [c for c in result_df.columns if c not in TARGETS+[f"AI_{t}" for t in TARGETS]+["Patient_ID"]]
    st.dataframe(pd.DataFrame({c:[row.get(c,"—")] for c in vitals_cols}).round(2), use_container_width=True, hide_index=True)

    st.markdown("#### 📈 Predicted vs Normal Range")
    fig, axes = plt.subplots(1, 5, figsize=(14, 3))
    fig.patch.set_facecolor("#132035")
    for ax, target in zip(axes, TARGETS):
        val = row.get(f"AI_{target}", np.nan)
        lo, hi = NORMAL_RANGES[target]
        color = "#0E9E8E" if lo <= val <= hi else ("#F4A623" if not pd.isna(val) else "#8A9BB8")
        ax.set_facecolor("#0A1628")
        ax.axhspan(lo, hi, alpha=0.15, color="#0E9E8E")
        ax.axhline(lo, color="#0E9E8E", lw=0.8, ls="--", alpha=0.5)
        ax.axhline(hi, color="#0E9E8E", lw=0.8, ls="--", alpha=0.5)
        if not pd.isna(val):
            ax.scatter([0], [val], color=color, s=120, zorder=5)
            ax.axhline(val, color=color, lw=1.5, alpha=0.7)
        ax.set_xlim(-0.5, 0.5); ax.set_xticks([])
        ax.set_title(target, color="#0E9E8E", fontsize=10, fontweight="bold")
        for sp in ax.spines.values(): sp.set_edgecolor("#1E3A5F")
        ax.tick_params(colors="#8A9BB8", labelsize=8)
    plt.tight_layout()
    st.pyplot(fig); plt.close(fig)

# ══════════════════════════════════════════════
# TAB 2 — ALL PATIENTS
# ══════════════════════════════════════════════
with tab_data:
    st.markdown("### 📋 Rekod Penuh Pesakit + Ramalan AI")
    ai_cols = [f"AI_{t}" for t in TARGETS]
    other_cols = [c for c in result_df.columns if c not in ai_cols]
    display_df = result_df[other_cols + ai_cols].copy()
    display_df["Clinical_Status"] = display_df["AI_pH"].apply(
        lambda v: ph_status(v).upper() if not pd.isna(v) else "UNKNOWN")
    st.dataframe(display_df.round(3), use_container_width=True, height=480)
    st.caption(f"Jumlah: {len(display_df)} pesakit · Kolum AI_ = ramalan model")

# ══════════════════════════════════════════════
# TAB 3 — ANALYTICS
# ══════════════════════════════════════════════
with tab_viz:
    col_l, col_r = st.columns(2, gap="large")

    with col_l:
        st.markdown("#### 🔗 Correlation Heatmap")
        numeric_df = processed_df.select_dtypes(include=np.number)
        corr = numeric_df.corr()
        fig_h, ax_h = plt.subplots(figsize=(8, 7))
        fig_h.patch.set_facecolor("#132035"); ax_h.set_facecolor("#0A1628")
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, cmap=sns.diverging_palette(180,10,s=80,l=40,as_cmap=True),
            center=0, annot=True, fmt=".1f", annot_kws={"size":7,"color":"#F4F7FB"},
            linewidths=0.4, linecolor="#0A1628", ax=ax_h, cbar_kws={"shrink":0.7})
        ax_h.tick_params(colors="#8A9BB8", labelsize=8)
        ax_h.set_title("Feature Correlation Matrix", color="#0E9E8E", fontsize=12, pad=12)
        plt.tight_layout(); st.pyplot(fig_h); plt.close(fig_h)

    with col_r:
        st.markdown("#### 🎯 Feature Importance")
        target_sel = st.selectbox("Pilih target", TARGETS, key="fi_target")
        imp_dict = importances.get(target_sel, {})
        if imp_dict:
            imp_s = pd.Series(imp_dict).sort_values(ascending=True)
            total = imp_s.sum()
            if total > 0: imp_s = imp_s / total * 100
            fig_b, ax_b = dark_fig(w=7, h=max(4, len(imp_s)*0.45))
            bars = ax_b.barh(imp_s.index, imp_s.values,
                color=["#0E9E8E" if v >= imp_s.quantile(0.66)
                       else ("#12C2B0" if v >= imp_s.quantile(0.33) else "#1E5A52")
                       for v in imp_s.values], edgecolor="none", height=0.65)
            ax_b.set_xlabel("Relative Importance (%)", color="#8A9BB8")
            ax_b.set_title(f"Feature Importance → {target_sel}", color="#0E9E8E", fontsize=11)
            ax_b.xaxis.grid(True, linestyle="--", alpha=0.3, color="#8A9BB8"); ax_b.set_axisbelow(True)
            for bar in bars:
                w = bar.get_width()
                if w > 0.5:
                    ax_b.text(w+0.3, bar.get_y()+bar.get_height()/2, f"{w:.1f}%", va="center", color="#F4F7FB", fontsize=7)
            plt.tight_layout(); st.pyplot(fig_b); plt.close(fig_b)

    st.markdown("---")
    st.markdown("#### 📊 Taburan Ramalan AI")
    fig_d, axes_d = plt.subplots(1, 5, figsize=(16, 4))
    fig_d.patch.set_facecolor("#132035")
    for ax_d, target in zip(axes_d, TARGETS):
        data = result_df[f"AI_{target}"].dropna()
        lo, hi = NORMAL_RANGES[target]
        ax_d.set_facecolor("#0A1628")
        ax_d.hist(data, bins=15, color="#0E9E8E", edgecolor="#0A1628", alpha=0.85)
        ax_d.axvline(lo, color="#F4A623", lw=1.2, ls="--")
        ax_d.axvline(hi, color="#F4A623", lw=1.2, ls="--")
        ax_d.set_title(target, color="#0E9E8E", fontsize=10)
        ax_d.tick_params(colors="#8A9BB8", labelsize=7)
        for sp in ax_d.spines.values(): sp.set_edgecolor("#1E3A5F")
    plt.suptitle("Population Distribution of AI Predictions", color="#F4F7FB", fontsize=12, y=1.02)
    plt.tight_layout(); st.pyplot(fig_d); plt.close(fig_d)

# ══════════════════════════════════════════════
# TAB 4 — FORMULA VALIDATION
# ══════════════════════════════════════════════
with tab_formula:
    st.markdown("### 🧮 Pengesahan Formula Fisiologi")
    st.markdown("Tab ini membandingkan ramalan ANN vs formula fisiologi yang betul secara klinikal.")

    col_f1, col_f2 = st.columns(2, gap="large")

    with col_f1:
        st.markdown("#### Henderson–Hasselbalch (pH)")
        st.markdown("""
<div class="formula-box">
pH = 6.1 + log₁₀( HCO₃ / (0.0307 × PaCO₂) )<br><br>
Koreksi Lactate Acidosis:<br>
pH_adjusted = pH − (Lactate × 0.02)
</div>""", unsafe_allow_html=True)

        st.markdown("#### Alveolar Gas Equation (PaO₂)")
        st.markdown("""
<div class="formula-box">
PAO₂ = FiO₂ × (760−47) − PaCO₂ / 0.8<br>
A-a gradient = 2.5 + 0.21 × Age<br>
PaO₂ = PAO₂ − A-a gradient
</div>""", unsafe_allow_html=True)

    with col_f2:
        st.markdown("#### Hill / ODC Equation (O₂ Saturation)")
        st.markdown("""
<div class="formula-box">
SaO₂ = PaO₂²·⁷ / (PaO₂²·⁷ + 26.8²·⁷) × 100<br><br>
n = 2.7 (koefisien Hill)<br>
P50 = 26.8 mmHg (normal)
</div>""", unsafe_allow_html=True)

        st.markdown("#### HCO₃ (Henderson)")
        st.markdown("""
<div class="formula-box">
HCO₃ = 0.0307 × PaCO₂ × 10^(pH − 6.1)<br><br>
Koreksi Lactate:<br>
HCO₃_adjusted = HCO₃ − (Lactate × 0.5)
</div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### 📊 R² Score — Ketepatan Model ANN")
    r2_data = {t: r2_scores.get(t, "Formula Fallback") for t in TARGETS}
    r2_cols = st.columns(5)
    for i, target in enumerate(TARGETS):
        val = r2_data[target]
        with r2_cols[i]:
            if isinstance(val, float):
                color = "#0E9E8E" if val >= 0.7 else ("#F4A623" if val >= 0.4 else "#E84C4C")
                st.markdown(
                    f"<div style='background:#132035;border:1px solid {color};border-radius:10px;"
                    f"padding:14px;text-align:center'>"
                    f"<div style='color:#8A9BB8;font-size:0.7rem;text-transform:uppercase'>{target}</div>"
                    f"<div style='color:{color};font-family:Space Mono,monospace;font-size:1.5rem;font-weight:700'>"
                    f"R²={val}</div></div>", unsafe_allow_html=True)
            else:
                st.markdown(
                    f"<div style='background:#132035;border:1px solid #8A9BB8;border-radius:10px;"
                    f"padding:14px;text-align:center'>"
                    f"<div style='color:#8A9BB8;font-size:0.7rem;text-transform:uppercase'>{target}</div>"
                    f"<div style='color:#8A9BB8;font-size:0.85rem'>Formula<br>Fallback</div></div>",
                    unsafe_allow_html=True)

    st.markdown("")
    st.markdown("#### 🔀 Blend Logic: ANN + Formula")
    st.markdown("""
<div class="formula-box">
Jika Ground Truth ≥ 5 baris:<br>
&nbsp;&nbsp;Final Prediction = 0.70 × ANN_Prediction + 0.30 × Formula_Prediction<br><br>
Jika Ground Truth &lt; 5 baris:<br>
&nbsp;&nbsp;Final Prediction = Formula Fisiologi (100%)
</div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════
# TAB 5 — EXPORT
# ══════════════════════════════════════════════
with tab_export:
    st.markdown("### ⬇️ Muat Turun Keputusan")
    st.markdown("""
<div class="section-card">
<p style="color:#8A9BB8">Fail CSV mengandungi data asal, semua features yang diproses,
5 kolum ramalan AI (<code>AI_pH</code>, <code>AI_PaCO2</code>, <code>AI_PaO2</code>,
<code>AI_O2_Saturation</code>, <code>AI_HCO3</code>), dan label klinikal status.</p>
</div>""", unsafe_allow_html=True)

    export_df = result_df.copy()
    export_df["Clinical_Status"] = export_df["AI_pH"].apply(
        lambda v: ph_status(v).upper() if not pd.isna(v) else "UNKNOWN")

    col_a, col_b, _ = st.columns([1, 1, 3])
    with col_a:
        st.download_button(
            label="⬇  Download Full CSV",
            data=export_df.round(4).to_csv(index=False).encode(),
            file_name="icu_blood_gas_predictions.csv", mime="text/csv")
    with col_b:
        st.metric("Jumlah Pesakit", len(export_df))

    st.markdown("#### Preview (10 baris pertama)")
    st.dataframe(export_df.head(10).round(3), use_container_width=True, hide_index=True)
