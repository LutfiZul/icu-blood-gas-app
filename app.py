"""
AI-Driven ICU Blood Gas Assistant
Streamlit Application
Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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
# CUSTOM CSS  — Teal / Navy / White palette
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=Space+Mono:wght@400;700&display=swap');

/* ── Root palette ── */
:root {
    --navy:   #0A1628;
    --navy2:  #112240;
    --teal:   #0E9E8E;
    --teal2:  #12C2B0;
    --white:  #F4F7FB;
    --muted:  #8A9BB8;
    --alert:  #F4A623;
    --crit:   #E84C4C;
    --ok:     #0E9E8E;
    --card:   #132035;
    --border: rgba(14,158,142,0.25);
}

/* ── Global ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--navy) !important;
    color: var(--white) !important;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: var(--navy2) !important;
    border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] * { color: var(--white) !important; }

/* ── Headers ── */
h1 { font-family: 'Space Mono', monospace; color: var(--teal2) !important; letter-spacing: -1px; }
h2, h3 { color: var(--teal) !important; }
h4, h5, h6 { color: var(--muted) !important; font-weight: 500; }

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    padding: 18px 22px !important;
}
[data-testid="stMetricLabel"]  { color: var(--muted)  !important; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 1px; }
[data-testid="stMetricValue"]  { color: var(--teal2)  !important; font-family: 'Space Mono', monospace; font-size: 1.9rem !important; }
[data-testid="stMetricDelta"]  { color: var(--muted)  !important; }

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, var(--teal), #0A7A6D) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.55rem 1.4rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.5px;
    transition: opacity 0.2s;
}
.stButton > button:hover { opacity: 0.85; }

/* ── Download button ── */
.stDownloadButton > button {
    background: var(--navy2) !important;
    color: var(--teal2) !important;
    border: 1px solid var(--teal) !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
}

/* ── Selectbox / inputs ── */
.stSelectbox > div > div, .stTextInput > div > input {
    background: var(--card) !important;
    color: var(--white) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
}

/* ── DataFrame ── */
.stDataFrame { border: 1px solid var(--border); border-radius: 10px; overflow: hidden; }

/* ── Dividers ── */
hr { border-color: var(--border) !important; }

/* ── Status badge helpers (HTML injected) ── */
.badge {
    display: inline-block;
    padding: 5px 14px;
    border-radius: 20px;
    font-weight: 700;
    font-size: 0.85rem;
    letter-spacing: 0.8px;
    font-family: 'Space Mono', monospace;
}
.badge-normal   { background: rgba(14,158,142,0.18); color: #0E9E8E; border: 1px solid #0E9E8E; }
.badge-alert    { background: rgba(244,166,35,0.18);  color: #F4A623; border: 1px solid #F4A623; }
.badge-critical { background: rgba(232,76,76,0.18);   color: #E84C4C; border: 1px solid #E84C4C; }

/* ── Section cards ── */
.section-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 24px;
    margin-bottom: 20px;
}

/* ── Top banner ── */
.top-banner {
    background: linear-gradient(135deg, #0E3D35 0%, #0A1628 60%);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 28px 32px;
    margin-bottom: 24px;
}
.top-banner h1 { margin: 0 0 6px 0; font-size: 2rem; }
.top-banner p  { color: var(--muted); margin: 0; font-size: 0.95rem; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# CONSTANTS — FYP: Fecal Peritonitis ICU
# ─────────────────────────────────────────────
TARGETS  = ["pH", "PaCO2", "PaO2", "O2_Saturation", "HCO3"]

# 10 features dari FYP korang
FEATURES = [
    "Age", "Gender", "Severity_Score", "Heart_Rate", "Temperature",
    "WBC_Count", "Lactate_Level", "Mechanical_Ventilation", "Systolic", "Diastolic",
]

NORMAL_RANGES = {
    "pH":            (7.35, 7.45),
    "PaCO2":         (35, 45),
    "PaO2":          (75, 100),
    "O2_Saturation": (95, 100),
    "HCO3":          (22, 26),
}

UNITS = {
    "pH": "",
    "PaCO2": "mmHg",
    "PaO2": "mmHg",
    "O2_Saturation": "%",
    "HCO3": "mEq/L",
}

# ─────────────────────────────────────────────
# HELPER: MATPLOTLIB DARK THEME
# ─────────────────────────────────────────────
def dark_fig(w=10, h=5):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor("#132035")
    ax.set_facecolor("#0A1628")
    for spine in ax.spines.values():
        spine.set_edgecolor("#1E3A5F")
    ax.tick_params(colors="#8A9BB8")
    ax.xaxis.label.set_color("#8A9BB8")
    ax.yaxis.label.set_color("#8A9BB8")
    ax.title.set_color("#0E9E8E")
    return fig, ax

# ─────────────────────────────────────────────
# STEP 1 — PRE-PROCESSING (dari FYP korang)
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_and_process(raw_bytes: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(raw_bytes))

    # Pecahkan Blood_Pressure "120/80" → Systolic / Diastolic
    if "Blood_Pressure" in df.columns:
        bp = df["Blood_Pressure"].astype(str).str.split("/", expand=True)
        df["Systolic"]  = pd.to_numeric(bp[0], errors="coerce")
        df["Diastolic"] = pd.to_numeric(bp[1], errors="coerce")
        df.drop(columns=["Blood_Pressure"], inplace=True)

    # Convert Categorical → Digital (sama macam FYP korang)
    if "Gender" in df.columns:
        df["Gender"] = df["Gender"].map({"Male": 1, "Female": 0, "M": 1, "F": 0})
    if "Mechanical_Ventilation" in df.columns:
        df["Mechanical_Ventilation"] = df["Mechanical_Ventilation"].map({"Yes": 1, "No": 0})

    # Coerce semua ke numeric
    for col in df.columns:
        if col != "Patient_ID":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df

# ─────────────────────────────────────────────
# STEP 2 — DEEP LEARNING INFERENCE ENGINE
#           (Logic terus dari FYP korang)
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def train_and_predict(raw_bytes: bytes):
    df = load_and_process(raw_bytes)

    # Tentukan features yang wujud dalam data
    available_features = [f for f in FEATURES if f in df.columns]

    # Pastikan tiada NaN dalam features sebelum training
    df_complete = df.dropna(subset=available_features)

    importances = {}
    log_messages = []

    for target in TARGETS:
        if target not in df.columns:
            df[target] = np.nan

        # Cari baris yang ada Ground Truth untuk target ini
        df_train = df_complete.dropna(subset=[target])

        # ── Statistical Imputation (logic FYP korang) ──
        # Jika data ground truth < 5, guna korelasi Lactate
        if len(df_train) < 5:
            log_messages.append(f"⚠️ {target}: Data terhad — Statistical Imputation digunakan.")
            if "Lactate_Level" in df.columns:
                if target == "pH":
                    df[target] = df[target].fillna(7.4 - (df["Lactate_Level"] * 0.02))
                elif target == "HCO3":
                    df[target] = df[target].fillna(24 - (df["Lactate_Level"] * 0.5))
                else:
                    fallback = df[target].mean() if not df[target].isnull().all() else 95.0
                    df[target] = df[target].fillna(fallback)
            else:
                fallback = df[target].mean() if not df[target].isnull().all() else 95.0
                df[target] = df[target].fillna(fallback)
            df_train = df.dropna(subset=[target] + available_features)
        else:
            log_messages.append(f"✅ {target}: {len(df_train)} rekod ground truth ditemui.")

        X_train = df_train[available_features]
        y_train = df_train[target]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        # ── ANN: 2 Hidden Layers (12, 12) — sama macam FYP ──
        model = MLPRegressor(
            hidden_layer_sizes=(12, 12),
            activation="relu",
            solver="adam",
            max_iter=5000,
            random_state=42,
        )
        model.fit(X_train_scaled, y_train)

        # ── PREDICT: Isi semua NaN dalam kolum tersebut ──
        nan_mask = df[target].isna()
        if nan_mask.any():
            X_pred = df.loc[nan_mask, available_features].fillna(
                df[available_features].mean()
            )
            X_pred_scaled = scaler.transform(X_pred)
            preds = model.predict(X_pred_scaled)

            # ── Had Fisiologi / Biological Capping ──
            if target == "pH":
                preds = np.clip(preds, 6.5, 8.0)
            elif target == "O2_Saturation":
                preds = np.clip(preds, 0, 100.0)
            else:
                preds = np.clip(preds, 0, None)

            df.loc[nan_mask, target] = preds

        # ── Permutation Feature Importance ──
        X_all_scaled = scaler.transform(
            df[available_features].fillna(df[available_features].mean())
        )
        y_all = df[target].values
        base_mse = np.mean((model.predict(X_all_scaled) - y_all) ** 2)
        imp = []
        for i in range(X_all_scaled.shape[1]):
            X_perm = X_all_scaled.copy()
            np.random.shuffle(X_perm[:, i])
            p_mse = np.mean((model.predict(X_perm) - y_all) ** 2)
            imp.append(max(p_mse - base_mse, 0))
        importances[target] = dict(zip(available_features, imp))

    # Simpan AI predictions dalam kolum berasingan
    result_df = df.copy()
    for t in TARGETS:
        result_df[f"AI_{t}"] = df[t]

    return result_df, importances, log_messages

# ─────────────────────────────────────────────
# STATUS BADGE
# ─────────────────────────────────────────────
def ph_status(ph_val: float) -> str:
    if pd.isna(ph_val):
        return "unknown"
    if ph_val < 7.2 or ph_val > 7.6:
        return "critical"
    if ph_val < 7.35 or ph_val > 7.45:
        return "alert"
    return "normal"

def status_badge_html(status: str) -> str:
    label = status.upper()
    cls   = f"badge-{status}"
    icons = {"normal": "✔", "alert": "⚠", "critical": "✖", "unknown": "?"}
    icon  = icons.get(status, "?")
    return f'<span class="badge {cls}">{icon} {label}</span>'

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🩺 ICU Blood Gas\n**AI Assistant**")
    st.markdown("---")

    uploaded = st.file_uploader(
        "Upload Patient CSV",
        type=["csv"],
        help="CSV must include vital signs columns. Blood_Pressure should be 'Systolic/Diastolic'.",
    )

    st.markdown("---")
    st.markdown("### About")
    st.markdown(
        "Predicts missing arterial blood gas parameters "
        "(pH, PaCO₂, PaO₂, O₂ Sat, HCO₃) using a deep "
        "Artificial Neural Network trained on vital signs.",
        unsafe_allow_html=False,
    )
    st.markdown("---")
    st.caption("AI predictions are decision-support tools only. Always apply clinical judgment.")

# ─────────────────────────────────────────────
# BANNER
# ─────────────────────────────────────────────
st.markdown("""
<div class="top-banner">
  <h1>🔬 ICU Blood Gas Assistant</h1>
  <p>AI-Driven Arterial Blood Gas Prediction · Deep Learning · Clinical Decision Support</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SAMPLE DATA GENERATOR (if no file uploaded)
# ─────────────────────────────────────────────
def generate_sample_csv() -> bytes:
    np.random.seed(0)
    n = 40
    data = {
        "Patient_ID":             [f"PT{1000+i}" for i in range(n)],
        "Age":                    np.random.randint(25, 85, n).astype(float),
        "Gender":                 np.random.choice(["Male", "Female"], n),
        "Severity_Score":         np.round(np.random.uniform(5, 25, n), 1),
        "Heart_Rate":             np.random.randint(55, 130, n).astype(float),
        "Temperature":            np.round(np.random.uniform(36.0, 39.5, n), 1),
        "WBC_Count":              np.round(np.random.uniform(4.0, 20.0, n), 1),
        "Lactate_Level":          np.round(np.random.uniform(0.5, 8.0, n), 2),
        "Mechanical_Ventilation": np.random.choice(["Yes", "No"], n),
        "Blood_Pressure":         [f"{np.random.randint(90,170)}/{np.random.randint(55,105)}" for _ in range(n)],
        # Targets — sebahagian sengaja dikosongkan (NaN) untuk AI predict
        "pH":            np.where(np.random.rand(n) < 0.35, np.nan, np.round(np.random.uniform(7.20, 7.55, n), 2)),
        "PaCO2":         np.where(np.random.rand(n) < 0.35, np.nan, np.round(np.random.uniform(28, 58, n), 1)),
        "PaO2":          np.where(np.random.rand(n) < 0.35, np.nan, np.round(np.random.uniform(55, 115, n), 1)),
        "O2_Saturation": np.where(np.random.rand(n) < 0.35, np.nan, np.round(np.random.uniform(85, 100, n), 1)),
        "HCO3":          np.where(np.random.rand(n) < 0.35, np.nan, np.round(np.random.uniform(15, 32, n), 1)),
    }
    df = pd.DataFrame(data)
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode()

# ─────────────────────────────────────────────
# MAIN APP LOGIC
# ─────────────────────────────────────────────
if uploaded is None:
    st.info("👈  Upload a CSV in the sidebar, or click below to load sample data.")
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

# ── Process ──────────────────────────────────
with st.spinner("🧠  Training AI models & predicting blood gas values…"):
    result_df, importances, ai_logs = train_and_predict(raw_bytes)
    processed_df = load_and_process(raw_bytes)

st.success(f"✅  Predictions complete — **{len(result_df)} patients** processed.")

# Tunjuk log AI engine (sama macam terminal output FYP korang)
with st.expander("🔬 AI Engine Log — Deep Learning Inference", expanded=False):
    for msg in ai_logs:
        color = "#F4A623" if "⚠️" in msg else "#0E9E8E"
        st.markdown(f"<span style='color:{color}; font-family:monospace'>{msg}</span>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# TAB LAYOUT
# ─────────────────────────────────────────────
tab_dash, tab_data, tab_viz, tab_export = st.tabs([
    "🏥  Patient Dashboard",
    "📋  All Patients",
    "📊  Analytics",
    "⬇️  Export",
])

# ╔════════════════════════════════════════════╗
# ║  TAB 1 — PATIENT DASHBOARD                ║
# ╚════════════════════════════════════════════╝
with tab_dash:
    st.markdown("### 🔍 Patient Lookup")

    patient_ids = result_df["Patient_ID"].astype(str).tolist() if "Patient_ID" in result_df.columns else result_df.index.astype(str).tolist()
    selected_id = st.selectbox("Select Patient ID", patient_ids)

    if "Patient_ID" in result_df.columns:
        row = result_df[result_df["Patient_ID"].astype(str) == selected_id].iloc[0]
    else:
        row = result_df.iloc[int(selected_id)]

    st.markdown("---")

    # Status badge
    ph_pred = row.get(f"AI_pH", np.nan)
    status  = ph_status(ph_pred)
    st.markdown(
        f"**Clinical Status:** {status_badge_html(status)}&nbsp;&nbsp;"
        f"<span style='color:#8A9BB8; font-size:0.85rem'>(based on predicted pH)</span>",
        unsafe_allow_html=True,
    )
    st.markdown("")

    # ── Predicted Blood Gas Metrics ───────────
    st.markdown("#### 🩸 Predicted Arterial Blood Gas Results")
    cols = st.columns(5)
    metric_icons = {"pH": "⚗", "PaCO2": "💨", "PaO2": "🫁", "O2_Saturation": "💉", "HCO3": "⚡"}

    for i, target in enumerate(TARGETS):
        val = row.get(f"AI_{target}", np.nan)
        unit = UNITS[target]
        lo, hi = NORMAL_RANGES[target]
        delta_str = f"Normal {lo}–{hi} {unit}".strip()
        with cols[i]:
            st.metric(
                label=f"{metric_icons[target]}  {target}",
                value=f"{val:.2f} {unit}".strip() if not pd.isna(val) else "N/A",
                delta=delta_str,
            )

    st.markdown("---")

    # ── Vitals Summary ────────────────────────
    st.markdown("#### 📟 Recorded Vitals")
    vitals_cols = [c for c in result_df.columns if c not in TARGETS + [f"AI_{t}" for t in TARGETS] + ["Patient_ID"]]
    vdata = {c: [row.get(c, "—")] for c in vitals_cols}
    st.dataframe(pd.DataFrame(vdata).round(2), use_container_width=True, hide_index=True)

    # ── Normal Range Mini Chart ───────────────
    st.markdown("#### 📈 Predicted vs Normal Range")

    fig, axes = plt.subplots(1, 5, figsize=(14, 3))
    fig.patch.set_facecolor("#132035")

    for ax, target in zip(axes, TARGETS):
        val  = row.get(f"AI_{target}", np.nan)
        lo, hi = NORMAL_RANGES[target]
        color = "#0E9E8E" if lo <= val <= hi else ("#F4A623" if not pd.isna(val) else "#8A9BB8")

        ax.set_facecolor("#0A1628")
        ax.axhspan(lo, hi, alpha=0.15, color="#0E9E8E")
        ax.axhline(lo, color="#0E9E8E", lw=0.8, ls="--", alpha=0.5)
        ax.axhline(hi, color="#0E9E8E", lw=0.8, ls="--", alpha=0.5)
        if not pd.isna(val):
            ax.scatter([0], [val], color=color, s=120, zorder=5)
            ax.axhline(val, color=color, lw=1.5, alpha=0.7)
        ax.set_xlim(-0.5, 0.5)
        ax.set_xticks([])
        ax.set_title(target, color="#0E9E8E", fontsize=10, fontweight="bold")
        for sp in ax.spines.values():
            sp.set_edgecolor("#1E3A5F")
        ax.tick_params(colors="#8A9BB8", labelsize=8)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# ╔════════════════════════════════════════════╗
# ║  TAB 2 — ALL PATIENTS TABLE               ║
# ╚════════════════════════════════════════════╝
with tab_data:
    st.markdown("### 📋 Full Patient Dataset with AI Predictions")

    ai_cols    = [f"AI_{t}" for t in TARGETS]
    other_cols = [c for c in result_df.columns if c not in ai_cols]
    display_df = result_df[other_cols + ai_cols].copy()

    # Add status column
    display_df["Clinical_Status"] = display_df[f"AI_pH"].apply(
        lambda v: ph_status(v).upper() if not pd.isna(v) else "UNKNOWN"
    )

    st.dataframe(
        display_df.round(3),
        use_container_width=True,
        height=480,
    )
    st.caption(f"Total: {len(display_df)} patients · AI_ prefixed columns are model predictions.")

# ╔════════════════════════════════════════════╗
# ║  TAB 3 — ANALYTICS                        ║
# ╚════════════════════════════════════════════╝
with tab_viz:
    col_l, col_r = st.columns([1, 1], gap="large")

    # ── Correlation Heatmap ───────────────────
    with col_l:
        st.markdown("#### 🔗 Correlation Heatmap")
        numeric_df = processed_df.select_dtypes(include=np.number)
        corr = numeric_df.corr()

        fig_h, ax_h = plt.subplots(figsize=(8, 7))
        fig_h.patch.set_facecolor("#132035")
        ax_h.set_facecolor("#0A1628")

        mask = np.triu(np.ones_like(corr, dtype=bool))
        cmap = sns.diverging_palette(180, 10, s=80, l=40, as_cmap=True)
        sns.heatmap(
            corr, mask=mask, cmap=cmap, center=0,
            annot=True, fmt=".1f", annot_kws={"size": 7, "color": "#F4F7FB"},
            linewidths=0.4, linecolor="#0A1628",
            ax=ax_h,
            cbar_kws={"shrink": 0.7},
        )
        ax_h.tick_params(colors="#8A9BB8", labelsize=8)
        ax_h.set_title("Feature Correlation Matrix", color="#0E9E8E", fontsize=12, pad=12)
        plt.tight_layout()
        st.pyplot(fig_h)
        plt.close(fig_h)

    # ── Feature Importance Bar Chart ──────────
    with col_r:
        st.markdown("#### 🎯 Feature Importance (by Target)")
        target_sel = st.selectbox("Choose target", TARGETS, key="fi_target")
        imp_dict   = importances.get(target_sel, {})

        if imp_dict:
            imp_s = pd.Series(imp_dict).sort_values(ascending=True)
            total = imp_s.sum()
            if total > 0:
                imp_s = imp_s / total * 100

            fig_b, ax_b = dark_fig(w=7, h=max(4, len(imp_s) * 0.45))
            bars = ax_b.barh(
                imp_s.index, imp_s.values,
                color=[
                    "#0E9E8E" if v >= imp_s.quantile(0.66)
                    else ("#12C2B0" if v >= imp_s.quantile(0.33) else "#1E5A52")
                    for v in imp_s.values
                ],
                edgecolor="none",
                height=0.65,
            )
            ax_b.set_xlabel("Relative Importance (%)", color="#8A9BB8")
            ax_b.set_title(f"Feature Importance → {target_sel}", color="#0E9E8E", fontsize=11)
            ax_b.xaxis.grid(True, linestyle="--", alpha=0.3, color="#8A9BB8")
            ax_b.set_axisbelow(True)

            # Value labels
            for bar in bars:
                w = bar.get_width()
                if w > 0.5:
                    ax_b.text(w + 0.3, bar.get_y() + bar.get_height() / 2,
                              f"{w:.1f}%", va="center", color="#F4F7FB", fontsize=7)

            plt.tight_layout()
            st.pyplot(fig_b)
            plt.close(fig_b)
        else:
            st.warning("No importance data available.")

    st.markdown("---")

    # ── Distribution of AI Predictions ───────
    st.markdown("#### 📊 Distribution of AI-Predicted Blood Gas Values")
    fig_d, axes_d = plt.subplots(1, 5, figsize=(16, 4))
    fig_d.patch.set_facecolor("#132035")

    for ax_d, target in zip(axes_d, TARGETS):
        col_name = f"AI_{target}"
        data = result_df[col_name].dropna()
        ax_d.set_facecolor("#0A1628")
        ax_d.hist(data, bins=15, color="#0E9E8E", edgecolor="#0A1628", alpha=0.85)
        lo, hi = NORMAL_RANGES[target]
        ax_d.axvline(lo, color="#F4A623", lw=1.2, ls="--", label="Normal range")
        ax_d.axvline(hi, color="#F4A623", lw=1.2, ls="--")
        ax_d.set_title(target, color="#0E9E8E", fontsize=10)
        ax_d.tick_params(colors="#8A9BB8", labelsize=7)
        for sp in ax_d.spines.values():
            sp.set_edgecolor("#1E3A5F")

    plt.suptitle("Population Distribution of AI Predictions", color="#F4F7FB", fontsize=12, y=1.02)
    plt.tight_layout()
    st.pyplot(fig_d)
    plt.close(fig_d)

# ╔════════════════════════════════════════════╗
# ║  TAB 4 — EXPORT                           ║
# ╚════════════════════════════════════════════╝
with tab_export:
    st.markdown("### ⬇️  Download Results")

    st.markdown("""
<div class="section-card">
<p style="color:#8A9BB8">The exported CSV contains the original patient data, all processed features,
and the 5 AI-predicted blood gas columns (<code>AI_pH</code>, <code>AI_PaCO2</code>, <code>AI_PaO2</code>,
<code>AI_O2_Saturation</code>, <code>AI_HCO3</code>), together with the clinical status label.</p>
</div>
""", unsafe_allow_html=True)

    export_df = result_df.copy()
    export_df["Clinical_Status"] = export_df["AI_pH"].apply(
        lambda v: ph_status(v).upper() if not pd.isna(v) else "UNKNOWN"
    )

    csv_bytes = export_df.round(4).to_csv(index=False).encode()

    col_a, col_b, _ = st.columns([1, 1, 3])
    with col_a:
        st.download_button(
            label="⬇  Download Full CSV",
            data=csv_bytes,
            file_name="icu_blood_gas_predictions.csv",
            mime="text/csv",
        )
    with col_b:
        st.metric("Total Patients", len(export_df))

    st.markdown("#### Preview (first 10 rows)")
    st.dataframe(export_df.head(10).round(3), use_container_width=True, hide_index=True)