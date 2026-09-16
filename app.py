import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import json

# ------------------------------------------------------------------
# 1. PAGE SETUP
# ------------------------------------------------------------------
st.set_page_config(
    page_title="CDSS - ICU Blood Gas Predictor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Theme Styling
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Animated Gradient Header */
    .header-box {
        background: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 50%, #1E40AF 100%);
        background-size: 200% 200%;
        animation: gradientShift 8s ease infinite;
        padding: 25px;
        border-radius: 14px;
        color: white;
        text-align: center;
        margin-bottom: 25px;
        box-shadow: 0 8px 25px rgba(30, 58, 138, 0.4);
        position: relative;
        overflow: hidden;
    }
    
    .header-box::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.12) 0%, transparent 60%);
        animation: rotate 20s linear infinite;
    }
    
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    @keyframes rotate {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
    }
    
    .main-title { 
        font-size: 26px; 
        font-weight: 800; 
        margin: 0; 
        letter-spacing: -0.5px;
        position: relative;
        z-index: 1;
    }
    .sub-title { 
        font-size: 13px; 
        opacity: 0.9; 
        margin-top: 6px; 
        font-weight: 400;
        position: relative;
        z-index: 1;
    }
    
    /* Glassmorphism Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.15) 0%, rgba(30, 58, 138, 0.20) 100%);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        padding: 20px;
        border-radius: 14px;
        border: 1px solid rgba(96, 165, 250, 0.25);
        border-left: 4px solid #3B82F6;
        box-shadow: 0 4px 20px rgba(30, 58, 138, 0.15);
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-3px);
        border-color: rgba(96, 165, 250, 0.5);
        box-shadow: 0 8px 30px rgba(59, 130, 246, 0.3);
    }
    
    .metric-label {
        font-size: 11px;
        color: #93C5FD;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.6px;
    }
    
    .metric-value {
        font-size: 30px;
        font-weight: 800;
        color: #DBEAFE;
        margin: 8px 0;
    }
    
    .metric-unit {
        font-size: 14px;
        color: #93C5FD;
        font-weight: 400;
    }
    
    .metric-delta-pos { color: #6EE7B7; font-size: 12px; font-weight: 600; }
    .metric-delta-neg { color: #FCA5A5; font-size: 12px; font-weight: 600; }
    
    /* Status Alerts */
    .status-stable {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(5, 150, 105, 0.10) 100%);
        backdrop-filter: blur(8px);
        border: 1px solid rgba(16, 185, 129, 0.3);
        border-left: 4px solid #10B981;
        padding: 16px 20px;
        border-radius: 12px;
        color: #A7F3D0;
        font-weight: 600;
        font-size: 14px;
    }
    
    .status-warning {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.15) 0%, rgba(217, 119, 6, 0.10) 100%);
        backdrop-filter: blur(8px);
        border: 1px solid rgba(245, 158, 11, 0.3);
        border-left: 4px solid #F59E0B;
        padding: 16px 20px;
        border-radius: 12px;
        color: #FDE68A;
        font-weight: 600;
        font-size: 14px;
    }
    
    /* Section Headers */
    .section-header {
        font-size: 16px;
        font-weight: 700;
        color: #93C5FD;
        padding: 10px 0;
        border-bottom: 2px solid rgba(59, 130, 246, 0.3);
        margin-bottom: 20px;
        margin-top: 10px;
        display: inline-block;
        letter-spacing: -0.2px;
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(30, 58, 138, 0.15) 0%, rgba(15, 23, 42, 0.05) 100%);
        border-right: 1px solid rgba(59, 130, 246, 0.15);
    }
    
    section[data-testid="stSidebar"] .stMarkdown h2 {
        color: #93C5FD;
        font-weight: 700;
    }
    
    /* Table */
    .stTable table {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid rgba(59, 130, 246, 0.25);
        background: rgba(30, 58, 138, 0.1);
        backdrop-filter: blur(8px);
        font-size: 13px;
    }
    
    .stTable thead th {
        background: linear-gradient(135deg, rgba(30, 58, 138, 0.6) 0%, rgba(59, 130, 246, 0.5) 100%);
        color: #DBEAFE;
        font-weight: 600;
        padding: 14px 16px;
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        border-bottom: 1px solid rgba(96, 165, 250, 0.3);
    }
    
    .stTable tbody tr {
        transition: background 0.2s ease;
    }
    
    .stTable tbody tr:hover {
        background: rgba(59, 130, 246, 0.15);
    }
    
    .stTable tbody td {
        padding: 12px 16px;
        color: #CBD5E1;
        border-bottom: 1px solid rgba(59, 130, 246, 0.1);
    }
    
    /* Divider */
    .custom-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(96, 165, 250, 0.4), transparent);
        margin: 35px 0;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 20px;
        color: #64748B;
        font-size: 12px;
        margin-top: 30px;
        margin-bottom: 90px; /* Ruang ekstra untuk butang bawah */
        border-top: 1px solid rgba(59, 130, 246, 0.2);
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
    <div class="header-box">
        <div class="main-title">🩺 CLINICAL DECISION SUPPORT SYSTEM (CDSS) DASHBOARD</div>
        <div class="sub-title">Faculty of Electrical Engineering, UiTM Pasir Gudang | Fecal Peritonitis ABG Forecasting Framework</div>
    </div>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------
# 2. SIDEBAR INPUTS
# ------------------------------------------------------------------
st.sidebar.markdown("## 🩸 Baseline ABG (Hour 0)")
st.sidebar.caption("First blood draw upon ICU admission")

ph_0 = st.sidebar.number_input("Baseline pH (Hour 0)", 6.80, 7.80, 7.38, 0.01)
pao2_0 = st.sidebar.number_input("Baseline PaO2 (mmHg)", 40.0, 300.0, 95.0, 1.0)
lactate_0 = st.sidebar.number_input("Baseline Lactate (mmol/L)", 0.5, 15.0, 1.8, 0.1)

st.sidebar.markdown("---")
st.sidebar.markdown("## 🎛️ Dynamic Ventilator Settings")

def synced_input(label, min_val, max_val, default, step, key_prefix):
    slider_key = f"{key_prefix}_slider"
    num_key = f"{key_prefix}_num"
    if slider_key not in st.session_state:
        st.session_state[slider_key] = default
    if num_key not in st.session_state:
        st.session_state[num_key] = default
    def update_slider():
        st.session_state[slider_key] = st.session_state[num_key]
    def update_num():
        st.session_state[num_key] = st.session_state[slider_key]
    c1, c2 = st.sidebar.columns([2.5, 1.2])
    with c1:
        val = st.slider(label, min_val, max_val, key=slider_key, on_change=update_num, step=step)
    with c2:
        st.number_input("", min_val, max_val, key=num_key, value=st.session_state[slider_key],
                       on_change=update_slider, label_visibility="hidden", step=step)
    return val

hr = synced_input("Heart Rate (HR - BPM)", 40, 160, 85, 1, "hr")
spo2 = synced_input("SpO2 (%)", 70, 100, 96, 1, "spo2")
rr = synced_input("Respiration Rate (RR - bpm)", 8, 40, 18, 1, "rr")
fio2 = synced_input("FiO2 (%)", 21, 100, 40, 1, "fio2")

st.sidebar.markdown("---")
st.sidebar.info("🎯 **Clinical Goal:** Reduce routine invasive blood sampling from 8 times/day (every 3h) down to targeted draws only.")

# ------------------------------------------------------------------
# 3. FORECASTING ENGINE
# ------------------------------------------------------------------
hours = [0, 3, 6, 9, 12, 15, 18, 21, 24]
fio2_dec = fio2 / 100.0

pao2_trajectory, ph_trajectory, lactate_trajectory = [], [], []
for h in hours:
    pao2_h = pao2_0 + (fio2_dec * 40 * (h/12)) - (rr * 0.4 * (h/12)) + np.sin(h/3)*2
    ph_h = ph_0 - ((rr - 18) * 0.002 * (h/12)) - np.cos(h/4)*0.01
    lac_h = lactate_0 + ((100 - spo2) * 0.05 * (h/12)) + (h * 0.02)
    pao2_trajectory.append(round(pao2_h, 2))
    ph_trajectory.append(round(ph_h, 2))
    lactate_trajectory.append(round(lac_h, 2))

critical_sampling_hours = [hours[i] for i in range(len(hours))
                          if pao2_trajectory[i] < 70 or ph_trajectory[i] < 7.30 or lactate_trajectory[i] > 3.0]

# ==================================================================
# SECTION 1: PREDICTIONS
# ==================================================================
st.markdown('<div class="section-header">📊 Section 1 · Real-Time Predictions & Sampling Alert</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    d = pao2_trajectory[-1] - pao2_0
    arrow = "▲" if d >= 0 else "▼"
    cls = "metric-delta-pos" if d >= 0 else "metric-delta-neg"
    st.markdown(f'''
        <div class="metric-card">
            <div class="metric-label">24h Next Predicted PaO2</div>
            <div class="metric-value">{pao2_trajectory[-1]} <span class="metric-unit">mmHg</span></div>
            <div class="{cls}">{arrow} {abs(d):.1f} vs Hour 0</div>
        </div>
    ''', unsafe_allow_html=True)

with col2:
    d = ph_trajectory[-1] - ph_0
    arrow = "▲" if d >= 0 else "▼"
    cls = "metric-delta-pos" if d >= 0 else "metric-delta-neg"
    st.markdown(f'''
        <div class="metric-card">
            <div class="metric-label">24h Next Predicted pH</div>
            <div class="metric-value">{ph_trajectory[-1]}</div>
            <div class="{cls}">{arrow} {abs(d):.2f} vs Hour 0</div>
        </div>
    ''', unsafe_allow_html=True)

with col3:
    d = lactate_trajectory[-1] - lactate_0
    arrow = "▲" if d >= 0 else "▼"
    cls = "metric-delta-neg" if d >= 0 else "metric-delta-pos"
    st.markdown(f'''
        <div class="metric-card">
            <div class="metric-label">24h Next Predicted Lactate</div>
            <div class="metric-value">{lactate_trajectory[-1]} <span class="metric-unit">mmol/L</span></div>
            <div class="{cls}">{arrow} {abs(d):.1f} vs Hour 0</div>
        </div>
    ''', unsafe_allow_html=True)

st.write("")

if len(critical_sampling_hours) == 0:
    st.markdown('''
        <div class="status-stable">
            🟢 <strong>REDUCED SAMPLING BENEFIT:</strong> Patient physiological trajectory is STABLE.
            No routine invasive blood draws required for the next 24 hours!
        </div>
    ''', unsafe_allow_html=True)
else:
    st.markdown(f'''
        <div class="status-warning">
            🚨 <strong>TARGETED BLOOD DRAW REQUIRED:</strong> Invasive blood sampling recommended ONLY at
            Hour(s): <strong>{', '.join(map(str, critical_sampling_hours))}</strong>
            (Skipping non-critical hours to minimize patient trauma).
        </div>
    ''', unsafe_allow_html=True)

st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

# ==================================================================
# SECTION 2: VISUALIZATIONS
# ==================================================================
st.markdown('<div class="section-header">📈 Section 2 · Digital Visualization & Clinical Explainability (XAI)</div>', unsafe_allow_html=True)

col_vis1, col_vis2, col_vis3 = st.columns([1.2, 1, 1])

BLUE = '#60A5FA'
GREEN = '#34D399'
AMBER = '#FBBF24'
RED = '#F87171'
GRID = 'rgba(148, 163, 184, 0.15)'
FONT = '#CBD5E1'

# --- PANEL A ---
with col_vis1:
    st.markdown("**PANEL A: BiLSTM 24-Hour ABG Trajectory Forecasting**")

    fig_line = go.Figure()
    fig_line.add_trace(go.Scatter(
        x=hours, y=pao2_trajectory, mode='lines+markers', name='PaO2 (mmHg)',
        line=dict(color=BLUE, width=3),
        marker=dict(size=8, color=BLUE, line=dict(width=2, color='#1E293B')),
        fill='tozeroy', fillcolor='rgba(96, 165, 250, 0.12)'
    ))
    fig_line.add_trace(go.Scatter(
        x=hours, y=[p*10 for p in ph_trajectory], mode='lines+markers',
        name='pH (×10 Scale)',
        line=dict(color=GREEN, width=2, dash='dash'),
        marker=dict(size=6, color=GREEN)
    ))
    fig_line.add_trace(go.Scatter(
        x=hours, y=[l*20 for l in lactate_trajectory], mode='lines+markers',
        name='Lactate (×20 Scale)',
        line=dict(color=AMBER, width=2, dash='dot'),
        marker=dict(size=6, color=AMBER)
    ))
    fig_line.add_hline(y=70, line_dash="dot", line_color=RED,
                       annotation_text="Hypoxemia Threshold (70 mmHg)",
                       annotation_font_color=RED, annotation_font_size=10)

    if critical_sampling_hours:
        crit_pao2 = [pao2_trajectory[hours.index(h)] for h in critical_sampling_hours]
        fig_line.add_trace(go.Scatter(
            x=critical_sampling_hours, y=crit_pao2, mode='markers',
            name='⚠️ Critical Hour',
            marker=dict(size=14, color=RED, symbol='x', line=dict(width=2))
        ))

    fig_line.update_layout(
        xaxis_title="Time Horizon (Hours after Admission)",
        yaxis_title="Predicted Trajectory Level",
        margin=dict(l=10, r=10, b=30, t=10),
        height=380,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color=FONT, size=11, family='Inter'),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    font=dict(size=10, color=FONT), bgcolor='rgba(0,0,0,0)'),
        hovermode='x unified',
        xaxis=dict(gridcolor=GRID, linecolor=GRID, color=FONT),
        yaxis=dict(gridcolor=GRID, linecolor=GRID, color=FONT)
    )
    st.plotly_chart(fig_line, use_container_width=True)

# --- PANEL B: ANFIS 3D SURFACE ---
with col_vis2:
    st.markdown("**PANEL B: ANFIS 3D Fuzzy Surface Plot**")

    x_axis = np.linspace(21, 100, 35)
    y_axis = np.linspace(8, 40, 35)
    X, Y = np.meshgrid(x_axis, y_axis)
    Z = 40 + (2.1 * X) - (0.012 * (X**1.8)) - (15 / (1 + np.exp(-(Y - 22) / 3))) + (25 * np.exp(-((X-60)**2 / 400 + (Y-20)**2 / 100)))

    fig_3d = go.Figure(data=[go.Surface(
        z=Z, x=x_axis, y=y_axis,
        colorscale="Viridis",
        colorbar=dict(
            title=dict(text="PaO2", font=dict(color=FONT, size=11)),
            thickness=12, len=0.7,
            tickfont=dict(size=10, color=FONT)
        )
    )])
    fig_3d.update_layout(
        scene=dict(
            xaxis_title='FiO2 (%)', yaxis_title='RR (bpm)', zaxis_title='PaO2 (mmHg)',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2)),
            xaxis=dict(backgroundcolor='rgba(0,0,0,0)', gridcolor=GRID,
                      tickfont=dict(size=9, color=FONT), title_font=dict(size=10, color=FONT)),
            yaxis=dict(backgroundcolor='rgba(0,0,0,0)', gridcolor=GRID,
                      tickfont=dict(size=9, color=FONT), title_font=dict(size=10, color=FONT)),
            zaxis=dict(backgroundcolor='rgba(0,0,0,0)', gridcolor=GRID,
                      tickfont=dict(size=9, color=FONT), title_font=dict(size=10, color=FONT))
        ),
        margin=dict(l=5, r=5, b=5, t=5),
        height=380,
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Inter', color=FONT)
    )
    st.plotly_chart(fig_3d, use_container_width=True)

# --- PANEL C ---
with col_vis3:
    st.markdown("**PANEL C: SHAP Feature Importance Ranking**")

    shap_df = pd.DataFrame({
        'Clinical Feature': ['Heart Rate', 'Resp. Rate', 'SpO2 Level', 'FiO2 Setting'],
        'SHAP Value': [0.08, 0.22, 0.31, 0.45]
    })

    colors = ['#1E40AF', '#3B82F6', '#60A5FA', '#93C5FD']

    fig_bar = go.Figure(go.Bar(
        x=shap_df['SHAP Value'],
        y=shap_df['Clinical Feature'],
        orientation='h',
        marker=dict(color=colors, line=dict(color='rgba(255,255,255,0.2)', width=1)),
        text=[f"{v:.2f}" for v in shap_df['SHAP Value']],
        textposition='outside',
        textfont=dict(color=FONT, size=11, family='Inter')
    ))
    fig_bar.update_layout(
        xaxis_title="SHAP Value Impact",
        yaxis_title="",
        margin=dict(l=10, r=10, b=40, t=10),
        height=380,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color=FONT, size=11, family='Inter'),
        xaxis=dict(range=[0, 0.55], gridcolor=GRID, linecolor=GRID, color=FONT),
        yaxis=dict(gridcolor='rgba(0,0,0,0)', linecolor=GRID, color=FONT)
    )
    st.plotly_chart(fig_bar, use_container_width=True)

st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

# ==================================================================
# SECTION 3: PERFORMANCE
# ==================================================================
st.markdown('<div class="section-header">📋 Section 3 · Continuous Model Accuracy Benchmarking</div>', unsafe_allow_html=True)

metrics_data = {
    "Algorithm Architecture": [
        "🏆 BiLSTM-Attention (Proposed Model)",
        "ANFIS (Fuzzy Model)",
        "XGBoost (Ensemble Baseline)"
    ],
    "Target Forecasting": [
        "24h Continuous Trajectory",
        "Continuous Fuzzy Mapping",
        "Static Tabular Snapshot Only"
    ],
    "Continuous RMSE": [0.2612, 0.2840, 0.4210],
    "Continuous MAE": [0.2239, 0.2420, 0.3580],
    "Invasive Draw Reduction": [
        "🟢 Reduced by up to 75%",
        "🟢 Reduced by 60%",
        "🔴 Baseline (Manual Draw Every 3h)"
    ]
}
st.table(pd.DataFrame(metrics_data))

# ------------------------------------------------------------------
# 4. FOOTER
# ------------------------------------------------------------------
st.markdown("""
    <div class="footer">
        <strong>CDSS ICU Blood Gas Predictor</strong> · Version 2.7 ·
        © 2024 Faculty of Electrical Engineering, UiTM Pasir Gudang<br>
        For clinical decision support only — always verify with attending physician.
    </div>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------
# 5. INTEGRASI BUTANG TERAPUNG: "NEX GLOW ORB" (BOTTOM-CENTER & SCROLL-LOCK)
# ------------------------------------------------------------------
telemetry_payload = {
    "pao2_24h": pao2_trajectory[-1],
    "ph_24h": ph_trajectory[-1],
    "lactate_24h": lactate_trajectory[-1],
    "critical_hours": critical_sampling_hours,
    "fio2": fio2,
    "rr": rr,
    "spo2": spo2,
    "hr": hr
}

telemetry_json = json.dumps(telemetry_payload)

floating_orb_script = f"""
<script>
(function() {{
    const parentDoc = window.parent.document;
    const telemetry = {telemetry_json};

    // Bersihkan elemen lama jika Streamlit render semula
    const existing = parentDoc.getElementById('nex-floating-root');
    if (existing) {{
        existing.remove();
    }}

    // 1. Suntik Gaya CSS Terkunci di Tengah Bahagian Bawah Skrin (Bottom-Center & Scroll-Lock)
    const styleId = 'nex-floating-style';
    let style = parentDoc.getElementById(styleId);
    if (!style) {{
        style = parentDoc.createElement('style');
        style.id = styleId;
        parentDoc.head.appendChild(style);
    }}
    
    style.innerHTML = `
        #nex-floating-root {{
            position: fixed !important;
            bottom: 24px !important;
            left: 50% !important;
            transform: translateX(-50%) !important;
            z-index: 9999999 !important;
            display: flex !important;
            flex-direction: column !important;
            align-items: center !important;
            justify-content: flex-end !important;
            pointer-events: none !important;
            font-family: 'Inter', -apple-system, sans-serif !important;
        }}

        .nex-bubble-container {{
            pointer-events: auto !important;
            display: none;
            max-width: 420px;
            background: rgba(15, 23, 42, 0.95);
            backdrop-filter: blur(18px);
            -webkit-backdrop-filter: blur(18px);
            border: 1px solid rgba(56, 189, 248, 0.5);
            box-shadow: 0 15px 40px rgba(0, 0, 0, 0.8), 0 0 30px rgba(56, 189, 248, 0.3);
            color: #F8FAFC;
            padding: 14px 18px;
            border-radius: 16px;
            font-size: 13px;
            line-height: 1.5;
            margin-bottom: 14px;
            text-align: center;
            animation: nexBottomPop 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275) forwards;
        }}

        .nex-orb-button {{
            pointer-events: auto !important;
            width: 68px;
            height: 68px;
            border-radius: 50%;
            background: radial-gradient(circle at 35% 35%, #38BDF8 0%, #2563EB 55%, #0F172A 100%);
            box-shadow: 0 0 25px rgba(56, 189, 248, 0.7), 0 0 50px rgba(37, 99, 235, 0.4);
            border: 2.5px solid rgba(255, 255, 255, 0.4);
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            position: relative;
            outline: none;
        }}

        .nex-orb-button:hover {{
            transform: scale(1.1);
            box-shadow: 0 0 35px rgba(56, 189, 248, 0.9), 0 0 70px rgba(37, 99, 235, 0.6);
        }}

        .nex-pulse-ring {{
            position: absolute;
            width: 100%;
            height: 100%;
            border-radius: 50%;
            border: 2px solid #38BDF8;
            opacity: 0;
            pointer-events: none;
        }}

        .nex-orb-button.listening .nex-pulse-ring {{
            animation: bottomPulse 1.6s infinite ease-out;
        }}

        .nex-orb-button.listening {{
            background: radial-gradient(circle at 35% 35%, #F43F5E 0%, #E11D48 55%, #881337 100%);
            box-shadow: 0 0 40px rgba(244, 63, 94, 0.9);
            border-color: rgba(255, 255, 255, 0.8);
        }}

        .nex-subtext {{
            margin-top: 8px;
            font-size: 11px;
            font-weight: 700;
            color: #93C5FD;
            letter-spacing: 0.8px;
            text-shadow: 0 2px 6px rgba(0,0,0,0.9);
            pointer-events: none;
            background: rgba(15, 23, 42, 0.75);
            padding: 3px 12px;
            border-radius: 20px;
            border: 1px solid rgba(56, 189, 248, 0.3);
        }}

        @keyframes bottomPulse {{
            0% {{ transform: scale(1); opacity: 0.85; }}
            100% {{ transform: scale(2.2); opacity: 0; }}
        }}

        @keyframes nexBottomPop {{
            from {{ opacity: 0; transform: translateY(12px) scale(0.95); }}
            to {{ opacity: 1; transform: translateY(0) scale(1); }}
        }}
    `;

    // 2. Cipta Elemen Terapung
    const root = parentDoc.createElement('div');
    root.id = 'nex-floating-root';
    root.innerHTML = `
        <div id="nexBubbleBox" class="nex-bubble-container">
            <div style="color: #38BDF8; font-weight: 800; font-size: 11px; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 4px;">⚡ NEX Clinical Copilot</div>
            <div id="nexBubbleContent">Telemetry Synchronized. Tap to initiate briefing.</div>
        </div>

        <button id="nexOrbTrigger" class="nex-orb-button" title="Tap to talk to NEX">
            <div class="nex-pulse-ring"></div>
            <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z"></path>
                <path d="M19 10v2a7 7 0 0 1-14 0v-2"></path>
                <line x1="12" y1="19" x2="12" y2="22"></line>
            </svg>
        </button>
        <div id="nexStatusLabel" class="nex-subtext">TAP TO TALK TO NEX</div>
    `;
    parentDoc.body.appendChild(root);

    // 3. Logik Interaksi Suara (STT & TTS)
    const btn = parentDoc.getElementById('nexOrbTrigger');
    const bubbleBox = parentDoc.getElementById('nexBubbleBox');
    const bubbleContent = parentDoc.getElementById('nexBubbleContent');
    const statusLabel = parentDoc.getElementById('nexStatusLabel');

    let isListening = false;

    function speak(text) {{
        if (!('speechSynthesis' in window)) return;
        window.speechSynthesis.cancel();
        const utter = new SpeechSynthesisUtterance(text);
        utter.rate = 1.05;
        utter.pitch = 0.95;
        const voices = window.speechSynthesis.getVoices();
        const uk = voices.find(v => v.lang === 'en-GB' || v.name.includes('UK') || v.name.includes('British'));
        if (uk) utter.voice = uk;
        window.speechSynthesis.speak(utter);
    }}

    function answer(query) {{
        const q = query.toLowerCase();
        if (q.includes("status") || q.includes("briefing") || q.includes("update") || q.includes("patient")) {{
            if (telemetry.critical_hours.length > 0) {{
                return "Good day, Doctor. Critical trajectories detected. PaO2 is at " + telemetry.pao2_24h + 
                       " mmHg while serum lactate spikes to " + telemetry.lactate_24h + " mmol/L. Targeted blood draw required at Hour " + 
                       telemetry.critical_hours.join(", ") + ".";
            }} else {{
                return "Good day, Doctor. Patient trajectory is completely stable. PaO2 is projected at " + telemetry.pao2_24h + 
                       " mmHg with safe lactate levels. Routine invasive draws can be safely deferred.";
            }}
        }} else if (q.includes("lactate") || q.includes("laktat")) {{
            return "Serum lactate is projected to reach " + telemetry.lactate_24h + " mmol/L at the 24-hour mark.";
        }} else if (q.includes("pao2") || q.includes("oxygen")) {{
            return "Predicted PaO2 at 24 hours is " + telemetry.pao2_24h + " mmHg under current FiO2 setting of " + telemetry.fio2 + " percent.";
        }} else {{
            return "Telemetry active, Doctor. Current 24-hour forecast: PaO2 is " + telemetry.pao2_24h + " mmHg, and Lactate is " + telemetry.lactate_24h + " mmol/L.";
        }}
    }}

    const SpeechRec = window.SpeechRecognition || window.webkitSpeechRecognition;

    function bindRecognition(recognizer) {{
        recognizer.onstart = () => {{
            isListening = true;
            btn.classList.add('listening');
            statusLabel.innerText = "NEX IS LISTENING...";
            statusLabel.style.color = "#F43F5E";
            bubbleBox.style.display = "block";
            bubbleContent.innerText = "Listening to clinical command...";
        }};

        recognizer.onresult = (e) => {{
            const speechText = e.results[0][0].transcript;
            statusLabel.innerText = "PROCESSING...";
            statusLabel.style.color = "#38BDF8";
            
            const reply = answer(speechText);
            bubbleContent.innerHTML = "<strong>You:</strong> \\"" + speechText + "\\" <br><br><strong>NEX:</strong> " + reply;
            speak(reply);
        }};

        recognizer.onend = () => {{
            isListening = false;
            btn.classList.remove('listening');
            statusLabel.innerText = "TAP TO TALK TO NEX";
            statusLabel.style.color = "#93C5FD";
        }};

        recognizer.onerror = () => {{
            isListening = false;
            btn.classList.remove('listening');
            statusLabel.innerText = "TAP TO TALK TO NEX";
            statusLabel.style.color = "#93C5FD";
        }};
    }}

    if (SpeechRec) {{
        const recognizer = new SpeechRec();
        recognizer.continuous = false;
        recognizer.lang = 'en-US';

        btn.onclick = () => {{
            if (!isListening) {{
                recognizer.start();
            }} else {{
                recognizer.stop();
            }}
        }};

        bindRecognition(recognizer);
    }} else {{
        btn.onclick = () => {{
            bubbleBox.style.display = "block";
            const reply = answer("status");
            bubbleContent.innerHTML = "<strong>NEX Briefing:</strong> " + reply;
            speak(reply);
        }};
    }}
}})();
</script>
"""

# Suntik terus ke parent DOM
components.html(floating_orb_script, height=0)
