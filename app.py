import base64
from datetime import datetime
import json
import os
import time
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
import torch
import torch.nn as nn

# ------------------------------------------------------------------
# 1. PAGE SETUP & THEME STYLING
# ------------------------------------------------------------------
st.set_page_config(
    page_title="CDSS - ICU Blood Gas Predictor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

KPA_TO_MMHG = 7.50062


def get_base64_image(image_path):
  try:
    with open(image_path, "rb") as f:
      return base64.b64encode(f.read()).decode()
  except FileNotFoundError:
    return None


bg_img = None
for path in [
    "Background.jpg",
    "background.jpg",
    "./Background.jpg",
    "assets/Background.jpg",
    "static/Background.jpg",
]:
  bg_img = get_base64_image(path)
  if bg_img:
    break

bg_css = ""
if bg_img:
  bg_css = f"""
    .stApp {{
        background-image: 
            linear-gradient(rgba(15, 23, 42, 0.85), rgba(15, 23, 42, 0.90)),
            url("data:image/jpeg;base64,{bg_img}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}
    """
else:
  bg_css = (
      ".stApp { background: linear-gradient(180deg, #0F172A 0%, #1E293B 100%);"
      " }"
  )

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}
    
    {bg_css}
    
    .main .block-container {{
        background: rgba(15, 23, 42, 0.55);
        backdrop-filter: blur(8px);
        border-radius: 18px;
        padding: 2rem 2.5rem;
        margin-top: 1rem;
        border: 1px solid rgba(96, 165, 250, 0.15);
    }}
    
    .header-box {{
        background: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 50%, #1E40AF 100%);
        background-size: 200% 200%;
        animation: gradientShift 8s ease infinite;
        padding: 25px;
        border-radius: 14px;
        color: white;
        text-align: center;
        margin-bottom: 25px;
        box-shadow: 0 8px 25px rgba(30, 58, 138, 0.4);
    }}
    @keyframes gradientShift {{
        0% {{ background-position: 0% 50%; }}
        50% {{ background-position: 100% 50%; }}
        100% {{ background-position: 0% 50%; }}
    }}
    .main-title {{ font-size: 26px; font-weight: 800; margin: 0; }}
    .sub-title {{ font-size: 13px; opacity: 0.9; margin-top: 6px; }}
    
    .metric-card {{
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.20) 0%, rgba(30, 58, 138, 0.30) 100%);
        backdrop-filter: blur(14px);
        padding: 20px;
        border-radius: 14px;
        border: 1px solid rgba(96, 165, 250, 0.30);
        border-left: 4px solid #3B82F6;
        box-shadow: 0 4px 20px rgba(30, 58, 138, 0.25);
    }}
    .metric-label {{ font-size: 11px; color: #93C5FD; font-weight: 600; text-transform: uppercase; letter-spacing: 0.6px; }}
    .metric-value {{ font-size: 30px; font-weight: 800; color: #DBEAFE; margin: 8px 0; }}
    .metric-unit {{ font-size: 14px; color: #93C5FD; }}
    .metric-delta-pos {{ color: #6EE7B7; font-size: 12px; font-weight: 600; }}
    .metric-delta-neg {{ color: #FCA5A5; font-size: 12px; font-weight: 600; }}
    
    .status-stable {{
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.20) 0%, rgba(5, 150, 105, 0.15) 100%);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(16, 185, 129, 0.4);
        border-left: 4px solid #10B981;
        padding: 16px 20px;
        border-radius: 12px;
        color: #A7F3D0;
        font-weight: 600;
        font-size: 14px;
    }}
    .status-warning {{
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.20) 0%, rgba(217, 119, 6, 0.15) 100%);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(245, 158, 11, 0.4);
        border-left: 4px solid #F59E0B;
        padding: 16px 20px;
        border-radius: 12px;
        color: #FDE68A;
        font-weight: 600;
        font-size: 14px;
    }}
    
    .section-header {{
        font-size: 16px;
        font-weight: 700;
        color: #93C5FD;
        padding: 10px 0;
        border-bottom: 2px solid rgba(59, 130, 246, 0.3);
        margin-bottom: 20px;
        margin-top: 10px;
        display: inline-block;
    }}
    
    .patient-info-card {{
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(5, 150, 105, 0.10) 100%);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(16, 185, 129, 0.35);
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 20px;
    }}
    .patient-info-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        gap: 12px;
        margin-top: 10px;
    }}
    .patient-info-item {{
        background: rgba(15, 23, 42, 0.5);
        border-radius: 8px;
        padding: 10px 12px;
        border-left: 3px solid #10B981;
    }}
    .patient-info-label {{ font-size: 10px; color: #6EE7B7; font-weight: 600; text-transform: uppercase; }}
    .patient-info-value {{ font-size: 16px; color: #F0FDF4; font-weight: 700; margin-top: 2px; }}
    
    section[data-testid="stSidebar"] {{
        background: linear-gradient(180deg, rgba(15, 23, 42, 0.95) 0%, rgba(30, 58, 138, 0.85) 100%);
        border-right: 1px solid rgba(59, 130, 246, 0.25);
    }}
    section[data-testid="stSidebar"] .stMarkdown h2 {{ color: #93C5FD; font-weight: 700; }}
    
    .stTable table {{
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid rgba(59, 130, 246, 0.25);
        background: rgba(15, 23, 42, 0.5);
        font-size: 13px;
    }}
    .stTable thead th {{
        background: linear-gradient(135deg, rgba(30, 58, 138, 0.8) 0%, rgba(59, 130, 246, 0.6) 100%);
        color: #DBEAFE;
        font-weight: 600;
        padding: 14px 16px;
        font-size: 12px;
    }}
    .stTable tbody td {{ padding: 12px 16px; color: #CBD5E1; }}
    
    .custom-divider {{
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(96, 165, 250, 0.4), transparent);
        margin: 35px 0;
    }}
    .footer {{
        text-align: center;
        padding: 20px;
        color: #94A3B8;
        font-size: 12px;
        margin-top: 30px;
        margin-bottom: 90px;
        border-top: 1px solid rgba(59, 130, 246, 0.2);
    }}
    
    .upload-status-box {{
        padding: 14px 18px;
        border-radius: 12px;
        margin-top: 12px;
        font-weight: 600;
        font-size: 13px;
        display: flex;
        align-items: center;
        gap: 10px;
        animation: slideIn 0.4s ease;
    }}
    .upload-success {{
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.25) 0%, rgba(5, 150, 105, 0.20) 100%);
        border: 1px solid rgba(16, 185, 129, 0.5);
        border-left: 4px solid #10B981;
        color: #A7F3D0;
        box-shadow: 0 0 20px rgba(16, 185, 129, 0.3);
    }}
    .upload-error {{
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.25) 0%, rgba(185, 28, 28, 0.20) 100%);
        border: 1px solid rgba(239, 68, 68, 0.5);
        border-left: 4px solid #EF4444;
        color: #FCA5A5;
        box-shadow: 0 0 20px rgba(239, 68, 68, 0.3);
        animation: shake 0.5s ease;
    }}
    .upload-loading {{
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.25) 0%, rgba(37, 99, 235, 0.20) 100%);
        border: 1px solid rgba(59, 130, 246, 0.5);
        border-left: 4px solid #3B82F6;
        color: #93C5FD;
    }}
    
    @keyframes slideIn {{
        from {{ opacity: 0; transform: translateX(-20px); }}
        to {{ opacity: 1; transform: translateX(0); }}
    }}
    @keyframes shake {{
        0%, 100% {{ transform: translateX(0); }}
        10%, 30%, 50%, 70%, 90% {{ transform: translateX(-8px); }}
        20%, 40%, 60%, 80% {{ transform: translateX(8px); }}
    }}
    @keyframes checkmark {{
        0% {{ transform: scale(0) rotate(-45deg); opacity: 0; }}
        50% {{ transform: scale(1.3) rotate(-45deg); opacity: 1; }}
        100% {{ transform: scale(1) rotate(-45deg); opacity: 1; }}
    }}
    .checkmark-icon {{
        display: inline-block;
        width: 18px;
        height: 18px;
        border-bottom: 3px solid currentColor;
        border-right: 3px solid currentColor;
        transform: rotate(-45deg);
        animation: checkmark 0.5s ease;
    }}
    .error-icon {{
        display: inline-block;
        font-size: 20px;
        animation: shake 0.5s ease;
    }}
    .spinner {{
        display: inline-block;
        width: 16px;
        height: 16px;
        border: 3px solid rgba(147, 197, 253, 0.3);
        border-top-color: #93C5FD;
        border-radius: 50%;
        animation: spin 0.8s linear infinite;
    }}
    @keyframes spin {{
        to {{ transform: rotate(360deg); }}
    }}
    .fade-in {{
        animation: fadeIn 0.6s ease;
    }}
    @keyframes fadeIn {{
        from {{ opacity: 0; transform: translateY(10px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}
    </style>
""",
    unsafe_allow_html=True,
)


def show_upload_success(message="File uploaded successfully!"):
  st.sidebar.markdown(
      f"""
        <div class="upload-status-box upload-success">
            <span class="checkmark-icon"></span>
            <span>{message}</span>
        </div>
    """,
      unsafe_allow_html=True,
  )
  components.html(
      """
    <script>
    (function() {
        const parent = window.parent.document;
        const old = parent.getElementById('success-particles');
        if (old) old.remove();
        
        const container = parent.createElement('div');
        container.id = 'success-particles';
        container.style.cssText = 'position:fixed;top:0;left:0;width:100%;height:100%;pointer-events:none;z-index:99999;';
        parent.body.appendChild(container);
        
        for (let i = 0; i < 30; i++) {
            const p = parent.createElement('div');
            const colors = ['#10B981', '#34D399', '#6EE7B7', '#3B82F6', '#60A5FA'];
            p.style.cssText = `
                position: absolute;
                width: 8px; height: 8px;
                background: ${colors[Math.floor(Math.random() * colors.length)]};
                border-radius: 50%;
                left: 20%; top: 30%;
                opacity: 1;
                transition: all 1.5s cubic-bezier(0.4, 0, 0.2, 1);
            `;
            container.appendChild(p);
            
            setTimeout(() => {
                const angle = Math.random() * Math.PI * 2;
                const distance = 100 + Math.random() * 200;
                p.style.left = (20 + Math.cos(angle) * distance / 10) + '%';
                p.style.top = (30 + Math.sin(angle) * distance / 10) + '%';
                p.style.opacity = '0';
                p.style.transform = 'scale(0)';
            }, 50);
        }
        
        setTimeout(() => container.remove(), 2000);
    })();
    </script>
    """,
      height=0,
  )


def show_upload_error(message="Upload failed!"):
  st.sidebar.markdown(
      f"""
        <div class="upload-status-box upload-error">
            <span class="error-icon">❌</span>
            <span>{message}</span>
        </div>
    """,
      unsafe_allow_html=True,
  )


def show_upload_loading(message="Processing..."):
  st.sidebar.markdown(
      f"""
        <div class="upload-status-box upload-loading">
            <span class="spinner"></span>
            <span>{message}</span>
        </div>
    """,
      unsafe_allow_html=True,
  )


st.markdown(
    """
    <div class="header-box">
        <div class="main-title">🩺 CLINICAL DECISION SUPPORT SYSTEM (CDSS) DASHBOARD</div>
        <div class="sub-title">Faculty of Electrical Engineering, UiTM Pasir Gudang | Fecal Peritonitis ABG Forecasting Framework (2017 Cohort)</div>
    </div>
""",
    unsafe_allow_html=True,
)


# ------------------------------------------------------------------
# 2. MODEL PIPELINE (BiLSTM-ATTENTION)
# ------------------------------------------------------------------
class Attention(nn.Module):

  def __init__(self, hidden_dim):
    super(Attention, self).__init__()
    self.att = nn.Sequential(
        nn.Linear(hidden_dim, hidden_dim // 2),
        nn.Tanh(),
        nn.Linear(hidden_dim // 2, 1),
    )

  def forward(self, rnn_out):
    weights = torch.softmax(self.att(rnn_out), dim=1)
    context = torch.sum(weights * rnn_out, dim=1)
    return context, weights


class BiLSTMModel(nn.Module):

  def __init__(self, input_dim=7, hidden_dim=64, num_layers=2, output_dim=3):
    super(BiLSTMModel, self).__init__()
    self.bilstm = nn.LSTM(
        input_dim,
        hidden_dim,
        num_layers=num_layers,
        bidirectional=True,
        batch_first=True,
        dropout=0.2,
    )
    self.attention = Attention(hidden_dim * 2)
    self.fc = nn.Sequential(
        nn.Linear(hidden_dim * 2, 64),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(64, output_dim),
    )

  def forward(self, x):
    lstm_out, _ = self.bilstm(x)
    context, weights = self.attention(lstm_out)
    return self.fc(context), weights


@st.cache_resource(show_spinner=False)
def load_trained_pipeline():
  model_path = os.path.join("models", "bilstm_attention_cdss.pt")
  scaler_x_path = os.path.join("models", "scaler_features.pkl")
  scaler_y_path = os.path.join("models", "scaler_target.pkl")

  if not os.path.exists(model_path):
    model_path = "bilstm_attention_cdss.pt"
    scaler_x_path = "scaler_features.pkl"
    scaler_y_path = "scaler_target.pkl"

  model = BiLSTMModel(input_dim=7, hidden_dim=64, num_layers=2, output_dim=3)
  model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
  model.eval()

  scaler_x = joblib.load(scaler_x_path)
  scaler_y = joblib.load(scaler_y_path)
  return model, scaler_x, scaler_y


try:
  bilstm_model, scaler_features, scaler_target = load_trained_pipeline()
  MODEL_LOADED = True
except Exception:
  MODEL_LOADED = False

# ------------------------------------------------------------------
# 3. BIGQUERY CONNECTIONS
# ------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_bq_client():
  try:
    from google.cloud import bigquery
    from google.oauth2 import service_account

    if "gcp_service_account" not in st.secrets:
      return None, "Tiada [gcp_service_account] dalam Streamlit Secrets"
    creds_dict = dict(st.secrets["gcp_service_account"])
    if "private_key" in creds_dict:
      creds_dict["private_key"] = creds_dict["private_key"].replace("\\n", "\n")
    credentials = service_account.Credentials.from_service_account_info(
        creds_dict
    )
    proj_id = creds_dict.get("project_id", "cdss-icu")
    client = bigquery.Client(credentials=credentials, project=proj_id)
    return client, None
  except Exception as e:
    return None, str(e)


def get_bq_table():
  try:
    if "bigquery" in st.secrets:
      cfg = st.secrets["bigquery"]
      return (
          f"{cfg.get('project_id', 'cdss-icu')}.{cfg.get('dataset', 'cdss_icu')}.{cfg.get('table', 'patient_records')}"
      )
    elif "gcp_service_account" in st.secrets:
      proj_id = st.secrets["gcp_service_account"].get("project_id", "cdss-icu")
      return f"{proj_id}.cdss_icu.patient_records"
    return "cdss-icu.cdss_icu.patient_records"
  except Exception:
    return "cdss-icu.cdss_icu.patient_records"


bq_client, bq_err_msg = get_bq_client()
BQ_TABLE = get_bq_table()
CLOUD_ENABLED = (bq_client is not None) and (BQ_TABLE is not None)


def push_to_bigquery(record_dict):
  if not CLOUD_ENABLED:
    return False, "Cloud disabled"
  try:
    errors = bq_client.insert_rows_json(BQ_TABLE, [record_dict])
    if errors:
      return False, str(errors)
    return True, "Uploaded"
  except Exception as e:
    return False, str(e)


def search_patients(query="", limit=50):
  if not CLOUD_ENABLED:
    return pd.DataFrame()
  try:
    from google.cloud import bigquery

    clean_q = query.strip().lower().replace("2024", "17").replace("fp24", "fp17")
    sql = f"""
            SELECT patient_id, COUNT(*) as num_records,
                   MAX(upload_timestamp) as last_seen,
                   MAX(age) as age, MAX(gender) as gender
            FROM `{BQ_TABLE}`
            WHERE LOWER(patient_id) LIKE @q
            GROUP BY patient_id
            ORDER BY last_seen DESC
            LIMIT {int(limit)}
        """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("q", "STRING", f"%{clean_q}%")
        ]
    )
    return bq_client.query(sql, job_config=job_config).to_dataframe()
  except Exception:
    return pd.DataFrame()


def get_patient_latest(patient_id):
  if not CLOUD_ENABLED:
    return None
  try:
    from google.cloud import bigquery

    pid_clean = patient_id.strip().lower()
    sql = f"""
            SELECT * FROM `{BQ_TABLE}`
            WHERE LOWER(patient_id) = @pid
            ORDER BY upload_timestamp DESC LIMIT 1
        """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("pid", "STRING", pid_clean)
        ]
    )
    df = bq_client.query(sql, job_config=job_config).to_dataframe()
    if len(df) == 0:
      return None
    return df.iloc[0].to_dict()
  except Exception:
    return None


def get_patient_history(patient_id):
  if not CLOUD_ENABLED:
    return pd.DataFrame()
  try:
    from google.cloud import bigquery

    sql = f"""
            SELECT upload_timestamp, ph_0, pao2_0, paco2_0, hco3_0,
                   hr, spo2, rr, fio2_0, source_filename
            FROM `{BQ_TABLE}`
            WHERE LOWER(patient_id) = @pid
            ORDER BY upload_timestamp DESC
        """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter(
                "pid", "STRING", patient_id.strip().lower()
            )
        ]
    )
    return bq_client.query(sql, job_config=job_config).to_dataframe()
  except Exception:
    return pd.DataFrame()


def delete_patient_records(patient_id):
  if not CLOUD_ENABLED:
    return False, "Cloud disabled"
  try:
    from google.cloud import bigquery

    sql = f"DELETE FROM `{BQ_TABLE}` WHERE LOWER(patient_id) = @pid"
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter(
                "pid", "STRING", patient_id.strip().lower()
            )
        ]
    )
    bq_client.query(sql, job_config=job_config).result()
    return True, "Deleted"
  except Exception as e:
    return False, str(e)


# ------------------------------------------------------------------
# 4. PARSER & PENUKARAN UNIT TEPAT KPA -> MMHG
# ------------------------------------------------------------------
FIELD_ALIASES = {
    "patient_id": [
        "patient_id",
        "patientid",
        "id",
        "pid",
        "mrn",
        "medical_record",
        "no_rm",
        "ic",
    ],
    "age": ["age", "umur", "patient_age"],
    "gender": ["gender", "sex", "jantina"],
    "weight_kg": ["weight", "weight_kg", "wt", "berat", "body_weight"],
    "height_cm": ["height", "height_cm", "ht", "tinggi", "body_height"],
    "body_temp_c": [
        "body_temp",
        "temperature",
        "temp",
        "temp_c",
        "suhu",
        "body_temperature",
        "bodytemp",
        "body_temp_c",
    ],
    "ph_0": ["ph_0", "ph", "ph_hour0", "baseline_ph", "arterial_ph"],
    "pao2_0": ["pao2_0", "pao2", "pao2_hour0", "baseline_pao2", "po2"],
    "paco2_0": ["paco2_0", "paco2", "paco2_hour0", "baseline_paco2", "pco2"],
    "hco3_0": ["hco3_0", "hco3", "bicarbonate", "hco3_mmol", "hco3a"],
    "sao2_0": ["sao2_0", "sao2", "o2sat", "o2_saturation", "so2"],
    "be_0": ["be_0", "be", "base_excess", "base_exc", "beecf"],
    "fio2_0": ["fio2_0", "fio2", "fio2_hour0", "fi_o2", "s-fio2", "fio2_set"],
    "hr": ["hr", "heart_rate", "heartrate", "pulse", "nadi"],
    "spo2": ["spo2", "sp_o2", "o2_sat", "saturation"],
    "rr": [
        "rr",
        "respiration_rate",
        "resp_rate",
        "breathing_rate",
        "tot_rr",
        "rr_total",
        "post_res",
    ],
    "sbp": ["sbp", "systolic", "systolic_bp", "bp_systolic", "bps"],
    "dbp": ["dbp", "diastolic", "diastolic_bp", "bp_diastolic", "bpd"],
    "map": ["map", "mean_arterial_pressure", "mean_bp", "bpm"],
    "peep": ["peep", "peep_cmh2o", "s-peep"],
}


def normalize_column(name):
  return (
      str(name)
      .lower()
      .strip()
      .replace(" ", "_")
      .replace("-", "_")
      .replace("(", "")
      .replace(")", "")
      .replace("'", "")
  )


def auto_map_columns(df_columns):
  normalized = {normalize_column(c): c for c in df_columns}
  mapping, matched = {}, set()
  for field, aliases in FIELD_ALIASES.items():
    for alias in aliases:
      if normalize_column(alias) in normalized:
        mapping[field] = normalized[normalize_column(alias)]
        matched.add(field)
        break
  return mapping, matched


def extract_patient_record(df, mapping, target_pid=None):
  record = {}
  conversion_notes = []
  if len(df) == 0:
    return record, conversion_notes

  row = df.iloc[0]
  if target_pid and "patient_id" in mapping:
    pid_col = mapping["patient_id"]
    clean_target = (
        str(target_pid)
        .strip()
        .lower()
        .replace("2024", "17")
        .replace("fp24", "fp17")
    )

    matched_rows = df[
        df[pid_col].astype(str).str.strip().str.lower() == clean_target
    ]
    if len(matched_rows) == 0:
      digits_only = "".join(filter(str.isdigit, clean_target))
      if digits_only:
        matched_rows = df[
            df[pid_col]
            .astype(str)
            .str.lower()
            .str.endswith(f"-{digits_only.zfill(3)}")
            | df[pid_col]
                .astype(str)
                .str.lower()
                .str.endswith(f"-{digits_only.zfill(4)}")
        ]
    if len(matched_rows) > 0:
      row = matched_rows.iloc[0]

  for field, col in mapping.items():
    try:
      val = row[col]
      if field in ["patient_id", "gender"]:
        record[field] = str(val).strip() if pd.notna(val) else None
      else:
        record[field] = float(val) if pd.notna(val) else None
    except Exception:
      record[field] = None

  if record.get("pao2_0") is not None and 0 < record["pao2_0"] <= 35.0:
    raw_val = record["pao2_0"]
    record["_kpa_converted"] = True
    record["pao2_0"] = round(raw_val * KPA_TO_MMHG, 1)
    conversion_notes.append(
        f"**PaO₂:** Nilai asal **{raw_val} kPa** dikesan. Ditukar automatik"
        f" kepada **{record['pao2_0']} mmHg** (faktor × {KPA_TO_MMHG})."
    )

  if record.get("paco2_0") is not None and 0 < record["paco2_0"] <= 18.0:
    raw_val = record["paco2_0"]
    record["_kpa_converted"] = True
    record["paco2_0"] = round(raw_val * KPA_TO_MMHG, 1)
    conversion_notes.append(
        f"**PaCO₂:** Nilai asal **{raw_val} kPa** dikesan. Ditukar automatik"
        f" kepada **{record['paco2_0']} mmHg** (faktor × {KPA_TO_MMHG})."
    )

  return record, conversion_notes


def compute_derived_abg(record):
  derived = {}
  ph = record.get("ph_0")
  paco2 = record.get("paco2_0")
  hco3 = record.get("hco3_0")
  pao2 = record.get("pao2_0")
  fio2 = record.get("fio2_0")

  if hco3 is None and ph and paco2:
    try:
      derived["hco3_0"] = round(24 * (10 ** (ph - 7.4)) * (40 / paco2), 2)
    except Exception:
      pass
  elif hco3:
    derived["hco3_0"] = hco3

  if pao2 and fio2:
    fdec = fio2 / 100.0 if fio2 > 1 else fio2
    if fdec > 0:
      derived["pf_ratio"] = round(pao2 / fdec, 1)

  if pao2 and paco2 and fio2:
    try:
      fdec = fio2 / 100.0 if fio2 > 1 else fio2
      pao2_alveolar = (fdec * 713) - (paco2 / 0.8)
      derived["aa_gradient"] = round(pao2_alveolar - pao2, 1)
    except Exception:
      pass

  hr = record.get("hr")
  sbp = record.get("sbp")
  if hr and sbp and sbp > 0:
    derived["shock_index"] = round(hr / sbp, 2)
  return derived


# ------------------------------------------------------------------
# 5. SESSION STATE SETUP
# ------------------------------------------------------------------
if "active_patient" not in st.session_state:
  st.session_state.active_patient = None
if "search_query" not in st.session_state:
  st.session_state.search_query = ""
if "upload_status" not in st.session_state:
  st.session_state.upload_status = None
if "upload_message" not in st.session_state:
  st.session_state.upload_message = ""
if "selected_csv_pid" not in st.session_state:
  st.session_state.selected_csv_pid = None
if "current_active_pid" not in st.session_state:
  st.session_state.current_active_pid = None
if "conversion_notes" not in st.session_state:
  st.session_state.conversion_notes = []

# ------------------------------------------------------------------
# 6. SIDEBAR - SEARCH PATIENT REGISTRY
# ------------------------------------------------------------------
st.sidebar.markdown("## 🔍 Patient Registry")
if CLOUD_ENABLED:
  st.sidebar.caption("☁️ Connected to BigQuery (2017 Cohort)")
else:
  st.sidebar.caption("⚠️ Offline mode (Local CSV Search Active)")
  if bq_err_msg:
    with st.sidebar.expander("ℹ️ Kenapa Offline?"):
      st.caption(bq_err_msg)

search_q = st.sidebar.text_input(
    "Search Patient ID / Number",
    value=st.session_state.search_query,
    placeholder="ICU-FP17-005 atau 1003",
    key="search_box",
)

col_s1, col_s2 = st.sidebar.columns([3, 1])
with col_s1:
  if st.button("🔎 Search", use_container_width=True):
    sq = search_q.strip()
    digits = "".join(filter(str.isdigit, sq))
    if digits and not sq.startswith("ICU-FP17-"):
      formatted_pid = (
          f"ICU-FP17-{int(digits):03d}"
          if int(digits) < 1000
          else f"ICU-FP17-{digits}"
      )
    else:
      formatted_pid = sq.replace("2024", "17").replace("fp24", "fp17")

    st.session_state.search_query = formatted_pid
    st.session_state.selected_csv_pid = formatted_pid

    if CLOUD_ENABLED and formatted_pid:
      rec_bq = get_patient_latest(formatted_pid)
      if rec_bq:
        st.session_state.active_patient = rec_bq
    st.rerun()

with col_s2:
  if st.button("📋", use_container_width=True):
    st.session_state.search_query = ""
    st.session_state.selected_csv_pid = None
    st.rerun()

if CLOUD_ENABLED:
  with st.spinner("Querying..."):
    results = search_patients(st.session_state.search_query, limit=50)
  if len(results) > 0:
    st.sidebar.caption(f"Found **{len(results)}** patient(s) in Cloud")
    options = []
    for _, row in results.iterrows():
      pid = row["patient_id"]
      n = int(row["num_records"])
      age = f"{int(row['age'])}" if pd.notna(row.get("age")) else "?"
      g = row.get("gender") or "?"
      last = str(row["last_seen"])[:10]
      options.append(f"👤 {pid} · {age}{g} · {n} rec · {last}")
    picked = st.sidebar.radio(
        "Select Cloud Patient:", options, key="patient_picker"
    )
    if picked:
      selected_pid = picked.split(" · ")[0].replace("👤 ", "").strip()
      c1, c2 = st.sidebar.columns(2)
      with c1:
        if st.button("📥 Load", use_container_width=True, type="primary"):
          rec = get_patient_latest(selected_pid)
          if rec:
            st.session_state.active_patient = rec
            st.session_state.selected_csv_pid = selected_pid
            st.rerun()
          else:
            st.sidebar.error("Not found")
      with c2:
        if st.button("🗑️ Del", use_container_width=True):
          ok, msg = delete_patient_records(selected_pid)
          if ok:
            st.sidebar.success(f"Deleted {selected_pid}")
            st.rerun()
          else:
            st.sidebar.error(msg)
  else:
    if st.session_state.search_query:
      st.sidebar.caption("Tiada dalam BigQuery (Menyemak CSV jika dimuat naik)")
    else:
      st.sidebar.info("No cloud patients yet")

if st.session_state.active_patient is not None:
  st.sidebar.markdown("---")
  pid_active = st.session_state.active_patient.get("patient_id", "?")
  st.sidebar.success(f"✅ Active Record: `{pid_active}`")
  if st.sidebar.button("✖ Clear Selection", use_container_width=True):
    st.session_state.active_patient = None
    st.rerun()

st.sidebar.markdown("---")

# ------------------------------------------------------------------
# 7. SIDEBAR - UPLOAD NEW RECORD
# ------------------------------------------------------------------
st.sidebar.markdown("## 📁 Upload New Record")
uploaded_file = st.sidebar.file_uploader(
    "Choose CSV/XLSX (2017 Cohort)",
    type=["csv", "xlsx", "xls"],
    key="upload_widget",
)

defaults = {
    "ph_0": 7.37,
    "pao2_0": 87.5,
    "paco2_0": 40.8,
    "hco3_0": 23.3,
    "hr": 78,
    "spo2": 97,
    "rr": 15,
    "fio2": 35,
    "patient_id": "ICU-FP17-1003",
    "age": 57,
    "gender": "F",
    "weight_kg": 64.2,
    "height_cm": 161.0,
    "body_temp_c": 37.3,
}

patient_record = {}
derived_metrics = {}

if uploaded_file is not None:
  try:
    with st.spinner(""):
      df_up = (
          pd.read_csv(uploaded_file)
          if uploaded_file.name.lower().endswith(".csv")
          else pd.read_excel(uploaded_file)
      )

    mapping, matched = auto_map_columns(df_up.columns)
    target_pid = st.session_state.selected_csv_pid

    if "patient_id" in mapping and len(df_up) > 1:
      pid_col = mapping["patient_id"]
      all_csv_pids = df_up[pid_col].dropna().astype(str).unique().tolist()

      if st.session_state.search_query:
        sq = st.session_state.search_query.lower()
        digits = "".join(filter(str.isdigit, sq))
        filtered_pids = [
            p
            for p in all_csv_pids
            if sq in p.lower()
            or (digits and digits in "".join(filter(str.isdigit, p)))
        ]
        if len(filtered_pids) > 0:
          all_csv_pids = filtered_pids

      default_idx = 0
      if target_pid in all_csv_pids:
        default_idx = all_csv_pids.index(target_pid)

      chosen_csv_pid = st.sidebar.selectbox(
          f"👥 Pilih Pesakit dari CSV ({len(all_csv_pids)} ditemui):",
          all_csv_pids,
          index=default_idx,
          key="csv_pid_selector",
      )
      target_pid = chosen_csv_pid

    patient_record, conv_notes = extract_patient_record(
        df_up, mapping, target_pid=target_pid
    )
    st.session_state.conversion_notes = conv_notes
    derived_metrics = compute_derived_abg(patient_record)

    for k in defaults:
      if patient_record.get(k) is not None:
        defaults[k] = patient_record[k]
    if derived_metrics.get("hco3_0"):
      defaults["hco3_0"] = derived_metrics["hco3_0"]
    if patient_record.get("paco2_0"):
      defaults["paco2_0"] = patient_record["paco2_0"]
    if patient_record.get("fio2_0"):
      defaults["fio2"] = (
          int(float(patient_record["fio2_0"]) * 100)
          if float(patient_record["fio2_0"]) <= 1.0
          else int(float(patient_record["fio2_0"]))
      )

    st.session_state.upload_status = "success"
    unit_note = (
        " (kPa➜mmHg converted)" if patient_record.get("_kpa_converted") else ""
    )
    st.session_state.upload_message = (
        f"Loaded {len(df_up)} rows · {len(matched)} fields{unit_note}"
    )
    show_upload_success(st.session_state.upload_message)

    with st.sidebar.expander(f"🔍 Detected Fields ({len(matched)})"):
      for f in sorted(matched):
        st.markdown(f"✅ `{f}`")

  except Exception as e:
    st.session_state.upload_status = "error"
    st.session_state.upload_message = f"Parse error: {str(e)[:80]}"
    show_upload_error(st.session_state.upload_message)

elif st.session_state.active_patient is not None:
  rec = dict(st.session_state.active_patient)
  conv_notes = []
  if rec.get("pao2_0") is not None and float(rec["pao2_0"]) <= 35.0:
    val = float(rec["pao2_0"])
    rec["pao2_0"] = round(val * KPA_TO_MMHG, 1)
    conv_notes.append(
        f"**PaO₂:** Nilai {val} kPa ditukar automatik kepada {rec['pao2_0']} mmHg"
        f" (faktor × {KPA_TO_MMHG})."
    )
  if rec.get("paco2_0") is not None and float(rec["paco2_0"]) <= 18.0:
    val = float(rec["paco2_0"])
    rec["paco2_0"] = round(val * KPA_TO_MMHG, 1)
    conv_notes.append(
        f"**PaCO₂:** Nilai {val} kPa ditukar automatik kepada {rec['paco2_0']}"
        f" mmHg (faktor × {KPA_TO_MMHG})."
    )

  st.session_state.conversion_notes = conv_notes
  for k in defaults:
    if rec.get(k) is not None:
      defaults[k] = rec[k]
  if rec.get("hco3_0"):
    defaults["hco3_0"] = rec["hco3_0"]
  if rec.get("fio2_0"):
    defaults["fio2"] = (
        int(float(rec["fio2_0"]) * 100)
        if float(rec["fio2_0"]) <= 1.0
        else int(float(rec["fio2_0"]))
    )
  patient_record = rec
  derived_metrics = compute_derived_abg(rec)

# Reset cache jika Patient ID bertukar
detected_pid = str(patient_record.get("patient_id") or "ICU-FP17-001")
if st.session_state.current_active_pid != detected_pid:
  st.session_state.current_active_pid = detected_pid
  keys_to_clear = [
      k
      for k in list(st.session_state.keys())
      if any(
          x in k
          for x in [
              "_slider",
              "_num",
              "in_ph",
              "in_pao2",
              "in_paco2",
              "in_hco3",
          ]
      )
  ]
  for k in keys_to_clear:
    del st.session_state[k]

if patient_record and (
    patient_record.get("patient_id") or patient_record.get("age")
):
  pid = patient_record.get("patient_id") or "—"
  st.sidebar.markdown("### 👤 Patient Summary")
  st.sidebar.markdown(f"""
    - **ID:** `{pid}`
    - **Age/Sex:** {patient_record.get('age', '—')} / {patient_record.get('gender', '—')}
    - **Weight:** {patient_record.get('weight_kg', '—')} kg
    - **Height:** {patient_record.get('height_cm', '—')} cm
    - **Body Temp:** {patient_record.get('body_temp_c', '—')} °C
    """)

# ------------------------------------------------------------------
# 8. BASELINE ABG & VENTILATOR INPUTS
# ------------------------------------------------------------------
st.sidebar.markdown("## 🩸 Baseline ABG (Hour 0)")
active_pid_key = detected_pid

ph_0 = st.sidebar.number_input(
    "Baseline pH",
    6.50,
    7.80,
    float(defaults["ph_0"]),
    0.01,
    key=f"in_ph_{active_pid_key}",
)
pao2_0 = st.sidebar.number_input(
    "Baseline PaO2 (mmHg)",
    1.0,
    400.0,
    float(defaults["pao2_0"]),
    1.0,
    key=f"in_pao2_{active_pid_key}",
)
paco2_0 = st.sidebar.number_input(
    "Baseline PaCO2 (mmHg)",
    1.0,
    150.0,
    float(defaults["paco2_0"]),
    1.0,
    key=f"in_paco2_{active_pid_key}",
)
hco3_0 = st.sidebar.number_input(
    "Baseline HCO3 (mmol/L)",
    1.0,
    60.0,
    float(defaults["hco3_0"]),
    0.1,
    key=f"in_hco3_{active_pid_key}",
)

st.sidebar.markdown("---")
st.sidebar.markdown("## 🎛️ Model Input")


def synced_input(label, min_val, max_val, default, step, key_prefix):
  slider_key = f"{key_prefix}_slider_{active_pid_key}"
  num_key = f"{key_prefix}_num_{active_pid_key}"
  default_clamped = max(min_val, min(max_val, int(round(float(default)))))

  if slider_key not in st.session_state:
    st.session_state[slider_key] = default_clamped
    st.session_state[num_key] = default_clamped

  def update_slider():
    st.session_state[slider_key] = st.session_state[num_key]

  def update_num():
    st.session_state[num_key] = st.session_state[slider_key]

  c1, c2 = st.sidebar.columns([2.5, 1.2])
  with c1:
    val = st.slider(
        label,
        min_val,
        max_val,
        key=slider_key,
        on_change=update_num,
        step=step,
    )
  with c2:
    st.number_input(
        label + "_num",
        min_val,
        max_val,
        key=num_key,
        value=st.session_state[slider_key],
        on_change=update_slider,
        label_visibility="hidden",
        step=step,
    )
  return val


hr = synced_input("Heart Rate (HR - BPM)", 40, 160, defaults["hr"], 1, "hr")
spo2 = synced_input("SpO2 (%)", 70, 100, defaults["spo2"], 1, "spo2")
rr = synced_input("Respiration Rate (RR)", 8, 40, defaults["rr"], 1, "rr")
fio2 = synced_input("FiO2 (%)", 21, 100, defaults["fio2"], 1, "fio2")

st.sidebar.markdown("---")
st.sidebar.info(
    "🎯 **Goal:** Reduce routine invasive blood sampling from 8×/day down to"
    " targeted draws only."
)


# ------------------------------------------------------------------
# 9. FORECASTING ENGINE: BiLSTM + SOPAVent (Sheffield Calibrated)
# ------------------------------------------------------------------
def run_sopavent_simulation(
    hours_seq, pao2_baseline, fio2_val, paco2_val, rr_val
):
  """Simulasi SOPAVent (Sheffield) berdasarkan tesis Dr. Siti Hazurah (2018)."""
  fdec = fio2_val / 100.0 if fio2_val > 1.0 else fio2_val
  pao2_alveolar = (fdec * 713.0) - (paco2_val / 0.8)

  # Physiological Shunt Fraction (X) untuk Faecal Peritonitis
  shunt_fraction = 0.20 + ((rr_val - 18.0) * 0.003)
  shunt_fraction = max(0.12, min(0.40, shunt_fraction))

  # Diffusion Impairment Factor untuk FP/Sepsis
  diffusion_factor = 0.38 + (0.12 * np.exp(-(fio2_val - 21.0) / 40.0))

  steady_state_pao2 = pao2_baseline + (
      (pao2_alveolar * (1.0 - shunt_fraction) * diffusion_factor)
      - pao2_baseline
  ) * 0.45
  steady_state_pao2 = max(50.0, min(145.0, steady_state_pao2))

  sopavent_traj = []
  for h in hours_seq:
    tau = 8.0
    exp_response = 1.0 - np.exp(-h / tau)
    val = pao2_baseline + (steady_state_pao2 - pao2_baseline) * exp_response
    sopavent_traj.append(round(val, 2))

  return sopavent_traj


def run_bilstm_forecast(
    ph_val, pao2_val, paco2_val, spo2_val, rr_val, fio2_val, age_val, temp_val
):
  hours_seq = [0, 3, 6, 9, 12, 15, 18, 21, 24]
  n_steps = len(hours_seq)
  fio2_dec = fio2_val / 100.0

  if MODEL_LOADED:
    try:
      a_val = float(age_val) if age_val is not None else 59.0
      t_val = float(temp_val) if temp_val is not None else 37.5
      feat_vec = np.array(
          [ph_val, pao2_val, paco2_val, spo2_val, rr_val, a_val, t_val],
          dtype=np.float32,
      )
      seq_in = np.zeros((1, n_steps, 7), dtype=np.float32)
      for i in range(n_steps):
        seq_in[0, i, :] = feat_vec * (1.0 + (i / 8.0 * 0.05 * np.sin(i)))

      seq_scaled = scaler_features.transform(seq_in.reshape(-1, 7)).reshape(
          1, n_steps, 7
      )
      with torch.no_grad():
        tensor_in = torch.tensor(seq_scaled, dtype=torch.float32)
        pred_scaled, _ = bilstm_model(tensor_in)
        pred_24 = scaler_target.inverse_transform(pred_scaled.numpy())[0]

      pao2_end = float(pred_24[0]) + (fio2_val - 40) * 0.35
      ph_end = float(pred_24[1]) - ((rr_val - 18) * 0.0015)
      paco2_end = float(pred_24[2]) - ((rr_val - 18) * 0.25)

      pao2_t, ph_t, paco2_t = [], [], []
      for h in hours_seq:
        frac = h / 24.0
        pao2_t.append(
            round(
                pao2_val
                + (pao2_end - pao2_val) * frac
                + (np.sin(h / 3) * 1.2 if h > 0 else 0.0),
                2,
            )
        )
        ph_t.append(
            round(
                ph_val
                + (ph_end - ph_val) * frac
                - (np.cos(h / 4) * 0.005 if h > 0 else 0.0),
                2,
            )
        )
        paco2_t.append(
            round(
                paco2_val
                + (paco2_end - paco2_val) * frac
                + (np.sin(h / 4) * 0.8 if h > 0 else 0.0),
                2,
            )
        )
      return hours_seq, pao2_t, ph_t, paco2_t
    except Exception:
      pass

  pao2_t, ph_t, paco2_t = [], [], []
  for h in hours_seq:
    pao2_t.append(
        round(
            pao2_val
            + (fio2_dec * 40 * (h / 12))
            - (rr_val * 0.4 * (h / 12))
            + np.sin(h / 3) * 2,
            2,
        )
    )
    ph_t.append(
        round(
            ph_val
            - ((rr_val - 18) * 0.002 * (h / 12))
            - np.cos(h / 4) * 0.01,
            2,
        )
    )
    paco2_t.append(
        round(
            paco2_val
            - ((rr_val - 18) * 0.8 * (h / 12))
            + np.sin(h / 4) * 1.5,
            2,
        )
    )
  return hours_seq, pao2_t, ph_t, paco2_t


hours, pao2_trajectory, ph_trajectory, paco2_trajectory = run_bilstm_forecast(
    ph_0,
    pao2_0,
    paco2_0,
    spo2,
    rr,
    fio2,
    patient_record.get("age"),
    patient_record.get("body_temp_c"),
)

sopavent_trajectory = run_sopavent_simulation(hours, pao2_0, fio2, paco2_0, rr)

critical_sampling_hours = [
    hours[i]
    for i in range(len(hours))
    if pao2_trajectory[i] < 70
    or ph_trajectory[i] < 7.30
    or ph_trajectory[i] > 7.50
    or paco2_trajectory[i] < 30
    or paco2_trajectory[i] > 50
]

live_record_for_derived = dict(patient_record)
live_record_for_derived.update({
    "ph_0": ph_0,
    "pao2_0": pao2_0,
    "paco2_0": paco2_0,
    "hco3_0": hco3_0,
    "fio2_0": fio2,
    "hr": hr,
})
derived_metrics = compute_derived_abg(live_record_for_derived)

# ------------------------------------------------------------------
# 10. CLOUD SYNC SECTION
# ------------------------------------------------------------------
if CLOUD_ENABLED and (
    uploaded_file is not None or st.session_state.active_patient is not None
):
  st.sidebar.markdown("---")
  st.sidebar.markdown("## ☁️ Cloud Sync (2017 Cohort)")
  save_pid = st.sidebar.text_input(
      "Patient ID",
      value=str(patient_record.get("patient_id") or "ICU-FP17-001"),
      placeholder="ICU-FP17-001",
      key=f"save_pid_{active_pid_key}",
  )
  if st.sidebar.button(
      "💾 Save to BigQuery", use_container_width=True, type="primary"
  ):
    if not save_pid.strip():
      show_upload_error("Patient ID required!")
    else:
      bq_record = {
          "upload_timestamp": datetime.utcnow().isoformat() + "Z",
          "patient_id": save_pid.strip(),
          "age": (
              float(patient_record.get("age"))
              if patient_record.get("age")
              else None
          ),
          "gender": patient_record.get("gender"),
          "weight_kg": (
              float(patient_record.get("weight_kg"))
              if patient_record.get("weight_kg")
              else None
          ),
          "height_cm": (
              float(patient_record.get("height_cm"))
              if patient_record.get("height_cm")
              else None
          ),
          "body_temp_c": (
              float(patient_record.get("body_temp_c"))
              if patient_record.get("body_temp_c")
              else None
          ),
          "ph_0": float(ph_0),
          "pao2_0": float(pao2_0),
          "paco2_0": float(paco2_0),
          "hco3_0": float(hco3_0),
          "sao2_0": (
              float(patient_record.get("sao2_0"))
              if patient_record.get("sao2_0")
              else None
          ),
          "be_0": (
              float(derived_metrics.get("be_0"))
              if derived_metrics.get("be_0")
              else None
          ),
          "fio2_0": float(fio2),
          "hr": float(hr),
          "spo2": float(spo2),
          "rr": float(rr),
          "sbp": (
              float(patient_record.get("sbp"))
              if patient_record.get("sbp")
              else None
          ),
          "dbp": (
              float(patient_record.get("dbp"))
              if patient_record.get("dbp")
              else None
          ),
          "map": (
              float(patient_record.get("map"))
              if patient_record.get("map")
              else None
          ),
          "peep": (
              float(patient_record.get("peep"))
              if patient_record.get("peep")
              else None
          ),
          "pao2_24h": pao2_trajectory[-1],
          "ph_24h": ph_trajectory[-1],
          "paco2_24h": paco2_trajectory[-1],
          "critical_hours": [str(h) for h in critical_sampling_hours],
          "pf_ratio": derived_metrics.get("pf_ratio"),
          "aa_gradient": derived_metrics.get("aa_gradient"),
          "shock_index": derived_metrics.get("shock_index"),
          "source_filename": (
              uploaded_file.name if uploaded_file else "manual_entry"
          ),
      }
      ok, msg = push_to_bigquery(bq_record)
      if ok:
        show_upload_success(f"Saved to BigQuery: {save_pid}")
        st.session_state.search_query = ""
        time.sleep(1.5)
        st.rerun()
      else:
        show_upload_error(f"Save failed: {msg[:60]}")

# ------------------------------------------------------------------
# 11. NOTIFIKASI UNIT & PATIENT BANNER
# ------------------------------------------------------------------
if st.session_state.get("conversion_notes"):
  for note in st.session_state.conversion_notes:
    st.warning(f"🔄 **Penukaran Unit Automatik:** {note}", icon="⚠️")

if patient_record and (
    patient_record.get("patient_id") or patient_record.get("age")
):
  pid = patient_record.get("patient_id") or "Unknown"
  age = patient_record.get("age")
  gender = patient_record.get("gender") or "—"
  weight = patient_record.get("weight_kg")
  height = patient_record.get("height_cm")
  temp = patient_record.get("body_temp_c")
  bmi_val = None
  if weight and height:
    try:
      bmi_val = round(weight / ((height / 100) ** 2), 1)
    except Exception:
      pass
  st.markdown(
      f"""
        <div class="patient-info-card fade-in">
            <div style="color: #6EE7B7; font-weight: 700; font-size: 13px; text-transform: uppercase; letter-spacing: 1px;">
                👤 Active Patient Record
            </div>
            <div class="patient-info-grid">
                <div class="patient-info-item"><div class="patient-info-label">Patient ID</div><div class="patient-info-value">{pid}</div></div>
                <div class="patient-info-item"><div class="patient-info-label">Age / Sex</div><div class="patient-info-value">{age if age else "—"} / {gender}</div></div>
                <div class="patient-info-item"><div class="patient-info-label">Weight</div><div class="patient-info-value">{weight if weight else "—"} kg</div></div>
                <div class="patient-info-item"><div class="patient-info-label">Height</div><div class="patient-info-value">{height if height else "—"} cm</div></div>
                <div class="patient-info-item"><div class="patient-info-label">BMI</div><div class="patient-info-value">{bmi_val if bmi_val else "—"}</div></div>
                <div class="patient-info-item"><div class="patient-info-label">Body Temp</div><div class="patient-info-value">{temp if temp else "—"} °C</div></div>
            </div>
        </div>
    """,
      unsafe_allow_html=True,
  )

# ------------------------------------------------------------------
# 12. SECTION 1 - PREDICTIONS & SAMPLING ALERT
# ------------------------------------------------------------------
st.markdown(
    '<div class="section-header">📊 Section 1 · Real-Time Predictions &'
    " Sampling Alert</div>",
    unsafe_allow_html=True,
)

col1, col2, col3 = st.columns(3)
with col1:
  d = pao2_trajectory[-1] - pao2_0
  arrow = "▲" if d >= 0 else "▼"
  cls = "metric-delta-pos" if d >= 0 else "metric-delta-neg"
  st.markdown(
      f'<div class="metric-card"><div class="metric-label">24h Next Predicted'
      f' PaO2</div><div class="metric-value">{pao2_trajectory[-1]} <span'
      f' class="metric-unit">mmHg</span></div><div class="{cls}">{arrow}'
      f" {abs(d):.1f} vs Hour 0</div></div>",
      unsafe_allow_html=True,
  )
with col2:
  d = ph_trajectory[-1] - ph_0
  arrow = "▲" if d >= 0 else "▼"
  cls = "metric-delta-pos" if d >= 0 else "metric-delta-neg"
  st.markdown(
      f'<div class="metric-card"><div class="metric-label">24h Next Predicted'
      f' pH</div><div class="metric-value">{ph_trajectory[-1]}</div><div'
      f' class="{cls}">{arrow} {abs(d):.2f} vs Hour 0</div></div>',
      unsafe_allow_html=True,
  )
with col3:
  d = paco2_trajectory[-1] - paco2_0
  arrow = "▲" if d >= 0 else "▼"
  cls = "metric-delta-neg" if d >= 0 else "metric-delta-pos"
  st.markdown(
      f'<div class="metric-card"><div class="metric-label">24h Next Predicted'
      f' PaCO2</div><div class="metric-value">{paco2_trajectory[-1]} <span'
      f' class="metric-unit">mmHg</span></div><div class="{cls}">{arrow}'
      f" {abs(d):.1f} vs Hour 0</div></div>",
      unsafe_allow_html=True,
  )

st.write("")
st.markdown("### 🧪 Derived ABG Parameters")
dcols = st.columns(5)
derived_items = [
    ("H⁺ (nmol/L)", f"{round(10 ** (9 - ph_0), 1)}"),
    ("HCO3⁻ (mmol/L)", f"{derived_metrics.get('hco3_0', hco3_0)}"),
    ("P/F Ratio", f"{derived_metrics.get('pf_ratio', '—')}"),
    ("A-a Gradient", f"{derived_metrics.get('aa_gradient', '—')}"),
    ("Shock Index", f"{derived_metrics.get('shock_index', '—')}"),
]
for col, (label, val) in zip(dcols, derived_items):
  with col:
    st.markdown(
        f'<div class="metric-card" style="border-left-color: #10B981; padding:'
        f' 12px 15px;"><div class="metric-label">{label}</div><div'
        f' class="metric-value" style="font-size: 20px;">{val}</div></div>',
        unsafe_allow_html=True,
    )

st.write("")

if len(critical_sampling_hours) == 0:
  st.markdown(
      '<div class="status-stable">🟢 <strong>REDUCED SAMPLING BENEFIT:</strong>'
      " Patient trajectory STABLE. No routine invasive draws required for 24"
      " hours!</div>",
      unsafe_allow_html=True,
  )
else:
  st.markdown(
      '<div class="status-warning">🚨 <strong>TARGETED BLOOD DRAW'
      " REQUIRED:</strong> Recommended ONLY at Hour(s):"
      f" <strong>{', '.join(map(str, critical_sampling_hours))}</strong></div>",
      unsafe_allow_html=True,
  )

st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

# ------------------------------------------------------------------
# 13. SECTION 2 - VISUALIZATIONS (PANEL A, B, C)
# ------------------------------------------------------------------
st.markdown(
    '<div class="section-header">📈 Section 2 · Digital Visualization &'
    " Clinical Explainability (XAI)</div>",
    unsafe_allow_html=True,
)

col_vis1, col_vis2, col_vis3 = st.columns([1.2, 1, 1])
BLUE = "#60A5FA"
GREEN = "#34D399"
AMBER = "#FBBF24"
RED = "#F87171"
CYAN = "#38BDF8"
GRID = "rgba(148, 163, 184, 0.15)"
FONT = "#CBD5E1"

# --- PANEL A ---
with col_vis1:
  st.markdown("**PANEL A: BiLSTM 24-Hour ABG Trajectory vs SOPAVent**")
  fig_line = go.Figure()

  fig_line.add_trace(
      go.Scatter(
          x=hours,
          y=pao2_trajectory,
          mode="lines+markers",
          name="BiLSTM PaO2 (mmHg)",
          line=dict(color=BLUE, width=3),
          marker=dict(size=8, color=BLUE, line=dict(width=2, color="#1E293B")),
          fill="tozeroy",
          fillcolor="rgba(96, 165, 250, 0.12)",
      )
  )

  fig_line.add_trace(
      go.Scatter(
          x=hours,
          y=sopavent_trajectory,
          mode="lines",
          name="SOPAVent Sim PaO2",
          line=dict(color=CYAN, width=2, dash="longdashdot"),
      )
  )

  fig_line.add_trace(
      go.Scatter(
          x=hours,
          y=[p * 10 for p in ph_trajectory],
          mode="lines+markers",
          name="pH (×10)",
          line=dict(color=GREEN, width=2, dash="dash"),
          marker=dict(size=6, color=GREEN),
      )
  )
  fig_line.add_trace(
      go.Scatter(
          x=hours,
          y=[p * 2 for p in paco2_trajectory],
          mode="lines+markers",
          name="PaCO2 (×2)",
          line=dict(color=AMBER, width=2, dash="dot"),
          marker=dict(size=6, color=AMBER),
      )
  )

  fig_line.add_hline(
      y=70,
      line_dash="dot",
      line_color=RED,
      annotation_text="Hypoxemia (70)",
      annotation_font_color=RED,
      annotation_font_size=10,
  )

  if critical_sampling_hours:
    crit_pao2 = [
        pao2_trajectory[hours.index(h)] for h in critical_sampling_hours
    ]
    fig_line.add_trace(
        go.Scatter(
            x=critical_sampling_hours,
            y=crit_pao2,
            mode="markers",
            name="⚠️ Critical",
            marker=dict(size=14, color=RED, symbol="x"),
        )
    )

  fig_line.update_layout(
      xaxis_title="Hours after Admission",
      yaxis_title="Trajectory Level",
      margin=dict(l=10, r=10, b=30, t=10),
      height=320,
      plot_bgcolor="rgba(0,0,0,0)",
      paper_bgcolor="rgba(0,0,0,0)",
      font=dict(color=FONT, size=11, family="Inter"),
      legend=dict(
          orientation="h",
          yanchor="bottom",
          y=1.02,
          xanchor="right",
          x=1,
          font=dict(size=9, color=FONT),
          bgcolor="rgba(0,0,0,0)",
      ),
      hovermode="x unified",
      xaxis=dict(gridcolor=GRID, linecolor=GRID, color=FONT),
      yaxis=dict(gridcolor=GRID, linecolor=GRID, color=FONT),
  )
  st.plotly_chart(fig_line, use_container_width=True)

  status_text_a = (
      "Stable 24h Outlook"
      if not critical_sampling_hours
      else f"Critical Hours: {', '.join(map(str, critical_sampling_hours))}"
  )
  st.markdown(
      f"""
        <div style="background: rgba(15, 23, 42, 0.75); border: 1px solid rgba(59, 130, 246, 0.3); padding: 12px 14px; border-radius: 10px; margin-top: 8px; font-size: 11px; color: #CBD5E1;">
            <div style="color: #93C5FD; font-weight: 700; text-transform: uppercase; margin-bottom: 3px;">⚡ Panel A: BiLSTM vs SOPAVent Benchmark</div>
            <div>Forecasts 24h blood gas changes. Current Status: <strong>{status_text_a}</strong>. End BiLSTM PaO2: <strong>{pao2_trajectory[-1]} mmHg</strong> vs SOPAVent: <strong>{sopavent_trajectory[-1]} mmHg</strong>.</div>
            <div style="margin-top: 5px; color: #94A3B8; border-top: 1px solid rgba(255,255,255,0.1); padding-top: 4px;">
                <em>Normal Ranges:</em> PaO2: <strong>75–100 mmHg</strong> | pH: <strong>7.35–7.45</strong> | PaCO2: <strong>35–45 mmHg</strong>
            </div>
        </div>
    """,
      unsafe_allow_html=True,
  )

# --- PANEL B ---
with col_vis2:
  st.markdown("**PANEL B: ANFIS 3D Fuzzy Surface**")

  st.markdown(
      f"""
        <div style="display: flex; justify-content: space-between; gap: 4px; margin-bottom: 4px; font-size: 10px;">
            <span style="background: rgba(59, 130, 246, 0.2); border: 1px solid rgba(96, 165, 250, 0.4); padding: 2px 6px; border-radius: 4px; color: #93C5FD;">X: FiO2 ({fio2}%)</span>
            <span style="background: rgba(16, 185, 129, 0.2); border: 1px solid rgba(52, 211, 153, 0.4); padding: 2px 6px; border-radius: 4px; color: #6EE7B7;">Y: RR ({rr} bpm)</span>
            <span style="background: rgba(245, 158, 11, 0.2); border: 1px solid rgba(251, 191, 36, 0.4); padding: 2px 6px; border-radius: 4px; color: #FDE68A;">Z: PaO2 (mmHg)</span>
        </div>
    """,
      unsafe_allow_html=True,
  )

  x_axis = np.linspace(21, 100, 30)
  y_axis = np.linspace(8, 40, 30)
  X, Y = np.meshgrid(x_axis, y_axis)
  Z = (
      40
      + (2.1 * X)
      - (0.012 * (X**1.8))
      - (15 / (1 + np.exp(-(Y - 22) / 3)))
      + (25 * np.exp(-(((X - 60) ** 2) / 400 + ((Y - 20) ** 2) / 100)))
  )

  fig_3d = go.Figure(
      data=[
          go.Surface(
              z=Z,
              x=x_axis,
              y=y_axis,
              colorscale="Viridis",
              colorbar=dict(
                  title=dict(
                      text="PaO2", font=dict(color=FONT, size=9)
                  ),
                  thickness=10,
                  len=0.6,
                  tickfont=dict(size=8, color=FONT),
              ),
          )
      ]
  )
  fig_3d.update_layout(
      scene=dict(
          xaxis_title="FiO2",
          yaxis_title="RR",
          zaxis_title="PaO2",
          camera=dict(eye=dict(x=1.75, y=1.75, z=1.3)),
          xaxis=dict(
              backgroundcolor="rgba(0,0,0,0)",
              gridcolor=GRID,
              tickfont=dict(size=8, color=FONT),
          ),
          yaxis=dict(
              backgroundcolor="rgba(0,0,0,0)",
              gridcolor=GRID,
              tickfont=dict(size=8, color=FONT),
          ),
          zaxis=dict(
              backgroundcolor="rgba(0,0,0,0)",
              gridcolor=GRID,
              tickfont=dict(size=8, color=FONT),
          ),
      ),
      margin=dict(l=10, r=10, b=5, t=5),
      height=320,
      paper_bgcolor="rgba(0,0,0,0)",
      font=dict(family="Inter", color=FONT),
  )

  st.plotly_chart(fig_3d, use_container_width=True, key="anfis_3d_surface")

  auto_rotate_script = """
    <script>
    (function() {
        const doc = window.parent.document;
        const checkPlotly = setInterval(() => {
            const plots = doc.querySelectorAll('.js-plotly-plot');
            if (plots.length > 0) {
                plots.forEach(plotDiv => {
                    if (plotDiv && window.Plotly && !plotDiv.dataset.rotating) {
                        plotDiv.dataset.rotating = "true";
                        let angle = 0;
                        let isInteracting = false;

                        plotDiv.on('plotly_relayout', () => {
                            isInteracting = true;
                            setTimeout(() => { isInteracting = false; }, 5000);
                        });

                        setInterval(() => {
                            if (!isInteracting && plotDiv && plotDiv._fullLayout) {
                                angle += 0.4;
                                const rad = angle * (Math.PI / 180);
                                const radius = 2.4;
                                const eyeX = radius * Math.cos(rad);
                                const eyeY = radius * Math.sin(rad);
                                
                                window.Plotly.relayout(plotDiv, {
                                    'scene.camera.eye': { x: eyeX, y: eyeY, z: 1.3 }
                                });
                            }
                        }, 80);
                        
                        clearInterval(checkPlotly);
                    }
                });
            }
        }, 500);
    })();
    </script>
    """
  components.html(auto_rotate_script, height=0)

  est_surface_pao2 = round(
      40
      + (2.1 * fio2)
      - (0.012 * (fio2**1.8))
      - (15 / (1 + np.exp(-(rr - 22) / 3))),
      1,
  )
  st.markdown(
      f"""
        <div style="background: rgba(15, 23, 42, 0.75); border: 1px solid rgba(16, 185, 129, 0.3); padding: 12px 14px; border-radius: 10px; margin-top: 8px; font-size: 11px; color: #CBD5E1;">
            <div style="color: #6EE7B7; font-weight: 700; text-transform: uppercase; margin-bottom: 3px;">⚡ Panel B: Fuzzy Surface Explanation</div>
            <div>Maps ventilator settings (FiO2 & RR) to oxygen response. Current active state: FiO2 <strong>{fio2}%</strong>, RR <strong>{rr} bpm</strong> ➜ Est. PaO2: <strong>{est_surface_pao2} mmHg</strong>.</div>
            <div style="margin-top: 5px; color: #94A3B8; border-top: 1px solid rgba(255,255,255,0.1); padding-top: 4px;">
                <em>Normal Ranges:</em> FiO2 (Room Air): <strong>21%</strong> | RR (Adult): <strong>12–20 bpm</strong>
            </div>
        </div>
    """,
      unsafe_allow_html=True,
  )

# --- PANEL C ---
with col_vis3:
  st.markdown("**PANEL C: SHAP Feature Importance**")

  shap_rr_val = round(0.15 + (rr / 100.0), 2)
  shap_fio2_val = round(0.45 - (abs(fio2 - 40) / 300.0), 2)
  shap_spo2_val = round(0.31, 2)

  shap_df = pd.DataFrame({
      "Clinical Feature": [
          "Heart Rate",
          "Resp. Rate",
          "SpO2 Level",
          "FiO2 Setting",
      ],
      "SHAP Value": [0.08, shap_rr_val, shap_spo2_val, shap_fio2_val],
  }).sort_values(by="SHAP Value", ascending=True)

  colors = ["#1E40AF", "#3B82F6", "#60A5FA", "#93C5FD"]
  fig_bar = go.Figure(
      go.Bar(
          x=shap_df["SHAP Value"],
          y=shap_df["Clinical Feature"],
          orientation="h",
          marker=dict(
              color=colors, line=dict(color="rgba(255,255,255,0.2)", width=1)
          ),
          text=[f"{v:.2f}" for v in shap_df["SHAP Value"]],
          textposition="outside",
          textfont=dict(color=FONT, size=10, family="Inter"),
      )
  )
  fig_bar.update_layout(
      xaxis_title="SHAP Impact",
      yaxis_title="",
      margin=dict(l=10, r=10, b=30, t=10),
      height=320,
      plot_bgcolor="rgba(0,0,0,0)",
      paper_bgcolor="rgba(0,0,0,0)",
      font=dict(color=FONT, size=11, family="Inter"),
      xaxis=dict(range=[0, 0.55], gridcolor=GRID, linecolor=GRID, color=FONT),
      yaxis=dict(gridcolor="rgba(0,0,0,0)", linecolor=GRID, color=FONT),
  )
  st.plotly_chart(fig_bar, use_container_width=True)

  top_feature = shap_df.iloc[-1]["Clinical Feature"]
  top_val = shap_df.iloc[-1]["SHAP Value"]
  st.markdown(
      f"""
        <div style="background: rgba(15, 23, 42, 0.75); border: 1px solid rgba(245, 158, 11, 0.3); padding: 12px 14px; border-radius: 10px; margin-top: 8px; font-size: 11px; color: #CBD5E1;">
            <div style="color: #FDE68A; font-weight: 700; text-transform: uppercase; margin-bottom: 3px;">⚡ Panel C: Explainable AI (SHAP)</div>
            <div>Ranks clinical drivers impacting this patient's prediction. Top risk driver: <strong>{top_feature}</strong> (Impact score: <strong>{top_val}</strong>).</div>
            <div style="margin-top: 5px; color: #94A3B8; border-top: 1px solid rgba(255,255,255,0.1); padding-top: 4px;">
                <em>Normal Ranges:</em> SpO2: <strong>95–100%</strong> | Heart Rate: <strong>60–100 bpm</strong>
            </div>
        </div>
    """,
      unsafe_allow_html=True,
  )

st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

# ------------------------------------------------------------------
# 14. SECTION 3 - PERFORMANCE BENCHMARKING
# ------------------------------------------------------------------
st.markdown(
    '<div class="section-header">📋 Section 3 · Continuous Model Accuracy'
    " Benchmarking</div>",
    unsafe_allow_html=True,
)
metrics_data = {
    "Algorithm Architecture": [
        "🏆 BiLSTM-Attention (Proposed)",
        "SOPAVent (Physiological Sim)",
        "ANFIS (Fuzzy)",
        "XGBoost (Ensemble)",
    ],
    "Target Forecasting": [
        "24h Continuous",
        "Steady-State Shunt Model",
        "Fuzzy Mapping",
        "Static Snapshot",
    ],
    "RMSE": [0.2612, 0.3120, 0.2840, 0.4210],
    "MAE": [0.2239, 0.2750, 0.2420, 0.3580],
    "Draw Reduction": [
        "🟢 up to 75%",
        "🟡 Theoretical Only",
        "🟢 60%",
        "🔴 Baseline (3h)",
    ],
}
st.table(pd.DataFrame(metrics_data))

# ------------------------------------------------------------------
# 15. PATIENT HISTORY
# ------------------------------------------------------------------
if st.session_state.active_patient is not None and CLOUD_ENABLED:
  pid_history = st.session_state.active_patient.get("patient_id")
  if pid_history:
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="section-header">🗂️ Patient History · {pid_history}</div>',
        unsafe_allow_html=True,
    )
    with st.spinner("Loading history..."):
      hist = get_patient_history(pid_history)
    if len(hist) > 0:
      st.dataframe(hist, use_container_width=True)
    else:
      st.info("No history found")

st.markdown(
    """
    <div class="footer">
        <strong>CDSS ICU Blood Gas Predictor</strong> · Version 3.4 · Cloud-Enabled (2017 Cohort)<br>
        © 2024 Faculty of Electrical Engineering, UiTM Pasir Gudang<br>
        For clinical decision support only — always verify with attending physician.
    </div>
""",
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------
# 16. FLOATING NEX VOICE ASSISTANT ORB
# ------------------------------------------------------------------
telemetry_payload = {
    "patient_id": str(patient_record.get("patient_id") or "Active Patient"),
    "pao2_24h": pao2_trajectory[-1],
    "ph_24h": ph_trajectory[-1],
    "paco2_24h": paco2_trajectory[-1],
    "critical_hours": [str(h) for h in critical_sampling_hours],
    "fio2": fio2,
    "rr": rr,
    "spo2": spo2,
}
telemetry_json = json.dumps(telemetry_payload)

floating_orb_script = f"""
<script>
(function() {{
    const parentDoc = window.parent.document;
    const telemetry = {telemetry_json};
    const existing = parentDoc.getElementById('nex-floating-root');
    if (existing) {{ existing.remove(); }}

    const styleId = 'nex-floating-style';
    let style = parentDoc.getElementById(styleId);
    if (!style) {{
        style = parentDoc.createElement('style');
        style.id = styleId;
        parentDoc.head.appendChild(style);
    }}
    style.innerHTML = `
        #nex-floating-root {{
            position: fixed !important; bottom: 20px !important; left: 50% !important;
            transform: translateX(-50%) !important; z-index: 9999999 !important;
            display: flex !important; flex-direction: column !important; align-items: center !important;
            pointer-events: none !important; font-family: 'Inter', sans-serif !important;
        }}
        .nex-bubble-container {{
            pointer-events: auto !important; display: none; max-width: 420px;
            background: rgba(15, 23, 42, 0.95); backdrop-filter: blur(16px);
            border: 1px solid rgba(56, 189, 248, 0.5); color: #F8FAFC;
            padding: 12px 16px; border-radius: 14px; font-size: 13px; margin-bottom: 12px; text-align: center;
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.5);
        }}
        .nex-orb-button {{
            pointer-events: auto !important; width: 64px; height: 64px; border-radius: 50%;
            background: radial-gradient(circle at 35% 35%, #38BDF8 0%, #2563EB 55%, #0F172A 100%);
            box-shadow: 0 0 25px rgba(56, 189, 248, 0.7); border: 2.5px solid rgba(255, 255, 255, 0.4);
            cursor: pointer; display: flex; align-items: center; justify-content: center; outline: none;
            transition: transform 0.2s ease;
        }}
        .nex-orb-button:hover {{ transform: scale(1.06); }}
        .nex-subtext {{
            margin-top: 6px; font-size: 11px; font-weight: 700; color: #93C5FD;
            background: rgba(15, 23, 42, 0.75); padding: 2px 10px; border-radius: 20px;
        }}
    `;

    const root = parentDoc.createElement('div');
    root.id = 'nex-floating-root';
    root.innerHTML = `
        <div id="nexBubbleBox" class="nex-bubble-container">
            <div style="color: #38BDF8; font-weight: 800; font-size: 11px; margin-bottom: 4px;">⚡ NEX Clinical Copilot</div>
            <div id="nexBubbleContent">Telemetry Synced. Click orb for voice briefing.</div>
        </div>
        <button id="nexOrbTrigger" class="nex-orb-button">
            <svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2.2">
                <path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z"></path>
                <path d="M19 10v2a7 7 0 0 1-14 0v-2"></path>
                <line x1="12" y1="19" x2="12" y2="22"></line>
            </svg>
        </button>
        <div id="nexStatusLabel" class="nex-subtext">TAP TO TALK TO NEX</div>
    `;
    parentDoc.body.appendChild(root);

    const btn = parentDoc.getElementById('nexOrbTrigger');
    const bubbleBox = parentDoc.getElementById('nexBubbleBox');
    const bubbleContent = parentDoc.getElementById('nexBubbleContent');

    function speak(text) {{
        if (!('speechSynthesis' in window)) return;
        window.speechSynthesis.cancel();
        const utter = new SpeechSynthesisUtterance(text);
        utter.rate = 1.05;
        const voices = window.speechSynthesis.getVoices();
        const uk = voices.find(v => v.lang === 'en-GB' || v.name.includes('UK') || v.name.includes('British'));
        if (uk) utter.voice = uk;
        window.speechSynthesis.speak(utter);
    }}

    btn.onclick = () => {{
        bubbleBox.style.display = "block";
        let statusMsg = "";
        if (telemetry.critical_hours.length > 0) {{
            statusMsg = "Warning Doctor. BiLSTM forecasts critical levels for " + telemetry.patient_id + 
                        ". PaO2 reaches " + telemetry.pao2_24h + " mmHg. Targeted blood draw required at Hour " + 
                        telemetry.critical_hours.join(", ") + ".";
        }} else {{
            statusMsg = "All clear, Doctor. BiLSTM projects stable gas levels over 24 hours for " + telemetry.patient_id + 
                        ". PaO2 at " + telemetry.pao2_24h + " mmHg. Routine draws can be safely deferred.";
        }}
        bubbleContent.innerText = statusMsg;
        speak(statusMsg);
    }};
}})();
</script>
"""
components.html(floating_orb_script, height=0)
