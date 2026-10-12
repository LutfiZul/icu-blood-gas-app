# test_bq.py
import streamlit as st
from google.cloud import bigquery
from google.oauth2 import service_account

st.markdown("## 🔌 BigQuery Connection Test")

try:
    creds_dict = dict(st.secrets["gcp_service_account"])
    # Fix newline if needed
    if "private_key" in creds_dict:
        creds_dict["private_key"] = creds_dict["private_key"].replace("\\n", "\n")
    
    credentials = service_account.Credentials.from_service_account_info(creds_dict)
    cfg = st.secrets["bigquery"]
    client = bigquery.Client(credentials=credentials, project=cfg["project_id"])
    
    st.success("✅ Credentials loaded!")
    
    table_id = f"{cfg['project_id']}.{cfg['dataset']}.{cfg['table']}"
    result = client.query(f"SELECT COUNT(*) as total FROM `{table_id}`").to_dataframe()
    st.success(f"✅ Query works! {result['total'][0]} rows")
    
    table = client.get_table(table_id)
    st.info(f"📊 Table `{table_id}` — {len(table.schema)} fields")
    
except Exception as e:
    st.error(f"❌ {e}")
    st.exception(e)
