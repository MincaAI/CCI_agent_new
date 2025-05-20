import gspread
import streamlit as st
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime

def store_lead_to_google_sheet(lead_data: dict):
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]

    # 🔐 Credentials from Streamlit secrets
    service_account_info = st.secrets["gcp_service_account"]
    creds = ServiceAccountCredentials.from_json_keyfile_dict(service_account_info, scope)
    client = gspread.authorize(creds)

    sheet = client.open("CCI-Leads").sheet1
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

    sheet.append_row([
        lead_data.get("prenom", "inconnu"),
        lead_data.get("nom", "inconnu"),
        lead_data.get("entreprise", "inconnu"),
        lead_data.get("date", now),
        lead_data.get("interet", "inconnu"),
        lead_data.get("email", "inconnu")
    ])

