# src/setup.py
import os
import streamlit as st
from dotenv import load_dotenv
import pandas as pd

def get_fred_api_key():
    try:
        return st.secrets["FRED_API_KEY"]
    except Exception:
        load_dotenv(dotenv_path=".env")
        return os.getenv("FRED_API_KEY")

def load_macro_data(path="monthly_macro_update.xlsx"):
    if not os.path.exists(path):
        st.warning(f"📁 File not found: {path}")
        return None
    try:
        return pd.read_excel(path, index_col="Name")
    except Exception as e:
        st.error(f"❌ Failed to load macro data: {e}")
        return None