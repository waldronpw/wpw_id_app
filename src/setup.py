# src/setup.py
import os
import streamlit as st
from dotenv import load_dotenv
import pandas as pd

def get_fred_api_key():
    try:
        return st.secrets["FRED_API_KEY"]
    except Exception:
        load_dotenv()
        return os.getenv("FRED_API_KEY")

def load_macro_data():
    path = "../monthly_macro_update.xlsx"
    if not os.path.exists(path):
        st.warning("Macro update file not found.")
        return None
    return pd.read_excel(path, index_col="Name")