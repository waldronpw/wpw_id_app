import streamlit as st
try:
    st.set_page_config(page_title="Waldron Private Wealth ID Team App", layout="wide")

    st.title("Waldron Private Wealth")
    st.caption("Welcome to the Investment Team Dashboard. Please use the sidebar to select a tool.")

except Exception as e:
    st.error(f"🚨 App failed to load: {e}")