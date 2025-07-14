import streamlit as st
import pandas as pd

st.title("Manager Analysis Tool")

uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])
if uploaded_file:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.subheader("Uploaded Data Preview")
    st.dataframe(df)

    # Example: Run some analysis
    st.subheader("Basic Stats")
    st.write(df.describe())