import streamlit as st

st.set_page_config(page_title="WPW ID App", layout="wide")
st.title("WPW ID App")

try:
    import sys, os
    ROOT = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(ROOT)
    sys.path.append(os.path.join(ROOT, "src"))
    sys.path.append(os.path.join(ROOT, "config"))

    st.success("Root path appended.")
    
    from src.setup import load_macro_data
    st.success("Imported load_macro_data successfully.")

    df = load_macro_data()
    if df is not None:
        st.dataframe(df.head())

except Exception as e:
    st.error(f"💥 Something failed in app.py: {e}")