import streamlit as st

from src.private_alts.pacing_model import pacing_model, plot_pacing_visual

st.title("Private Equity Pacing Model Simulation")
st.caption("Simulate a basic private equity pacing model to understand capital calls, distributions, and NAV growth over time.")
st.markdown("###")
st.subheader("Pacing Model Parameters")

col1, col2 = st.columns(2)
with col1:
    aum = st.number_input("Total Portfolio Value", min_value=1_000_000, value=10_000_000, step=100_000)
    target_pe_alloc = st.number_input("Target Private Equity Allocation (%)", min_value=0, max_value=100, value=15)
    years = st.number_input("Investment Horizon (years)", min_value=1, max_value=20, value=5, step=1)
with col2:
    growth = st.number_input("Portfolio Growth Rate (%)", min_value=0.0, max_value=100.0, value=6.5, step=0.5)
    pe_nav_growth = st.number_input("Private Equity NAV Growth Rate (%)", min_value=0.0, max_value=100.0, value=9.5, step=0.5)

results = pacing_model(
    aum=int(aum),
    target_pe_alloc=target_pe_alloc / 100,
    years=years,
    growth=growth / 100,
    pe_nav_growth=pe_nav_growth / 100
)

st.markdown("###")
st.subheader("Pacing Model Results")
col1, col2, col3 = st.columns(3)
col1.metric("Future Value of Portfolio", f"${results[0]:,.0f}")
col2.metric("Target Private Equity NAV", f"${results[1]:,.0f}")
col3.metric("Annual Commitment to PE Sleeve", f"${results[2]:,.0f}")
st.markdown("###")

fig = plot_pacing_visual(results[3], results[1])
st.pyplot(fig)