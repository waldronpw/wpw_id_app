import streamlit as st
from src.portfolio.monte_carlo import monte_carlo_simulation


st.set_page_config(page_title="Portfolio Monte Carlo Simulator", layout="centered")

st.title("🎲 Monte Carlo Simulation for Goal Planning")

st.write(
    "This tool estimates the probability of reaching a financial goal using Monte Carlo simulations. "
    "Customize your assumptions below:"
)

with st.form("mc_form"):
    col1, col2 = st.columns(2)

    with col1:
        current_assets = st.number_input("Current Portfolio Value ($)", min_value=0, value=500000, step=10000)
        annual_contribution = st.number_input("Annual Contribution ($)", min_value=0, value=10000, step=1000)
        investment_horizon = st.number_input("Investment Horizon (Years)", min_value=1, value=30)
        expected_return = st.number_input("Expected Annual Return (%)", value=6.0) / 100

    with col2:
        asset_goal = st.number_input("Future Asset Goal ($)", min_value=0, value=2000000, step=50000)
        annual_withdrawal = st.number_input("Annual Withdrawal ($)", min_value=0, value=0, step=1000)
        expected_volatility = st.number_input("Expected Volatility (%)", value=12.0) / 100
        inflation_rate = st.number_input("Inflation Rate (%)", value=2.5) / 100

    submitted = st.form_submit_button("Run Simulation")

if submitted:
    with st.spinner("Running simulations..."):
        results = monte_carlo_simulation(
            investment_horizon=investment_horizon,
            asset_goal=asset_goal,
            current_assets=current_assets,
            exp_ret=expected_return,
            exp_vol=expected_volatility,
            ann_contribution=annual_contribution,
            ann_withdrawal=annual_withdrawal,
            inflation=inflation_rate
        )

    if results:
        st.success(f"✅ Simulation Complete! Success Probability: **{results['probability_success'] * 100:.1f}%**")

        st.metric("Median Ending Value", f"${results['median_ending_assets']:,.0f}")
        st.metric("5th Percentile", f"${results['5th_percentile']:,.0f}")
        st.metric("95th Percentile", f"${results['95th_percentile']:,.0f}")

    else:
        st.error("⚠️ Simulation failed. Check your inputs or try again.")