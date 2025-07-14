import streamlit as st
import pandas as pd
from src.portfolio.portfolio_class import Portfolio
from config.cme import models, capital_market_expectations, covariance_matrix
from src.utils.formatter import parse_dollar_input

st.header("📊 Monte Carlo Results: WPW Risk Based Asset Allocations")

with st.form("multi_model_mc_form"):
    col1, col2 = st.columns(2)

    with col1:
        initial_value = parse_dollar_input("Initial Portfolio Value ($)", 1_000_000)
        annual_contribution = parse_dollar_input("Annual Contribution ($)", 25_000)
        investment_horizon = st.number_input("Investment Horizon (Years)", min_value=1, value=30)

    with col2:
        goal = parse_dollar_input("Asset Goal ($)", 5_000_000)
        annual_withdrawal = parse_dollar_input("Annual Withdrawal ($)", 0)
        inflation_rate = st.number_input("Inflation Rate (%)", value=2.5) / 100

    submitted = st.form_submit_button("Run Monte Carlo Across All Allocations")

if submitted:
    with st.spinner("Running simulations..."):
        results = []

        for name, allocation in models.items():
            portfolio = Portfolio(name=name, asset_allocation=allocation)

            exp_ret = portfolio.expected_return(capital_market_expectations)
            exp_vol = portfolio.expected_std(covariance_matrix)
            sharpe = portfolio.sharpe_ratio(capital_market_expectations, covariance_matrix)

            sim = portfolio.monte_carlo_goal_projection(
                years=investment_horizon,
                goal=goal,
                initial_value=initial_value,
                annual_contribution=annual_contribution,
                annual_withdrawal=annual_withdrawal,
                expected_return=exp_ret,
                expected_volatility=exp_vol,
                inflation_rate=inflation_rate
            )

            results.append({
                "Allocation": name,
                "Sharpe Ratio": sharpe,
                "Success Rate": sim.get("success_rate"),
                "Median Ending Value": sim.get("median_ending_value"),
                "5th Percentile": sim.get("5th_percentile"),
                "95th Percentile": sim.get("95th_percentile"),
                "Goal": sim.get("goal")
            })

        df = pd.DataFrame(results).set_index("Allocation")

        # Format display
        df_display = df.copy()
        df_display["Sharpe Ratio"] = df_display["Sharpe Ratio"].map("{:.2f}".format)
        df_display["Success Rate"] = df_display["Success Rate"].map("{:.2%}".format)
        for col in ["Median Ending Value", "5th Percentile", "95th Percentile", "Goal"]:
            df_display[col] = df_display[col].map("${:,.0f}".format)

        st.success("✅ Simulation Complete")
        st.dataframe(df_display)