import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns

import sys
import os
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT)
sys.path.append(os.path.join(ROOT, "src"))
sys.path.append(os.path.join(ROOT, "config"))

from src.portfolio.portfolio_class import Portfolio
from config.cme import capital_market_expectations, covariance_matrix, etf_proxies, models

st.title("Monte Carlo Simulation Tool for Goals Based Investing")
st.caption("Estimate an asset allocation's probability of meeting a client's goals using sophisticated statistics.")

col1, col2 = st.columns(2)
with col1:
    st.markdown("### Define Simulation Parameters")
    years = st.number_input("Investment Horizon (years)", 1, 99, 15, 1)
    goal = st.number_input("Future Asset Goal", value = 10_000_000, step=10_000)
    assets = st.number_input("Current Portfolio Value", value = 4_000_000)
    contributions = st.number_input("Annual Contributions", value = 120_000)
    withdrawls = st.number_input("Annual Withdrawals", value = 0)
    n_sims = st.number_input("# Simulations", 1_000, 1_000_000, 10_000, 1_000)
with col2:
    # --- Input: Define current portfolio allocation ---
    st.markdown("### Define Current Portfolio Allocation")
    dom_eq = st.number_input("US Equity (%)", 0, 100, 20)
    intl_eq = st.number_input("International Equity (%)", 0, 100, 20)
    em_eq = st.number_input("Emerging Markets Equity (%)", 0, 100, 20)
    bonds = st.number_input("Fixed Income (%)", 0, 100, 20)
    cash = st.number_input("Cash (%)", 0, 100, 20)

current_alloc = {
    "US Equity": dom_eq / 100,
    "Intl Equity": intl_eq / 100,
    "Emerging Markets": em_eq / 100,
    "Bonds": bonds / 100,
    "Cash": cash / 100,
}

if sum(current_alloc.values()) != 1:
    st.error("Current portfolio allocation must total 100%")
else:
    current_portfolio = Portfolio(name="Current", asset_allocation=current_alloc)
    sim_results = {}

    # --- Run simulation for current portfolio ---
    sim_results["Current"] = current_portfolio.monte_carlo_goal_projection(
        years=years,
        goal = goal,
        initial_value = assets,
        num_simulations=n_sims,
        annual_contribution = contributions,
        annual_withdrawal = withdrawls,
        expected_return = current_portfolio.expected_return(cme=capital_market_expectations),
        expected_volatility= current_portfolio.expected_std(covariance_matrix=covariance_matrix),
    )

    # --- Run simulations for model portfolios ---
    for model_name, alloc in models.items():
        model_port = Portfolio(name=model_name, asset_allocation=alloc)
        sim_results[model_name] = model_port.monte_carlo_goal_projection(
            years=years,
            goal = goal,
            initial_value = assets,
            num_simulations=n_sims,
            annual_contribution = contributions,
            annual_withdrawal = withdrawls,
            expected_return = model_port.expected_return(cme=capital_market_expectations),
            expected_volatility= model_port.expected_std(covariance_matrix=covariance_matrix),
        )

    # --- Create summary table ---
    summary = []
    for name, result in sim_results.items():
        summary.append({
            "Portfolio": name,
            "Prob. of Success": f"{result['success_rate']:.2%}",
            "Exp. Return": f"{result["exp_ret"]:.2%}",
            "Median Ending Value": f"${result['median_ending_value']:,.0f}",
            "5th Percentile": f"${result['5th_percentile']:,.0f}",
            "95th Percentile": f"${result['95th_percentile']:,.0f}",
        })

    if summary:
        summary_df = pd.DataFrame(summary).set_index("Portfolio")
        st.subheader("Summary of Monte Carlo Results")
        st.dataframe(summary_df)
    else:
        st.warning("No valid simulation results available to display.")
    

    st.markdown("###")
    st.subheader("Distribution of Ending Portfolio Values")

    # Dropdown to select portfolio to visualize
    selected_hist = st.selectbox(
        "Select a portfolio to visualize ending value distribution:",
        options=list(sim_results.keys())
    )

    # Retrieve the ending values from simulation results
    if selected_hist and "ending_values" in sim_results[selected_hist]:
        values = sim_results[selected_hist]["ending_values"]
        
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.histplot(values, bins=50, kde=True, ax=ax, color="#1f77b4")

        ax.axvline(goal, color="red", linestyle="--", label=f"Goal (${goal:,.0f})")
        ax.set_xlabel("Ending Portfolio Value", fontsize=6)
        ax.set_ylabel("Frequency", fontsize=6)
        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(which="both", length=0)
        ax.xaxis.set_major_formatter(mtick.StrMethodFormatter("${x:,.0f}"))
        ax.legend(frameon=False, fontsize=6, title=f"{selected_hist} Portfolio ({summary_df.loc[selected_hist, 'Prob. of Success']} POS)")
        ax.grid(axis="y", alpha=0.3)
        ax.tick_params(axis="both", labelsize=6)

        st.pyplot(fig)
    else:
        st.warning("No simulation data available for this portfolio.")