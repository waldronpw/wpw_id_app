import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

import sys
import os
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT)
sys.path.append(os.path.join(ROOT, "src"))
sys.path.append(os.path.join(ROOT, "config"))

from src.portfolio.portfolio_class import Portfolio
from config.cme import capital_market_expectations, covariance_matrix, etf_proxies, models
from src.utils.charting import plot_pie

st.title("Portfolio Risk and Return Analysis")
st.caption("Compare a current vs. proposed portfolio, caclulate expected rates of return and volatility for selected asset classes, and calculate risk metrics.")

# Select model (outside of form so it updates immediately)
selected_model = st.selectbox(
    "Select a Risk-Based Portfolio for Proposed Allocation",
    options=["Custom"] + list(models.keys()),
    index=0
)

# Now derive the model allocations right away
if selected_model != "Custom":
    model_alloc = models[selected_model]
else:
    model_alloc = {k: 0.0 for k in ["US Equity", "Intl Equity", "Emerging Markets", "Bonds", "Cash"]}

with st.form("current_portfolio_form"):
    st.subheader("Define Portfolio Allocations:")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Current")
        dom_eq = st.number_input("US Equity (%)", min_value=0, max_value=100, value=20, key="us_current")
        intl_eq = st.number_input("International Equity (%)", min_value=0, max_value=100, value=20, key="intl_current")
        em_eq = st.number_input("Emerging Markets Equity (%)", min_value=0, max_value=100, value=20, key="em_current")
        bonds = st.number_input("Fixed Income (%)", min_value=0, max_value=100, value=20, key="bonds_current")
        cash = st.number_input("Cash (%)", min_value=0, max_value=100, value=20, key="cash_current")
    
    with col2:
        disabled = selected_model != "Custom"
        st.subheader("Proposed")
        dom_eq_prop = st.number_input("US Equity (%)", min_value=0, max_value=100, value=int(model_alloc["US Equity"] * 100), key="us_prop", disabled=disabled)
        intl_eq_prop = st.number_input("International Equity (%)", min_value=0, max_value=100, value=int(model_alloc["Intl Equity"] * 100), key="intl_prop", disabled=disabled)
        em_eq_prop = st.number_input("Emerging Markets Equity (%)", min_value=0, max_value=100, value=int(model_alloc["Emerging Markets"] * 100), key="em_prop", disabled=disabled)
        bonds_prop = st.number_input("Fixed Income (%)", min_value=0, max_value=100, value=int(model_alloc["Bonds"] * 100), key="bonds_prop", disabled=disabled)
        cash_prop = st.number_input("Cash (%)", min_value=0, max_value=100, value=int(model_alloc["Cash"] * 100), key="cash_prop", disabled=disabled)

    submit = st.form_submit_button("Create Portfolios")

if submit:
    current_alloc = {
        "US Equity": dom_eq / 100,
        "Intl Equity": intl_eq / 100,
        "Emerging Markets": em_eq / 100,
        "Bonds": bonds / 100,
        "Cash": cash / 100
    }

    proposed_alloc = {
        "US Equity": dom_eq_prop / 100,
        "Intl Equity": intl_eq_prop / 100,
        "Emerging Markets": em_eq_prop / 100,
        "Bonds": bonds_prop / 100,
        "Cash": cash_prop / 100
    }


    if sum(current_alloc.values()) != 1:
        st.error("Current portfolio allocations must total 100%")
    
    elif sum(proposed_alloc.values()) != 1:
        st.error("Proposed portfolio allocations must total 100%")
    
    else:
        current_portfolio = Portfolio(name="Current Portfolio", asset_allocation=current_alloc)
        returns = current_portfolio.simulate_historical_performance(etf_proxies, "2015-06-30", "2025-06-30")
        exp_ret = current_portfolio.expected_return(capital_market_expectations)
        exp_std = current_portfolio.expected_std(covariance_matrix)
        sharpe_ratio = current_portfolio.sharpe_ratio(capital_market_expectations, covariance_matrix)
        cvar = current_portfolio.conditional_value_at_risk(capital_market_expectations, covariance_matrix)
        max_dd = current_portfolio.calculate_max_drawdown()

        proposed_portfolio = Portfolio(name="Proposed Portfolio", asset_allocation=proposed_alloc)
        returns_prop = proposed_portfolio.simulate_historical_performance(etf_proxies, "2015-06-30", "2025-06-30")
        exp_ret_prop = proposed_portfolio.expected_return(cme=capital_market_expectations)
        exp_std_prop = proposed_portfolio.expected_std(covariance_matrix=covariance_matrix)
        sharpe_ratio_prop = proposed_portfolio.sharpe_ratio(capital_market_expectations, covariance_matrix)
        cvar_prop = proposed_portfolio.conditional_value_at_risk(capital_market_expectations, covariance_matrix)
        max_dd_prop = proposed_portfolio.calculate_max_drawdown()

        # Pie Charts
        st.markdown("###")
        st.divider()
        st.markdown("###")
        st.subheader("Asset Allocation Comparison")

        labels = list(current_alloc.keys())
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']  # consistent asset class colors

        # Create two pie charts side by side
        fig, axes = plt.subplots(1, 2, figsize=(8, 2))

        plot_pie(current_alloc, "Current", ax=axes[0], colors=colors)
        plot_pie(proposed_alloc, "Proposed", ax=axes[1], colors=colors)

        # Shared legend below
        fig.legend(
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.15),
            ncol=5,
            frameon=False,
            fontsize=6
        )

        plt.tight_layout()
        st.pyplot(fig)
        
        st.markdown("###")
        st.divider()
        st.markdown("###")

        st. subheader("Portfolio Return and Risk Comparison")
        col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
        with col1:
            col1.metric(label="Allocation", value="Current")
            col1.metric(label="", value="Proposed")
        
        with col2:
            col2.metric("Expected Return", f"{exp_ret:.2%}" if exp_ret else "N/A")
            col2.metric(label="", value=f"{exp_ret_prop:.2%}" if exp_ret_prop else "N/A", delta=f"{(exp_ret_prop - exp_ret):.2%}", delta_color="normal")
        
        with col3:
            col3.metric("Expected Volatility", f"{exp_std:.2%}" if exp_std else "N/A")
            col3.metric(label="", value=f"{exp_std_prop:.2%}" if exp_std_prop else "N/A", delta=f"{(exp_std_prop - exp_std):.2%}", delta_color="inverse")
        
        with col4:
            col4.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}" if sharpe_ratio else "N/A")
            col4.metric(label="", value=f"{sharpe_ratio_prop:.2f}" if sharpe_ratio_prop else "N/A", delta=f"{(sharpe_ratio_prop - sharpe_ratio):.2f}", delta_color="normal")
        
        with col5:
            col5.metric("CVaR", f"{cvar:.2%}" if cvar else "N/A")
            col5.metric(label="", value=f"{cvar_prop:.2%}" if cvar_prop else "N/A", delta=f"{(cvar_prop - cvar):.2%}", delta_color="normal")
        
        with col6:
            col6.metric("Max Drawdown", f"{max_dd["max_drawdown"]:.2%}" if max_dd else "N/A")
            col6.metric(label="", value=f"{max_dd_prop["max_drawdown"]:.2%}" if max_dd_prop else "N/A", delta=f"{(max_dd_prop["max_drawdown"] - max_dd["max_drawdown"]):.2%}", delta_color="normal")
        
        with col7:
            col7.metric("Recovery", f"{max_dd["recovery_months"]:.0f} months" if max_dd else "N/A")
            col7.metric(label="", value=f"{max_dd_prop["recovery_months"]:.0f} months" if max_dd_prop else "N/A", delta=f"{(max_dd_prop["recovery_months"] - max_dd["recovery_months"]):.0f} months", delta_color="inverse")

    st.markdown("###")
    st.divider()
    st.markdown("###")

    st.subheader("Growth of $100,000: Current vs Proposed Portfolio")
    start_value = 100_000
    returns_df = (
        pd.DataFrame({
            "Current": current_portfolio.historical_returns,
            "Proposed": proposed_portfolio.historical_returns
        })
        .dropna()
    )

    cumulative = (1 + returns_df).cumprod() * start_value

    fig, ax = plt.subplots(figsize=(10, 4))
    cumulative.plot(ax=ax)

    ax.set_xlabel("")

    ax.yaxis.set_major_formatter(mtick.StrMethodFormatter('${x:,.0f}'))
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=6)
    ax.tick_params(axis="x", which="both", length=0)
    ax.tick_params(axis="y", which="both", length=0)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(frameon=False, fontsize=6)
    ax.tick_params(axis="x", labelrotation=0)

    st.pyplot(fig)


    st.markdown("###")
    st.divider()
    st.markdown("###")

    st.subheader("Drawdown Analysis: Current vs Proposed Portfolio")

    drawdown = cumulative / cumulative.cummax() - 1

    fig_dd, ax_dd = plt.subplots(figsize=(10, 4))
    for col in drawdown.columns:
        line, = ax_dd.plot(drawdown.index, drawdown[col], label=col)
        ax_dd.fill_between(drawdown.index, drawdown[col], 0, color=line.get_color(), alpha=0.2)

    # ax_dd.set_ylabel("Drawdown", fontsize=9)
    ax_dd.set_xlabel("")
    ax_dd.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax_dd.spines["top"].set_visible(False)
    ax_dd.spines["bottom"].set_visible(False)
    ax_dd.spines["left"].set_visible(False)
    ax_dd.spines["right"].set_visible(False)
    ax_dd.tick_params(axis="both", labelsize=6)
    ax_dd.tick_params(axis="x", which="both", length=0)
    ax_dd.tick_params(axis="y", which="both", length=0)
    ax_dd.grid(True, axis="y", alpha=0.3)
    ax_dd.legend(frameon=False, fontsize=6)

    st.pyplot(fig_dd)