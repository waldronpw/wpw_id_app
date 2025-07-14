import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pandas as pd

from src.portfolio.portfolio_class import Portfolio
from config.cme import capital_market_expectations, covariance_matrix, etf_proxies


st.set_page_config(
    page_title="Waldron Private Wealth ID Team App",
    layout="wide",
    initial_sidebar_state="expanded"
)

page = st.sidebar.radio(
    "Select a Tool",
    [
        "Portfolio Analysis",
        "Monte Carlo Simulation",
        "Historical Backtest",
        "Stress Testing",
        "Fund Scoring"
    ]
)

if page == "Portfolio Analysis":
    st.title("Portfolio Risk and Return Analysis")
    st.caption("Compare a current vs. proposed portfolio, caclulate expected rates of return and volatility for selected asset classes, and calculate risk metrics.")

    with st.form("current_portfolio_form"):
        st.subheader("Define Portfolio Allocations:")
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Current Asset Allocation")
            dom_eq = st.number_input("US Equity (%)")
            intl_eq = st.number_input("International Equity (%)")
            em_eq = st.number_input("Emerging Markets Equity (%)")
            bonds = st.number_input("Fixed Income (%)")
            cash = st.number_input("Cash (%)")
        
        with col2:
            st.subheader("Proposed Asset Allocation")
            dom_eq_prop = st.number_input("US Equity (%)", key="us_prop")
            intl_eq_prop = st.number_input("International Equity (%)", key="intl_prop")
            em_eq_prop = st.number_input("Emerging Markets Equity (%)", key="em_prop")
            bonds_prop = st.number_input("Fixed Income (%)", key="fi_prop")
            cash_prop = st.number_input("Cash (%)", key="cash_prop")

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

            st. subheader("Portfolio Metrics")
            col1, col2, = st.columns(2)
            with col1:
                col1.metric("Expected Return", f"{exp_ret:.2%}" if exp_ret else "N/A")
                col1.metric("Expected Volatility", f"{exp_std:.2%}" if exp_std else "N/A")
                col1.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}" if sharpe_ratio else "N/A")
                col1.metric("CVaR", f"{cvar:.2%}" if cvar else "N/A")
                col1.metric("Max Drawdown", f"{max_dd["max_drawdown"]:.2%}" if max_dd else "N/A")
                col1.metric("Recovery", f"{max_dd["recovery_months"]:.0f} months" if max_dd else "N/A")
            
            with col2:
                col2.metric("Expected Return", f"{exp_ret_prop:.2%}" if exp_ret_prop else "N/A")
                col2.metric("Expected Volatility", f"{exp_std_prop:.2%}" if exp_std_prop else "N/A")
                col2.metric("Sharpe Ratio", f"{sharpe_ratio_prop:.2f}" if sharpe_ratio_prop else "N/A")
                col2.metric("CVaR", f"{cvar_prop:.2%}" if cvar_prop else "N/A")
                col2.metric("Max Drawdown", f"{max_dd_prop["max_drawdown"]:.2%}" if max_dd_prop else "N/A")
                col2.metric("Recovery", f"{max_dd_prop["recovery_months"]:.0f} months" if max_dd_prop else "N/A")

        st.markdown("###")
        st.divider()
        st.markdown("###")
        st.subheader("📈 Growth of $10,000: Current vs Proposed Portfolio")
        # Align and calculate cumulative performance
        start_value = 100_000
        returns_df = (
            pd.DataFrame({
                "Current": current_portfolio.historical_returns,
                "Proposed": proposed_portfolio.historical_returns
            })
            .dropna()
        )

        cumulative = (1 + returns_df).cumprod() * start_value

        # Plot using matplotlib
        fig, ax = plt.subplots(figsize=(10, 4))
        cumulative.plot(ax=ax)

        # ax.set_title(f"Growth of ${start_value:.0f} (2015–2025)")
        # ax.set_ylabel("Portfolio Value ($)")
        # ax.set_xlabel("Date")
        ax.yaxis.set_major_formatter(mtick.StrMethodFormatter('${x:,.0f}'))
        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="x", which="both", length=0)
        ax.tick_params(axis="y", which="both", length=0)
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend()
        ax.tick_params(axis="x", labelrotation=0)

        st.pyplot(fig)



if page == "Monte Carlo Simulation":
    st.title("Monte Carlo Simulation Tool for Goals Based Investing")
    st.caption("Estimate an asset allocation's probability of meeting a client's goals using sophisticated statistics.")

if page == "Historical Backtest":
    st.caption("Please come back later. This section of the application is still in development.")

if page == "Stress Testing":
    st.caption("Please come back later. This section of the application is still in development.")

if page == "Fund Scoring":
    st.caption("Please come back later. This section of the application is still in development.")