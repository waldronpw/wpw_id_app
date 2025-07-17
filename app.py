import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pandas as pd
import seaborn as sns
from fredapi import Fred

from dotenv import load_dotenv
import os

try:
    api_key = st.secrets["FRED_API_KEY"]
except Exception:
    load_dotenv()
    api_key = os.getenv("FRED_API_KEY")

from src.portfolio.portfolio_class import Portfolio
from src.private_alts.pacing_model import pacing_model
from src.utils.charting import plot_pie
from config.cme import capital_market_expectations, covariance_matrix, etf_proxies, models


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
        # "Historical Backtest",
        "Private Alts Pacing Model",
        "Stress Testing",
        "Fund Scoring",
        "Economic Research"
    ]
)

if page == "Portfolio Analysis":
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
        col1, col2, col3 = st.columns(3)
        with col1:
            st.subheader("###")
            st.markdown("### US Equity")
            st.markdown("### International Equity")
            st.markdown("### Emerging Markets Equity")
            st.markdown("### Fixed Income")
            st.markdown("### Cash")

        with col2:
            st.subheader("Portfolio 1")
            # dom_eq = st.slider("US Equity", min_value=0, max_value=100, format="%0.1f%%", value=20)
            # intl_eq = st.slider("International Equity", min_value=0, max_value=100, format="%0.1f%%", value=20)
            # em_eq = st.slider("Emerging Markets Equity", min_value=0, max_value=100, format="%0.1f%%", value=20)
            # bonds = st.slider("Fixed Income", min_value=0, max_value=100, format="%0.1f%%", value=20)
            # cash = st.slider("Cash (%)", min_value=0, max_value=100, format="%0.1f%%", value=20)
            dom_eq = st.number_input("US Equity (%)", min_value=0, max_value=100, value=int(model_alloc["US Equity"] * 100), key="us_current", width=300)
            intl_eq = st.number_input("International Equity (%)", min_value=0, max_value=100, value=int(model_alloc["Intl Equity"] * 100), key="intl_current", width=300)
            em_eq = st.number_input("Emerging Markets Equity (%)", min_value=0, max_value=100, value=int(model_alloc["Emerging Markets"] * 100), key="em_current", width=300)
            bonds = st.number_input("Fixed Income (%)", min_value=0, max_value=100, value=int(model_alloc["Bonds"] * 100), key="bonds_current", width=300)
            cash = st.number_input("Cash (%)", min_value=0, max_value=100, value=int(model_alloc["Cash"] * 100), key="cash_current", width=300)
        
        with col3:
            st.subheader("Portfolio 2")
            
            disabled = selected_model != "Custom"

            dom_eq_prop = st.slider("US Equity", 0, 100, int(model_alloc["US Equity"] * 100), key="us_prop", disabled=disabled)
            intl_eq_prop = st.slider("International Equity", 0, 100, int(model_alloc["Intl Equity"] * 100), key="intl_prop", disabled=disabled)
            em_eq_prop = st.slider("Emerging Markets Equity", 0, 100, int(model_alloc["Emerging Markets"] * 100), key="em_prop", disabled=disabled)
            bonds_prop = st.slider("Fixed Income", 0, 100, int(model_alloc["Bonds"] * 100), key="bonds_prop", disabled=disabled)
            cash_prop = st.slider("Cash (%)", 0, 100, int(model_alloc["Cash"] * 100), key="cash_prop", disabled=disabled)

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

if page == "Monte Carlo Simulation":
    st.title("Monte Carlo Simulation Tool for Goals Based Investing")
    st.caption("Estimate an asset allocation's probability of meeting a client's goals using sophisticated statistics.")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Define Simulation Parameters")
        years = st.number_input("Investment Horizon (years)", 1, 99, 15, 1)
        goal = st.number_input("Future Asset Goal", value = 10_000_000, step=10_000)
        assets = st.number_input("Current Portfolio Value", value = 4_000_000)
        contributions = st.number_input("Annual Contributions", value = 0)
        withdrawls = st.number_input("Annual Withdrawals", value = 0)
        n_sims = st.number_input("# Simulations", 1_000, 1_000_000, 10_000, 1_000)
    with col2:
        # --- Input: Define current portfolio allocation ---
        st.markdown("### Define Current Portfolio Allocation")
        dom_eq = st.slider("US Equity", 0, 100, 0)
        intl_eq = st.slider("International Equity", 0, 100, 0)
        em_eq = st.slider("Emerging Markets", 0, 100, 0)
        bonds = st.slider("Fixed Income", 0, 100, 100)
        cash = st.slider("Cash", 0, 100, 0)

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
            ax.legend(frameon=False, fontsize=6, title=f"{selected_hist} Portfolio")
            ax.grid(axis="y", alpha=0.3)
            ax.tick_params(axis="both", labelsize=6)

            st.pyplot(fig)
        else:
            st.warning("No simulation data available for this portfolio.")

if page == "Historical Backtest":
    st.caption("Please come back later. This section of the application is still in development.")

if page == "Private Alts Pacing Model":
    st.title("Private Equity Pacing Model Simulation")
    st.caption("Simulate a basic private equity pacing model to understand capital calls, distributions, and NAV growth over time.")
    st.markdown("###")
    st.subheader("Pacing Model Parameters")
    col1, col2 = st.columns(2)
    with col1:
        total_aum = st.number_input("Total Investable Assets ($)", min_value=500_000, value=10_000_000, step=100_000)
        st.caption(f"Total AUM: ${total_aum:,.0f}")
        target_pe_alloc = st.slider("Target PE Allocation (%)", 0.0, 100.0, 10.0, step=5.0) / 100.0
        # st.caption(f"Target PE Allocation: ${total_aum * target_pe_alloc:,.0f}")
        initial_commitments = st.number_input("Initial Commitments ($)", min_value=0, value=500_000, step=10_000)
        st.caption(f"Initial Commitments: ${initial_commitments:,.0f}")
    with col2:
        years = st.number_input("Years to Simulate", min_value=1, max_value=50, value=5, step=1)
        call_rate = st.slider("Annual Capital Call Rate (%)", 0.0, 100.0, 25.0, step=0.1) / 100.0
        dist_rate = st.slider("Annual Distribution Rate (%)", 0.0, 100.0, 5.0, step=0.1) / 100.0
        growth_rate = st.slider("Annual Growth Rate on NAV (%)", 0.0, 100.0, 10.0, step=0.1) / 100.0
    
    if st.button("Run Pacing Model Simulation"):
        results = pacing_model(
            total_aum=total_aum,
            target_pe_alloc=target_pe_alloc,
            initial_commitments=initial_commitments,
            years=years,
            call_rate=call_rate,
            dist_rate=dist_rate,
            growth_rate=growth_rate
        )

        st.subheader("Simulation Results")
        st.dataframe(results)

        st.markdown("###")
        st.subheader("Yearly NAV and Cashflows")
        
        fig, ax = plt.subplots(figsize=(10, 4))
        results.plot(x="Year", y=["NAV", "Capital Called", "Distributions"], ax=ax)
        
        ax.set_xlabel("Year", fontsize=6)
        ax.set_ylabel("Amount ($)", fontsize=6)
        ax.yaxis.set_major_formatter(mtick.StrMethodFormatter("${x:,.0f}"))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(which="both", length=0)
        ax.tick_params(axis="both", labelsize=6)
        ax.grid(True, axis="y", alpha=0.3)
        
        st.pyplot(fig)

if page == "Stress Testing":
    st.caption("Please come back later. This section of the application is still in development.")

if page == "Fund Scoring":
    st.caption("Please come back later. This section of the application is still in development.")

if page =="Economic Research":
    st.title("Macroeconomic Research & Analysis")
    st.caption("Regularly updated macroeconomic data series and visuals for use in conversations with clients.")

    # st.subheader("Macro Data Table")
    macro_df = pd.read_excel("monthly_macro_update.xlsx", index_col="Name")
    # st.dataframe(macro_df)
    
    st.divider()
    st.subheader("Macro Data")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("GDP", macro_df.loc["Gross Domestic Product (GDP)", "Most Recent Observation"], macro_df.loc["Gross Domestic Product (GDP)", "Percent Change"])
    with col2:
        st.metric("Real GDP", macro_df.loc["Real Gross Domestic Product (rGDP)", "Most Recent Observation"], macro_df.loc["Real Gross Domestic Product (rGDP)", "Percent Change"])
    with col3:
        st.metric("Corporate Profits", macro_df.loc["Corporate Profits (% share of GDI)", "Most Recent Observation"], macro_df.loc["Corporate Profits (% share of GDI)", "Percent Change"])
    with col4:
        st.metric("Manufacturing Activity", macro_df.loc["Manufacturing Activity Index (Chi. Fed Survey)", "Most Recent Observation"], macro_df.loc["Manufacturing Activity Index (Chi. Fed Survey)", "Percent Change"])
    
    st.markdown("###")
    api_key = "d8dbd3216491c011edb8f42b52ee82f4"
    fred = Fred(api_key=api_key)

    gdp_raw = fred.get_series("GDPC1")
    gdp_yoy = gdp_raw.pct_change(4).dropna()[-16:]
    quarter_labels = [f"Q{d.quarter} '{str(d.year)[2:]}" for d in gdp_yoy.index]

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.bar(quarter_labels, gdp_yoy.values)

    ax.set_ylabel("Real GDP Growth YoY", fontsize=6)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(which="both", length=0)
    ax.tick_params(axis="x", labelrotation=45, labelsize=6)
    ax.tick_params(axis="y", labelsize=6)
    ax.grid(True, axis="y", alpha=0.25)

    st.pyplot(fig)
    
    
    st.divider()
    st.subheader("Consumer Data")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Consumer Sentiment", macro_df.loc["Consumer Sentiment", "Most Recent Observation"], macro_df.loc["Consumer Sentiment", "Change"])
    with col2:
        st.metric("Personal Savings", macro_df.loc["Personal Savings (% of Disp. Inc.)", "Most Recent Observation"], macro_df.loc["Personal Savings (% of Disp. Inc.)", "Change"])
    with col3:
        st.metric("New Housing Starts", macro_df.loc["New Housing Starts", "Most Recent Observation"], macro_df.loc["New Housing Starts", "Change"])

    st.divider()
    st.subheader("Inflation Data")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("CPI", macro_df.loc["Consumer Price Index (CPI)", "Most Recent Observation"], macro_df.loc["Consumer Price Index (CPI)", "Percent Change"])
        st.metric("Core CPI", macro_df.loc["CPI ex Food & Energy (Core CPI)", "Most Recent Observation"], macro_df.loc["CPI ex Food & Energy (Core CPI)", "Percent Change"])
    with col2:
        st.metric("PCE", macro_df.loc["Personal Consumption Expenditures (PCE)", "Most Recent Observation"], macro_df.loc["Personal Consumption Expenditures (PCE)", "Percent Change"])
        st.metric("PPI", macro_df.loc["Producer Price Index (PPI): All Commodities", "Most Recent Observation"], macro_df.loc["Producer Price Index (PPI): All Commodities", "Percent Change"])
    with col3:
        st.metric("1-Year Inflation Expectations", macro_df.loc["1-Year Expected Inflation", "Most Recent Observation"], macro_df.loc["1-Year Expected Inflation", "Percent Change"])
        st.metric("3-Year Inflation Expectations", macro_df.loc["3-Year Expected Inflation", "Most Recent Observation"], macro_df.loc["3-Year Expected Inflation", "Percent Change"])
    
    st.markdown("###")
    cpi = fred.get_series("CPIAUCSL").pct_change(12).dropna()
    cpi_core = fred.get_series("CPILFESL").pct_change(12).dropna()
    pce = fred.get_series("PCE").pct_change(12).dropna()

    df = pd.DataFrame([cpi, cpi_core, pce]).T.dropna()
    df.columns = ["CPI", "Core CPI", "PCE"]
    
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(df["CPI"][-48:], label = "CPI", linewidth=0.75)
    ax.plot(df["Core CPI"][-48:], label = "Core CPI (ex Food & Energy)", linewidth=0.75)
    ax.plot(df["PCE"][-48:], label = "PCE", linewidth=0.75)
    ax.axhline(y=0.02, color="r", linestyle="--", label = "2% Inflation Target", linewidth=0.75)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    xticks = df.index[-48:]
    ax.set_xticks(xticks[::3])
    ax.set_xticklabels([d.strftime("%b '%y") for d in xticks[::3]])
    ax.set_ylabel("% Change YoY", fontsize=6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(which="both", length=0)
    ax.tick_params(axis="x", labelrotation=45, labelsize=6)
    ax.tick_params(axis="y", labelsize=6)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=6)

    st.pyplot(fig)
    st.markdown("###")

    st.divider()
    st.subheader("Labor Market Data")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Labor Force Participation", macro_df.loc["Labor Force Participation Rate", "Most Recent Observation"], macro_df.loc["Labor Force Participation Rate", "Percent Change"])
    with col2:
        st.metric("Unemployment Claims", macro_df.loc["Weekly Initial Claims for Unemployment", "Most Recent Observation"], macro_df.loc["Weekly Initial Claims for Unemployment", "Percent Change"])
    with col3:
        st.metric("Nonfarm Payrolls", macro_df.loc["Total Nonfarm Payroll", "Most Recent Observation"], macro_df.loc["Total Nonfarm Payroll", "Percent Change"])
    with col4:
        st.metric("Unemployment Rate", macro_df.loc["Unemployment Rate", "Most Recent Observation"], macro_df.loc["Unemployment Rate", "Percent Change"])


    st.divider()
    st.subheader("Interest Rates & Spread Data")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("2-Year Treasury Yield", macro_df.loc["2-Year Treasury Yield", "Most Recent Observation"], macro_df.loc["2-Year Treasury Yield", "% Change 12mo"])
        st.metric("AAA OAS", macro_df.loc["AAA OAS", "Most Recent Observation"], macro_df.loc["AAA OAS", "% Change 12mo"]) 
    with col2:
        st.metric("10-Year Treasury Yield", macro_df.loc["10-Year Treasury Yield", "Most Recent Observation"], macro_df.loc["10-Year Treasury Yield", "% Change 12mo"])
        st.metric("BB OAS", macro_df.loc["BB OAS", "Most Recent Observation"], macro_df.loc["BB OAS", "% Change 12mo"]) 
    with col3:
        st.metric("2s10s Spread", macro_df.loc["2s10s Spread", "Most Recent Observation"], macro_df.loc["2s10s Spread", "% Change 12mo"])
        st.metric("CCC & Lower OAS", macro_df.loc["CCC & Lower OAS", "Most Recent Observation"], macro_df.loc["CCC & Lower OAS", "% Change 12mo"])
    
    hy_spreads = fred.get_series("BAMLH0A0HYM2")
    ig_spreads = fred.get_series("BAMLC0A0CM")

    spread_df = pd.DataFrame([hy_spreads, ig_spreads]).T
    spread_df.columns = ["IG Spread", "HY Spread"]
    spread_df = spread_df * 100
    st.markdown("###")
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(spread_df["HY Spread"][-1000:], label="High Yield OAS", linewidth=0.7)
    ax.plot(spread_df["IG Spread"][-1000:], label="Investment Grade OAS", linewidth=0.7)
    xticks = spread_df.index[-1000:]
    xticks_shown = xticks[::90]
    ax.set_xticks(xticks_shown)
    ax.set_xticklabels([d.strftime("%b '%y") for d in xticks_shown])
    ax.set_ylabel("Spread (bps)", fontsize=6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(which="both", length=0)
    ax.tick_params(axis="x", labelrotation=45, labelsize=6)
    ax.tick_params(axis="y", labelsize=6)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=6)

    st.pyplot(fig)