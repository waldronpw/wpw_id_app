import streamlit as st
import pandas as pd
from fredapi import Fred
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from dotenv import load_dotenv
import os

try:
    api_key = st.secrets["FRED_API_KEY"]
except Exception:
    load_dotenv()
    api_key = os.getenv("FRED_API_KEY")

st.title("Macroeconomic Research & Analysis")
st.caption("Regularly updated macroeconomic data series and visuals for use in conversations with clients.")

# st.subheader("Macro Data Table")
macro_df = pd.read_excel("../monthly_macro_update.xlsx", index_col="Name")
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