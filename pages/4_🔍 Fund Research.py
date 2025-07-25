import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt

try:
    st.title("Mutual Fund Research Tool")
    ticker = st.text_input("Enter Mutual Fund Ticker", key="ticker_input").strip().upper()

    if ticker:
        try:
            data = yf.Ticker(ticker).funds_data
            info = yf.Ticker(ticker).info
            hx = yf.download(ticker, period="10y", interval="1d", progress=False, auto_adjust=True)["Close"]
            ytd = yf.download(ticker, period="ytd", interval="1d", progress=False, auto_adjust=True)["Close"]
            one_year = yf.download(ticker, period="1y", interval="1d", progress=False, auto_adjust=True)["Close"]
            three_year = yf.download(ticker, period="3y", interval="1d", progress=False, auto_adjust=True)["Close"]
            five_year = yf.download(ticker, period="5y", interval="1d", progress=False, auto_adjust=True)["Close"]

            exp_ratio = info.get("netExpenseRatio", None)
            aum = info.get("netAssets", None)

            overview = data.fund_overview # dictionary
            description = data.description # string
            sector_weights = data.sector_weightings # dataframe
            top_holdings = data.top_holdings # dataframe

            st.header(f"{info.get('longName', 'No Name Found')} ({ticker.upper()})")
            col1, col2 = st.columns(2)
            col1.metric("Category", overview.get("categoryName", "N/A"))
            col1.metric("Expense Ratio", f"{exp_ratio:.2f}%" if exp_ratio is not None else "N/A")
            col1.metric("AUM (billions)", f"${aum/1_000_000_000:,.2f}" if aum is not None else "N/A")
            col2.subheader("Fund Overview")
            col2.caption(description)
            
            st.divider()

            st.subheader("Historical Performance")
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(hx.index, hx.values)
            ax.set_ylabel("NAV/Price")
            ax.spines["top"].set_visible(False)
            ax.spines["left"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.tick_params(which="both", length=0)
            ax.grid(axis="y", linestyle="--", alpha=0.5)
            st.pyplot(fig)

            ytd_return = (ytd.iloc[-1] / ytd.iloc[0]).values[0] - 1
            one_year_return = (one_year.iloc[-1] / one_year.iloc[0]).values[0] - 1
            three_year_return = ((three_year.iloc[-1] / three_year.iloc[0]).values[0] ** (1/3)) - 1
            five_year_return = ((five_year.iloc[-1] / five_year.iloc[0]).values[0] ** (1/5)) - 1
            ten_year_return = ((hx.iloc[-1] / hx.iloc[0]).values[0] ** (1/10)) - 1
            st.markdown("###")
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("YTD", f"{ytd_return * 100:.2f}%")
            col2.metric("1 Year", f"{one_year_return * 100:.2f}%")
            col3.metric("3 Year", f"{three_year_return * 100:.2f}%")
            col4.metric("5 Year", f"{five_year_return * 100:.2f}%")
            col5.metric("10 Year", f"{ten_year_return * 100:.2f}%")

            st.divider()
            
            st.subheader("Sector Weights and Top Holdings")
            st.markdown("####")
            col1, col2 = st.columns(2)
            sector_weights = dict(sorted(sector_weights.items(), key=lambda x: x[1]))
            sectors = list(sector_weights.keys())
            weights = [v * 100 for v in sector_weights.values()]
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.barh(sectors, weights)
            ax.xaxis.set_visible(False)
            ax.spines["top"].set_visible(False)
            ax.spines["bottom"].set_visible(False)
            ax.spines["left"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.tick_params(which="both", length=0)
            for i, v in enumerate(weights):
                ax.text(v + 0.5, i, f"{v:.1f}%", va='center', fontsize=10)
            col1.pyplot(fig)


            df = top_holdings.copy()
            df["Holding"] = df["Name"] + " (" + df.index + ")"
            df["Weight"] = df["Holding Percent"] * 100
            df = df[["Holding", "Weight"]].sort_values(by="Weight", ascending=False).reset_index(drop=True)
            df.set_index("Holding", inplace=True)
            df.columns = ["Weight (%)"]
            col2.dataframe(df.style.format({"Weight (%)": "{:.2f}%"}))
        
        except Exception as e:
            st.error(f"Error fetching data for {ticker}: {e}")

except Exception as e:
    st.error(f"❌ Page failed to load: {e}")