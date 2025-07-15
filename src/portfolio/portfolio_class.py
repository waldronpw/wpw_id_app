import sys
import os
import pandas as pd
from typing import Dict, Optional
import scipy.stats as stats
import yfinance as yf
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

# Add the project root directory to the path (one level up from /src)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))


class Portfolio:
    def __init__(self, name: str, asset_allocation: Dict[str, float]):
        self.name = name
        self.asset_allocation = asset_allocation
        self.historical_returns: Optional[pd.Series] = None
    
    def summary(self) -> None:
        """Prints a basic summary of the portfolio."""
        print(f"Portfolio: {self.name}")
        print("Asset Allocation:")
        for asset, weight in self.asset_allocation.items():
            print(f"{asset}: {weight:.1%}")
    
    def compare_allocation(self, other: "Portfolio") -> Dict[str, float]:
        """
        Compares asset allocation with another portfolio.

        Args:
            other (Portfolio): The other (likely proposed) portfolio to compare.

        Returns:
            dic: Differences in weights by asset class.
        """
        all_assets = set(self.asset_allocation) | set(other.asset_allocation)
        return {
            asset: self.asset_allocation.get(asset, 0.0) - other.asset_allocation.get(asset, 0.0) for asset in all_assets
        }
    
    def expected_return(self, cme: Dict[str, float]) -> Optional[float]:
        """
        Calculates expected return using provided capital market expectations.
        
        Args:
            cme(dict): Asset calss -> expected returns (decimal form)
        
        Returns:
            float: Weighted porfolio return, or None if missing data.
        """

        try:
            return sum(self.asset_allocation[asset] * cme[asset] for asset in self.asset_allocation)
        except KeyError as missing:
            print(f"Missing CME data for {missing}")
            return None
    
    def expected_std(self, covariance_matrix: Dict[str, Dict[str, float]]) -> Optional[float]:
        """
        Calculates the portfolio's standard deviation using a covariance matrix.

        Args:
            covariance_matrix (dict): Nested dict of covariances between asset classes.

        Returns:
            float: Portfolio standard deviation, or None if missing data.
        """
        try:
            variance = 0.0
            for asset_i in self.asset_allocation:
                for asset_j in self.asset_allocation:
                    weight_i = self.asset_allocation[asset_i]
                    weight_j = self.asset_allocation[asset_j]
                    cov_ij = covariance_matrix[asset_i][asset_j]
                    variance += weight_i * weight_j * cov_ij
            return variance**0.5
        except KeyError as missing:
            print(f"Missing covariance data for {missing}")
            return None
    
    def sharpe_ratio(self, cme_returns: Dict[str, float], covariance_matrix: Dict[str, Dict[str, float]], risk_free_rate: float = 0.02) -> Optional[float]:
        """
        Calculates the Sharpe ratio of the portfolio.

        Args:
            cme_returns (dict): Asset class -> expected return.
            covariance_matrix (dict): Asset class -> covariance with others.
            risk_free_rate (float): Annual risk-free rate.

        Returns:
            float: Sharpe ratio, or None if data is missing.
        """
        exp_ret = self.expected_return(cme_returns)
        exp_std = self.expected_std(covariance_matrix)
        if exp_ret is None or exp_std is None or exp_std == 0:
            return None
        return (exp_ret - risk_free_rate) / exp_std
    
    def value_at_risk(self, cme: Dict[str, float], cov: Dict[str, Dict[str, float]], confidence_level: float = 0.95) -> Optional[float]:
        """
        Calculates parametric (normal) Value at Risk (VaR).

        Args:
            cme (dict): Capital market expectations (mean returns).
            cov (dict): Covariance matrix.
            confidence_level (float): Confidence level (e.g., 0.95 for 95% VaR).

        Returns:
            float: One-period VaR (as a positive number), or None if missing data.
        """
        mu = self.expected_return(cme)
        sigma = self.expected_std(cov)
        if mu is None or sigma is None:
            return None
        z = stats.norm.ppf(1 - confidence_level)
        return -(mu + z * sigma)  # Flip sign to express as positive potential loss

    def conditional_value_at_risk(self, cme: Dict[str, float], cov: Dict[str, Dict[str, float]], confidence_level: float = 0.95) -> Optional[float]:
        """
        Calculates parametric Conditional Value at Risk (CVaR), or Expected Shortfall.

        Args:
            cme (dict): Capital market expectations.
            cov (dict): Covariance matrix.
            confidence_level (float): Confidence level (e.g., 0.95).

        Returns:
            float: One-period CVaR (as a positive number), or None if missing data.
        """
        mu = self.expected_return(cme)
        sigma = self.expected_std(cov)
        if mu is None or sigma is None:
            return None
        z = stats.norm.ppf(1 - confidence_level)
        pdf = stats.norm.pdf(z)
        cvar = -(mu + (pdf / (1 - confidence_level)) * sigma)
        return cvar
    
    def compare_to(self, other: "Portfolio", cme: Dict[str, float], cov: Dict[str, Dict[str, float]], risk_free_rate: float = 0.02) -> None:
        """
        Prints a summary comparison between this portfolio and another, including
        allocation differences and a risk/return comparison table.

        Args:
            other (Portfolio): Portfolio to compare against.
            cme (dict): Capital market expectations.
            cov (dict): Covariance matrix.
            risk_free_rate (float): Risk-free rate for Sharpe ratio.
        """
        print(f"\n--- Portfolio Comparison: {self.name} vs {other.name} ---")

        # 1. Allocation Differences
        print("\nOver/Underweights (Current vs. Proposed):")
        diff = self.compare_allocation(other)
        for asset, delta in diff.items():
            print(f" - {asset}: {delta:.1%}")

        # 2. Risk/Return/Sharpe
        metrics = {
            "Expected Return": {
                self.name: self.expected_return(cme),
                other.name: other.expected_return(cme)
            },
            "Standard Deviation": {
                self.name: self.expected_std(cov),
                other.name: other.expected_std(cov)
            },
            "Sharpe Ratio": {
                self.name: self.sharpe_ratio(cme, cov, risk_free_rate),
                other.name: other.sharpe_ratio(cme, cov, risk_free_rate)
                },
            "VaR (95%)": {
                self.name: self.value_at_risk(cme, cov, 0.95),
                other.name: other.value_at_risk(cme, cov, 0.95)
            },
            "CVaR (95%)": {
                self.name: self.conditional_value_at_risk(cme, cov, 0.95),
                other.name: other.conditional_value_at_risk(cme, cov, 0.95)
            }
        }

        df = pd.DataFrame(metrics).T
        df.columns.name = "Portfolio"

        def format_row(row):
            label = row.name
            if "Return" in label:
                return row.apply(lambda x: f"{x:.2%}" if pd.notnull(x) else "N/A")
            elif "Deviation" in label:
                return row.apply(lambda x: f"{x:.2%}" if pd.notnull(x) else "N/A")
            elif "Sharpe" in label:
                return row.apply(lambda x: f"{x:.2f}" if pd.notnull(x) else "N/A")
            elif "VaR" in label:
                return row.apply(lambda x: f"{x:.2%}" if pd.notnull(x) else "N/A")
            else:
                return row.apply(lambda x: f"{x:.2f}" if pd.notnull(x) else "N/A")

        df = df.apply(format_row, axis=1)

        print("\nRisk/Return Metrics:")
        print(df.T)
    
    def simulate_historical_performance(self, etf_proxies: Dict[str, str], start: str, end: Optional[str] = None) -> pd.Series:
        """
        Simulates historical monthly returns for the portfolio using ETF proxies.

        Args:
            etf_proxies (dict): Asset class -> ETF ticker mapping.
            start (str): Start date (e.g., '2005-01-01').
            end (str): End date (default: today).

        Returns:
            pd.Series: Simulated monthly portfolio returns.
        """
        if end is None:
            end = dt.date.today().strftime("%Y-%m-%d")

        # Step 1: Get tickers
        tickers = [etf_proxies[asset] for asset in self.asset_allocation if asset in etf_proxies]
        price_data = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)["Close"]
        price_data = price_data.resample("ME").last()

        # Step 2: Monthly returns
        monthly_returns = price_data.pct_change().dropna()

        # Step 3: Apply weights
        weights = {
            etf_proxies[asset]: weight
            for asset, weight in self.asset_allocation.items()
            if asset in etf_proxies
        }

        # Step 4: Compute weighted returns
        monthly_returns = monthly_returns[weights.keys()]
        weighted_returns = monthly_returns.mul(pd.Series(weights), axis=1).sum(axis=1)

        self.historical_returns = weighted_returns
        return weighted_returns
    
    def plot_cumulative_return(self, starting_value: float = 10000.0, benchmark: Optional[pd.Series] = None, benchmark_label: str = "60/40 Benchmark") -> None:
        """
        Plots growth of a $10,000 investment vs. an optional benchmark.

        Args:
            starting_value (float): Initial investment amount.
            benchmark (pd.Series): Optional benchmark return series.
            benchmark_label (str): Label for the benchmark line.
        """
        if self.historical_returns is None:
            print("No historical return data. Please run simulate_historical_performance() first.")
            return

        cumulative = (1 + self.historical_returns).cumprod() * starting_value

        fig, ax = plt.subplots(figsize=(10, 5))
        cumulative.plot(ax=ax, label=self.name)

        if benchmark is not None:
            aligned_benchmark = benchmark[cumulative.index.min():cumulative.index.max()]
            benchmark_growth = (1 + aligned_benchmark).cumprod() * starting_value
            benchmark_growth.plot(ax=ax, label=benchmark_label, linestyle="--", color="gray")

        # Aesthetics
        ax.set_title(f"{self.name} – Growth of ${int(starting_value):,}")
        ax.set_ylabel("Portfolio Value ($)")
        ax.yaxis.set_major_formatter(mtick.StrMethodFormatter('${x:,.0f}'))
        ax.spines["top"].set_visible(False)
        # ax.spines["bottom"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="x", which="both", length=0)
        ax.tick_params(axis="y", which="both", length=0)
        ax.legend()
        plt.grid(axis="y", alpha=0.25)
        plt.tight_layout()
        plt.show()
    
    def calculate_max_drawdown(self) -> Optional[Dict[str, any]]:
        """
        Calculates the maximum drawdown and related metadata.

        Returns:
            dict: {
                'max_drawdown': float (as negative decimal),
                'start_date': Timestamp,
                'recovery_date': Timestamp or None,
                'recovery_months': int or None
            }
        """
        if self.historical_returns is None or self.historical_returns.empty:
            print("No historical return data. Please run simulate_historical_performance() first.")
            return None

        cumulative = (1 + self.historical_returns).cumprod()
        peak = cumulative.cummax()
        drawdown = (cumulative - peak) / peak

        if drawdown.empty or drawdown.isna().all():
            print("Drawdown series is empty or all NaNs.")
            return None

        min_dd = drawdown.min()
        dd_end = drawdown.idxmin()

        dd_start_candidates = cumulative[:dd_end][cumulative[:dd_end] == peak[:dd_end]].index
        dd_start = dd_start_candidates[-1] if not dd_start_candidates.empty else drawdown.index[0]

        # Try to find recovery date: when cumulative value returns to previous peak
        recovery = cumulative[dd_end:]
        recovery_date = recovery[recovery >= peak[dd_start]].index
        recovery_date = recovery_date[0] if not recovery_date.empty else None

        recovery_months = (
            (recovery_date.to_period("M") - dd_start.to_period("M")).n
            if recovery_date else None
        )

        return {
            "max_drawdown": min_dd,
            "start_date": dd_start,
            "recovery_date": recovery_date,
            "recovery_months": recovery_months
        }
    
    def summarize_backtest(self) -> None:
        """
        Prints summary statistics of historical performance, including drawdown details.
        """
        if self.historical_returns is None:
            print("No historical return data. Please run simulate_historical_performance() first.")
            return

        total_return = (1 + self.historical_returns).prod() - 1
        annualized_return = (1 + total_return)**(12 / len(self.historical_returns)) - 1
        annualized_vol = self.historical_returns.std() * (12 ** 0.5)
        sharpe_ratio = annualized_return / annualized_vol if annualized_vol else None
        dd_info = self.calculate_max_drawdown()

        print(f"\n--- {self.name} Historical Backtest Summary ---")
        print(f"Annualized Return:      {annualized_return:.2%}")
        print(f"Annualized Volatility:  {annualized_vol:.2%}")
        print(f"Sharpe Ratio:           {sharpe_ratio:.2f}" if sharpe_ratio else "Sharpe Ratio:   N/A")

        if dd_info:
            print(f"Max Drawdown:           {dd_info['max_drawdown']:.2%}")
            print(f"Drawdown Start:         {dd_info['start_date'].strftime('%b %Y')}")
            print(f"Recovery Date:          {dd_info['recovery_date'].strftime('%b %Y') if dd_info['recovery_date'] else 'Not yet recovered'}")
            print(f"Drawdown Recovery:      {dd_info['recovery_months']} months" if dd_info['recovery_months'] else "Recovery: N/A")
        else:
            print("Max Drawdown:           N/A")
    
    def crisis_performance(self, periods: Dict[str, Dict[str, pd.Timestamp]], risk_free_rate: float = 0.0) -> pd.DataFrame:
        """
        Evaluates performance over predefined crisis periods.

        Args:
            periods (dict): Dictionary of crisis names and their start/end timestamps.
            risk_free_rate (float): Assumed annual risk-free rate.

        Returns:
            pd.DataFrame: Performance stats for each crisis.
        """
        if self.historical_returns is None:
            print("No historical return data. Please run simulate_historical_performance() first.")
            return pd.DataFrame()

        results = []
        for name, window in periods.items():
            returns = self.historical_returns[window["start"]:window["end"]]

            if returns.empty:
                results.append({
                    "Period": name,
                    "Return": None,
                    # "Volatility": None,
                    # "Sharpe": None,
                    "Max Drawdown": None
                })
                continue

            cumulative_return = (1 + returns).prod() - 1
            # vol = returns.std() * (12 ** 0.5)
            # sharpe = ((1 + cumulative_return) ** (12 / len(returns)) - 1 - risk_free_rate) / vol if vol else None

            # Drawdown
            cumulative = (1 + returns).cumprod()
            peak = cumulative.cummax()
            drawdown = (cumulative - peak) / peak
            max_dd = drawdown.min()

            results.append({
                "Period": name,
                "Return": f"{cumulative_return:.2%}",
                # "Volatility": vol,
                # "Sharpe": sharpe,
                "Max Drawdown": f"{max_dd:.2%}"
            })

        return pd.DataFrame(results).set_index("Period")
    
    def stress_test(self, shock_type: str, shock_magnitude: float, sensitivities: Dict[str, float]) -> float:
        """
        Estimates portfolio return under a macro shock scenario.

        Args:
            shock_type (str): Label for the type of shock (e.g., 'Inflation Shock').
            shock_magnitude (float): Size of the shock (e.g., +1.5 for +1.5% CPI).
            sensitivities (dict): Mapping of asset classes to sensitivity per 1% shock.

        Returns:
            float: Estimated portfolio return in the stress scenario.
        """
        est_return = 0.0
        for asset, weight in self.asset_allocation.items():
            sensitivity = sensitivities.get(asset, 0.0)
            impact = sensitivity * shock_magnitude
            est_return += weight * impact

        print(f"\nStress Test – {shock_type}:")
        print(f"  Shock Magnitude:     {shock_magnitude:+.2f}%")
        print(f"  Estimated Portfolio Return: {est_return:.2%}")
        return est_return
    
    def monte_carlo_goal_projection(
        self,
        years: int,
        goal: float,
        initial_value: float = 1_000_000,
        num_simulations: int = 10_000,
        annual_contribution: float = 0.0,
        annual_withdrawal: float = 0.0,
        expected_return: Optional[float] = None,
        expected_volatility: Optional[float] = None,
        inflation_rate: float = 0.0258,
        seed: int = 42
    ) -> dict:
        """
        Monte Carlo simulation to estimate the probability of funding a future goal.

        Returns:
            dict: Success rate and summary stats for ending portfolio values.
        """
        import numpy as np

        # --- Input validation ---
        if initial_value <= 0:
            print("Initial value must be greater than 0.")
            return {}

        if years <= 0 or goal <= 0 or num_simulations <= 0:
            print("Invalid simulation parameters.")
            return {}

        np.random.seed(seed)

        # Use expected return and volatility if provided
        if expected_return is None:
            expected_return = self.expected_return(cme={})
        if expected_volatility is None:
            expected_volatility = self.expected_std(covariance_matrix={})

        if expected_return is None or expected_volatility is None:
            print("Missing expected return or volatility. Provide overrides or check CME inputs.")
            return {}

        try:
            # Simulate return paths
            returns = np.random.normal(
                loc=expected_return,
                scale=expected_volatility,
                size=(num_simulations, years)
            )

            # Initialize all simulations with the starting portfolio value
            values = np.full(num_simulations, initial_value, dtype=np.float64)

            for t in range(years):
                infl_adj_contribution = annual_contribution * ((1 + inflation_rate) ** t)
                infl_adj_withdrawal = annual_withdrawal * ((1 + inflation_rate) ** t)

                values *= (1 + returns[:, t])
                values += infl_adj_contribution
                values -= infl_adj_withdrawal
                values = np.maximum(0, values)  # Avoid negative balances

            if np.isnan(values).any():
                print("NaNs found in simulation output.")
                print(f"Sample values: {values[:5]}")
                return {}

            success_rate = np.mean(values >= goal)

            return {
                "success_rate": round(success_rate, 4),
                "goal": round(goal, 2),
                "median_ending_value": round(np.median(values), 2),
                "5th_percentile": round(np.percentile(values, 5), 2),
                "95th_percentile": round(np.percentile(values, 95), 2)
            }

        except Exception as e:
            print(f"Monte Carlo simulation error: {e}")
            return {}
    

def simulate_6040_benchmark(start: str, end: Optional[str] = None) -> pd.Series:
    """
    Simulates a 60/40 portfolio (60% VTI / 40% AGG) using historical monthly returns.

    Args:
        start (str): Start date (e.g., '2005-01-01').
        end (str): End date (default: today).

    Returns:
        pd.Series: Monthly returns of the 60/40 benchmark.
    """
    import yfinance as yf
    import datetime as dt

    if end is None:
        end = dt.date.today().strftime("%Y-%m-%d")

    tickers = ["VTI", "AGG"]
    data = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)["Close"]
    data = data.resample("ME").last()
    monthly_returns = data.pct_change().dropna()

    weights = {"VTI": 0.60, "AGG": 0.40}
    weighted_returns = monthly_returns.mul(pd.Series(weights), axis=1).sum(axis=1)

    return weighted_returns



# Example Usage

def main():
    from config.cme import (
        models,
        capital_market_expectations,
        covariance_matrix,
        etf_proxies,
        crisis_periods,
        inflation_shock_sensitivities,
        rate_shock_sensitivities
        )

    current = Portfolio("Client Current", {"US Equity": 0.40, "Intl Equity": 0.50, "Bonds": 0.10})
    proposed = Portfolio("Proposed",
                        models["Moderate"]
                         )

    # --- Summary of comparative statistics ---
    current.compare_to(proposed, capital_market_expectations, covariance_matrix)

    # --- Simulated Historical Performance ---
    proposed.simulate_historical_performance(etf_proxies, start="2015-01-01")
    benchmark_returns = simulate_6040_benchmark(start="2015-01-01")
    proposed.plot_cumulative_return(starting_value=1_000_000, benchmark=benchmark_returns)
    proposed.summarize_backtest()

    # # --- Crisis Shock Analysis ---
    print("\n--- Crisis Period Performance ---")
    crisis_df = proposed.crisis_performance(crisis_periods)
    print(crisis_df)

    # --- Stress Testing Portfolio ---
    print("\n--- Asset Allocation Stress Testing ---")
    # Estimate for a +1.5% inflation shock
    proposed.stress_test("Inflation Shock", shock_magnitude=1.5, sensitivities=inflation_shock_sensitivities)

    # Estimate for a +2.0% interest rate shock
    proposed.stress_test("Rate Shock", shock_magnitude=-2.0, sensitivities=rate_shock_sensitivities)

    # --- Monte Carlo Simulation for GBI ---
    # Client assumptions
    years = 20
    goal = 10_000_000
    initial_value = 3_500_000
    annual_contribution = 120_000

    rows = []
    print("\n--- Model Allocation Expcted Performance Summary ---\n")
    for model_name, allocation in models.items():
        portfolio = Portfolio(name=model_name, asset_allocation=allocation)

        exp_ret = portfolio.expected_return(capital_market_expectations)
        exp_std = portfolio.expected_std(covariance_matrix)
        risk_free_rate = 0.02
        sharpe_ratio = (exp_ret - risk_free_rate) / exp_std if exp_ret is not None and exp_std else None

        print(f"{model_name} | Expected Return: {exp_ret:.2%} | Std Dev: {exp_std:.2%}")

        sim_result = portfolio.monte_carlo_goal_projection(
            years=years,
            goal=goal,
            initial_value=initial_value,
            annual_contribution=annual_contribution,
            expected_return=exp_ret,
            expected_volatility=exp_std
        )

        row = {
            "Allocation": model_name,
            "Success Rate": sim_result.get("success_rate", 0),
            "Sharpe Ratio": sharpe_ratio,
            "Median Ending Value": sim_result.get("median_ending_value", 0),
            "5th Percentile": sim_result.get("5th_percentile", 0),
            "95th Percentile": sim_result.get("95th_percentile", 0),
            "Goal": sim_result.get("goal", goal)
        }

        rows.append(row)

    # Build DataFrame
    df_results = pd.DataFrame(rows).set_index("Allocation")

    # Format nicely for display
    df_display = df_results.copy()
    df_display["Success Rate"] = df_display["Success Rate"].map("{:.2%}".format)
    df_display["Sharpe Ratio"] = df_display["Sharpe Ratio"].map("{:.2f}".format)

    for col in ["Median Ending Value", "5th Percentile", "95th Percentile", "Goal"]:
        df_display[col] = df_display[col].map("${:,.0f}".format)

    # Print results
    print("\n\n--- Monte Carlo Results by Potential Allocation ---\n")
    print(df_display)




if __name__ == "__main__":
    main()