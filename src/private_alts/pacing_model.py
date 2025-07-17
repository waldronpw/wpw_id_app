from typing import Tuple, List
import pandas as pd

def pacing_model(
    aum: int,
    target_pe_alloc: float,
    years: int,
    growth: float,
    pe_nav_growth: float,
    verbose: bool = False
) -> Tuple[float, float, float, List[float]]:
    """
    Calculates the annual PE commitment needed to achieve a target private equity allocation in the future.

    Parameters:
        aum (int): Current assets under management (AUM).
        target_pe_alloc (float): Target PE allocation (e.g., 0.15 for 15%).
        years (int): Investment horizon in years.
        growth (float): Annual growth rate of the total portfolio (e.g., 0.05 for 5%).
        pe_nav_growth (float): Assumed annual growth rate of PE NAV (e.g., 0.12 for 12%).
        verbose (bool): If True, print year-by-year NAV accumulation.

    Returns:
        fv_portfolio (float): Future value of the portfolio.
        target_pe_nav (float): Target PE NAV in the future.
        annual_commitment (float): Even annual commitment required to hit target NAV.
        nav_accumulation (List[float]): Cumulative NAV buildup each year (for inspection/plotting).
    """
    fv_portfolio = aum * (1 + growth) ** years
    target_pe_nav = fv_portfolio * target_pe_alloc

    discount_factors = [(1 + pe_nav_growth) ** (years - i) for i in range(years)]
    factor_sum = sum(discount_factors)

    annual_commitment = target_pe_nav / factor_sum

    nav_accumulation = []
    cumulative_nav = 0
    for i in range(years):
        nav = annual_commitment * (1 + pe_nav_growth) ** (years - i)
        cumulative_nav += nav
        nav_accumulation.append(cumulative_nav)
        if verbose:
            print(f"Year {i+1}: Annual Commitment ${annual_commitment:,.2f} → NAV: ${cumulative_nav:,.2f}")

    return (fv_portfolio, target_pe_nav, annual_commitment, nav_accumulation)


# Sample global schedule: % of capital returned each year after commitment
GLOBAL_DISTRIBUTION_SCHEDULE = [0.00, 0.00, 0.25, 0.3, 0.25, 0.15, 0.05]

def pacing_model_vintages(
    aum: float,
    target_pe_alloc: float,
    years: int,
    growth: float,
    pe_nav_growth: float,
    verbose: bool = False
) -> pd.DataFrame:
    """
    Vintage-level pacing model for private equity.

    Returns a DataFrame with yearly contributions, NAV, and distributions by vintage.
    """
    fv_portfolio = aum * (1 + growth) ** years
    target_pe_nav = fv_portfolio * target_pe_alloc

    discount_factors = [(1 + pe_nav_growth) ** (years - i) for i in range(years)]
    factor_sum = sum(discount_factors)
    annual_commitment = target_pe_nav / factor_sum

    all_records = []

    for vintage in range(years):
        contribution = annual_commitment
        nav = contribution
        for age in range(len(GLOBAL_DISTRIBUTION_SCHEDULE)):
            year = vintage + age
            nav *= (1 + pe_nav_growth)

            dist_pct = GLOBAL_DISTRIBUTION_SCHEDULE[age]
            distribution = contribution * dist_pct
            nav -= distribution

            all_records.append({
                "Year": year + 1,
                "Vintage": vintage + 1,
                "Fund Age": age,
                "Contribution": contribution if age == 0 else 0,
                "NAV": nav,
                "Distributions": distribution
            })

    df = pd.DataFrame(all_records)
    summary = df.groupby("Year").agg({
        "Contribution": "sum",
        "NAV": "sum",
        "Distributions": "sum"
    })

    if verbose:
        print(summary)

    return summary


def pacing_model_dynamic_reinvestment(
    aum: float,
    target_pe_alloc: float,
    years: int,
    growth: float,
    pe_nav_growth: float,
    distribution_schedule: List[float]
) -> pd.DataFrame:
    """
    A dynamic pacing model that reinvests distributions and adds capital as needed
    to maintain the target private equity allocation over time.
    """
    records = []
    vintages = []  # Track vintages as list of dicts with 'commit_year', 'contribution', etc.

    for year in range(years + len(distribution_schedule)):
        # 1. Update total AUM for year
        total_portfolio_value = aum * (1 + growth) ** year
        target_nav = total_portfolio_value * target_pe_alloc

        # 2. Loop over all vintages to compute NAV and distributions
        total_pe_nav = 0
        distributions_this_year = 0

        for v in vintages:
            age = year - v["commit_year"]
            if age < 0:
                continue
            if age == 0:
                v["nav"] = v["commitment"]
            else:
                v["nav"] *= (1 + pe_nav_growth)

            # Apply distribution
            dist_pct = distribution_schedule[age] if age < len(distribution_schedule) else 0.05
            distribution = v["commitment"] * dist_pct
            v["nav"] -= distribution
            distributions_this_year += distribution

            total_pe_nav += v["nav"]

        # 3. Determine funding need to meet target
        nav_gap = target_nav - total_pe_nav
        reinvestment = distributions_this_year
        additional_commitment = max(nav_gap - reinvestment, 0)

        new_commitment = reinvestment + additional_commitment

        if new_commitment > 0:
            vintages.append({
                "commit_year": year,
                "commitment": new_commitment,
                "nav": new_commitment
            })

        # 4. Record this year's summary
        records.append({
            "Year": year + 1,
            "Target PE NAV": target_nav,
            "Actual PE NAV": total_pe_nav,
            "Distributions Reinvested": reinvestment,
            "Additional Capital": additional_commitment,
            "Total New Commitments": new_commitment
        })

    return pd.DataFrame(records)


# Example usage:
if __name__ == "__main__":
    aum = 10_000_000  # Example AUM
    target_pe_alloc = 0.15  # 15% target allocation
    years = 5  # Investment horizon in years
    growth = 0.065  # Portfolio growth rate (6.5%)
    pe_nav_growth = 0.095  # PE NAV growth rate (9.5%)

    # results = pacing_model(aum, target_pe_alloc, years, growth, pe_nav_growth, verbose=True)
    # print(f"\nFuture Value of Portfolio: ${results[0]:,.2f}")
    # print(f"Target Private Equity NAV: ${results[1]:,.2f}")
    # print(f"Annual Commitment to Private Equity: ${results[2]:,.2f}")

    # vintage_results = pacing_model_vintages(aum, target_pe_alloc, years, growth, pe_nav_growth, verbose=True)
    # print("\nVintage-Level Pacing Model Results:")
    # print(vintage_results)

    dynamic_results = pacing_model_dynamic_reinvestment(aum, target_pe_alloc, years, growth, pe_nav_growth, distribution_schedule=GLOBAL_DISTRIBUTION_SCHEDULE)
    print("\nDynamic Pacing Model Results with Reinvestment:")
    print(dynamic_results)