from typing import Tuple, List
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator

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


def plot_pacing_visual(nav_accumulation: list, target_pe_nav: float):
    years = list(range(1, len(nav_accumulation) + 1))
    target_line = [target_pe_nav] * len(years)

    fig, ax = plt.subplots(figsize=(8, 3))

    # Plot lines
    ax.plot(years, nav_accumulation, marker='o', label="Projected PE NAV", linewidth=2)
    ax.plot(years, target_line, linestyle='--', color='gray', label="Target PE NAV")

    # Axis formatting
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"${x:,.0f}"))

    ax.tick_params(which="both", labelsize=6, length=0)

    ax.set_xlabel("Year", fontsize=6)
    # ax.set_ylabel("NAV")

    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(frameon=False, fontsize=6)

    # Remove spines and title
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.set_title("")
    return fig