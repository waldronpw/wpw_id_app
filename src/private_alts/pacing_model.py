import pandas as pd

def pacing_model(
    total_aum: float,
    target_pe_alloc: float,
    initial_commitments: float,
    years: int = 15,
    call_rate: float = 0.25,         # % of committed capital called per year
    dist_rate: float = 0.15,         # % of NAV distributed per year
    growth_rate: float = 0.08        # annual growth on invested capital (NAV)
):
    """
    Simulate a basic PE pacing model.

    Parameters:
        total_aum (float): Total investable assets (e.g., 1e9 for $1B)
        target_pe_alloc (float): Target allocation to private equity (e.g., 0.1 for 10%)
        initial_commitments (float): Committed capital at year 0
        years (int): Number of years to simulate
        call_rate (float): Annual capital call rate (e.g., 0.25 = 25% of commitments)
        dist_rate (float): Annual distribution rate on NAV
        growth_rate (float): Annual growth of NAV

    Returns:
        pd.DataFrame: Yearly summary of PE cashflows, NAV, and commitments
    """

    data = []
    unfunded_commitments = initial_commitments
    nav = 0.0
    total_commitments = initial_commitments

    for year in range(1, years + 1):
        capital_called = min(unfunded_commitments, call_rate * total_commitments)
        nav = (nav + capital_called) * (1 + growth_rate)
        distributions = nav * dist_rate
        nav -= distributions
        unfunded_commitments -= capital_called

        # Determine new commitments to hit target
        target_pe_value = total_aum * target_pe_alloc
        gap = target_pe_value - (nav + unfunded_commitments)
        new_commitment = max(gap, 0)
        total_commitments = new_commitment
        unfunded_commitments += new_commitment

        data.append({
            "Year": year,
            "Capital Called": capital_called,
            "Distributions": distributions,
            "NAV": nav,
            "Unfunded Commitments": unfunded_commitments,
            "New Commitments": new_commitment
        })

    return pd.DataFrame(data)