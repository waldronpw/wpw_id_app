# Standard WPW Model Asset Allocations
models = {
    "Conservative": {
        "US Equity": 0.270,
        "Intl Equity": 0.110,
        "Emerging Markets": 0.000,
        "Bonds": 0.600,
        "Cash": 0.020
    },
    "Moderate Conservative": {
        "US Equity": 0.320,
        "Intl Equity": 0.140,
        "Emerging Markets": 0.020,
        "Bonds": 0.500,
        "Cash": 0.020
    },
    "Moderate": {
        "US Equity": 0.380,
        "Intl Equity": 0.160,
        "Emerging Markets": 0.040,
        "Bonds": 0.400,
        "Cash": 0.020
    },
    "Growth": {
        "US Equity": 0.420,
        "Intl Equity": 0.200,
        "Emerging Markets": 0.060,
        "Bonds": 0.300,
        "Cash": 0.020
    },
    "Aggressive": {
        "US Equity": 0.480,
        "Intl Equity": 0.220,
        "Emerging Markets": 0.080,
        "Bonds": 0.200,
        "Cash": 0.020
    },
}

# WPW Capital Market Expectations
capital_market_expectations = {
    "US Equity": 0.065,
    "Intl Equity": 0.075,
    "Bonds": 0.045,
    "Cash": 0.035,
    "Emerging Markets": 0.085
}

# WPW Covariance Matrix for selected asset classes
covariance_matrix = {
    "US Equity": {
        "US Equity": 0.0225,
        "Intl Equity": 0.0180,
        "Emerging Markets": 0.0250,
        "Bonds": 0.0040,
        "Cash": 0.0002
    },
    "Intl Equity": {
        "US Equity": 0.0180,
        "Intl Equity": 0.0300,
        "Emerging Markets": 0.0280,
        "Bonds": 0.0050,
        "Cash": 0.0003
    },
    "Emerging Markets": {
        "US Equity": 0.0250,
        "Intl Equity": 0.0280,
        "Emerging Markets": 0.0400,
        "Bonds": 0.0060,
        "Cash": 0.0005
    },
    "Bonds": {
        "US Equity": 0.0040,
        "Intl Equity": 0.0050,
        "Emerging Markets": 0.0060,
        "Bonds": 0.0100,
        "Cash": 0.0004
    },
    "Cash": {
        "US Equity": 0.0002,
        "Intl Equity": 0.0003,
        "Emerging Markets": 0.0005,
        "Bonds": 0.0004,
        "Cash": 0.0001
    }
}

# WPW ETF Proxies for asset class performance
etf_proxies = {
    "US Equity": "VTI",               # Total US Market
    "Intl Equity": "VXUS",            # Total Intl Market
    "Emerging Markets": "VWO",        # EM Stocks
    "Bonds": "BND",                   # US Aggregate Bond
    "Cash": "SHV"                     # Short-term Treasuries (proxy for cash)
}