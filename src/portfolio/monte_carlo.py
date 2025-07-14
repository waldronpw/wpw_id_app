import numpy as np

def monte_carlo_simulation(
        investment_horizon: int,
        asset_goal: int,
        current_assets: int,
        exp_ret: float,
        exp_vol: float,
        n_sims: int = 10_000,
        ann_contribution: int = 0,
        ann_withdrawal: int = 0,
        inflation: float = 0.0258,
        seed: int = 42
        ) -> dict:
    
        """
        Monte Carlo simulation to estimate the probability of funding a future goal.

        Returns:
            dict: Success rate and summary stats for ending portfolio values.
        """

        np.random.seed(seed)

        try:
            returns = np.random.normal(loc=exp_ret, scale=exp_vol, size=(n_sims, investment_horizon))

            values = np.full(n_sims, current_assets, dtype=np.float64)

            for t in range(investment_horizon):
                inflation_adj_contribution = ann_contribution * ((1 + inflation) ** t)
                inflation_adj_withdrawal = ann_withdrawal * ((1 + inflation) ** t)

                values *= (1 + returns[:, t])
                values += inflation_adj_contribution
                values -+ inflation_adj_withdrawal
                values = np.maximum(0, values)
            
            if np.isnan(values).any():
                 print("NaNs found in simulation output.")
                 return {}
            
            probability_success = np.mean(values >= asset_goal)

            return {
                 "probability_success": round(probability_success, 4),
                 "asset_goal": round(asset_goal, 2),
                 "median_ending_assets": round(np.median(values), 2),
                 "5th_percentile": round(np.percentile(values, 5), 2),
                 "95th_percentile": round(np.percentile(values, 95), 2)
            }
         
        except Exception as e:
             print(f"Monte Carlo simulation error: {e}")
             return {}