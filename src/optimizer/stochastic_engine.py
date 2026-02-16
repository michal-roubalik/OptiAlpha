import pulp
import numpy as np
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

class StochasticOptimizer:
    """
    Constructs a portfolio using Mixed-Integer Linear Programming (MILP).
    Optimizes for Expected Return subject to CVaR (risk) and Neutrality constraints.
    """
    def __init__(self, 
                 tickers: List[str], 
                 scenarios: Dict[str, np.ndarray], 
                 betas: Dict[str, float],
                 alpha_confidence: float = 0.95):
        
        self.tickers = tickers
        self.scenarios = scenarios
        self.betas = betas
        self.alpha_confidence = alpha_confidence
        self.n_scenarios = len(next(iter(scenarios.values())))

    def solve(self, max_positions: int = 4) -> Dict[str, float]:
        """
        Solves the optimization problem. Returns a dictionary of asset weights.
        """
        prob = pulp.LpProblem("Stochastic_Portfolio_Opt", pulp.LpMaximize)

        # --- Variables ---
        # w: Portfolio weights (-100% to +100%)
        w = pulp.LpVariable.dicts("w", self.tickers, lowBound=-1.0, upBound=1.0)
        # z: Binary indicator (1 if asset is traded, 0 otherwise)
        z = pulp.LpVariable.dicts("z", self.tickers, cat=pulp.LpBinary)
        # CVaR Auxiliaries
        zeta = pulp.LpVariable("zeta", lowBound=-10, upBound=10)
        d = pulp.LpVariable.dicts("loss_deviation", range(self.n_scenarios), lowBound=0)

        # --- Objective: Maximize Expected Return ---
        exp_returns = {t: np.mean(self.scenarios[t]) for t in self.tickers}
        prob += pulp.lpSum([exp_returns[t] * w[t] for t in self.tickers])

        # --- Constraints ---
        
        # 1. Market Neutrality (Net Beta = 0)
        prob += pulp.lpSum([w[t] * self.betas[t] for t in self.tickers]) == 0
        
        # 2. Dollar Neutrality (Longs = Shorts)
        prob += pulp.lpSum([w[t] for t in self.tickers]) == 0
        
        # 3. Cardinality (Max active positions)
        prob += pulp.lpSum([z[t] for t in self.tickers]) <= max_positions
        
        # 4. Linking Weights to Binary Indicators
        # If z=0, w must be 0. If z=1, w can be between -1 and 1.
        for t in self.tickers:
            prob += w[t] <= z[t]
            prob += w[t] >= -z[t]

        # 5. CVaR Risk Constraint (95% Confidence)
        # Defines the tail loss distribution
        for s in range(self.n_scenarios):
            port_return_s = pulp.lpSum([w[t] * self.scenarios[t][s] for t in self.tickers])
            prob += d[s] >= -port_return_s - zeta

        avg_tail_loss = zeta + (1 / ((1 - self.alpha_confidence) * self.n_scenarios)) * pulp.lpSum([d[s] for s in range(self.n_scenarios)])
        
        # Hard limit: Conditional Value at Risk cannot exceed 2% daily
        prob += avg_tail_loss <= 0.02 

        # --- Solve ---
        solver = pulp.PULP_CBC_CMD(msg=False)
        prob.solve(solver)

        if pulp.LpStatus[prob.status] == 'Optimal':
            return {t: round(pulp.value(w[t]), 4) for t in self.tickers if abs(pulp.value(w[t])) > 1e-4}
        else:
            return {}