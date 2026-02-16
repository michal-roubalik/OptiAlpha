import numpy as np
import pandas as pd
from sklearn.linear_model import BayesianRidge
import logging

logger = logging.getLogger(__name__)

class BayesianAlphaModel:
    """
    Production-grade Bayesian Inference.
    
    Uses Analytical Bayesian Linear Regression (BayesianRidge) instead of MCMC.
    This provides the EXACT posterior distribution instantly, without 
    incurring the massive compilation overhead of PyMC inside loops.
    """
    def __init__(self, n_samples: int = 1000):
        # n_samples is used for scenario generation, not fitting
        self.n_samples = n_samples
        # The 'Engine': Solves P(Beta | Data) analytically
        # compute_score=True allows us to access the marginal likelihood if needed
        self.model = BayesianRidge(
            compute_score=True, 
            fit_intercept=True
        )

    def fit(self, returns: pd.Series, signal: pd.Series):
        """
        Fits the Bayesian model analytically.
        Time complexity: O(N) instead of O(N * Samples).
        """
        # Scikit-learn expects shape (N_samples, N_features)
        X = signal.values.reshape(-1, 1)
        y = returns.values
        
        self.model.fit(X, y)

    def predict_scenarios(self, current_signal: float, n_scenarios: int = 1000) -> np.ndarray:
        """
        Generates probabilistic scenarios from the Posterior Predictive Distribution.
        """
        # 1. Get the Mean and Standard Deviation of the prediction
        # 'return_std=True' captures the Bayesian uncertainty (Epistemic + Aleatoric)
        X_pred = np.array([[current_signal]])
        y_mean, y_std = self.model.predict(X_pred, return_std=True)
        
        # 2. Sample from this analytical distribution
        # This is mathematically equivalent to the MCMC trace but instant
        scenarios = np.random.normal(loc=y_mean[0], scale=y_std[0], size=n_scenarios)
        
        return scenarios