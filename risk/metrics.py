
import numpy as np
import pandas as pd
from scipy import stats

class RiskMetrics:
    def __init__(self, confidence_level=0.95):
        self.confidence_level = confidence_level
        
    def calculate_var(self, returns, method='historical'):
        """Calculate Value at Risk"""
        if method == 'historical':
            return np.percentile(returns, (1 - self.confidence_level) * 100)
        elif method == 'parametric':
            mean = returns.mean()
            std = returns.std()
            return mean + std * stats.norm.ppf(1 - self.confidence_level)
        elif method == 'monte_carlo':
            # Monte Carlo VaR
            simulations = 10000
            mean = returns.mean()
            std = returns.std()
            simulated_returns = np.random.normal(mean, std, simulations)
            return np.percentile(simulated_returns, (1 - self.confidence_level) * 100)
    
    def calculate_cvar(self, returns):
        """Calculate Conditional Value at Risk (Expected Shortfall)"""
        var = self.calculate_var(returns)
        return returns[returns <= var].mean()
    
    def calculate_beta(self, portfolio_returns, market_returns):
        """Calculate portfolio beta"""
        covariance = np.cov(portfolio_returns, market_returns)[0, 1]
        market_variance = np.var(market_returns)
        return covariance / market_variance
    
    def calculate_information_ratio(self, portfolio_returns, benchmark_returns):
        """Calculate Information Ratio"""
        active_returns = portfolio_returns - benchmark_returns
        tracking_error = active_returns.std()
        return active_returns.mean() / tracking_error if tracking_error != 0 else 0
    
    def calculate_sortino_ratio(self, returns, target_return=0):
        """Calculate Sortino Ratio"""
        excess_returns = returns - target_return
        downside_returns = excess_returns[excess_returns < 0]
        downside_deviation = np.sqrt((downside_returns ** 2).mean())
        return excess_returns.mean() / downside_deviation if downside_deviation != 0 else 0
    
    def calculate_calmar_ratio(self, returns, max_drawdown):
        """Calculate Calmar Ratio"""
        annual_return = returns.mean() * 252
        return annual_return / abs(max_drawdown) if max_drawdown != 0 else 0