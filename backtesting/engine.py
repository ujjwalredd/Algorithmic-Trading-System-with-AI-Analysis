
import numpy as np
import pandas as pd
from datetime import datetime

class BacktestEngine:
    def __init__(self, initial_capital=100000, commission=0.001):
        self.initial_capital = initial_capital
        self.commission = commission
        self.results = {}
        
    def run_backtest(self, data, signals, symbol):
        """Run backtest on strategy signals"""
        positions = pd.DataFrame(index=signals.index)
        positions[symbol] = signals['position']
        
        # Calculate returns
        portfolio = pd.DataFrame(index=signals.index)
        portfolio['positions'] = positions[symbol]
        portfolio['price'] = data['Close']
        portfolio['holdings'] = positions[symbol] * portfolio['price'] * 1000
        portfolio['cash'] = self.initial_capital - (positions[symbol].diff().fillna(0) * portfolio['price'] * 1000).cumsum()
        portfolio['total'] = portfolio['holdings'] + portfolio['cash']
        portfolio['returns'] = portfolio['total'].pct_change()
        
        # Transaction costs
        portfolio['trades'] = positions[symbol].diff().fillna(0).abs()
        portfolio['commission'] = portfolio['trades'] * portfolio['price'] * 1000 * self.commission
        portfolio['net_total'] = portfolio['total'] - portfolio['commission'].cumsum()
        portfolio['net_returns'] = portfolio['net_total'].pct_change()
        
        return portfolio
    
    def calculate_metrics(self, portfolio):
        """Calculate performance metrics"""
        returns = portfolio['net_returns'].dropna()
        
        metrics = {
            'total_return': (portfolio['net_total'].iloc[-1] / self.initial_capital - 1) * 100,
            'annualized_return': returns.mean() * 252 * 100,
            'volatility': returns.std() * np.sqrt(252) * 100,
            'sharpe_ratio': (returns.mean() * 252) / (returns.std() * np.sqrt(252)),
            'max_drawdown': self.calculate_max_drawdown(portfolio['net_total']),
            'win_rate': (returns > 0).sum() / len(returns) * 100,
            'profit_factor': returns[returns > 0].sum() / abs(returns[returns < 0].sum()),
            'trades': portfolio['trades'].sum() / 2
        }
        
        # Value at Risk (VaR) - 95% confidence
        metrics['var_95'] = np.percentile(returns, 5) * 100
        
        # Conditional Value at Risk (CVaR)
        metrics['cvar_95'] = returns[returns <= np.percentile(returns, 5)].mean() * 100
        
        return metrics
    
    def calculate_max_drawdown(self, equity_curve):
        """Calculate maximum drawdown"""
        cumulative = (1 + equity_curve.pct_change()).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min() * 100