
import numpy as np
import pandas as pd

class MomentumStrategy:
    def __init__(self, lookback=20, holding_period=5):
        self.lookback = lookback
        self.holding_period = holding_period
        
    def calculate_momentum(self, prices):
        """Calculate price momentum"""
        return prices.pct_change(self.lookback)
    
    def generate_signals(self, data):
        """Generate momentum-based trading signals"""
        signals = pd.DataFrame(index=data.index)
        signals['price'] = data['Close']
        signals['returns'] = data['Close'].pct_change()
        signals['momentum'] = self.calculate_momentum(data['Close'])
        
        # Volatility adjustment
        signals['volatility'] = signals['returns'].rolling(self.lookback).std()
        signals['vol_adjusted_momentum'] = signals['momentum'] / signals['volatility']
        
        # Generate signals
        signals['signal'] = 0
        momentum_threshold = signals['vol_adjusted_momentum'].rolling(252).quantile(0.8)
        
        signals.loc[signals['vol_adjusted_momentum'] > momentum_threshold, 'signal'] = 1
        signals.loc[signals['vol_adjusted_momentum'] < -momentum_threshold, 'signal'] = -1
        
        # Position sizing based on momentum strength
        signals['position'] = signals['signal'] * np.minimum(
            abs(signals['vol_adjusted_momentum']) / 2, 1
        )
        
        return signals