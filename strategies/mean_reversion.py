import numpy as np
import pandas as pd
from scipy import stats

class MeanReversionStrategy:
    def __init__(self, lookback=20, entry_zscore=2.0, exit_zscore=0.5):
        self.lookback = lookback
        self.entry_zscore = entry_zscore
        self.exit_zscore = exit_zscore
        self.positions = {}
        
    def calculate_zscore(self, prices):
        rolling_mean = prices.rolling(window=self.lookback).mean()
        rolling_std = prices.rolling(window=self.lookback).std()
        zscore = (prices - rolling_mean) / rolling_std
        return zscore
    
    def generate_signals(self, data):
        signals = pd.DataFrame(index=data.index)
        signals['price'] = data['Close']
        signals['zscore'] = self.calculate_zscore(data['Close'])
        
        signals['long_entry'] = signals['zscore'] < -self.entry_zscore
        signals['short_entry'] = signals['zscore'] > self.entry_zscore
        
        signals['long_exit'] = signals['zscore'] > -self.exit_zscore
        signals['short_exit'] = signals['zscore'] < self.exit_zscore
        
        signals['position'] = 0
        signals.loc[signals['long_entry'], 'position'] = 1
        signals.loc[signals['short_entry'], 'position'] = -1
        signals.loc[signals['long_exit'] & (signals['position'].shift(1) > 0), 'position'] = 0
        signals.loc[signals['short_exit'] & (signals['position'].shift(1) < 0), 'position'] = 0
        
        signals['position'] = signals['position'].fillna(method='ffill').fillna(0)
        
        return signals