
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint, adfuller
from scipy import stats

class PairsTradingStrategy:
    def __init__(self, lookback=60, entry_zscore=2.0, exit_zscore=0.5):
        self.lookback = lookback
        self.entry_zscore = entry_zscore
        self.exit_zscore = exit_zscore
        
    def find_cointegrated_pairs(self, data_dict, p_value_threshold=0.05):
        """Find cointegrated pairs using Engle-Granger test"""
        symbols = list(data_dict.keys())
        n = len(symbols)
        pairs = []
        
        for i in range(n):
            for j in range(i+1, n):
                try:
                    stock1 = data_dict[symbols[i]]['Close']
                    stock2 = data_dict[symbols[j]]['Close']
                    
                    # Check if data is valid
                    if stock1.empty or stock2.empty:
                        continue
                    
                    # Align the series
                    aligned = pd.DataFrame({
                        'stock1': stock1,
                        'stock2': stock2
                    }).dropna()
                    
                    if len(aligned) < self.lookback:
                        continue
                    
                    # Test for cointegration
                    score, p_value, _ = coint(aligned['stock1'], aligned['stock2'])
                    
                    if p_value < p_value_threshold:
                        # Calculate hedge ratio using OLS
                        hedge_ratio = np.polyfit(aligned['stock2'], aligned['stock1'], 1)[0]
                        spread = aligned['stock1'] - hedge_ratio * aligned['stock2']
                        
                        # Test spread for stationarity
                        adf_result = adfuller(spread)
                        if adf_result[1] < p_value_threshold:
                            pairs.append({
                                'pair': (symbols[i], symbols[j]),
                                'p_value': p_value,
                                'hedge_ratio': hedge_ratio,
                                'half_life': self.calculate_half_life(spread)
                            })
                except Exception as e:
                    # Silently skip pairs that can't be processed
                    continue
        
        return sorted(pairs, key=lambda x: x['p_value'])
    
    def calculate_half_life(self, spread):
        """Calculate mean reversion half-life using Ornstein-Uhlenbeck process"""
        try:
            spread_lag = spread.shift(1)
            spread_diff = spread - spread_lag
            spread_lag = spread_lag[1:]
            spread_diff = spread_diff[1:]
            
            if len(spread_lag) < 2:
                return 30  # Default half-life
            
            model = np.polyfit(spread_lag, spread_diff, 1)
            
            if model[0] == 0:
                return 30  # Default half-life if no mean reversion
            
            half_life = -np.log(2) / model[0]
            return max(1, int(half_life))  # Ensure positive half-life
        except Exception:
            return 30  # Default half-life on error
    
    def generate_signals(self, stock1_data, stock2_data, hedge_ratio):
        """Generate pairs trading signals"""
        signals = pd.DataFrame(index=stock1_data.index)
        
        # Calculate spread
        signals['stock1_price'] = stock1_data['Close']
        signals['stock2_price'] = stock2_data['Close']
        signals['spread'] = signals['stock1_price'] - hedge_ratio * signals['stock2_price']
        
        # Calculate z-score
        spread_mean = signals['spread'].rolling(self.lookback).mean()
        spread_std = signals['spread'].rolling(self.lookback).std()
        signals['zscore'] = (signals['spread'] - spread_mean) / spread_std
        
        # Generate signals
        signals['long_spread'] = signals['zscore'] < -self.entry_zscore
        signals['short_spread'] = signals['zscore'] > self.entry_zscore
        signals['exit_spread'] = abs(signals['zscore']) < self.exit_zscore
        
        # Positions
        signals['position1'] = 0
        signals['position2'] = 0
        
        # Long spread: buy stock1, sell stock2
        signals.loc[signals['long_spread'], 'position1'] = 1
        signals.loc[signals['long_spread'], 'position2'] = -hedge_ratio
        
        # Short spread: sell stock1, buy stock2
        signals.loc[signals['short_spread'], 'position1'] = -1
        signals.loc[signals['short_spread'], 'position2'] = hedge_ratio
        
        # Exit positions
        signals.loc[signals['exit_spread'], 'position1'] = 0
        signals.loc[signals['exit_spread'], 'position2'] = 0
        
        signals['position1'] = signals['position1'].fillna(method='ffill').fillna(0)
        signals['position2'] = signals['position2'].fillna(method='ffill').fillna(0)
        
        return signals