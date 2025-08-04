import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class DataLoader:
    def __init__(self):
        self.cache = {}
    
    def get_stock_data(self, symbols, start_date, end_date, interval='1d'):
        if isinstance(symbols, str):
            symbols = [symbols]
        
        data = {}
        for symbol in symbols:
            if symbol not in self.cache:
                df = yf.download(symbol, start=start_date, end=end_date, interval=interval)
                df['Returns'] = df['Close'].pct_change()
                df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
                self.cache[symbol] = df
            data[symbol] = self.cache[symbol]
        
        return data
    
    def get_sp500_symbols(self, limit=50):
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 
                  'BRK-B', 'JPM', 'JNJ', 'V', 'PG', 'UNH', 'HD', 'MA',
                  'DIS', 'BAC', 'ADBE', 'CRM', 'XOM', 'CVX', 'PFE', 'CSCO',
                  'TMO', 'ABBV', 'ACN', 'COST', 'NKE', 'MRK', 'LLY']
        return symbols[:limit]