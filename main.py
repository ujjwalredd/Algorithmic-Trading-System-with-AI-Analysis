import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from data.data_loader import DataLoader
from strategies.mean_reversion import MeanReversionStrategy
from strategies.momentum import MomentumStrategy
from strategies.pairs_trading import PairsTradingStrategy
from backtesting.engine import BacktestEngine
from risk.metrics import RiskMetrics
from utils.technical_indicators import TechnicalIndicators
from ollama_integration import analyze_with_ollama

class TradingSystem:
    def __init__(self):
        self.data_loader = DataLoader()
        self.backtest_engine = BacktestEngine()
        self.risk_metrics = RiskMetrics()
        self.results = {}
        
    def run_all_strategies(self):
        """Run all trading strategies"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365*2)
        
        print("Loading data...")
        symbols = self.data_loader.get_sp500_symbols(limit=30)
        stock_data = self.data_loader.get_stock_data(symbols, start_date, end_date)
        
        print("\nRunning Mean Reversion Strategy...")
        mean_rev_strategy = MeanReversionStrategy()
        mean_rev_results = []
        
        for symbol in symbols[:10]:
            if symbol in stock_data:
                signals = mean_rev_strategy.generate_signals(stock_data[symbol])
                portfolio = self.backtest_engine.run_backtest(stock_data[symbol], signals, symbol)
                metrics = self.backtest_engine.calculate_metrics(portfolio)
                metrics['symbol'] = symbol
                mean_rev_results.append(metrics)
        
        self.results['mean_reversion'] = pd.DataFrame(mean_rev_results)
        
        print("\nRunning Momentum Strategy...")
        momentum_strategy = MomentumStrategy()
        momentum_results = []
        
        for symbol in symbols[:10]:
            if symbol in stock_data:
                signals = momentum_strategy.generate_signals(stock_data[symbol])
                portfolio = self.backtest_engine.run_backtest(stock_data[symbol], signals, symbol)
                metrics = self.backtest_engine.calculate_metrics(portfolio)
                metrics['symbol'] = symbol
                momentum_results.append(metrics)
        
        self.results['momentum'] = pd.DataFrame(momentum_results)
        
        print("\nRunning Pairs Trading Strategy...")
        pairs_strategy = PairsTradingStrategy()
        pairs = pairs_strategy.find_cointegrated_pairs(stock_data)
        
        if pairs:
            best_pair = pairs[0]
            print(f"Best pair found: {best_pair['pair']} with p-value: {best_pair['p_value']:.4f}")
            
            stock1_data = stock_data[best_pair['pair'][0]]
            stock2_data = stock_data[best_pair['pair'][1]]
            signals = pairs_strategy.generate_signals(stock1_data, stock2_data, best_pair['hedge_ratio'])
            
            self.results['pairs_trading'] = {
                'pair': best_pair['pair'],
                'p_value': best_pair['p_value'],
                'hedge_ratio': best_pair['hedge_ratio'],
                'half_life': best_pair['half_life']
            }
        
        return self.results
    
    def analyze_with_ollama(self, base_url: str = "http://localhost:11434", model: str = "llama2"):
        """
        Analyze results with Ollama AI.
        
        Args:
            base_url (str): Ollama API base URL
            model (str): Model name to use
        """
        if not self.results:
            print("❌ No trading results available. Run strategies first.")
            return
        
        print("\n🤖 Starting Ollama Analysis...")
        analyze_with_ollama(self.results, base_url, model)
    
    def generate_report(self):
        """Generate comprehensive performance report"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Strategy Performance Comparison
        ax1 = axes[0, 0]
        strategies = []
        sharpe_ratios = []
        returns = []
        
        for strategy_name, results in self.results.items():
            if isinstance(results, pd.DataFrame) and not results.empty:
                strategies.append(strategy_name)
                sharpe_ratios.append(results['sharpe_ratio'].mean())
                returns.append(results['annualized_return'].mean())
        
        x = np.arange(len(strategies))
        width = 0.35
        
        ax1.bar(x - width/2, returns, width, label='Annualized Return (%)')
        ax1.bar(x + width/2, sharpe_ratios, width, label='Sharpe Ratio')
        ax1.set_xlabel('Strategy')
        ax1.set_ylabel('Value')
        ax1.set_title('Strategy Performance Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels(strategies)
        ax1.legend()
        
        # 2. Risk Metrics
        ax2 = axes[0, 1]
        if 'mean_reversion' in self.results:
            risk_data = self.results['mean_reversion'][['volatility', 'max_drawdown', 'var_95']].mean()
            risk_data.plot(kind='bar', ax=ax2)
            ax2.set_title('Risk Metrics (Mean Reversion)')
            ax2.set_ylabel('Percentage')
            
        # 3. Win Rate Distribution
        ax3 = axes[1, 0]
        win_rates = []
        for strategy_name, results in self.results.items():
            if isinstance(results, pd.DataFrame) and 'win_rate' in results.columns:
                win_rates.extend(results['win_rate'].values)
        
        ax3.hist(win_rates, bins=20, alpha=0.7, color='green')
        ax3.axvline(x=50, color='red', linestyle='--', label='50% threshold')
        ax3.set_xlabel('Win Rate (%)')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Win Rate Distribution Across Strategies')
        ax3.legend()
        
        # 4. Monte Carlo Simulation
        ax4 = axes[1, 1]
        self.run_monte_carlo_simulation(ax4)
        
        plt.tight_layout()
        plt.savefig('trading_strategy_report.png', dpi=300)
        plt.show()
        
        # Print summary statistics
        self.print_summary()
    
    def run_monte_carlo_simulation(self, ax, n_simulations=1000):
        """Run Monte Carlo simulation for portfolio outcomes"""
        initial_value = 100000
        n_days = 252
        
        # Use mean reversion results for simulation parameters
        if 'mean_reversion' in self.results:
            mean_return = self.results['mean_reversion']['annualized_return'].mean() / 252 / 100
            volatility = self.results['mean_reversion']['volatility'].mean() / np.sqrt(252) / 100
        else:
            mean_return = 0.0001
            volatility = 0.02
        
        simulations = np.zeros((n_simulations, n_days))
        
        for i in range(n_simulations):
            daily_returns = np.random.normal(mean_return, volatility, n_days)
            simulations[i] = initial_value * (1 + daily_returns).cumprod()
        
        # Plot percentiles
        percentiles = [10, 25, 50, 75, 90]
        for p in percentiles:
            values = np.percentile(simulations, p, axis=0)
            ax.plot(values, label=f'{p}th percentile')
        
        ax.set_xlabel('Days')
        ax.set_ylabel('Portfolio Value ($)')
        ax.set_title('Monte Carlo Simulation (1000 paths)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def print_summary(self):
        """Print summary of all strategies"""
        print("\n" + "="*80)
        print("ALGORITHMIC TRADING STRATEGY SUMMARY")
        print("="*80)
        
        for strategy_name, results in self.results.items():
            print(f"\n{strategy_name.upper()} STRATEGY:")
            print("-"*40)
            
            if isinstance(results, pd.DataFrame) and not results.empty:
                print(f"Average Sharpe Ratio: {results['sharpe_ratio'].mean():.3f}")
                print(f"Average Annual Return: {results['annualized_return'].mean():.2f}%")
                print(f"Average Max Drawdown: {results['max_drawdown'].mean():.2f}%")
                print(f"Average Win Rate: {results['win_rate'].mean():.2f}%")
                print(f"Best Performing Symbol: {results.loc[results['sharpe_ratio'].idxmax(), 'symbol']}")
            elif isinstance(results, dict):
                for key, value in results.items():
                    print(f"{key}: {value}")

if __name__ == "__main__":
    trading_system = TradingSystem()
    results = trading_system.run_all_strategies()
    trading_system.generate_report()
    
    print("\n" + "="*60)
    print("🤖 AI ANALYSIS")
    print("="*60)
    
    use_ai = input("Would you like to analyze results with AI? (y/n): ").strip().lower()
    
    if use_ai in ['y', 'yes']:
        base_url = input("Enter Ollama base URL (default: http://localhost:11434): ").strip()
        if not base_url:
            base_url = "http://localhost:11434"
        
        model = input("Enter model name (default: llama2): ").strip()
        if not model:
            model = "llama2"
        
        trading_system.analyze_with_ollama(base_url, model)
    else:
        print("👋 Analysis complete!")