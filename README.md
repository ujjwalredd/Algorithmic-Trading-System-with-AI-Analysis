# Algorithmic Trading System with AI Analysis

A comprehensive algorithmic trading system featuring multiple trading strategies, backtesting capabilities, risk management, and **AI-powered analysis using Ollama**.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-brightgreen.svg)]()

## ✨ Features

### 📈 Trading Strategies
- **Mean Reversion**: Z-score based entry/exit signals
- **Momentum**: Volatility-adjusted price momentum
- **Pairs Trading**: Statistical arbitrage with cointegration analysis

### 🔧 Core Components
- **Data Management**: Yahoo Finance integration with caching
- **Backtesting Engine**: Portfolio simulation with transaction costs
- **Risk Metrics**: VaR, CVaR, Beta, Information Ratio, Sortino Ratio, Calmar Ratio
- **Technical Indicators**: SMA, EMA, RSI, Bollinger Bands, MACD

### 🤖 AI Analysis
- **Ollama Integration**: Ask questions about your trading results with streaming responses
- **Interactive Mode**: Real-time Q&A about performance
- **Intelligent Insights**: Strategy comparisons, risk analysis, recommendations

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Ollama (for AI analysis)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/algorithmic-trading.git
cd algorithmic-trading
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Setup Ollama (for AI analysis)**
```bash
# Install Ollama from https://ollama.ai
# Start Ollama service
ollama serve

# Pull a model (in another terminal)
ollama pull llama2
```

4. **Run the system**
```bash
python main.py
```

## 📊 Usage Examples

### Basic Trading Analysis
```python
from main import TradingSystem

# Initialize and run all strategies
trading_system = TradingSystem()
results = trading_system.run_all_strategies()

# Generate comprehensive report
trading_system.generate_report()
```

### AI-Powered Analysis
```python
from ollama_integration import OllamaAnalyzer

# Setup analyzer
analyzer = OllamaAnalyzer()
analyzer.load_trading_results(results)

# Get quick insights
insights = analyzer.get_quick_insights()
for insight in insights:
    print(insight)

# Ask specific questions with streaming
response = analyzer.ask_question("How can we improve the momentum strategy?", stream=True)
print(response)

# Interactive mode
analyzer.interactive_mode()
```

### Individual Strategy Analysis
```python
from data.data_loader import DataLoader
from strategies.mean_reversion import MeanReversionStrategy
from backtesting.engine import BacktestEngine

# Load data
data_loader = DataLoader()
data = data_loader.get_stock_data('AAPL', '2023-01-01', '2024-01-01')

# Run strategy
strategy = MeanReversionStrategy()
signals = strategy.generate_signals(data)

# Backtest
engine = BacktestEngine()
results = engine.run_backtest(data, signals)
metrics = engine.calculate_metrics(results)
```

## 📁 Project Structure

```
algorithmic-trading/
├── README.md                       # Project documentation
├── requirements.txt                # Python dependencies
├── main.py                         # Main application
├── ollama_integration.py           # AI analysis module
├── interactive_ollama.py           # Interactive Ollama chat
├── trading_chat.py                 # Trading-specific chat
├── data/
│   ├── __init__.py
│   └── data_loader.py              # Data fetching and management
├── strategies/
│   ├── __init__.py
│   ├── mean_reversion.py           # Mean reversion strategy
│   ├── momentum.py                 # Momentum strategy
│   └── pairs_trading.py            # Pairs trading strategy
├── backtesting/
│   ├── __init__.py
│   └── engine.py                   # Backtesting engine
├── risk/
│   ├── __init__.py
│   └── metrics.py                  # Risk management metrics
└── utils/
    ├── __init__.py
    └── technical_indicators.py     # Technical analysis indicators
```

## 🔧 Configuration

### Trading Parameters
- **Default Symbols**: S&P 500 stocks
- **Time Period**: 1 year historical data
- **Transaction Costs**: 0.1% per trade
- **Initial Capital**: $100,000

### AI Analysis Settings
- **Default Model**: llama2
- **Ollama URL**: http://localhost:11434
- **Streaming**: Enabled by default
- **Temperature**: 0.7 (configurable)

## 📈 Performance Metrics

The system calculates comprehensive performance metrics:

- **Returns**: Annualized returns, cumulative returns
- **Risk**: Maximum drawdown, Value at Risk (VaR), Conditional VaR (CVaR)
- **Ratios**: Sharpe ratio, Sortino ratio, Calmar ratio, Information ratio
- **Trade Statistics**: Win rate, profit factor, average trade duration

## 🤖 AI Analysis Features

### Current Capabilities (Ollama Mode)
✅ **Streaming Responses**: Real-time AI responses as they're generated  
✅ **Smart Analysis**: Intelligent insights about trading performance  
✅ **Question Types**: Strategy comparison, risk analysis, recommendations  
✅ **Interactive Mode**: Real-time Q&A about your results

### Sample Questions
- "Which strategy performed best and why?"
- "How can we improve the momentum strategy?"
- "What are the main risks in our portfolio?"
- "Compare the Sharpe ratios of all strategies"
- "What would you recommend for risk management?"

## 🔧 Setup: Ollama AI

For AI analysis, you need Ollama running:

### 1. Install Ollama
```bash
# Visit: https://ollama.ai
# Download and install Ollama for your platform
```

### 2. Start Ollama and Pull Model
```bash
# Start Ollama
ollama serve

# Pull a model (in another terminal)
ollama pull llama2
```

### 3. Use Ollama Analysis
```python
from main import TradingSystem
from ollama_integration import OllamaAnalyzer

# Run strategies
trading_system = TradingSystem()
results = trading_system.run_all_strategies()

# Setup analyzer
analyzer = OllamaAnalyzer()
analyzer.load_trading_results(results)

# Ask questions with streaming
response = analyzer.ask_question("Which strategy performed best and why?", stream=True)
print(response)
```

## 📊 Example Output

```
================================================================================
ALGORITHMIC TRADING STRATEGY SUMMARY
================================================================================

MEAN_REVERSION STRATEGY:
----------------------------------------
Average Sharpe Ratio: 0.057
Average Annual Return: 2.01%
Average Max Drawdown: -45.47%
Average Win Rate: 5.10%
Best Performing Symbol: NVDA

MOMENTUM STRATEGY:
----------------------------------------
Average Sharpe Ratio: 0.180
Average Annual Return: 9.96%
Average Max Drawdown: -36.66%
Average Win Rate: 7.85%
Best Performing Symbol: NVDA
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


## 🔄 Version History

### Version 2.0 - AI Analysis Integration
- ✅ **Ollama AI Integration**: Ask questions about trading results with streaming
- ✅ **Interactive Q&A**: Real-time analysis of performance
- ✅ **Smart Insights**: Strategy comparisons and recommendations
- ✅ **Streaming Responses**: Real-time AI responses as they're generated

### Version 1.0 - Core Trading System
- ✅ **Multiple Strategies**: Mean reversion, momentum, pairs trading
- ✅ **Backtesting Engine**: Comprehensive portfolio simulation
- ✅ **Risk Management**: Advanced risk metrics and analysis
- ✅ **Technical Indicators**: Complete set of trading indicators
- ✅ **Data Management**: Yahoo Finance integration with caching

