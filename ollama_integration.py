

import json
import pandas as pd
from datetime import datetime
import requests
from typing import Dict, Any, List
import warnings
warnings.filterwarnings('ignore')


class OllamaAnalyzer:
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama2"):
        self.base_url = base_url
        self.model = model
        self.results = {}
        self.analysis_data = {}
        
        # Test connection
        self._test_connection()
    
    def _test_connection(self):
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                models = response.json().get("models", [])
                available_models = [m["name"] for m in models]
                print(f"✅ Connected to Ollama at {self.base_url}")
                print(f"📋 Available models: {', '.join(available_models)}")
                
                if self.model not in available_models:
                    print(f"⚠️ Model '{self.model}' not found. Using first available model.")
                    if available_models:
                        self.model = available_models[0]
                        print(f"🔄 Switched to model: {self.model}")
            else:
                print(f"❌ Failed to connect to Ollama at {self.base_url}")
                print("Make sure Ollama is running: ollama serve")
        except Exception as e:
            print(f"❌ Error connecting to Ollama: {e}")
            print("Make sure Ollama is running: ollama serve")
    
    def load_trading_results(self, results: Dict[str, Any]):
        self.results = results
        self._prepare_analysis_data()
        print("✅ Trading results loaded for analysis")
    
    def _prepare_analysis_data(self):
        analysis_data = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "strategies": {},
            "summary": {}
        }
        
        for strategy_name, results in self.results.items():
            if isinstance(results, pd.DataFrame) and not results.empty:
                strategy_data = {
                    "name": strategy_name,
                    "num_symbols": len(results),
                    "metrics": {
                        "avg_sharpe_ratio": float(results['sharpe_ratio'].mean()),
                        "avg_annual_return": float(results['annualized_return'].mean()),
                        "avg_max_drawdown": float(results['max_drawdown'].mean()),
                        "avg_win_rate": float(results['win_rate'].mean()),
                        "avg_volatility": float(results['volatility'].mean()),
                        "best_symbol": results.loc[results['sharpe_ratio'].idxmax(), 'symbol'],
                        "worst_symbol": results.loc[results['sharpe_ratio'].idxmin(), 'symbol']
                    },
                    "top_performers": results.nlargest(3, 'sharpe_ratio')[['symbol', 'sharpe_ratio', 'annualized_return']].to_dict('records'),
                    "risk_metrics": {
                        "var_95": float(results['var_95'].mean()),
                        "cvar_95": float(results['cvar_95'].mean()),
                        "profit_factor": float(results['profit_factor'].mean())
                    }
                }
                analysis_data["strategies"][strategy_name] = strategy_data
        
        # Overall summary
        if analysis_data["strategies"]:
            all_sharpe = [s["metrics"]["avg_sharpe_ratio"] for s in analysis_data["strategies"].values()]
            all_returns = [s["metrics"]["avg_annual_return"] for s in analysis_data["strategies"].values()]
            
            analysis_data["summary"] = {
                "best_strategy": max(analysis_data["strategies"].keys(), 
                                   key=lambda x: analysis_data["strategies"][x]["metrics"]["avg_sharpe_ratio"]),
                "highest_return": max(all_returns),
                "highest_sharpe": max(all_sharpe),
                "total_symbols_tested": sum(s["num_symbols"] for s in analysis_data["strategies"].values())
            }
        
        self.analysis_data = analysis_data
    
    def ask_question(self, question: str, stream: bool = True) -> str:
        """
        Ask a question about the trading results using Ollama.
        
        Args:
            question (str): Question about the trading results
            stream (bool): Whether to stream the response
            
        Returns:
            str: Ollama's response
        """
        if not self.analysis_data:
            return "❌ No trading results loaded. Please run the trading system first."
        
        try:
            # Create prompt
            prompt = f"""You are a financial analyst specializing in algorithmic trading. Analyze the following trading results and answer the user's question.

TRADING RESULTS:
{json.dumps(self.analysis_data, indent=2)}

USER QUESTION: {question}

Please provide a detailed, professional analysis based on the data above. Focus on:
- Key performance metrics
- Risk analysis
- Strategy comparisons
- Actionable insights

ANSWER:"""
            
            if stream:
                return self._ask_question_streaming(prompt)
            else:
                return self._ask_question_non_streaming(prompt)
        
        except Exception as e:
            return f"❌ Error communicating with Ollama: {e}"
    
    def _ask_question_streaming(self, prompt: str) -> str:
        """Ask question with streaming response."""
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": True,
                    "options": {
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "max_tokens": 1000
                    }
                },
                stream=True
            )
            
            if response.status_code == 200:
                full_response = ""
                print("🤖 Ollama is responding: ", end="", flush=True)
                
                for line in response.iter_lines():
                    if line:
                        try:
                            data = json.loads(line.decode('utf-8'))
                            if 'response' in data:
                                chunk = data['response']
                                print(chunk, end="", flush=True)
                                full_response += chunk
                            
                            if data.get('done', False):
                                print()  # New line at end
                                break
                        except json.JSONDecodeError:
                            continue
                
                return full_response.strip()
            else:
                return f"❌ Error getting response from Ollama: {response.status_code}"
        
        except Exception as e:
            return f"❌ Error in streaming response: {e}"
    
    def _ask_question_non_streaming(self, prompt: str) -> str:
        """Ask question without streaming response."""
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "max_tokens": 1000
                    }
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "No response from Ollama").strip()
            else:
                return f"❌ Error getting response from Ollama: {response.status_code}"
        
        except Exception as e:
            return f"❌ Error in non-streaming response: {e}"
    
    def get_quick_insights(self) -> List[str]:
        """
        Get quick insights about the trading results.
        
        Returns:
            List[str]: List of insights
        """
        insights = []
        
        if not self.analysis_data:
            return ["No trading results available for analysis."]
        
        # Strategy comparison
        strategies = self.analysis_data["strategies"]
        if len(strategies) > 1:
            best_strategy = max(strategies.keys(), 
                              key=lambda x: strategies[x]["metrics"]["avg_sharpe_ratio"])
            insights.append(f"🏆 Best performing strategy: {best_strategy} (Sharpe: {strategies[best_strategy]['metrics']['avg_sharpe_ratio']:.3f})")
        
        # Risk analysis
        for strategy_name, data in strategies.items():
            if data["metrics"]["avg_max_drawdown"] < -30:
                insights.append(f"⚠️ {strategy_name} shows high drawdown risk ({data['metrics']['avg_max_drawdown']:.1f}%)")
            elif data["metrics"]["avg_sharpe_ratio"] > 0.5:
                insights.append(f"✅ {strategy_name} shows good risk-adjusted returns (Sharpe: {data['metrics']['avg_sharpe_ratio']:.3f})")
        
        # Best performers
        for strategy_name, data in strategies.items():
            best_symbol = data["metrics"]["best_symbol"]
            best_return = data["top_performers"][0]["annualized_return"]
            insights.append(f"📈 {strategy_name} best performer: {best_symbol} ({best_return:.1f}% return)")
        
        return insights
    
    def interactive_mode(self):
        """Start interactive mode for asking questions."""
        if not self.analysis_data:
            print("❌ No trading results loaded. Please run the trading system first.")
            return
        
        print("\n" + "="*60)
        print("🤖 OLLAMA TRADING ANALYSIS - INTERACTIVE MODE")
        print("="*60)
        print(f"Using model: {self.model}")
        print("Ask questions about your trading results!")
        print("Type 'quit' to exit, 'insights' for quick insights, 'help' for suggestions")
        print("="*60)
        
        # Show quick insights first
        insights = self.get_quick_insights()
        print("\n📊 QUICK INSIGHTS:")
        for insight in insights:
            print(f"  {insight}")
        
        while True:
            try:
                question = input("\n🤔 Your question: ").strip()
                
                if question.lower() == 'quit':
                    print("👋 Goodbye!")
                    break
                elif question.lower() == 'insights':
                    insights = self.get_quick_insights()
                    print("\n📊 QUICK INSIGHTS:")
                    for insight in insights:
                        print(f"  {insight}")
                elif question.lower() == 'help':
                    print("\n💡 SUGGESTED QUESTIONS:")
                    print("  - Which strategy performed best and why?")
                    print("  - What are the main risks in these results?")
                    print("  - How do the strategies compare in terms of risk-adjusted returns?")
                    print("  - Which stocks performed best across all strategies?")
                    print("  - What insights can we draw about market conditions?")
                    print("  - How can we improve these strategies?")
                elif question:
                    print(f"\n🤖 Ollama ({self.model}) is analyzing...")
                    response = self.ask_question(question, stream=True)
                    print(f"\n")  # Extra line after streaming response
                else:
                    print("Please enter a question or type 'help' for suggestions.")
                    
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")


def setup_ollama_analyzer(base_url: str = "http://localhost:11434", model: str = "llama2") -> OllamaAnalyzer:
    """
    Setup Ollama analyzer.
    
    Args:
        base_url (str): Ollama API base URL
        model (str): Model name to use
        
    Returns:
        OllamaAnalyzer: Configured analyzer
    """
    return OllamaAnalyzer(base_url=base_url, model=model)


def analyze_with_ollama(trading_results: Dict[str, Any], base_url: str = "http://localhost:11434", model: str = "llama2"):
    """
    Analyze trading results with Ollama.
    
    Args:
        trading_results (Dict[str, Any]): Results from trading system
        base_url (str): Ollama API base URL
        model (str): Model name to use
    """
    analyzer = setup_ollama_analyzer(base_url, model)
    analyzer.load_trading_results(trading_results)
    
    print("🤖 Starting Ollama analysis...")
    analyzer.interactive_mode()


if __name__ == "__main__":
    # Example usage
    print("🤖 Ollama Trading Analysis Module")
    print("Use this module to analyze trading results with Ollama!")
    print("\nExample usage:")
    print("1. from ollama_integration import analyze_with_ollama")
    print("2. analyze_with_ollama(trading_results, 'http://localhost:11434', 'llama2')") 