"""
Trading Chat with Ollama
Interactive chat about trading results with streaming responses.
"""

from main import TradingSystem
from ollama_integration import OllamaAnalyzer

def trading_chat():
    """Interactive chat about trading results."""
    
    print("🚀 TRADING CHAT WITH OLLAMA")
    print("="*50)
    
    # Run trading strategies
    print("1. Loading trading data...")
    trading_system = TradingSystem()
    results = trading_system.run_all_strategies()
    
    # Setup Ollama analyzer
    print("\n2. Setting up Ollama...")
    analyzer = OllamaAnalyzer()
    analyzer.load_trading_results(results)
    
    print("\n3. Starting interactive chat...")
    print("="*50)
    print("Ask questions about your trading results!")
    print("Type 'quit' to exit, 'insights' for quick insights, 'help' for suggestions")
    print("="*50)
    
    # Show quick insights first
    insights = analyzer.get_quick_insights()
    print("\n📊 QUICK INSIGHTS:")
    for insight in insights:
        print(f"  {insight}")
    
    while True:
        try:
            question = input("\n🤔 Your question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            elif question.lower() == 'insights':
                insights = analyzer.get_quick_insights()
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
                print("  - What would be a good stop-loss level for the momentum strategy?")
                print("  - Which stocks should I avoid based on these results?")
            elif question:
                print(f"\n🤖 Ollama ({analyzer.model}) is analyzing...")
                response = analyzer.ask_question(question, stream=True)
                print(f"\n")  # Extra line after streaming
            else:
                print("Please enter a question or type 'help' for suggestions.")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    trading_chat() 