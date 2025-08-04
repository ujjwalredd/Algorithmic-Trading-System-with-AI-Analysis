"""
Interactive Ollama Chat
Ask questions directly to Ollama with streaming responses.
"""

import requests
import json

def interactive_ollama_chat():
    """Interactive chat with Ollama."""
    
    base_url = "http://localhost:11434"
    
    # Test connection and get model
    try:
        response = requests.get(f"{base_url}/api/tags")
        if response.status_code == 200:
            models = response.json().get("models", [])
            if models:
                model_name = models[0]["name"]
                print(f"✅ Connected to Ollama at {base_url}")
                print(f"📋 Using model: {model_name}")
                print("="*50)
            else:
                print("❌ No models available")
                return
        else:
            print(f"❌ Failed to connect to Ollama at {base_url}")
            return
    except Exception as e:
        print(f"❌ Error connecting to Ollama: {e}")
        return
    
    print("🤖 OLLAMA INTERACTIVE CHAT")
    print("="*50)
    print("Ask any question! Type 'quit' to exit.")
    print("="*50)
    
    while True:
        try:
            # Get user input
            question = input("\n🤔 You: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not question:
                print("Please enter a question.")
                continue
            
            # Send to Ollama with streaming
            print(f"\n🤖 {model_name}: ", end="", flush=True)
            
            response = requests.post(
                f"{base_url}/api/generate",
                json={
                    "model": model_name,
                    "prompt": question,
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
                for line in response.iter_lines():
                    if line:
                        try:
                            data = json.loads(line.decode('utf-8'))
                            if 'response' in data:
                                chunk = data['response']
                                print(chunk, end="", flush=True)
                            
                            if data.get('done', False):
                                print()  # New line at end
                                break
                        except json.JSONDecodeError:
                            continue
            else:
                print(f"❌ Error: {response.status_code}")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    interactive_ollama_chat() 