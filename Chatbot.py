import sys
from openai import OpenAI
import os

# Ensure UTF-8 output encoding for terminal compatibility with emojis
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# Initialize client for local Ollama instance
client = OpenAI(
    base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
    api_key="ollama"  # Ollama does not require a real API key
)

MODEL = os.getenv("OLLAMA_MODEL", "qwen3:latest")

def run_chatbox():
    print(f"Chatbox initialized ({MODEL}). Type 'quit' to exit.")
    
    # The system prompt gives the chatbox its personality
    messages = [
        {"role": "system", "content": "You are a helpful, conversational assistant."}
    ]
    
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == 'quit':
            break
            
        # 1. Add user message to history
        messages.append({"role": "user", "content": user_input})
        
        # 2. Call the LLM
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages
        )
        
        # 3. Extract and print the response
        bot_reply = response.choices[0].message.content
        print(f"\nChatbox: {bot_reply}")
        
        # 4. Add the bot's response to history so it remembers the context
        messages.append({"role": "assistant", "content": bot_reply})

if __name__ == "__main__":
    run_chatbox()