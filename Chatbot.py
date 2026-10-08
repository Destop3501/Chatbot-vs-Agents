import os
import sys
import json
from openai import OpenAI

# Ensure UTF-8 output encoding for terminal compatibility with emojis and degree symbols
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

def load_env(env_path=".env"):
    """Load key-value pairs from .env into os.environ if not already set."""
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, val = line.split("=", 1)
                    key = key.strip()
                    val = val.strip().strip("'\"")
                    if key and key not in os.environ:
                        os.environ[key] = val

load_env()

# Configure model endpoint (defaults to local Ollama if not configured)
BASE_URL = os.getenv("OPENAI_BASE_URL", "http://localhost:11434/v1")
API_KEY = os.getenv("OPENAI_API_KEY", "ollama")
MODEL = os.getenv("OPENAI_MODEL", "qwen3:latest")

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

# 1. The underlying Python functions (same tools as Agent)
def get_user_location(name: str) -> str:
    return {"Alice": "London", "Bob": "Tokyo"}.get(name, "Unknown")

def get_weather(city: str) -> str:
    return {"London": "Raining, 12°C", "Tokyo": "Sunny, 25°C"}.get(city, "Unknown")

# 2. The mapping dictionary for dynamic execution
available_functions = {
    "get_user_location": get_user_location,
    "get_weather": get_weather
}

# 3. The JSON Schema passed to the LLM (identical to Agent)
tools_schema = [
    {
        "type": "function",
        "function": {
            "name": "get_user_location",
            "description": "Get the current city location of a user by name.",
            "parameters": {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a given city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"]
            }
        }
    }
]

# ============================================================================
# CHATBOT ARCHITECTURE: Single-pass tool calling (NO reasoning loop)
#
# Step 1: Send user message to LLM (with tools available)
# Step 2: If LLM calls tools, execute them and feed results back
# Step 3: Get ONE final text response (NO more tool calls allowed)
#
# This means: If the answer requires CHAINING tools (tool A's output feeds
# into tool B), the chatbot CANNOT do it. It only gets ONE round of tools.
# ============================================================================

def run_chatbot(user_prompt: str) -> str:
    messages = [{"role": "user", "content": user_prompt}]
    
    # Step 1: First LLM call — tools are available
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=tools_schema,
        tool_choice="auto"
    )
    
    msg = response.choices[0].message
    messages.append(msg)
    
    # Step 2: If tools were called, execute them (ONE round only)
    if msg.tool_calls:
        for tool_call in msg.tool_calls:
            func_name = tool_call.function.name
            args = json.loads(tool_call.function.arguments)
            print(f"  [Tool Call]: {func_name}({args})")
            
            result = available_functions[func_name](**args)
            print(f"  [Tool Result]: {result}")
            
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": str(result)
            })
        
        # Step 3: Final LLM call — NO tools allowed, just synthesize a text answer
        # This is the key limitation: chatbot does NOT loop back for more tool calls
        final_response = client.chat.completions.create(
            model=MODEL,
            messages=messages
            # Notice: no 'tools' parameter here — the chatbot is DONE with tools
        )
        return final_response.choices[0].message.content
    else:
        return msg.content

if __name__ == "__main__":
    print(f"=== Tool-Calling Chatbot (Model: {MODEL} @ {BASE_URL}) ===")
    print("This chatbot HAS tools but can only use them in ONE round (no chaining).")
    print("Try: 'What is the weather in London?'        (It CAN answer — single tool)")
    print("Try: 'What is the weather where Alice lives?' (It CANNOT fully answer — needs chaining)")
    print("Type 'quit' or 'exit' to stop.\n")
    
    while True:
        try:
            user_input = input("You: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("quit", "exit"):
                print("Exiting Chatbot.")
                break
            
            response = run_chatbot(user_input)
            print(f"\nChatbot: {response}\n")
        except KeyboardInterrupt:
            print("\nExiting Chatbot.")
            break
        except Exception as e:
            print(f"\n[Error]: {e}\n")