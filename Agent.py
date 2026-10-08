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

# 1. The underlying Python functions
def get_user_location(name: str) -> str:
    return {"Alice": "London", "Bob": "Tokyo"}.get(name, "Unknown")

def get_weather(city: str) -> str:
    return {"London": "Raining, 12°C", "Tokyo": "Sunny, 25°C"}.get(city, "Unknown")

# 2. The mapping dictionary for dynamic execution
available_functions = {
    "get_user_location": get_user_location,
    "get_weather": get_weather
}

# 3. The JSON Schema passed to the LLM
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

def run_agent(task_goal: str) -> str:
    # The system prompt forces the ReAct (Reason + Act) architecture
    system_instruction = """
    You are an autonomous agent. Before calling any tool, you MUST write down your 
    step-by-step reasoning in the text response. Evaluate the current state, determine 
    what information is missing, and then call the appropriate tool.
    """
    
    messages = [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": task_goal}
    ]
    
    while True:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=tools_schema,
            tool_choice="auto"
        )
        
        msg = response.choices[0].message
        messages.append(msg)
        
        # AGENT DIFFERENCE: Log the Agent's internal reasoning scratchpad
        if msg.content:
            print(f"[Agent Thought Process]: {msg.content}")
            
        if msg.tool_calls:
            for tool_call in msg.tool_calls:
                func_name = tool_call.function.name
                args = json.loads(tool_call.function.arguments)
                print(f"[Agent Action]: Executing {func_name} with {args}")
                
                result = available_functions[func_name](**args)
                print(f"[Agent Observation]: {result}\n")
                
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": str(result)
                })
        else:
            return msg.content

if __name__ == "__main__":
    print(f"=== ReAct Agent Initialized (Model: {MODEL} @ {BASE_URL}) ===")
    print("Example: 'What is the weather for Alice?'")
    print("Type 'quit' or 'exit' to stop.\n")
    
    while True:
        try:
            user_input = input("You: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("quit", "exit"):
                print("Exiting Agent.")
                break
                
            final_answer = run_agent(user_input)
            print(f"\n[Agent Final Answer]: {final_answer}\n")
        except KeyboardInterrupt:
            print("\nExiting Agent.")
            break
        except Exception as e:
            print(f"\n[Error]: {e}\n")