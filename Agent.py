import sys
from openai import OpenAI
import json
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

# ---------------------------------------------------------
# Step 1: Define the Tool (The Action)
# ---------------------------------------------------------
def get_current_weather(location):
    """A simulated API call to get the weather."""
    # In a real agent, this would call a real weather API
    print(f"\n[System: Agent is calling weather API for {location}...]")
    return json.dumps({"location": location, "temperature": "72", "condition": "Sunny"})

# ---------------------------------------------------------
# Step 2: Define the Agent
# ---------------------------------------------------------
def run_agent(user_query):
    # Tell the LLM what tools it has available
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g., San Francisco, CA",
                        }
                    },
                    "required": ["location"],
                },
            }
        }
    ]

    messages = [{"role": "user", "content": user_query}]

    # Loop #1: The Agent decides if it needs to use a tool
    response = client.chat.completions.create(
        model=MODEL, 
        messages=messages, 
        tools=tools,
        tool_choice="auto" # Let the model decide if it should call the tool
    )

    response_message = response.choices[0].message
    
    # Check if the Agent decided to use a tool
    if response_message.tool_calls:
        # Step 3: Execute the tool on behalf of the agent
        messages.append(response_message) # Append the agent's request to use the tool
        
        for tool_call in response_message.tool_calls:
            if tool_call.function.name == "get_current_weather":
                # Parse the arguments the agent decided to pass to the tool
                function_args = json.loads(tool_call.function.arguments)
                
                # Run the actual python function
                tool_result = get_current_weather(location=function_args.get("location"))
                
                # Step 4: Send the tool's result back to the Agent
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": tool_call.function.name,
                    "content": tool_result,
                })

        # Loop #2: The Agent synthesizes the tool data into a final answer
        final_response = client.chat.completions.create(
            model=MODEL,
            messages=messages
        )
        print(f"\nAgent Final Answer: {final_response.choices[0].message.content}")
    else:
        # The agent decided it didn't need tools to answer the question
        print(f"\nAgent: {response_message.content}")

if __name__ == "__main__":
    # Test 1: Needs a tool
    print("User: What is the weather like in Tokyo?")
    run_agent("What is the weather like in Tokyo?")
    
    # Test 2: Doesn't need a tool
    print("\nUser: What is 2 + 2?")
    run_agent("What is 2 + 2?")