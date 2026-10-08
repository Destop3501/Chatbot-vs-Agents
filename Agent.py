import json
from openai import OpenAI

client = OpenAI(api_key="YOUR_API_KEY")

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

def run_agent(task_goal):
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
            model="gpt-4o",
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
            # Task completion criteria met
            return msg.content