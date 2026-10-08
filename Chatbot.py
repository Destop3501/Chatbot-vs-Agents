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

def run_chatbot(user_prompt):
    messages = [{"role": "user", "content": user_prompt}]
    
    while True:
        # tool_choice="auto" allows the LLM to decide whether to call a tool or reply with text
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=tools_schema,
            tool_choice="auto" 
        )
        
        msg = response.choices[0].message
        messages.append(msg) # Append the assistant's message (which contains the tool_call)
        
        # If the LLM decided to call tools, execute them
        if msg.tool_calls:
            for tool_call in msg.tool_calls:
                func_name = tool_call.function.name
                args = json.loads(tool_call.function.arguments)
                
                # Execute the actual Python function
                result = available_functions[func_name](**args)
                
                # Append the result back to the conversation history
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": str(result)
                })
        else:
            # If no tools were called, the chatbot has provided its final text answer
            return msg.content