import sys
import os
import json
from openai import OpenAI

# Ensure UTF-8 output encoding for terminal compatibility with emojis
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# Initialize OpenAI client (pointing to local Ollama by default)
client = OpenAI(
    base_url=os.getenv("OLLAMA_BASE_URL", os.getenv("OPENAI_BASE_URL", "http://localhost:11434/v1")),
    api_key=os.getenv("OPENAI_API_KEY", "ollama"),
    timeout=2.0
)
MODEL = os.getenv("OPENAI_MODEL", os.getenv("OLLAMA_MODEL", "qwen3:latest"))

# ==============================================================================
# ENVIRONMENT: Simulated Kitchen State
# ==============================================================================
KITCHEN_CUPBOARDS = {
    "Cupboard 1 (Spices)": ["Salt", "Black Pepper", "Oregano", "Paprika"],
    "Cupboard 2 (Under Sink)": ["Dish Soap", "Sponges", "Trash Bags"],
    "Cupboard 3 (Pantry)": ["Flour", "White Rice", "Olive Oil", "Canned Beans"],
}


# ==============================================================================
# [GUIDELINE 4: SEPARATE PERMISSIONS AND TOOLS]
# ==============================================================================
# WORK POSITION: Tool functions & schema definition.
# Only give an agent the tools it strictly needs.
# This agent has READ-ONLY cupboard access. It cannot touch the stove, delete items,
# or access internet APIs.
def tool_list_cupboards():
    """Lists all cupboards present in the kitchen."""
    return list(KITCHEN_CUPBOARDS.keys())

def tool_inspect_cupboard(cupboard_name: str):
    """Opens a specific cupboard and returns the items found inside."""
    items = KITCHEN_CUPBOARDS.get(cupboard_name, [])
    print(f"    -> [Tool Action] Opening '{cupboard_name}'... Found: {items}")
    return items

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "tool_list_cupboards",
            "description": "List all cupboards in the kitchen",
            "parameters": {"type": "object", "properties": {}},
        }
    },
    {
        "type": "function",
        "function": {
            "name": "tool_inspect_cupboard",
            "description": "Inspect and read contents of a specific kitchen cupboard",
            "parameters": {
                "type": "object",
                "properties": {
                    "cupboard_name": {
                        "type": "string",
                        "description": "Exact name of the cupboard to search"
                    }
                },
                "required": ["cupboard_name"],
            }
        }
    }
]


# ==============================================================================
# [GUIDELINE 2: ONE CLEAR JOB (CLEAR SCOPE)]
# ==============================================================================
# WORK POSITION: Agent System Prompt / Role definition.
# Every agent focuses on ONE specific area. This agent is strictly a KitchenSearchAgent.
# It does NOT cook, order groceries, or handle unrelated features.
AGENT_SYSTEM_PROMPT = """You are a dedicated KitchenSearchAgent.
Your ONE job is to find a requested item by searching kitchen cupboards one by one.
Rules:
1. Discover the available cupboards using tool_list_cupboards.
2. Inspect the most likely cupboard using tool_inspect_cupboard.
3. If not found, inspect the next cupboard.
4. Stop as soon as the target item is located.
"""


# ==============================================================================
# AGENT AUTONOMOUS EXECUTION
# ==============================================================================
def run_agentic_search(target_item: str):
    print("=" * 70)
    print("--- AGENTIC AI EXECUTION: KITCHEN CUPBOARD SEARCH ---")
    print("=" * 70)
    print(f"Goal: Find '{target_item}' in the kitchen cupboards.\n")

    messages = [
        {"role": "system", "content": AGENT_SYSTEM_PROMPT},
        {"role": "user", "content": f"Find '{target_item}' in the kitchen cupboards."}
    ]

    max_steps = 6
    step_count = 0
    found_result = None

    # [GUIDELINE 3: WORKING IN THE BACKGROUND (PROACTIVE BEHAVIOR)]
    # WORK POSITION: The Autonomous Loop (while-loop).
    # Normal code sits until called. This agent actively loops through tasks on its own,
    # observing results and triggering the next action without human prompting.
    while step_count < max_steps:
        step_count += 1
        print(f"\n[Cycle #{step_count}]: Agent is thinking and planning next step...")

        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=TOOLS_SCHEMA,
                tool_choice="auto"
            )
            response_msg = response.choices[0].message
        except Exception:
            # Simulated autonomous agent decisions if Ollama is not active locally
            response_msg = simulated_agent_step(step_count, target_item)

        # [GUIDELINE 1: SELF DECISION-MAKING]
        # WORK POSITION: Result inspection & dynamic replanning.
        # The agent inspects tool output. If the item wasn't in Cupboard 1,
        # it adapts its plan and selects the next cupboard on its own.
        if response_msg.tool_calls:
            messages.append(response_msg)

            for tool_call in response_msg.tool_calls:
                func_name = tool_call.function.name
                func_args = json.loads(tool_call.function.arguments) if isinstance(tool_call.function.arguments, str) else tool_call.function.arguments

                print(f"  [Agent Decision]: Decided to call '{func_name}' with args {func_args}")

                # Execute the tool
                if func_name == "tool_list_cupboards":
                    result = tool_list_cupboards()
                elif func_name == "tool_inspect_cupboard":
                    cupboard = func_args.get("cupboard_name")
                    result = tool_inspect_cupboard(cupboard)

                    # Inspect observation to check if goal is reached
                    if any(target_item.lower() in str(i).lower() for i in result):
                        found_result = {
                            "status": "FOUND",
                            "item": target_item,
                            "location": cupboard,
                            "contents": result
                        }
                else:
                    result = "Unknown tool"

                # Feed observation back to agent's memory
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id if hasattr(tool_call, 'id') else f"call_{step_count}",
                    "name": func_name,
                    "content": json.dumps(result)
                })

            if found_result:
                print(f"\n[Agent Perception]: Goal satisfied! '{target_item}' found in '{found_result['location']}'.")
                break
        else:
            print(f"  [Agent]: {response_msg.content}")
            break

    # [GUIDELINE 5: TEAMWORK AND HANDOFFS]
    # WORK POSITION: Structured output payload generation.
    # An agent works with other agents by passing structured data back and forth.
    # It hands off the verified item location to the ChefAgent.
    print("\n" + "=" * 70)
    print("--- [GUIDELINE 5: TEAMWORK AND HANDOFFS] ---")
    if found_result:
        handoff_packet = {
            "from_agent": "KitchenSearchAgent",
            "to_agent": "ChefAgent",
            "action": "PROCEED_RECIPE",
            "payload": found_result
        }
        print("Handoff to ChefAgent generated:")
        print(json.dumps(handoff_packet, indent=2))
    else:
        handoff_packet = {
            "from_agent": "KitchenSearchAgent",
            "to_agent": "GroceryOrderAgent",
            "action": "PURCHASE_ITEM",
            "payload": {"item": target_item, "status": "NOT_FOUND"}
        }
        print("Item missing. Handoff to GroceryOrderAgent generated:")
        print(json.dumps(handoff_packet, indent=2))
    print("=" * 70)
    return handoff_packet


def simulated_agent_step(step: int, target_item: str):
    """Fallback step logic mirroring tool decisions if local LLM is offline."""
    class MockCall:
        def __init__(self, name, args, call_id):
            self.id = call_id
            self.function = type('obj', (object,), {'name': name, 'arguments': json.dumps(args)})

    class MockMessage:
        def __init__(self, tool_calls=None, content=None):
            self.tool_calls = tool_calls
            self.content = content

    if step == 1:
        return MockMessage(tool_calls=[MockCall("tool_list_cupboards", {}, "call_1")])
    elif step == 2:
        return MockMessage(tool_calls=[MockCall("tool_inspect_cupboard", {"cupboard_name": "Cupboard 1 (Spices)"}, "call_2")])
    elif step == 3:
        # Self decision-making: noticed oil was not in spices, Under Sink is cleaning supplies, so inspect Pantry
        return MockMessage(tool_calls=[MockCall("tool_inspect_cupboard", {"cupboard_name": "Cupboard 3 (Pantry)"}, "call_3")])
    else:
        return MockMessage(content=f"Found {target_item} in Cupboard 3 (Pantry).")


if __name__ == "__main__":
    print("Starting Kitchen Agentic AI demo (Press Enter to search 'Olive Oil', or type item):")
    item = input("\nTarget Item: ").strip()
    if not item:
        item = "Olive Oil"
    run_agentic_search(item)