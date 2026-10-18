import sys
import os
import json
from openai import OpenAI

# Ensure UTF-8 output encoding for terminal compatibility with emojis
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# Initialize OpenAI client with a short timeout so it gracefully uses simulated fallback if Ollama isn't started
client = OpenAI(
    base_url=os.getenv("OLLAMA_BASE_URL", os.getenv("OPENAI_BASE_URL", "http://localhost:11434/v1")),
    api_key=os.getenv("OPENAI_API_KEY", "ollama"),
    timeout=1.5
)
MODEL = os.getenv("OPENAI_MODEL", os.getenv("OLLAMA_MODEL", "qwen3:latest"))

# ==============================================================================
# ENVIRONMENT: Simulated Kitchen Cupboards
# ==============================================================================
KITCHEN_CUPBOARDS = {
    "Cupboard 1 (Spices)": ["Salt", "Black Pepper", "Oregano", "Paprika"],
    "Cupboard 2 (Under Sink)": ["Dish Soap", "Sponges", "Trash Bags"],
    "Cupboard 3 (Pantry)": ["Flour", "White Rice", "Olive Oil", "Canned Beans"],
}


# ==============================================================================
# PART 1: TRADITIONAL CHATBOT IMPLEMENTATION
# ==============================================================================
# [GUIDELINE 2: ONE CLEAR JOB - WORK POSITION IN CHATBOT]
# Chatbot has no specific boundary; it is a general conversationalist.
# It attempts to handle any request with general language text generation.
def run_chatbot_search(target_item: str):
    print("\n" + "="*70)
    print("--- [1] TRADITIONAL CHATBOT EXECUTION ---")
    print("="*70)

    # [GUIDELINE 4: SEPARATE PERMISSIONS AND TOOLS - WORK POSITION IN CHATBOT]
    # WORK POSITION: Chatbot has ZERO tools and ZERO permissions.
    # It cannot interact with the kitchen environment or read any real cupboard states.
    messages = [
        {"role": "system", "content": "You are a helpful conversational assistant."},
        {"role": "user", "content": f"Find the '{target_item}' in my kitchen cupboards."}
    ]

    # [GUIDELINE 3: WORKING IN THE BACKGROUND - WORK POSITION IN CHATBOT]
    # WORK POSITION: Strictly passive / single-turn.
    # It does not run in the background. It takes 1 prompt, returns 1 text blob, and stops.
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages
        )
        reply = response.choices[0].message.content
    except Exception as e:
        reply = (
            f"(Simulated fallback due to connection error): To find '{target_item}', "
            "check your pantry or the spice cabinet near the stove."
        )

    # [GUIDELINE 1: SELF DECISION-MAKING - WORK POSITION IN CHATBOT]
    # WORK POSITION: Absent.
    # The chatbot makes NO decisions, cannot observe results, and cannot replan if wrong.
    print(f"User Request: Find '{target_item}' in the kitchen cupboards.")
    print(f"Chatbot Output:\n\"{reply}\"")

    # [GUIDELINE 5: TEAMWORK AND HANDOFFS - WORK POSITION IN CHATBOT]
    # WORK POSITION: Dead end.
    # It cannot pass structured information (e.g. JSON data) to another agent or system.
    print("\n[Chatbot Limitation]: Olive oil was NOT actually located or verified in any cupboard.")
    return None


# ==============================================================================
# PART 2: AGENTIC AI IMPLEMENTATION
# ==============================================================================

# [GUIDELINE 4: SEPARATE PERMISSIONS AND TOOLS - WORK POSITION IN AGENTIC AI]
# WORK POSITION: Tool definitions.
# We give the agent ONLY the tools it strictly needs to inspect storage.
# It does NOT have permissions to ignite the stove, delete items, or access internet APIs.
def tool_list_cupboards():
    """Tool: Lists all cupboards present in the kitchen."""
    return list(KITCHEN_CUPBOARDS.keys())

def tool_inspect_cupboard(cupboard_name: str):
    """Tool: Opens a specific cupboard and returns the items found inside."""
    items = KITCHEN_CUPBOARDS.get(cupboard_name, [])
    print(f"    -> [Tool Action] Opening '{cupboard_name}'... Found: {items}")
    return items

AGENT_TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "tool_list_cupboards",
            "description": "Get a list of all cupboards in the kitchen",
            "parameters": {"type": "object", "properties": {}},
        }
    },
    {
        "type": "function",
        "function": {
            "name": "tool_inspect_cupboard",
            "description": "Inspect and read the contents of a specific kitchen cupboard",
            "parameters": {
                "type": "object",
                "properties": {
                    "cupboard_name": {
                        "type": "string",
                        "description": "The exact name of the cupboard to search"
                    }
                },
                "required": ["cupboard_name"],
            }
        }
    }
]


# [GUIDELINE 2: ONE CLEAR JOB - WORK POSITION IN AGENTIC AI]
# WORK POSITION: Agent Persona & System Prompt.
# Scope is strictly restricted: Locate items in kitchen cupboards.
# It does NOT try to cook, write recipes, or perform general chit-chat.
AGENT_SYSTEM_PROMPT = """You are a dedicated KitchenSearchAgent.
Your ONLY job is to locate a specified target item by searching the kitchen cupboards one by one.
Follow these rules:
1. First, list the cupboards.
2. Inspect the most promising cupboard.
3. If the item is not there, inspect the next cupboard.
4. Stop immediately once the item is found.
"""

def run_agentic_search(target_item: str):
    print("\n" + "="*70)
    print("--- [2] AGENTIC AI EXECUTION (STEP-BY-STEP AUTONOMOUS LOOP) ---")
    print("="*70)

    messages = [
        {"role": "system", "content": AGENT_SYSTEM_PROMPT},
        {"role": "user", "content": f"Find '{target_item}' in the kitchen cupboards."}
    ]

    max_steps = 6
    step_count = 0
    found_result = None

    # [GUIDELINE 3: WORKING IN THE BACKGROUND / PROACTIVE - WORK POSITION IN AGENTIC AI]
    # WORK POSITION: The autonomous execution loop (while-loop).
    # The agent proactively executes steps on its own schedule without requiring
    # a human user to click 'next' or prompt it after every door is opened.
    while step_count < max_steps:
        step_count += 1
        print(f"\n[Cycle #{step_count}]: Agent is thinking and planning next action...")

        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=AGENT_TOOLS_SCHEMA,
                tool_choice="auto"
            )
            response_msg = response.choices[0].message
        except Exception:
            # Fallback simulator if local LLM service is offline
            response_msg = simulated_llm_agent_step(step_count, target_item, messages)

        # [GUIDELINE 1: SELF DECISION-MAKING - WORK POSITION IN AGENTIC AI]
        # WORK POSITION: Tool call evaluation & dynamic replanning.
        # The agent inspects its perception/observations and decides what to do next:
        # If tool returned no match -> change plan and pick a different cupboard.
        # If tool returned match -> conclude and stop.
        if response_msg.tool_calls:
            messages.append(response_msg)

            for tool_call in response_msg.tool_calls:
                func_name = tool_call.function.name
                func_args = json.loads(tool_call.function.arguments) if isinstance(tool_call.function.arguments, str) else tool_call.function.arguments

                print(f"  [Agent Decision]: Calling tool '{func_name}' with args {func_args}")

                if func_name == "tool_list_cupboards":
                    result = tool_list_cupboards()
                elif func_name == "tool_inspect_cupboard":
                    cupboard = func_args.get("cupboard_name")
                    result = tool_inspect_cupboard(cupboard)

                    # Check if target item is observed
                    if any(target_item.lower() in str(i).lower() for i in result):
                        found_result = {
                            "status": "FOUND",
                            "item": target_item,
                            "location": cupboard,
                            "contents": result
                        }
                else:
                    result = "Unknown tool"

                # Feed observation back into Agent memory
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id if hasattr(tool_call, 'id') else f"call_{step_count}",
                    "name": func_name,
                    "content": json.dumps(result)
                })

            if found_result:
                print(f"\n[Agent Perception]: Goal satisfied! '{target_item}' was observed.")
                break
        else:
            # Model decided no more tools are needed
            print(f"  [Agent Response]: {response_msg.content}")
            break

    # [GUIDELINE 5: TEAMWORK AND HANDOFFS - WORK POSITION IN AGENTIC AI]
    # WORK POSITION: Output payload and downstream handoff.
    # The agent produces a structured, machine-readable artifact and hands it off
    # to the next agent (e.g. ChefAgent or GroceryAgent) to continue the workflow.
    print("\n" + "-"*50)
    print("--- [GUIDELINE 5] TEAMWORK & HANDOFF PAYLOAD ---")
    if found_result:
        handoff_packet = {
            "from_agent": "KitchenSearchAgent",
            "to_agent": "ChefAgent",
            "action": "PROCEED_RECIPE",
            "data": found_result
        }
        print("Structured Handoff Packet generated for ChefAgent:")
        print(json.dumps(handoff_packet, indent=2))
    else:
        handoff_packet = {
            "from_agent": "KitchenSearchAgent",
            "to_agent": "GroceryOrderAgent",
            "action": "ORDER_MISSING_ITEM",
            "data": {"item": target_item, "status": "NOT_FOUND"}
        }
        print("Item not found. Handoff to GroceryOrderAgent:")
        print(json.dumps(handoff_packet, indent=2))
    print("-"*50)
    return handoff_packet


# Helper simulation for environments without an active LLM daemon
def simulated_llm_agent_step(step: int, target_item: str, messages: list):
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
        # Self decision-making: Not in spices, skip Under Sink (cleaning), search Pantry
        return MockMessage(tool_calls=[MockCall("tool_inspect_cupboard", {"cupboard_name": "Cupboard 3 (Pantry)"}, "call_3")])
    else:
        return MockMessage(content=f"Found {target_item} in Cupboard 3 (Pantry).")


if __name__ == "__main__":
    TARGET = "Olive Oil"
    
    # 1. Run Chatbot demo
    run_chatbot_search(TARGET)

    # 2. Run Agentic AI demo
    run_agentic_search(TARGET)
