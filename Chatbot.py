import sys
import os
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
# TRADITIONAL CHATBOT IMPLEMENTATION: KITCHEN SEARCH
# ==============================================================================

# [GUIDELINE 2: ONE CLEAR JOB (CLEAR SCOPE) - WORK POSITION IN CHATBOT]
# Chatbot has no restricted boundary or specialized scope; it is a general conversationalist.
# It tries to answer everything through natural language generation.
def run_chatbot(user_query: str):
    print("=" * 70)
    print("--- TRADITIONAL CHATBOT EXECUTION ---")
    print("=" * 70)
    print(f"User Goal: {user_query}\n")

    # [GUIDELINE 4: SEPARATE PERMISSIONS AND TOOLS - WORK POSITION IN CHATBOT]
    # WORK POSITION: Chatbot has ZERO tools and ZERO permissions.
    # It cannot touch the physical kitchen, query a cupboard database, or read actual state.
    messages = [
        {"role": "system", "content": "You are a helpful, conversational assistant."},
        {"role": "user", "content": user_query}
    ]

    # [GUIDELINE 3: WORKING IN THE BACKGROUND (PROACTIVE) - WORK POSITION IN CHATBOT]
    # WORK POSITION: Strictly passive / single-turn prompt-response.
    # Code just sits there until called. It CANNOT proactively check cupboards one by one
    # in the background or trigger actions on its own schedule.
    print("[Chatbot Status]: Generating single text response...")
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages
        )
        reply = response.choices[0].message.content
    except Exception:
        # Fallback if local Ollama daemon is offline
        reply = (
            "Olive oil is typically stored in a cool, dark place. "
            "You should check your pantry cupboard or near the stove. "
            "Make sure to look behind taller containers!"
        )

    # [GUIDELINE 1: SELF DECISION-MAKING - WORK POSITION IN CHATBOT]
    # WORK POSITION: Absent.
    # The chatbot makes NO decisions. It cannot observe whether Cupboard 1 actually has
    # the item, cannot learn from negative results, and cannot replan.
    print(f"\nChatbot Reply:\n\"{reply}\"")

    # [GUIDELINE 5: TEAMWORK AND HANDOFFS - WORK POSITION IN CHATBOT]
    # WORK POSITION: Dead end.
    # Emits unstructured text to a human. It cannot produce structured data (JSON)
    # or hand off tasks to another agent (e.g. ChefAgent).
    print("\n" + "-" * 70)
    print("[Chatbot Limitation Summary]:")
    print("1. Did it open any cupboard? NO (No tools / Guideline 4)")
    print("2. Did it check one by one? NO (No proactive loop / Guideline 3)")
    print("3. Does it actually know where Olive Oil is? NO (No self decision-making / Guideline 1)")
    print("4. Can it pass verified location to a chef agent? NO (No handoff / Guideline 5)")
    print("-" * 70)


if __name__ == "__main__":
    print("Starting Kitchen Chatbot demo (Press Enter to search 'Find Olive Oil in the cupboards', or type your query):")
    query = input("\nYou: ").strip()
    if not query:
        query = "Find the Olive Oil in my kitchen cupboards."
    run_chatbot(query)