import os
import sys
import json
import uuid
from datetime import datetime
from openai import OpenAI
from gmail_api import GmailAPIClient

# Ensure UTF-8 output encoding for terminal compatibility with emojis and symbols
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

# Initialize Gmail REST API client
gmail_client = GmailAPIClient()

# ============================================================================
# Local Email Inbox Storage (Fallback / Local Mode)
# ============================================================================
DB_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "emails_db.json")

DEFAULT_SEED_EMAILS = [
    {
        "id": "msg-101",
        "sender": "sarah.chen@techcorp.com (Sarah Chen, Lead Architect)",
        "to": "me@company.com",
        "subject": "Q4 Architecture Review - Action Required Before Friday",
        "date": "2026-10-10 09:15 AM",
        "body": "Hi team,\n\nPlease find the updated system design document for our Q4 architecture review. We need each team lead to submit their feedback by Friday 5:00 PM. Key areas of focus include microservices latency reduction and the new Agentic AI workflow integration.\n\nBest,\nSarah",
        "is_read": False
    },
    {
        "id": "msg-102",
        "sender": "marcus.vance@investorpartners.com (Marcus Vance, Investment Director)",
        "to": "me@company.com",
        "subject": "Follow-up: Series B Due Diligence & Financial Metrics",
        "date": "2026-10-09 04:45 PM",
        "body": "Dear founders,\n\nThanks for the productive discussion yesterday. Could you please share the monthly recurring revenue (MRR) breakdown and customer retention rates for the last two quarters? Our committee would like to review them before finalizing the term sheet next Tuesday.\n\nRegards,\nMarcus Vance",
        "is_read": False
    },
    {
        "id": "msg-103",
        "sender": "alex.rivera@devops-cloud.io (Alex Rivera, Senior SRE)",
        "to": "engineering-all@company.com",
        "subject": "Heads-Up: Scheduled PostgreSQL Cluster Maintenance this Sunday",
        "date": "2026-10-08 11:30 AM",
        "body": "Hello everyone,\n\nThis is a notification that our primary PostgreSQL cluster will undergo maintenance this Sunday between 02:00 UTC and 04:00 UTC. Expect approximately 10-15 minutes of read-only mode during the failover.\n\nFeel free to reach out on Slack #devops if you have concerns.\nAlex",
        "is_read": False
    },
    {
        "id": "msg-104",
        "sender": "dr.elena.rostova@ai-research.org (Dr. Elena Rostova)",
        "to": "me@company.com",
        "subject": "Accepted Paper: Autonomous ReAct Agents in Production",
        "date": "2026-10-07 02:20 PM",
        "body": "Dear colleague,\n\nOur paper on Autonomous ReAct Agents in Production has been officially accepted for oral presentation at NeurAI 2026! Let's arrange a brief sync next week to coordinate slides.\n\nWarm regards,\nDr. Elena",
        "is_read": True
    }
]

def load_emails_db():
    if not os.path.exists(DB_FILE):
        save_emails_db(DEFAULT_SEED_EMAILS)
        return list(DEFAULT_SEED_EMAILS)
    try:
        with open(DB_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return list(DEFAULT_SEED_EMAILS)

def save_emails_db(emails):
    with open(DB_FILE, "w", encoding="utf-8") as f:
        json.dump(emails, f, indent=2, ensure_ascii=False)


# ============================================================================
# Agent Tools Implementation (Gmail REST API + Local Fallback)
# ============================================================================

def get_unread_emails(limit: int = 10) -> str:
    """Retrieve all unread emails with sender (writer), date, subject, and preview."""
    if gmail_client.is_configured():
        try:
            print("[Email Service]: Fetching unread emails via official Gmail REST API...")
            emails = gmail_client.list_unread_emails(limit=limit)
            return json.dumps({
                "status": "success",
                "source": "Gmail REST API",
                "count": len(emails),
                "unread_emails": emails
            }, indent=2)
        except Exception as e:
            print(f"[Gmail API Error]: {e}. Falling back to local mailbox...")

    # Fallback to local storage
    emails = load_emails_db()
    unread = [e for e in emails if not e.get("is_read", False)][:limit]
    
    results = []
    for item in unread:
        body = item.get("body", "")
        preview = (body[:120] + "...") if len(body) > 120 else body
        results.append({
            "id": item.get("id"),
            "writer": item.get("sender"),
            "date": item.get("date"),
            "subject": item.get("subject"),
            "preview": preview
        })
    
    return json.dumps({
        "status": "success",
        "source": "Local Mailbox (Add credentials.json for live Gmail API)",
        "count": len(results),
        "unread_emails": results
    }, indent=2)

def read_email(email_id: str, mark_as_read: bool = True) -> str:
    """Read the full content of an email by its ID, and optionally mark it as read."""
    if gmail_client.is_configured():
        try:
            print(f"[Email Service]: Reading email '{email_id}' via official Gmail REST API...")
            data = gmail_client.read_email(email_id=email_id, mark_as_read=mark_as_read)
            data["source"] = "Gmail REST API"
            return json.dumps(data, indent=2)
        except Exception as e:
            print(f"[Gmail API Error]: {e}. Checking local mailbox...")

    emails = load_emails_db()
    matched = None
    
    for item in emails:
        if item.get("id", "").lower() == email_id.strip().lower():
            matched = item
            if mark_as_read:
                item["is_read"] = True
            break
            
    if not matched:
        for item in emails:
            if email_id.lower() in item.get("subject", "").lower() or email_id.lower() in item.get("sender", "").lower():
                matched = item
                if mark_as_read:
                    item["is_read"] = True
                break
                
    if not matched:
        return json.dumps({"status": "error", "message": f"Email with ID or search term '{email_id}' was not found."})
        
    if mark_as_read:
        save_emails_db(emails)
        
    return json.dumps({
        "status": "success",
        "source": "Local Mailbox",
        "id": matched.get("id"),
        "writer": matched.get("sender"),
        "to": matched.get("to"),
        "date": matched.get("date"),
        "subject": matched.get("subject"),
        "body": matched.get("body"),
        "marked_as_read": mark_as_read
    }, indent=2)

def write_and_send_email(to: str, subject: str, body: str) -> str:
    """Compose and send a new email via Gmail REST API or local mailbox."""
    if not to or not subject or not body:
        return json.dumps({"status": "error", "message": "Missing 'to', 'subject', or 'body' field."})
        
    if gmail_client.is_configured():
        try:
            print(f"[Email Service]: Sending email to {to} via official Gmail REST API...")
            send_res = gmail_client.send_email(to=to, subject=subject, body=body)
            return json.dumps(send_res, indent=2)
        except Exception as e:
            print(f"[Gmail API Error]: {e}. Saving to local sent box...")

    emails = load_emails_db()
    now_str = datetime.now().strftime("%Y-%m-%d %I:%M %p")
    new_id = f"sent-{uuid.uuid4().hex[:6]}"
    
    new_email = {
        "id": new_id,
        "sender": "me@company.com (Current User)",
        "to": to.strip(),
        "subject": subject.strip(),
        "date": now_str,
        "body": body.strip(),
        "is_read": True,
        "is_sent": True
    }
    
    emails.append(new_email)
    save_emails_db(emails)
    
    return json.dumps({
        "status": "success",
        "source": "Local Mailbox",
        "message": f"Email successfully sent and stored with ID: {new_id}",
        "email_details": {
            "id": new_id,
            "to": to,
            "subject": subject,
            "date": now_str,
            "body": body
        }
    }, indent=2)

def search_emails(query: str, unread_only: bool = False) -> str:
    """Search emails by sender, subject, or content."""
    if gmail_client.is_configured():
        try:
            print(f"[Email Service]: Searching emails for '{query}' via official Gmail REST API...")
            matches = gmail_client.search_emails(query=query, unread_only=unread_only)
            return json.dumps({
                "status": "success",
                "source": "Gmail REST API",
                "matches_count": len(matches),
                "results": matches
            }, indent=2)
        except Exception as e:
            print(f"[Gmail API Error]: {e}. Searching local mailbox...")

    emails = load_emails_db()
    q = query.lower()
    matches = []
    
    for item in emails:
        if unread_only and item.get("is_read", False):
            continue
        text = f"{item.get('sender', '')} {item.get('subject', '')} {item.get('body', '')}".lower()
        if q in text:
            matches.append({
                "id": item.get("id"),
                "writer": item.get("sender"),
                "date": item.get("date"),
                "subject": item.get("subject"),
                "is_read": item.get("is_read", False)
            })
            
    return json.dumps({
        "status": "success",
        "source": "Local Mailbox",
        "matches_count": len(matches),
        "results": matches
    }, indent=2)


# ============================================================================
# OpenAI Function Calling Tools Schema
# ============================================================================

available_functions = {
    "get_unread_emails": get_unread_emails,
    "read_email": read_email,
    "write_and_send_email": write_and_send_email,
    "search_emails": search_emails
}

tools_schema = [
    {
        "type": "function",
        "function": {
            "name": "get_unread_emails",
            "description": "Fetch all unread emails with their sender (writer), date received, subject, and preview using Gmail REST API.",
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of unread emails to retrieve (default is 10)."
                    }
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_email",
            "description": "Read the complete body and details of a specific email by its ID, and marks it as read in Gmail.",
            "parameters": {
                "type": "object",
                "properties": {
                    "email_id": {
                        "type": "string",
                        "description": "The unique ID (e.g. Gmail message ID) or subject/sender keyword of the email to read."
                    },
                    "mark_as_read": {
                        "type": "boolean",
                        "description": "Whether to mark the email as read (default is True)."
                    }
                },
                "required": ["email_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write_and_send_email",
            "description": "Compose and send a new email to a recipient with subject and body using Gmail REST API.",
            "parameters": {
                "type": "object",
                "properties": {
                    "to": {
                        "type": "string",
                        "description": "Recipient email address."
                    },
                    "subject": {
                        "type": "string",
                        "description": "Email subject line."
                    },
                    "body": {
                        "type": "string",
                        "description": "Email content / body text."
                    }
                },
                "required": ["to", "subject", "body"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_emails",
            "description": "Search inbox emails matching a specific keyword in writer/sender, subject, or body.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search term or keyword."
                    },
                    "unread_only": {
                        "type": "boolean",
                        "description": "If true, only returns matching unread emails."
                    }
                },
                "required": ["query"]
            }
        }
    }
]

# ============================================================================
# ReAct Agent Loop
# ============================================================================

def run_email_agent(user_goal: str) -> str:
    """
    Autonomous ReAct Agent for Email management.
    Performs reasoning, calls tools iteratively, synthesizes results.
    """
    system_instruction = """
You are an intelligent, autonomous Email Agentic AI assistant connected to the user's email system (Gmail REST API).
Your capabilities:
1. Write/Compose and send emails using 'write_and_send_email'.
2. Read individual emails with full details using 'read_email'.
3. Fetch and summarize unread emails using 'get_unread_emails' (or 'search_emails').

CRITICAL INSTRUCTIONS FOR SUMMARIZING UNREAD EMAILS:
When asked to summarize unread emails, you MUST:
- Retrieve the unread emails using the 'get_unread_emails' tool.
- Provide a clear, well-structured executive summary.
- For EVERY unread email, prominently present:
  * Date received
  * Writer / Sender
  * Subject
  * Summary of the content & Key points
  * Any action items or deadlines mentioned
- If an email's preview is insufficient, you can autonomously call 'read_email' to read the full body before answering.

REASONING GUIDELINES (ReAct):
- Before taking any action or calling a tool, explain your step-by-step reasoning in plain text.
- After receiving tool outputs, examine the observation and determine if further steps are required.
- Provide clean, professional, markdown-formatted final answers.
"""

    messages = [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": user_goal}
    ]
    
    max_steps = 10
    step = 0
    
    while step < max_steps:
        step += 1
        
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=tools_schema,
            tool_choice="auto"
        )
        
        msg = response.choices[0].message
        messages.append(msg)
        
        # Log Agent's internal reasoning
        if msg.content:
            print(f"\n[Agent Thought Process]:\n{msg.content.strip()}\n")
            
        if msg.tool_calls:
            for tool_call in msg.tool_calls:
                func_name = tool_call.function.name
                raw_args = tool_call.function.arguments or "{}"
                try:
                    args = json.loads(raw_args)
                except Exception:
                    args = {}
                    
                print(f"[Agent Action]: Calling `{func_name}` with arguments: {args}")
                
                if func_name in available_functions:
                    try:
                        result = available_functions[func_name](**args)
                    except Exception as err:
                        result = json.dumps({"status": "error", "message": str(err)})
                else:
                    result = json.dumps({"status": "error", "message": f"Unknown tool '{func_name}'."})
                    
                # Print preview of observation
                print(f"[Agent Observation]: {result[:200]}..." if len(result) > 200 else f"[Agent Observation]: {result}")
                
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": str(result)
                })
        else:
            return msg.content or "No response generated."
            
    return "Agent reached maximum execution steps without finishing."


# ============================================================================
# Interactive CLI
# ============================================================================

def show_inbox_status():
    if gmail_client.is_configured():
        try:
            unread = gmail_client.list_unread_emails(limit=10)
            print(f"\n--- Live Gmail Inbox Status (Unread: {len(unread)}) ---")
            for e in unread:
                print(f"  [UNREAD] {e.get('id')}: {e.get('subject')} | From: {e.get('writer')} ({e.get('date')})")
            print("------------------------------------------------------\n")
            return
        except Exception as e:
            print(f"[Gmail Error]: {e}")

    emails = load_emails_db()
    unread_count = sum(1 for e in emails if not e.get("is_read", False))
    print(f"\n--- Local Mailbox Status (Total: {len(emails)} | Unread: {unread_count}) ---")
    for e in emails:
        status = "[UNREAD]" if not e.get("is_read", False) else "[READ]  "
        print(f"  {status} {e.get('id')}: {e.get('subject')} | From: {e.get('sender')} ({e.get('date')})")
    print("------------------------------------------------------\n")

if __name__ == "__main__":
    is_gmail = gmail_client.is_configured()
    mode_str = "🟢 Live Gmail REST API" if is_gmail else "🟡 Local Mailbox Mode (Drop 'credentials.json' for Live Gmail)"
    
    print(f"================================================================")
    print(f"   🤖 Autonomous Email Agentic AI (Model: {MODEL})")
    print(f"   Status: {mode_str}")
    print(f"================================================================")
    print("Capabilities:")
    print(" • Summarize unread emails with Date and Writers")
    print(" • Read specific emails in full")
    print(" • Draft & send new emails")
    print(" • Search emails by keyword or sender")
    print("\nSpecial commands:")
    print(" • 'inbox'    -> Show inbox status")
    print(" • 'auth'     -> Authorize / authenticate with Google Gmail OAuth")
    print(" • 'reset'    -> Reset local seed emails")
    print(" • 'exit'     -> Quit\n")
    
    load_emails_db()
    
    while True:
        try:
            user_input = input("You: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("quit", "exit"):
                print("Exiting Email Agent. Goodbye!")
                break
            if user_input.lower() == "inbox":
                show_inbox_status()
                continue
            if user_input.lower() == "auth":
                try:
                    gmail_client.authenticate()
                    print("[Gmail Auth]: Successfully authenticated with Google Gmail REST API!")
                except Exception as err:
                    print(f"\n[Auth Instructions]:\n{err}\n")
                continue
            if user_input.lower() == "reset":
                save_emails_db(DEFAULT_SEED_EMAILS)
                print("[System]: Local email inbox reset to default seed emails.\n")
                continue
                
            final_answer = run_email_agent(user_input)
            print(f"\n[Agent Final Answer]:\n{final_answer}\n")
            print("-" * 64)
            
        except KeyboardInterrupt:
            print("\nExiting Email Agent.")
            break
        except Exception as e:
            print(f"\n[Error encountered]: {e}\n")
