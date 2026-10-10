# Autonomous Email Agentic AI (Official Google Gmail REST API)

An autonomous, tool-calling ReAct (Reasoning + Acting) Agent for managing emails using the **Official Google Gmail REST API v1**. Built to seamlessly work with local LLMs (Ollama with `qwen3:latest`) as well as OpenAI / Gemini.

---

## 🚀 Key Capabilities

1. **Summarize Unread Emails with Date & Writers**:
   - Queries Gmail using `is:unread`.
   - Extracts and presents: **Date received**, **Writer / Sender**, **Subject**, and **Executive Summary** with action items and deadlines.
2. **Read Emails**:
   - Fetches and inspects full email content by ID, sender, or subject keyword.
   - Automatically marks emails as read by removing the Gmail `UNREAD` label.
3. **Write & Send Emails**:
   - Composes professional emails and replies with recipient, subject, and body.
   - Sends real emails directly through Google's `https://gmail.googleapis.com/gmail/v1/users/me/messages/send` endpoint.
4. **Autonomous Tool Chaining**:
   - The Agentic AI can chain multiple actions in a single session (e.g. search for an unread email &rarr; read its details &rarr; draft a contextual reply &rarr; send it via Gmail API).

---

## 🔑 Setting Up Official Google Gmail REST API

To connect the agent to your live Gmail account:

### 1. Enable Gmail API in Google Cloud Console
1. Go to the [Google Cloud Console](https://console.cloud.google.com/).
2. Create a new project (e.g., `Email-Agent-AI`).
3. Navigate to **APIs & Services** > **Library**, search for **Gmail API**, and click **Enable**.

### 2. Configure OAuth Consent Screen
1. Go to **APIs & Services** > **OAuth consent screen**.
2. Select **External** (or Internal for Google Workspace) and click **Create**.
3. Fill in the App Name (e.g., `Email Agent`) and your email address.
4. Under **Test Users**, add your personal Gmail address (this allows your account to log in during development).

### 3. Create Desktop OAuth Credentials
1. Go to **APIs & Services** > **Credentials**.
2. Click **Create Credentials** > **OAuth client ID**.
3. For Application type, select **Desktop app**.
4. Download the generated JSON credentials file and rename it to:
   ```
   credentials.json
   ```
5. Place `credentials.json` directly in the project root directory (`e:\Git\Chatbot-vs-Agents\credentials.json`).

### 4. Authorize
When you run `EmailAgent.py` for the first time or type `auth` in the CLI:
- A browser window will open asking you to sign in with your Google account.
- Once approved, a `token.json` file is automatically saved so you never have to sign in again.

> **Note:** If `credentials.json` is not yet placed in the project folder, the agent automatically runs in local mailbox mode with sample data so you can test it immediately without interruptions.

---

## 🏃 Running the Agent

```bash
python EmailAgent.py
```

### Commands in Interactive Mode:
- `auth`: Authorize with Google OAuth 2.0.
- `inbox`: Display live unread inbox status.
- `reset`: Reset sample seed data if in local mode.
- `exit` or `quit`: Exit the agent.

---

## 💬 Example Natural Language Queries

- **Summary of Unread Emails:**
  > *"Please give me a summary of all my unread emails with their dates and writers."*

- **Read Specific Email:**
  > *"Read the full email from Marcus Vance about Series B."*

- **Write an Email:**
  > *"Write an email to team@example.com confirming that our project report is ready."*

- **Autonomous Multi-Step Task:**
  > *"Check if I have any unread emails from Sarah. Read what she needs, and write a reply telling her I will send the feedback by tomorrow."*
