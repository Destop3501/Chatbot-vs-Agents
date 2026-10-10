import os
import json
import base64
import warnings
from typing import Dict, Any, List, Optional
from email.message import EmailMessage

# Suppress Google Auth Python 3.9 EOL advisory notice
warnings.filterwarnings("ignore", category=FutureWarning)
import requests
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow

# OAuth 2.0 Scopes required for reading, modifying read status, and sending emails via Gmail REST API
SCOPES = [
    'https://www.googleapis.com/auth/gmail.modify',
    'https://www.googleapis.com/auth/gmail.send'
]

CREDENTIALS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "credentials.json")
TOKEN_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "token.json")
GMAIL_BASE_URL = "https://gmail.googleapis.com/gmail/v1/users/me"

class GmailAPIClient:
    """Official Google Gmail REST API Client."""

    def __init__(self, credentials_path: str = CREDENTIALS_PATH, token_path: str = TOKEN_PATH):
        self.credentials_path = credentials_path
        self.token_path = token_path
        self.creds: Optional[Credentials] = None

    def is_configured(self) -> bool:
        """Check if OAuth token or credentials.json exists."""
        return os.path.exists(self.token_path) or os.path.exists(self.credentials_path)

    def authenticate(self) -> Credentials:
        """Authenticate with Google OAuth 2.0 and return valid credentials."""
        creds = None
        if os.path.exists(self.token_path):
            try:
                creds = Credentials.from_authorized_user_file(self.token_path, SCOPES)
            except Exception:
                creds = None

        if creds and creds.valid:
            self.creds = creds
            return creds

        # Refresh expired token if possible
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
                with open(self.token_path, "w", encoding="utf-8") as f:
                    f.write(creds.to_json())
                self.creds = creds
                return creds
            except Exception as e:
                print(f"[Gmail Auth Warning]: Refresh failed: {e}. Re-authenticating...")

        # Run local server flow if credentials.json is present
        if not os.path.exists(self.credentials_path):
            raise FileNotFoundError(
                f"Google OAuth credentials file '{os.path.basename(self.credentials_path)}' not found in workspace.\n"
                "To enable real Gmail REST API access:\n"
                "1. Go to Google Cloud Console (https://console.cloud.google.com/)\n"
                "2. Create a project and enable 'Gmail API'\n"
                "3. Configure OAuth Consent Screen (External/Testing, add your email as Test User)\n"
                "4. Go to 'Credentials' -> 'Create Credentials' -> 'OAuth client ID' -> 'Desktop app'\n"
                "5. Download the JSON file and save it as 'credentials.json' in this folder."
            )

        flow = InstalledAppFlow.from_client_secrets_file(self.credentials_path, SCOPES)
        creds = flow.run_local_server(port=0)
        with open(self.token_path, "w", encoding="utf-8") as f:
            f.write(creds.to_json())

        self.creds = creds
        return creds

    def _get_auth_headers(self) -> Dict[str, str]:
        """Get HTTP authorization headers with refreshed access token."""
        creds = self.authenticate()
        if creds.expired and creds.refresh_token:
            creds.refresh(Request())
            with open(self.token_path, "w", encoding="utf-8") as f:
                f.write(creds.to_json())
        return {
            "Authorization": f"Bearer {creds.token}",
            "Accept": "application/json"
        }

    def list_unread_emails(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Fetch unread emails using Gmail REST API (q='is:unread').
        Returns list of emails with Date, Writer (sender), Subject, and Preview.
        """
        headers = self._get_auth_headers()
        params = {"q": "is:unread", "maxResults": limit}
        
        resp = requests.get(f"{GMAIL_BASE_URL}/messages", headers=headers, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        messages = data.get("messages", [])
        if not messages:
            return []

        results = []
        for item in messages:
            msg_id = item["id"]
            # Fetch message metadata and snippet
            msg_resp = requests.get(
                f"{GMAIL_BASE_URL}/messages/{msg_id}",
                headers=headers,
                params={"format": "metadata", "metadataHeaders": ["From", "Date", "Subject"]},
                timeout=15
            )
            if msg_resp.status_code != 200:
                continue

            msg_data = msg_resp.json()
            payload = msg_data.get("payload", {})
            headers_list = payload.get("headers", [])

            header_dict = {h["name"].lower(): h["value"] for h in headers_list}

            results.append({
                "id": msg_id,
                "writer": header_dict.get("from", "Unknown"),
                "date": header_dict.get("date", "Unknown"),
                "subject": header_dict.get("subject", "(No Subject)"),
                "preview": msg_data.get("snippet", "")
            })

        return results

    def read_email(self, email_id: str, mark_as_read: bool = True) -> Dict[str, Any]:
        """
        Fetch full body and details of a specific email by ID.
        Optionally marks as read by removing UNREAD label.
        """
        headers = self._get_auth_headers()
        resp = requests.get(f"{GMAIL_BASE_URL}/messages/{email_id}?format=full", headers=headers, timeout=15)
        resp.raise_for_status()
        msg_data = resp.json()

        payload = msg_data.get("payload", {})
        headers_list = payload.get("headers", [])
        header_dict = {h["name"].lower(): h["value"] for h in headers_list}

        body_text = self._extract_body(payload)

        # Mark as read if requested
        if mark_as_read:
            modify_url = f"{GMAIL_BASE_URL}/messages/{email_id}/modify"
            requests.post(modify_url, headers=headers, json={"removeLabelIds": ["UNREAD"]}, timeout=15)

        return {
            "id": email_id,
            "writer": header_dict.get("from", "Unknown"),
            "to": header_dict.get("to", "Unknown"),
            "date": header_dict.get("date", "Unknown"),
            "subject": header_dict.get("subject", "(No Subject)"),
            "body": body_text or msg_data.get("snippet", ""),
            "marked_as_read": mark_as_read
        }

    def _extract_body(self, payload: Dict[str, Any]) -> str:
        """Helper to recursively decode text parts from email payload."""
        body = payload.get("body", {})
        data = body.get("data")
        if data:
            try:
                return base64.urlsafe_b64decode(data).decode("utf-8", errors="replace")
            except Exception:
                pass

        parts = payload.get("parts", [])
        text_parts = []
        for part in parts:
            mime_type = part.get("mimeType", "")
            if mime_type == "text/plain":
                part_data = part.get("body", {}).get("data")
                if part_data:
                    try:
                        text_parts.append(base64.urlsafe_b64decode(part_data).decode("utf-8", errors="replace"))
                    except Exception:
                        pass
            elif "parts" in part:
                sub = self._extract_body(part)
                if sub:
                    text_parts.append(sub)

        return "\n".join(text_parts).strip()

    def send_email(self, to: str, subject: str, body: str) -> Dict[str, Any]:
        """Compose and send an email via Gmail REST API."""
        headers = self._get_auth_headers()
        headers["Content-Type"] = "application/json"

        message = EmailMessage()
        message["To"] = to
        message["Subject"] = subject
        message.set_content(body)

        raw_b64 = base64.urlsafe_b64encode(message.as_bytes()).decode("utf-8")

        resp = requests.post(
            f"{GMAIL_BASE_URL}/messages/send",
            headers=headers,
            json={"raw": raw_b64},
            timeout=20
        )
        resp.raise_for_status()
        send_data = resp.json()

        return {
            "status": "success",
            "message": "Email sent successfully via Gmail REST API.",
            "gmail_message_id": send_data.get("id"),
            "thread_id": send_data.get("threadId"),
            "to": to,
            "subject": subject
        }

    def search_emails(self, query: str, unread_only: bool = False, limit: int = 10) -> List[Dict[str, Any]]:
        """Search emails using Gmail query syntax."""
        headers = self._get_auth_headers()
        full_query = f"is:unread {query}" if unread_only else query
        params = {"q": full_query, "maxResults": limit}

        resp = requests.get(f"{GMAIL_BASE_URL}/messages", headers=headers, params=params, timeout=15)
        resp.raise_for_status()
        messages = resp.json().get("messages", [])

        results = []
        for item in messages:
            msg_id = item["id"]
            msg_resp = requests.get(
                f"{GMAIL_BASE_URL}/messages/{msg_id}",
                headers=headers,
                params={"format": "metadata", "metadataHeaders": ["From", "Date", "Subject"]},
                timeout=15
            )
            if msg_resp.status_code == 200:
                data = msg_resp.json()
                h_list = data.get("payload", {}).get("headers", [])
                h_dict = {h["name"].lower(): h["value"] for h in h_list}
                results.append({
                    "id": msg_id,
                    "writer": h_dict.get("from", "Unknown"),
                    "date": h_dict.get("date", "Unknown"),
                    "subject": h_dict.get("subject", "(No Subject)"),
                    "preview": data.get("snippet", "")
                })

        return results
