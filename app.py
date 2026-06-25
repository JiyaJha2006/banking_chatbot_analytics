from __future__ import annotations

import json
import logging
import mimetypes
import os
import threading
import time
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from backend.auth_service import ensure_auth_tables, get_analytics_dashboard, get_user_profile, log_question, revoke_auth_token, signin_user, signup_user, verify_auth_token
from backend.chat_service import answer_message, load_models, load_vector_db

PROJECT_ROOT = Path(__file__).resolve().parent
FRONTEND_DIST = PROJECT_ROOT / "frontend" / "dist"
FRONTEND_PUBLIC = PROJECT_ROOT / "frontend" / "public"
HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", "8000"))
LIGHTWEIGHT_MODE = os.getenv("LIGHTWEIGHT_MODE", "0").lower() in {"1", "true", "yes"}
logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)

PAGE_LINKS = {
    "login": "/login",
    "signup": "/signup",
    "chat": "/chat",
    "new_chat": "/chat/new",
    "profile": "/profile",
    "analytics": "/analytics",
    "logout": "/logout",
}

API_LINKS = {
    "health": "/api/health",
    "pages": "/api/pages",
    "token": "/api/token",
    "signin": "/api/login",
    "signup": "/api/register",
    "logout": "/api/logout",
    "chat": "/api/chat",
    "profile": "/api/profile?user_id={user_id}",
    "analytics": "/api/analytics",
}

BUTTON_LINKS = {
    "signin_tab": "/login",
    "signup_tab": "/signup",
    "submit_signin": "/api/login",
    "submit_signup": "/api/register",
    "new_chat": "/chat/new",
    "chat_view": "/chat",
    "profile_view": "/profile",
    "analytics_view": "/analytics",
    "logout": "/logout",
    "collapse_sidebar": "/api/buttons/collapse_sidebar",
    "send_message": "/api/chat",
    "voice_input": "/api/buttons/voice_input",
    "english_language": "/chat?language=English",
    "hindi_language": "/chat?language=Hindi",
    "health_check": "/api/health",
}


class BankingChatbotHandler(SimpleHTTPRequestHandler):
    server_version = "BankingChatbotAPI/1.0"

    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(204)
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"
        if path == "/api/health":
            self.send_json({"ok": True, "message": "Banking chatbot backend is running."})
            return
        if path == "/api/token":
            self.handle_token_status()
            return
        if path in {"/api/pages", "/api/buttons", "/api/actions", "/api/links"}:
            self.send_json({"pages": PAGE_LINKS, "apis": API_LINKS, "buttons": BUTTON_LINKS})
            return
        if path.startswith("/api/pages/"):
            page_name = path.rsplit("/", 1)[-1]
            page_link = PAGE_LINKS.get(page_name)
            if not page_link:
                self.send_json({"error": "Page not found"}, status=404)
                return
            self.send_json({"name": page_name, "link": page_link})
            return
        if path.startswith("/api/buttons/"):
            button_name = path.rsplit("/", 1)[-1]
            button_link = BUTTON_LINKS.get(button_name)
            if not button_link:
                self.send_json({"error": "Button not found"}, status=404)
                return
            self.send_json({"name": button_name, "link": button_link})
            return
        if path == "/api/profile":
            try:
                auth_user = self.require_auth()
                query = parse_qs(parsed.query)
                user_id = query.get("user_id", [""])[0]
                if str(auth_user["id"]) != str(user_id):
                    self.send_json({"error": "You are not authorized to view this profile."}, status=403)
                    return
                self.send_json({"profile": get_user_profile(user_id)})
            except ValueError as exc:
                self.send_json({"error": str(exc)}, status=401)
            except Exception as exc:
                self.send_json({"error": "Could not load profile.", "details": str(exc)}, status=500)
            return
        if path == "/api/analytics":
            try:
                self.require_auth()
                self.send_json({"analytics": get_analytics_dashboard()})
            except ValueError as exc:
                self.send_json({"error": str(exc)}, status=401)
            except Exception as exc:
                self.send_json({"error": "Could not load analytics.", "details": str(exc)}, status=500)
            return
        self.serve_frontend(path)

    def do_POST(self):
        path = urlparse(self.path).path.rstrip("/") or "/"
        if path in {"/api/signup", "/api/signin", "/api/register", "/api/login"}:
            self.handle_auth(path)
            return
        if path == "/api/logout":
            self.handle_logout()
            return
        if path != "/api/chat":
            self.send_json({"error": "Not found"}, status=404)
            return
        try:
            auth_user = self.require_auth()
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
            message = str(payload.get("message", ""))
            language = str(payload.get("language", "English"))
            start_time = time.perf_counter()
            result = answer_message(
                message=message,
                language=language,
                session_id=payload.get("session_id"),
            )
            response_time_ms = int((time.perf_counter() - start_time) * 1000)
            log_question(
                user_id=auth_user["id"],
                username=auth_user["username"],
                question=message,
                language=language,
                session_id=payload.get("session_id"),
                topic=result.get("topic"),
                response_time_ms=response_time_ms,
            )
            result["response_time_ms"] = response_time_ms
            self.send_json(result)
        except ValueError as exc:
            self.send_json({"error": str(exc)}, status=401 if "token" in str(exc).lower() else 400)
        except Exception as exc:
            self.send_json({"error": "The chatbot backend failed to answer.", "details": str(exc)}, status=500)

    def get_bearer_token(self):
        header = self.headers.get("Authorization", "")
        if header.lower().startswith("bearer "):
            return header.split(" ", 1)[1].strip()
        return ""

    def require_auth(self):
        return verify_auth_token(self.get_bearer_token())

    def handle_token_status(self):
        try:
            token = self.get_bearer_token()
            user = self.require_auth()
            self.send_json({"ok": True, "token": token, "token_type": "Bearer", "user": user})
        except ValueError as exc:
            self.send_json({"ok": False, "error": str(exc)}, status=401)

    def handle_logout(self):
        revoke_auth_token(self.get_bearer_token())
        self.send_json({"ok": True})

    def handle_auth(self, path):
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
            username = str(payload.get("username", ""))
            password = str(payload.get("password", ""))
            user = signup_user(username, password) if path in {"/api/signup", "/api/register"} else signin_user(username, password)
            self.send_json({"user": user})
        except ValueError as exc:
            self.send_json({"error": str(exc)}, status=400)
        except Exception as exc:
            self.send_json({"error": "Authentication failed.", "details": str(exc)}, status=500)

    def send_json(self, data, status=200):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def serve_frontend(self, request_path):
        root = FRONTEND_DIST if FRONTEND_DIST.exists() else FRONTEND_PUBLIC
        file_path = root / "index.html" if request_path in {"", "/"} else (root / request_path.lstrip("/")).resolve()
        if root.resolve() not in file_path.parents and file_path != root.resolve():
            self.send_error(403)
            return
        if not file_path.exists() or file_path.is_dir():
            file_path = root / "index.html"
        if not file_path.exists():
            self.send_json({
                "error": "React frontend is not built yet.",
                "hint": "Run: cd frontend; npm install; npm run build. Or use npm run dev for development.",
            }, status=404)
            return
        content = file_path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", mimetypes.guess_type(str(file_path))[0] or "application/octet-stream")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)


def main():
    url = f"http://{HOST}:{PORT}"
    server = ThreadingHTTPServer((HOST, PORT), BankingChatbotHandler)

    def warm_up_chatbot():
        try:
            ensure_auth_tables()
            if LIGHTWEIGHT_MODE:
                print("Lightweight mode enabled. Skipping chatbot model startup.", flush=True)
                return
            print("Loading chatbot models in the background. First answer may take a while...", flush=True)
            load_models()
            load_vector_db()
            print("Chatbot models are ready.", flush=True)
        except Exception as exc:
            print(f"Chatbot model startup failed: {exc}", flush=True)

    print(f"Banking Chatbot running at {url}", flush=True)
    threading.Thread(target=warm_up_chatbot, daemon=True).start()
    if os.getenv("OPEN_BROWSER", "1") == "1":
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    server.serve_forever()


if __name__ == "__main__":
    main()
