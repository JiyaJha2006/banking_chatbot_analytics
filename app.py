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
logger = logging.getLogger("banking_chatbot.api")

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
        logger.info("http.options path=%s client=%s", self.path, self.client_address[0])
        self.send_response(204)
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"
        logger.info("http.get path=%s query=%s client=%s", path, parsed.query, self.client_address[0])
        if path == "/api/health":
            logger.info("http.route endpoint=health method=GET")
            self.send_json({"ok": True, "message": "Banking chatbot backend is running."})
            return
        if path == "/api/token":
            logger.info("http.route endpoint=token_status method=GET")
            self.handle_token_status()
            return
        if path in {"/api/pages", "/api/buttons", "/api/actions", "/api/links"}:
            logger.info("http.route endpoint=links method=GET")
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
                logger.info("http.route endpoint=profile method=GET auth_user_id=%s requested_user_id=%s", auth_user["id"], user_id)
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
                auth_user = self.require_auth()
                logger.info("http.route endpoint=analytics method=GET auth_user_id=%s", auth_user["id"])
                self.send_json({"analytics": get_analytics_dashboard()})
            except ValueError as exc:
                self.send_json({"error": str(exc)}, status=401)
            except Exception as exc:
                self.send_json({"error": "Could not load analytics.", "details": str(exc)}, status=500)
            return
        self.serve_frontend(path)

    def do_POST(self):
        path = urlparse(self.path).path.rstrip("/") or "/"
        logger.info(
            "http.post path=%s client=%s content_length=%s",
            path,
            self.client_address[0],
            self.headers.get("Content-Length", "0"),
        )
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
            logger.info(
                "http.chat.start user_id=%s username=%s session_id=%s language=%s message=%s",
                auth_user["id"],
                auth_user["username"],
                payload.get("session_id"),
                language,
                " ".join(message.split())[:160],
            )
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
            logger.info(
                "http.chat.done user_id=%s session_id=%s response_time_ms=%s topic=%s methods=%s sources=%s reply_words=%s",
                auth_user["id"],
                result.get("session_id"),
                response_time_ms,
                result.get("topic"),
                result.get("search_methods"),
                [
                    {
                        "question": source.get("question", ""),
                        "score": source.get("hybrid_score", ""),
                        "methods": source.get("search_methods", []),
                    }
                    for source in result.get("sources", [])[:3]
                ],
                len(str(result.get("reply", "")).split()),
            )
            self.send_json(result)
        except ValueError as exc:
            logger.warning("http.chat.auth_or_value_error path=%s error=%s", path, exc)
            self.send_json({"error": str(exc)}, status=401 if "token" in str(exc).lower() else 400)
        except Exception as exc:
            logger.exception("http.chat.failed path=%s", path)
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
            logger.info("auth.token_status ok user_id=%s username=%s", user["id"], user["username"])
            self.send_json({"ok": True, "token": token, "token_type": "Bearer", "user": user})
        except ValueError as exc:
            logger.warning("auth.token_status failed error=%s", exc)
            self.send_json({"ok": False, "error": str(exc)}, status=401)

    def handle_logout(self):
        logger.info("auth.logout requested has_token=%s", bool(self.get_bearer_token()))
        revoke_auth_token(self.get_bearer_token())
        self.send_json({"ok": True})

    def handle_auth(self, path):
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
            username = str(payload.get("username", ""))
            password = str(payload.get("password", ""))
            logger.info("auth.start endpoint=%s username=%s password_present=%s", path, username, bool(password))
            user = signup_user(username, password) if path in {"/api/signup", "/api/register"} else signin_user(username, password)
            logger.info("auth.done endpoint=%s user_id=%s username=%s", path, user.get("id"), user.get("username"))
            self.send_json({"user": user})
        except ValueError as exc:
            logger.warning("auth.failed endpoint=%s error=%s", path, exc)
            self.send_json({"error": str(exc)}, status=400)
        except Exception as exc:
            logger.exception("auth.exception endpoint=%s", path)
            self.send_json({"error": "Authentication failed.", "details": str(exc)}, status=500)

    def send_json(self, data, status=200):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        logger.info("http.response status=%s bytes=%s keys=%s", status, len(body), list(data.keys()) if isinstance(data, dict) else [])
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def serve_frontend(self, request_path):
        root = FRONTEND_DIST if FRONTEND_DIST.exists() else FRONTEND_PUBLIC
        file_path = root / "index.html" if request_path in {"", "/"} else (root / request_path.lstrip("/")).resolve()
        logger.info("frontend.serve request_path=%s root=%s resolved=%s", request_path, root, file_path)
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
