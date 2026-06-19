import os
from pathlib import Path
from urllib.parse import unquote, urlparse

import mysql.connector

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if load_dotenv:
    load_dotenv(PROJECT_ROOT / ".env")


def get_mysql_config():
    mysql_url = os.getenv("MYSQL_URL", "").strip()
    if mysql_url:
        parsed = urlparse(mysql_url)
        return {
            "host": parsed.hostname or "localhost",
            "port": parsed.port or 3306,
            "user": unquote(parsed.username or ""),
            "password": unquote(parsed.password or ""),
            "database": parsed.path.lstrip("/") or os.getenv("MYSQL_DATABASE", "banking_chatbot_db"),
        }

    return {
        "host": os.getenv("MYSQL_HOST", "localhost"),
        "port": int(os.getenv("MYSQL_PORT", "3306")),
        "user": os.getenv("MYSQL_USER", "root"),
        "password": os.getenv("MYSQL_PASSWORD", "root123"),
        "database": os.getenv("MYSQL_DATABASE", "banking_chatbot_db"),
    }


def get_mysql_connection():
    return mysql.connector.connect(**get_mysql_config())
