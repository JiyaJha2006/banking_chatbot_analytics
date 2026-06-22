from __future__ import annotations

import hashlib
import hmac
import os
import secrets
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

from mysql.connector import Error, errorcode

from .db import get_mysql_connection

TOKEN_TTL_HOURS = 24
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SQLITE_DB_PATH = PROJECT_ROOT / "data" / "chatbot_app.sqlite3"


def using_sqlite():
    return os.getenv("DATABASE_BACKEND", "mysql").strip().lower() == "sqlite"


def sql_placeholders(query):
    return query.replace("%s", "?") if using_sqlite() else query


def get_auth_connection():
    if not using_sqlite():
        return get_mysql_connection()
    sqlite_path = Path(os.getenv("SQLITE_DB_PATH", str(DEFAULT_SQLITE_DB_PATH)))
    sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(sqlite_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def dict_row(row):
    return dict(row) if row else None


def cursor_dict(conn):
    return conn.cursor() if using_sqlite() else conn.cursor(dictionary=True)


def parse_datetime(value):
    if isinstance(value, datetime) or value is None:
        return value
    return datetime.fromisoformat(str(value))


def hash_password(password: str, salt: bytes | None = None) -> tuple[str, str]:
    salt = salt or os.urandom(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 120000)
    return salt.hex(), digest.hex()


def verify_password(password: str, salt_hex: str, password_hash: str) -> bool:
    _, attempted_hash = hash_password(password, bytes.fromhex(salt_hex))
    return hmac.compare_digest(attempted_hash, password_hash)


def ensure_auth_tables():
    conn = get_auth_connection()
    cursor = conn.cursor()
    id_type = "INTEGER PRIMARY KEY AUTOINCREMENT" if using_sqlite() else "INT AUTO_INCREMENT PRIMARY KEY"
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS chatbot_users (
            id {id_type},
            username VARCHAR(100) NOT NULL UNIQUE,
            password_salt VARCHAR(64) NOT NULL,
            password_hash VARCHAR(128) NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_signin_at DATETIME NULL
        )
    """)
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS chatbot_questions (
            id {id_type},
            user_id INT NULL,
            username VARCHAR(100) NULL,
            question TEXT NOT NULL,
            language VARCHAR(30) NOT NULL,
            session_id VARCHAR(80) NULL,
            topic VARCHAR(160) NULL,
            response_time_ms INT NULL,
            asked_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES chatbot_users(id)
                ON DELETE SET NULL
        )
    """)
    for alter_sql in [
        "ALTER TABLE chatbot_questions ADD COLUMN topic VARCHAR(160) NULL",
        "ALTER TABLE chatbot_questions ADD COLUMN response_time_ms INT NULL",
    ]:
        try:
            cursor.execute(alter_sql)
        except (Error, sqlite3.OperationalError) as exc:
            is_duplicate = (
                getattr(exc, "errno", None) == errorcode.ER_DUP_FIELDNAME
                or "duplicate column" in str(exc).lower()
            )
            if not is_duplicate:
                raise
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS chatbot_auth_tokens (
            id {id_type},
            user_id INT NOT NULL,
            token_hash VARCHAR(128) NOT NULL UNIQUE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            expires_at DATETIME NOT NULL,
            revoked_at DATETIME NULL,
            FOREIGN KEY (user_id) REFERENCES chatbot_users(id)
                ON DELETE CASCADE
        )
    """)
    conn.commit()
    cursor.close()
    conn.close()


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def create_auth_token(user_id: int) -> dict:
    ensure_auth_tables()
    token = secrets.token_urlsafe(32)
    expires_at = datetime.now() + timedelta(hours=TOKEN_TTL_HOURS)
    conn = get_auth_connection()
    cursor = conn.cursor()
    cursor.execute(sql_placeholders(
        """
        INSERT INTO chatbot_auth_tokens (user_id, token_hash, expires_at)
        VALUES (%s, %s, %s)
        """),
        (user_id, hash_token(token), expires_at),
    )
    conn.commit()
    cursor.close()
    conn.close()
    return {
        "token": token,
        "token_type": "Bearer",
        "expires_at": expires_at.isoformat(sep=" ", timespec="seconds"),
        "expires_in_seconds": TOKEN_TTL_HOURS * 60 * 60,
    }


def verify_auth_token(token: str):
    if not token:
        raise ValueError("Authorization token is required.")

    ensure_auth_tables()
    conn = get_auth_connection()
    cursor = cursor_dict(conn)
    cursor.execute(sql_placeholders(
        """
        SELECT u.id, u.username, t.expires_at, t.revoked_at
        FROM chatbot_auth_tokens t
        JOIN chatbot_users u ON u.id = t.user_id
        WHERE t.token_hash = %s
        """),
        (hash_token(token),),
    )
    row = dict_row(cursor.fetchone())
    cursor.close()
    conn.close()
    if not row or row.get("revoked_at"):
        raise ValueError("Invalid authorization token.")
    if parse_datetime(row["expires_at"]) <= datetime.now():
        raise ValueError("Authorization token has expired.")
    return {"id": row["id"], "username": row["username"]}


def revoke_auth_token(token: str):
    if not token:
        return
    ensure_auth_tables()
    conn = get_auth_connection()
    cursor = conn.cursor()
    cursor.execute(sql_placeholders(
        """
        UPDATE chatbot_auth_tokens
        SET revoked_at = %s
        WHERE token_hash = %s AND revoked_at IS NULL
        """),
        (datetime.now(), hash_token(token)),
    )
    conn.commit()
    cursor.close()
    conn.close()


def signup_user(username: str, password: str):
    username = username.strip()
    if not username or not password:
        raise ValueError("Username and password are required.")

    ensure_auth_tables()
    salt, password_hash = hash_password(password)
    conn = get_auth_connection()
    cursor = cursor_dict(conn)
    try:
        cursor.execute(sql_placeholders(
            """
            INSERT INTO chatbot_users (username, password_salt, password_hash, last_signin_at)
            VALUES (%s, %s, %s, %s)
            """),
            (username, salt, password_hash, datetime.now()),
        )
        conn.commit()
        user_id = cursor.lastrowid
    except (Error, sqlite3.IntegrityError) as exc:
        if getattr(exc, "errno", None) == errorcode.ER_DUP_ENTRY or "unique" in str(exc).lower():
            raise ValueError("That username already exists. Please sign in instead.") from exc
        raise
    finally:
        cursor.close()
        conn.close()

    user = {"id": user_id, "username": username, "is_new_user": True}
    return {**user, **create_auth_token(user_id)}


def signin_user(username: str, password: str):
    username = username.strip()
    if not username or not password:
        raise ValueError("Username and password are required.")

    ensure_auth_tables()
    conn = get_auth_connection()
    cursor = cursor_dict(conn)
    cursor.execute(sql_placeholders(
        """
        SELECT id, username, password_salt, password_hash
        FROM chatbot_users
        WHERE username = %s
        """),
        (username,),
    )
    user = dict_row(cursor.fetchone())
    if not user or not verify_password(password, user["password_salt"], user["password_hash"]):
        cursor.close()
        conn.close()
        raise ValueError("Invalid username or password.")

    cursor.execute(
        sql_placeholders("UPDATE chatbot_users SET last_signin_at = %s WHERE id = %s"),
        (datetime.now(), user["id"]),
    )
    conn.commit()
    cursor.close()
    conn.close()
    auth_token = create_auth_token(user["id"])
    return {"id": user["id"], "username": user["username"], "is_new_user": False, **auth_token}


def log_question(user_id, username: str, question: str, language: str, session_id=None, topic=None, response_time_ms=None):
    if not question.strip():
        return

    ensure_auth_tables()
    conn = get_auth_connection()
    cursor = conn.cursor()
    cursor.execute(sql_placeholders(
        """
        INSERT INTO chatbot_questions (user_id, username, question, language, session_id, topic, response_time_ms)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        """),
        (user_id, username, question.strip(), language, session_id, topic, response_time_ms),
    )
    conn.commit()
    cursor.close()
    conn.close()


def get_user_profile(user_id):
    ensure_auth_tables()
    conn = get_auth_connection()
    cursor = cursor_dict(conn)
    cursor.execute(sql_placeholders(
        """
        SELECT id, username, created_at, last_signin_at
        FROM chatbot_users
        WHERE id = %s
        """),
        (user_id,),
    )
    user = dict_row(cursor.fetchone())
    if not user:
        cursor.close()
        conn.close()
        raise ValueError("User not found.")

    cursor.execute(sql_placeholders(
        """
        SELECT COUNT(*) AS total_questions,
               MAX(asked_at) AS last_question_at
        FROM chatbot_questions
        WHERE user_id = %s
        """),
        (user_id,),
    )
    stats = dict_row(cursor.fetchone()) or {}
    cursor.execute(sql_placeholders(
        """
        SELECT question, language, asked_at
        FROM chatbot_questions
        WHERE user_id = %s
        ORDER BY asked_at DESC
        LIMIT 5
        """),
        (user_id,),
    )
    recent_questions = [dict_row(row) for row in cursor.fetchall()]
    cursor.close()
    conn.close()

    def clean_datetime(value):
        if isinstance(value, str):
            return value[:16]
        return value.isoformat(sep=" ", timespec="minutes") if value else None

    return {
        "id": user["id"],
        "username": user["username"],
        "created_at": clean_datetime(user.get("created_at")),
        "last_signin_at": clean_datetime(user.get("last_signin_at")),
        "total_questions": stats.get("total_questions", 0),
        "last_question_at": clean_datetime(stats.get("last_question_at")),
        "recent_questions": [
            {
                "question": row["question"],
                "language": row["language"],
                "asked_at": clean_datetime(row.get("asked_at")),
            }
            for row in recent_questions
        ],
    }


def get_analytics_dashboard():
    ensure_auth_tables()
    conn = get_auth_connection()
    cursor = cursor_dict(conn)

    cursor.execute("SELECT COUNT(*) AS total_users FROM chatbot_users")
    user_stats = dict_row(cursor.fetchone()) or {}

    cursor.execute(
        """
        SELECT COUNT(*) AS total_questions,
               ROUND(AVG(response_time_ms)) AS average_response_time_ms
        FROM chatbot_questions
        """
    )
    question_stats = dict_row(cursor.fetchone()) or {}

    cursor.execute(
        """
        SELECT question, COUNT(*) AS count
        FROM chatbot_questions
        GROUP BY LOWER(TRIM(question)), question
        ORDER BY count DESC, MAX(asked_at) DESC
        LIMIT 1
        """
    )
    most_asked = dict_row(cursor.fetchone())

    cursor.execute(
        """
        SELECT COALESCE(NULLIF(topic, ''), 'general') AS topic, COUNT(*) AS count
        FROM chatbot_questions
        GROUP BY COALESCE(NULLIF(topic, ''), 'general')
        ORDER BY count DESC
        LIMIT 1
        """
    )
    most_topic = dict_row(cursor.fetchone())

    cursor.execute(
        """
        SELECT question AS label, COUNT(*) AS value
        FROM chatbot_questions
        GROUP BY LOWER(TRIM(question)), question
        ORDER BY value DESC, MAX(asked_at) DESC
        LIMIT 5
        """
    )
    top_questions = [dict_row(row) for row in cursor.fetchall()]

    cursor.execute(
        """
        SELECT COALESCE(NULLIF(topic, ''), 'general') AS label, COUNT(*) AS value
        FROM chatbot_questions
        GROUP BY COALESCE(NULLIF(topic, ''), 'general')
        ORDER BY value DESC
        LIMIT 6
        """
    )
    top_topics = [dict_row(row) for row in cursor.fetchall()]

    cursor.execute(
        """
        SELECT DATE(asked_at) AS label, COUNT(*) AS value
        FROM chatbot_questions
        GROUP BY DATE(asked_at)
        ORDER BY DATE(asked_at) DESC
        LIMIT 7
        """
    )
    daily_rows = list(reversed([dict_row(row) for row in cursor.fetchall()]))

    cursor.close()
    conn.close()

    def clean_number(value):
        return int(value or 0)

    return {
        "most_asked_question": {
            "question": (most_asked or {}).get("question", "No questions yet"),
            "count": clean_number((most_asked or {}).get("count")),
        },
        "most_searched_topic": {
            "topic": (most_topic or {}).get("topic", "No topics yet"),
            "count": clean_number((most_topic or {}).get("count")),
        },
        "number_of_users": clean_number(user_stats.get("total_users")),
        "total_questions": clean_number(question_stats.get("total_questions")),
        "average_response_time_ms": clean_number(question_stats.get("average_response_time_ms")),
        "charts": {
            "top_questions": [{"label": row["label"], "value": clean_number(row["value"])} for row in top_questions],
            "top_topics": [{"label": row["label"], "value": clean_number(row["value"])} for row in top_topics],
            "daily_questions": [{"label": str(row["label"]), "value": clean_number(row["value"])} for row in daily_rows],
        },
    }
