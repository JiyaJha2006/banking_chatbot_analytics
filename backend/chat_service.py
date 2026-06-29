from __future__ import annotations

import json
import logging
import os
import uuid
import re
from difflib import SequenceMatcher, get_close_matches
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import chromadb
from deep_translator import GoogleTranslator
from sentence_transformers import CrossEncoder, SentenceTransformer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from .official_kb import load_official_kb_documents

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DB_PATH = str(DATA_DIR / "vector_db")
BANKING_COLLECTION_NAME = "banking_knowledge_base"
SESSIONS = {}
BANKING_METADATA_CACHE = {"count": -1, "metadatas": []}
MODEL_BUNDLE = None
LIGHTWEIGHT_MODE = os.getenv("LIGHTWEIGHT_MODE", "0").lower() in {"1", "true", "yes"}
logger = logging.getLogger("banking_chatbot.chat")


def log_text(value, max_length=160):
    text = " ".join(str(value or "").split())
    return text if len(text) <= max_length else f"{text[:max_length]}..."

BANKING_TOPIC_ALIASES = {
    "savings account": "savings account",
    "saving account": "savings account",
    "student account": "student savings account",
    "salary account": "salary account",
    "current account": "current account",
    "zero balance account": "zero balance savings account",
    "fixed deposit": "fixed deposit",
    "fd": "fixed deposit",
    "recurring deposit": "recurring deposit",
    "rd": "recurring deposit",
    "home loan": "home loan",
    "housing loan": "home loan",
    "personal loan": "personal loan",
    "education loan": "education loan",
    "car loan": "car loan",
    "vehicle loan": "vehicle loan",
    "gold loan": "gold loan",
    "credit card": "credit card",
    "axis credit card": "credit card",
    "axis bank credit card": "credit card",
    "debit card": "debit card",
    "atm card": "debit card",
    "net banking": "net banking",
    "netbanking": "net banking",
    "internet banking": "net banking",
    "mobile banking": "mobile banking",
    "upi": "UPI",
    "kyc": "KYC",
    "neft": "NEFT",
    "rtgs": "RTGS",
    "imps": "IMPS",
    "cheque": "cheque",
    "checkbook": "cheque book",
    "cheque book": "cheque book",
    "passbook": "passbook",
    "bank statement": "bank statement",
    "minimum balance": "minimum balance",
}

BANKING_VOCABULARY = set(
    " ".join(BANKING_TOPIC_ALIASES).split()
) | {
    "account", "bank", "banking", "balance", "branch", "cash", "charge",
    "complaint", "deposit", "dispute", "document", "eligibility", "fee", "fraud",
    "interest", "loan", "money",
    "nominee", "password", "payment", "pin", "statement", "transaction",
    "transfer", "withdrawal", "apply", "open", "activate", "documents",
    "required", "salaried", "self-employed", "compare", "benefits", "fees",
    "charges", "steps", "chargeback", "unauthorized", "unauthorised",
    "suspicious", "ombudsman", "escalate", "escalation", "reward", "rewards",
    "points", "edge", "redeem", "redemption", "cashback", "bill", "billing",
    "due", "emi", "approval", "approved", "consent", "liability", "rbi",
    "annual", "joining", "limit", "limits", "cardholder", "phishing", "vishing",
    "skimming", "malware", "spoofing", "harassment", "mile", "miles", "waiver",
    "waived", "travel", "traveller", "flight", "hotel", "replacement", "replace",
    "block", "blocking", "blocked", "lost", "stolen", "missing", "paper",
    "papers", "website", "online", "qualify", "qualified", "merchant", "valid",
    "investigate", "investigation", "record", "records", "evidence",
}

QUERY_WORDS = {
    "what", "when", "where", "which", "who", "why", "how", "can", "could",
    "do", "does", "is", "are", "help", "make", "create", "start", "tell",
    "explain", "need", "needed", "required", "about", "for", "with", "my",
    "me", "i", "it", "this", "that", "them", "they", "more", "much",
}

COMMON_QUERY_CORRECTIONS = {
    "hlp": "help",
    "pls": "please",
    "acount": "account",
    "accnt": "account",
    "savngs": "savings",
    "documnts": "documents",
    "eligiblity": "eligibility",
    "intrest": "interest",
    "pasword": "password",
    "transction": "transaction",
}

SENSITIVE_PERSONAL_TERMS = [
    "password", "passcode", "pin", "otp", "one time password", "cvv", "card number",
    "debit card number", "credit card number", "account number", "ifsc", "routing number",
    "sort code", "aadhaar", "aadhar", "pan number", "passport number", "social security",
    "ssn", "date of birth", "dob", "address", "phone number", "mobile number", "email",
    "login id", "user id", "username", "security answer", "secret answer", "mother's maiden",
    "balance", "transaction", "transactions", "transaction history", "bank statement",
]

SENSITIVE_REQUEST_PATTERNS = [
    r"\b(what|tell|show|give|send|share|find|check|display|reveal)\b.*\b(my|me|mine|this user|that user|someone|customer|account holder)\b",
    r"\b(my|mine|me|this user|that user|someone|customer|account holder)\b.*\b(what|tell|show|give|send|share|find|check|display|reveal)\b",
    r"\b(can|could|will|would)\s+you\b.*\b(access|open|see|view|retrieve|fetch)\b.*\b(my|someone|customer)\b",
]

SENSITIVE_ACTION_EXCEPTIONS = [
    "reset", "change", "update", "recover", "forgot", "protect", "secure", "block",
    "unblock", "report", "lost", "stolen", "apply", "register", "create", "set up",
    "activate", "deactivate", "close", "download", "generate statement",
]


def is_sensitive_personal_question(question):
    q = question.lower().strip()
    if not q:
        return False
    if any(exception in q for exception in SENSITIVE_ACTION_EXCEPTIONS):
        return False
    has_sensitive_term = any(re.search(rf"\b{re.escape(term)}\b", q) for term in SENSITIVE_PERSONAL_TERMS)
    if not has_sensitive_term:
        return False
    return any(re.search(pattern, q) for pattern in SENSITIVE_REQUEST_PATTERNS)


def build_sensitive_personal_refusal(language="English"):
    if language == "Hindi":
        return (
            "Maaf kijiye, main passwords, OTP, PIN, card/account numbers, balance, "
            "transaction history ya kisi bhi private personal detail ko bata ya verify nahi kar sakta. "
            "Aisi information ke liye apne bank ki official app, website, ya branch ka use karein."
        )
    return (
        "I cannot answer questions that ask for private personal information such as passwords, OTPs, PINs, "
        "card or account numbers, balances, transaction history, addresses, or identity details. "
        "Please use your bank's official app, website, or branch for private account information."
    )


@lru_cache(maxsize=1)
def load_models():
    global MODEL_BUNDLE
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    reranker_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    llm_model_name = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
    llm_model = AutoModelForSeq2SeqLM.from_pretrained(llm_model_name)
    MODEL_BUNDLE = (embedding_model, reranker_model, tokenizer, llm_model)
    return MODEL_BUNDLE


def get_ready_models():
    return MODEL_BUNDLE


@lru_cache(maxsize=1)
def load_vector_db():
    client = chromadb.PersistentClient(path=DB_PATH)
    banking_collection = client.get_or_create_collection(name=BANKING_COLLECTION_NAME)
    return ensure_official_banking_collection(client, banking_collection)


def collection_uses_official_kb(collection, expected_count):
    if collection.count() != expected_count:
        return False
    if expected_count == 0:
        return False
    sample = collection.get(limit=1, include=["metadatas"])
    metadatas = sample.get("metadatas") or []
    return bool(metadatas and metadatas[0].get("dataset") == "official_kb")


def ensure_official_banking_collection(client, collection):
    official_documents = load_official_kb_documents()
    if not official_documents:
        return collection
    if collection_uses_official_kb(collection, len(official_documents)):
        return collection

    client.delete_collection(name=BANKING_COLLECTION_NAME)
    collection = client.get_or_create_collection(name=BANKING_COLLECTION_NAME)
    ready_models = get_ready_models()
    embedding_model = ready_models[0] if ready_models is not None else SentenceTransformer("all-MiniLM-L6-v2")

    ids = []
    documents = []
    embeddings = []
    metadatas = []
    for item in official_documents:
        document_text = (
            f"Section: {item['section']}\n"
            f"Question: {item['question']}\n"
            f"Answer: {item['answer']}\n"
            f"Source: {item['source']}"
        )
        ids.append(item["id"])
        documents.append(document_text)
        embeddings.append(embedding_model.encode(document_text).tolist())
        metadatas.append(
            {
                "section": item["section"],
                "question": item["question"],
                "answer": item["answer"],
                "source": item["source"],
                "source_file": item["source_file"],
                "dataset": "official_kb",
            }
        )
    collection.upsert(ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas)
    BANKING_METADATA_CACHE["count"] = -1
    BANKING_METADATA_CACHE["metadatas"] = []
    return collection


def get_session(session_id=None):
    if not session_id:
        session_id = str(uuid.uuid4())
    SESSIONS.setdefault(session_id, {
        "chat_history": [],
        "current_topic": "",
        "current_intent": "general",
        "topic_history": [],
        "last_resolved_question": "",
        "last_kb_source": {},
        "last_entity_kb_source": {},
        "kb_source_history": [],
        "last_comparison": {},
    })
    return session_id, SESSIONS[session_id]


def is_follow_up_question(user_question):
    q = user_question.lower().strip()
    phrases = [
        "it", "that", "this", "they", "them", "those", "these", "for it",
        "about it", "for that", "about that", "how much", "what documents",
        "documents needed", "required documents", "explain more", "tell me more",
        "what about", "does it", "can it", "is it", "which is better",
        "better one", "how long", "what next", "then what", "what should i do next",
        "where do i file", "file online", "how long should i wait", "how many days",
        "what if", "where do i", "how do i", "does this", "can this", "is this",
        "should i keep", "records should i keep",
    ]
    if any(re.search(rf"\b{re.escape(phrase)}\b", q) for phrase in phrases):
        return True
    short_follow_up_terms = {
        "late", "interest", "fee", "fees", "charge", "charges", "cashback", "reward",
        "rewards", "points", "documents", "limit", "limits", "online", "next",
        "complaint", "resolve", "resolved", "solve", "solved", "delay", "delayed",
        "waiver", "waived", "miles", "mile", "proof", "record", "records",
        "evidence", "replacement", "replace",
    }
    if len(q.split()) <= 3 and search_tokens(q) & short_follow_up_terms:
        return True
    intent = detect_question_intent(q)
    return len(q.split()) <= 7 and intent in {
        "documents", "fees", "interest", "tenure", "limits", "eligibility", "opening", "process", "dispute"
    }


def build_recent_chat_history(session):
    text = ""
    for chat in session["chat_history"][-6:]:
        role = "User" if chat["role"] == "user" else "Assistant"
        text += f"{role}: {chat['message']}\n"
    return text


def resolve_question_context(user_question, session):
    current_topic = str(session.get("current_topic", "")).strip()
    explicit_topic = extract_topic_from_question(user_question)
    follow_up = is_follow_up_question(user_question)
    previous_source = session.get("last_kb_source") or {}
    if follow_up and not explicit_topic and previous_source:
        source_context = " ".join([
            previous_source.get("section", ""),
            previous_source.get("question", ""),
        ]).strip()
        if source_context:
            resolved = " ".join(f"{user_question} about {source_context}".split())
            logger.info(
                "context.resolve follow_up_kb_source original=%s resolved=%s previous_section=%s previous_question=%s",
                log_text(user_question),
                log_text(resolved),
                previous_source.get("section", ""),
                log_text(previous_source.get("question", "")),
            )
            return resolved

    if explicit_topic or not current_topic or not follow_up:
        logger.info(
            "context.resolve passthrough explicit_topic=%s current_topic=%s follow_up=%s question=%s",
            explicit_topic or "",
            current_topic or "",
            follow_up,
            log_text(user_question),
        )
        return user_question

    resolved = f" {user_question.strip()} "
    for pronoun in ["it", "this", "that", "they", "them", "those", "these"]:
        resolved = re.sub(rf"\b{pronoun}\b", current_topic, resolved, flags=re.IGNORECASE)
    resolved = " ".join(resolved.split())
    if current_topic.lower() not in resolved.lower():
        resolved = f"{resolved} about {current_topic}"
    logger.info(
        "context.resolve follow_up current_topic=%s original=%s resolved=%s",
        current_topic,
        log_text(user_question),
        log_text(resolved),
    )
    return resolved


def remember_conversation_context(session, topic="", intent="general", resolved_question=""):
    topic = normalize_topic_label(topic)
    previous_topic = session.get("current_topic", "")
    previous_intent = session.get("current_intent", "general")
    if topic and topic.lower() not in {"account recommendation", "banking", "bank"}:
        previous = session.get("current_topic", "")
        if previous and previous != topic:
            history = session.setdefault("topic_history", [])
            if previous not in history:
                history.append(previous)
            session["topic_history"] = history[-4:]
        session["current_topic"] = topic
    if intent and intent != "general":
        session["current_intent"] = intent
    if resolved_question:
        session["last_resolved_question"] = resolved_question
    logger.info(
        "context.remember previous_topic=%s new_topic=%s previous_intent=%s new_intent=%s resolved_question=%s topic_history=%s",
        previous_topic or "",
        session.get("current_topic", ""),
        previous_intent,
        session.get("current_intent", "general"),
        log_text(resolved_question),
        session.get("topic_history", []),
    )


def remember_kb_source_context(session, source):
    if not source:
        return
    answer_context = str(source.get("context_answer") or source.get("answer", ""))
    remembered_source = {
        "section": str(source.get("section", "")),
        "question": str(source.get("question", "")),
        "source_file": str(source.get("source_file", "")),
        "source": str(source.get("source", "")),
        "answer": answer_context,
        "answer_preview": log_text(answer_context, 280),
    }
    session["last_kb_source"] = remembered_source
    if source_entity_label(remembered_source):
        session["last_entity_kb_source"] = remembered_source
    history = session.setdefault("kb_source_history", [])
    source_key = (
        remembered_source.get("source_file", ""),
        remembered_source.get("question", ""),
    )
    history = [
        item for item in history
        if (item.get("source_file", ""), item.get("question", "")) != source_key
    ]
    history.append(remembered_source)
    session["kb_source_history"] = history[-5:]
    logger.info(
        "context.kb_source_remember section=%s question=%s source_file=%s",
        session["last_kb_source"].get("section", ""),
        log_text(session["last_kb_source"].get("question", "")),
        session["last_kb_source"].get("source_file", ""),
    )


def build_contextual_retrieval_query(query, session):
    source = session.get("last_kb_source") or {}
    if not source:
        return query
    context_text = " ".join([
        source.get("section", ""),
        source.get("question", ""),
    ]).strip()
    if not context_text:
        return query
    return f"{query} previous context: {context_text}"


def detect_question_intent(question):
    q = question.lower().strip()
    intent_rules = [
        ("dispute", ["dispute", "complaint", "complain", "chargeback", "fraud", "fraudulent", "unauthorized", "unauthorised", "suspicious", "strange activity", "report", "block card", "lost card", "stolen card", "ombudsman", "escalate", "escalation", "resolve", "resolved", "unresolved", "solve", "solved"]),
        ("documents", ["document", "documents", "proof", "kyc", "required", "requirement", "need to carry", "needed"]),
        ("fees", ["fee", "fees", "charge", "charges", "cost", "minimum balance", "penalty"]),
        ("interest", ["interest rate", "rate of interest", "returns", "interest earned", "how much interest"]),
        ("tenure", ["how long", "how many days", "tenure", "duration", "maturity period", "term period"]),
        ("limits", ["limit", "limits", "maximum amount", "minimum amount", "transaction limit", "withdrawal limit"]),
        ("eligibility", ["eligible", "eligibility", "who can", "can i get", "can i qualify", "am i eligible", "allowed", "qualify"]),
        ("opening", ["open", "opening", "make", "create", "start", "set up", "setup", "apply", "register", "get an account", "get a card"]),
        ("process", ["how can", "how do", "how to", "steps", "procedure", "process", "way to", "ways to", "help me", "apply from website", "apply online"]),
        ("definition", ["what is", "what are", "meaning", "define", "explain", "tell me about"]),
    ]
    for intent, phrases in intent_rules:
        if any(phrase in q for phrase in phrases):
            return intent
    return "general"


def build_intent_search_query(search_query, intent):
    if intent == "opening":
        return f"how to open apply register steps documents required {search_query}"
    if intent == "documents":
        return f"documents required proof KYC needed for {search_query}"
    if intent == "process":
        return f"steps process procedure how to {search_query}"
    if intent == "dispute":
        return f"dispute chargeback fraud complaint ombudsman unauthorized suspicious lost stolen block report {search_query}"
    if intent == "fees":
        return f"fees charges minimum balance cost for {search_query}"
    if intent == "eligibility":
        return f"eligibility who can apply allowed for {search_query}"
    if intent == "interest":
        return f"interest rate returns earned for {search_query}"
    if intent == "tenure":
        return f"tenure duration maturity period for {search_query}"
    if intent == "limits":
        return f"minimum maximum transaction withdrawal limits for {search_query}"
    if intent == "definition":
        return f"what is meaning definition explanation {search_query}"
    return search_query


def normalize_topic_label(topic):
    return " ".join(str(topic or "").split()).strip()


def detect_product_category(topic):
    text = f" {str(topic or '').lower()} "
    if " loan" in text:
        return "loan"
    if " account" in text:
        return "account"
    if " card" in text:
        return "card"
    if " deposit" in text or " fd" in text or " fixed deposit" in text:
        return "deposit"
    if "netbanking" in text or "net banking" in text or "mobile banking" in text:
        return "digital_banking"
    return ""


def parse_employment_type(message):
    text = str(message or "").lower()
    if any(word in text for word in ["salary", "salaried", "employee", "job", "working"]):
        return "salaried"
    if any(word in text for word in ["self", "business", "owner", "shop", "company", "freelance", "professional"]):
        return "self-employed"
    return ""


def build_suggested_questions(topic="", intent="general"):
    topic = normalize_topic_label(topic)
    if topic:
        category = detect_product_category(topic)
        if category == "loan":
            return [
                f"What documents are required for {topic}?",
                f"Who is eligible for {topic}?",
                f"What are the charges for {topic}?",
            ]
        if category == "account":
            return [
                f"How do I open {topic}?",
                f"What documents are required for {topic}?",
                f"What is the minimum balance for {topic}?",
            ]
        if category == "card":
            return [
                f"How do I apply for {topic}?",
                f"What documents are required for {topic}?",
                f"What are the charges for {topic}?",
            ]
        return [
            f"What documents are required for {topic}?",
            f"What are the charges for {topic}?",
            f"Who is eligible for {topic}?",
        ]
    return ["What is a savings account?", "What is FD?", "How to activate net banking?"]


def build_contextual_suggested_questions(source=None, topic="", intent="general"):
    if source:
        current_question = normalize_match_text(source.get("question", ""))
        source_file = source.get("source_file", "")
        section = source.get("section", "")
        candidates = []
        for item in load_official_kb_documents():
            question = item.get("question", "")
            if not question or normalize_match_text(question) == current_question:
                continue
            score = 0
            if source_file and item.get("source_file") == source_file:
                score += 8
            if section and item.get("section") == section:
                score += 5
            score += score_intent_match(intent, f"{item.get('question', '')} {item.get('answer', '')} {item.get('section', '')}")
            score += score_query_specific_match(
                f"{source.get('question', '')} {source.get('answer', '')}",
                item.get("question", "").lower(),
                f"{item.get('question', '')} {item.get('answer', '')} {item.get('section', '')}".lower(),
            )
            if score > 0:
                candidates.append((score, question))
        candidates.sort(key=lambda item: item[0], reverse=True)
        suggestions = []
        for _, question in candidates:
            if question not in suggestions:
                suggestions.append(question)
            if len(suggestions) == 3:
                logger.info(
                    "suggestions.contextual source_question=%s suggestions=%s",
                    log_text(source.get("question", "")),
                    suggestions,
                )
                return suggestions
    suggestions = build_suggested_questions(topic, intent)
    logger.info("suggestions.fallback topic=%s intent=%s suggestions=%s", topic or "", intent, suggestions)
    return suggestions


def build_form_assistant_answer(topic, intent="general", user_profile=""):
    topic = normalize_topic_label(topic) or "this banking service"
    category = detect_product_category(topic)
    profile = parse_employment_type(user_profile) or user_profile

    if category == "loan":
        documents = [
            "Aadhaar or another identity proof",
            "PAN card",
            "Address proof",
            "Recent bank statements",
            "Loan application form",
            "Property or collateral documents, if the bank asks for them",
        ]
        if profile == "salaried":
            documents.insert(3, "Latest salary slips or Form 16")
            lead = f"For a salaried applicant, these are usually needed to apply for {topic}:"
        elif profile == "self-employed":
            documents.insert(3, "ITR, business proof, and income statements")
            lead = f"For a self-employed applicant, these are usually needed to apply for {topic}:"
        else:
            documents.insert(3, "Income proof such as salary slips, ITR, or business income documents")
            lead = f"To apply for {topic}, banks usually ask for:"
        steps = ["Check eligibility", "Fill the application form", "Submit KYC and income documents", "Wait for verification and approval"]
    elif category == "account":
        lead = f"To open {topic}, banks usually ask for:"
        documents = ["Aadhaar or another identity proof", "PAN card or Form 60", "Address proof", "Passport-size photo", "Completed KYC/account opening form", "Initial deposit, if required"]
        steps = ["Choose the account type", "Complete KYC", "Submit documents online or at a branch", "Activate debit card, mobile banking, and net banking if needed"]
    elif category == "card":
        lead = f"To apply for {topic}, banks usually ask for:"
        documents = ["Identity proof", "Address proof", "PAN card", "Income proof", "Recent bank statements", "Completed card application form"]
        steps = ["Check eligibility", "Fill the application", "Submit KYC and income documents", "Wait for approval and card dispatch"]
    elif category == "deposit":
        lead = f"To start {topic}, banks usually ask for:"
        documents = ["Active savings/current account or customer ID", "PAN card", "KYC documents if not already updated", "Deposit amount and tenure choice", "Nominee details"]
        steps = ["Choose amount and tenure", "Select payout or reinvestment option", "Confirm nominee details", "Submit the request online or at a branch"]
    else:
        lead = f"For {topic}, banks usually ask for:"
        documents = ["Identity proof", "Address proof", "PAN card if applicable", "Completed application/request form", "Any product-specific documents requested by the bank"]
        steps = ["Check eligibility", "Fill the form", "Submit documents", "Wait for bank verification"]

    return (
        f"{lead}\n\nRequired documents:\n"
        + "\n".join(f"- {item}" for item in documents)
        + "\n\nBasic steps:\n"
        + "\n".join(f"{index}. {step}" for index, step in enumerate(steps, 1))
    )


def handle_pending_flow(message, session):
    flow = session.get("pending_flow") or {}
    if not flow:
        return ""
    if flow.get("field") == "employment_type":
        employment_type = parse_employment_type(message)
        if employment_type:
            session.pop("pending_flow", None)
            return build_form_assistant_answer(flow.get("topic", ""), flow.get("intent", "general"), employment_type)
        return "Please tell me whether you are salaried or self-employed, so I can list the right documents."
    session.pop("pending_flow", None)
    return ""


def build_clarifying_question(message, topic, intent, session):
    category = detect_product_category(topic)
    if category != "loan" or session.get("pending_flow"):
        return ""
    if parse_employment_type(message):
        return ""
    if intent not in {"general", "opening", "documents", "process"}:
        return ""
    session["pending_flow"] = {"topic": normalize_topic_label(topic), "intent": intent, "field": "employment_type"}
    topic_label = normalize_topic_label(topic) or "this loan"
    return f"Before I list the exact steps for {topic_label}, are you salaried or self-employed?"


def should_use_form_assistant(question, topic, intent):
    category = detect_product_category(topic)
    if not category:
        logger.info("route.form_assistant skip reason=no_category topic=%s intent=%s question=%s", topic, intent, log_text(question))
        return False
    q = str(question or "").lower()
    security_words = [
        "block", "blocked", "lost", "stolen", "fraud", "fraudulent", "unauthorized",
        "unauthorised", "dispute", "chargeback", "phishing", "vishing", "skimming",
        "suspicious", "security", "replace card", "replacement card", "complaint",
        "complain", "ombudsman", "escalate", "escalation", "report",
    ]
    if any(word in q for word in security_words):
        logger.info(
            "route.form_assistant skip reason=security_or_dispute topic=%s intent=%s question=%s",
            topic,
            intent,
            log_text(question),
        )
        return False
    action_words = ["apply", "open", "make", "create", "start", "documents", "required", "needed", "how do", "how can", "steps", "process"]
    decision = intent in {"opening", "documents", "process"} or any(word in q for word in action_words)
    logger.info(
        "route.form_assistant decision=%s category=%s topic=%s intent=%s question=%s",
        decision,
        category,
        topic,
        intent,
        log_text(question),
    )
    return decision


def is_credential_help_question(question):
    q = str(question or "").lower()
    credential_terms = ["password", "passcode", "pin", "mpin", "login id", "user id", "username"]
    help_terms = [
        "reset", "change", "update", "recover", "forgot", "forget", "blocked",
        "unblock", "protect", "secure", "safe", "not working", "locked",
    ]
    return any(term in q for term in credential_terms) and any(term in q for term in help_terms)


def build_credential_help_answer(question, language="English"):
    q = str(question or "").lower()
    credential = "PIN" if "pin" in q or "mpin" in q else "banking password"
    if language == "Hindi":
        return (
            f"Apna {credential} reset ya change karne ke liye:\n"
            "1. Bank ki official app ya website kholiye.\n"
            "2. Login page par Forgot Password / Reset Password option select kijiye.\n"
            "3. Apna customer ID, registered mobile number, ya username enter kijiye.\n"
            "4. OTP sirf official bank screen par enter kijiye. Kisi ko OTP, PIN, ya password mat batayein.\n"
            "5. Naya strong password set kijiye aur confirmation message check kijiye.\n\n"
            "Agar account locked hai ya OTP nahi aa raha, bank customer care ya branch se contact kijiye."
        )
    return (
        f"To reset or change your {credential}:\n"
        "1. Open your bank's official app or website.\n"
        "2. Choose Forgot Password / Reset Password on the login page.\n"
        "3. Enter your customer ID, registered mobile number, or username.\n"
        "4. Verify using OTP only on the official bank screen. Do not share your OTP, PIN, or password with anyone.\n"
        "5. Set a new strong password and confirm the update.\n\n"
        "If your account is locked or OTP is not arriving, contact the bank's customer care or visit a branch."
    )


def build_credential_help_suggestions(language="English"):
    return translate_suggested_questions([
        "How do I protect my bank account?",
        "How do I report a lost card?",
        "How do I unblock net banking?",
    ], language)


def extract_topic_from_question(user_question):
    text = re.sub(r"[^a-z0-9\s]", " ", user_question.lower())
    text = " ".join(text.split())
    for alias in sorted(BANKING_TOPIC_ALIASES, key=len, reverse=True):
        if re.search(rf"\b{re.escape(alias)}\b", text):
            return BANKING_TOPIC_ALIASES[alias]

    remove_phrases = [
        "what is", "what are", "how can i", "how do i", "how to", "help me",
        "tell me about", "explain", "define", "meaning of", "documents required",
        "documents", "required", "requirement", "requirements", "need to", "needed",
        "open", "opening", "make", "create", "start", "set up", "setup", "apply",
        "register", "get", "can i", "who can", "eligible", "eligibility", "fees",
        "fee", "charges", "charge", "cost", "steps", "process", "procedure",
    ]
    for phrase in remove_phrases:
        text = re.sub(rf"\b{re.escape(phrase)}\b", " ", text)
    words = [word for word in text.split() if word not in {
        "a", "an", "the", "for", "of", "to", "in", "on", "my", "me", "i", "we",
        "you", "your", "it", "this", "that", "please", "with", "and", "or",
    }]
    if not words:
        return ""

    domain_nouns = {
        "account", "card", "loan", "deposit", "facility", "service", "cheque",
        "payment", "banking", "insurance", "investment", "fund", "statement",
        "password", "pin", "branch", "netbanking", "transaction", "mortgage",
    }
    candidates = []
    for index, word in enumerate(words):
        if word in domain_nouns:
            start = max(0, index - 3)
            candidates.append(" ".join(words[start:index + 1]))
    return max(candidates, key=len) if candidates else ""


def normalize_banking_spelling(question):
    tokens = re.findall(r"[a-z0-9']+|[^a-z0-9']+", question.lower())
    corrected = []
    vocabulary = BANKING_VOCABULARY | QUERY_WORDS
    for token in tokens:
        if token in COMMON_QUERY_CORRECTIONS:
            corrected.append(COMMON_QUERY_CORRECTIONS[token])
            continue
        if not token.isalpha() or len(token) < 4 or token in vocabulary:
            corrected.append(token)
            continue
        match = get_close_matches(token, vocabulary, n=1, cutoff=0.78)
        corrected.append(match[0] if match else token)
    return "".join(corrected).strip()


def question_clarity_score(question, session):
    words = re.findall(r"[a-z0-9]+", question.lower())
    if not words:
        return 0.0
    recognized = sum(word in BANKING_VOCABULARY or word in QUERY_WORDS for word in words)
    score = recognized / len(words)
    if extract_topic_from_question(question):
        score += 0.45
    if session.get("current_topic") and is_follow_up_question(question):
        score += 0.35
    if detect_question_intent(question) != "general":
        score += 0.2
    return score


def question_needs_llm_rewrite(question, session):
    words = re.findall(r"[a-z0-9]+", question.lower())
    if len(words) <= 1 and not extract_topic_from_question(question):
        return True
    return question_clarity_score(question, session) < 0.42


def is_banking_related_question(question, session):
    text = question.lower().strip()
    if extract_topic_from_question(text):
        return True
    words = set(re.findall(r"[a-z0-9]+", text))
    if words & BANKING_VOCABULARY:
        return True
    banking_phrases = {
        "credit score", "bank account", "send money", "receive money",
        "save money", "borrow money", "monthly instalment", "monthly installment",
        "account number", "routing number", "ifsc code",
    }
    if any(phrase in text for phrase in banking_phrases):
        return True
    if session.get("last_comparison") and answer_comparison_followup(text, session):
        return True
    return bool(
        is_follow_up_question(text)
        and (session.get("current_topic") or session.get("last_kb_source") or session.get("kb_source_history"))
    )


def is_general_factual_question(message):
    text = message.lower().strip()
    return bool(re.match(
        r"^(what|who|where|when|why|which|how many|how much|name|tell me (what|who|where|when))\b",
        text,
    )) or text.endswith("?")


def retrieve_general_knowledge_context(question):
    params = urlencode({
        "action": "query",
        "generator": "search",
        "gsrsearch": question,
        "gsrlimit": 3,
        "prop": "extracts",
        "exintro": 1,
        "explaintext": 1,
        "exchars": 800,
        "redirects": 1,
        "format": "json",
    })
    request = Request(
        f"https://en.wikipedia.org/w/api.php?{params}",
        headers={"User-Agent": "BankingChatbotAnalytics/1.0 (educational project)"},
    )
    try:
        with urlopen(request, timeout=4) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception:
        return ""
    pages = list(payload.get("query", {}).get("pages", {}).values())
    pages.sort(key=lambda page: page.get("index", 999))
    context_parts = []
    for page in pages:
        title = str(page.get("title", "")).strip()
        extract = " ".join(str(page.get("extract", "")).split())
        if title and extract:
            context_parts.append(f"{title}: {extract}")
    return "\n".join(context_parts)[:2400]


def extract_direct_general_fact(question, context):
    capital_match = re.search(r"\bcapital of ([a-z][a-z .'-]+?)(?:\?|$)", question.lower().strip())
    if not capital_match:
        return ""
    place = " ".join(capital_match.group(1).split())
    escaped_place = re.escape(place)
    place_pattern = rf"(?i:{escaped_place})"
    patterns = [
        rf"\b([A-Z][a-zA-Z.'-]+(?:\s+[A-Z][a-zA-Z.'-]+){{0,3}}), the capital of {place_pattern}\b",
        rf"\b([A-Z][a-zA-Z.'-]+(?:\s+[A-Z][a-zA-Z.'-]+){{0,3}}) is the capital of {place_pattern}\b",
        rf"\bthe capital of {place_pattern} is ([A-Z][a-zA-Z.'-]+(?:\s+[A-Z][a-zA-Z.'-]+){{0,3}})\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, context)
        if match:
            capital = " ".join(word.capitalize() for word in match.group(1).split())
            return f"The capital of {place.title()} is {capital}."
    return ""


def select_relevant_reference_answer(message, context):
    query_terms = search_tokens(message)
    text = " ".join(line.split(": ", 1)[-1] for line in context.splitlines())
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", text)
        if sentence.strip()
    ]
    intent_cues = []
    lowered = message.lower()
    if re.search(r"\b(who|invented|inventor|created|discovered|founded)\b", lowered):
        intent_cues = ["invent", "patent", "created", "discovered", "founded", "credited", "first"]
    elif re.search(r"\b(when|year|date)\b", lowered):
        intent_cues = ["year", "date", "century", "founded", "invented", "born"]
    elif re.search(r"\b(where|location|located)\b", lowered):
        intent_cues = ["located", "country", "city", "region", "capital"]

    ranked = []
    for index, sentence in enumerate(sentences):
        sentence_lower = sentence.lower()
        overlap = len(query_terms & search_tokens(sentence_lower))
        cue_score = sum(2 for cue in intent_cues if cue in sentence_lower)
        ranked.append((overlap * 3 + cue_score, -index, sentence))
    ranked.sort(reverse=True)
    selected = [item[2] for item in ranked[:2] if item[0] > 0]
    return " ".join(selected).strip()


def build_general_reference_answer(message, language):
    text = message.lower().strip()
    if re.search(r"\b(hello|hi|hey|good morning|good afternoon|good evening)\b", text):
        answer = "Hello! How can I help you today?"
        return translate_answer(answer, language), ["conversation"]
    if re.search(r"\b(how are you|how is it going)\b", text):
        answer = "I'm doing well, thank you. What would you like to know?"
        return translate_answer(answer, language), ["conversation"]
    if re.search(r"\b(thank you|thanks)\b", text):
        answer = "You're welcome!"
        return translate_answer(answer, language), ["conversation"]

    context = retrieve_general_knowledge_context(message)
    if not context:
        answer = build_offline_general_answer(message) or "I couldn't verify that answer right now. Please try rephrasing the question."
        return translate_answer(answer, language), ["general_reference"]

    direct_fact = extract_direct_general_fact(message, context)
    if direct_fact:
        return translate_answer(direct_fact, language), ["wikipedia", "fact_extraction"]

    answer = select_relevant_reference_answer(message, context)
    if not answer:
        answer = "I couldn't verify that answer right now. Please try rephrasing the question."
        return translate_answer(answer, language), ["general_reference"]
    return translate_answer(answer, language), ["wikipedia"]


def build_offline_general_answer(message):
    text = message.lower().strip()
    capital_match = re.search(r"\bcapital of india\b", text)
    if capital_match:
        return "The capital of India is New Delhi."
    if "invented the telephone" in text or "inventor of the telephone" in text:
        return "Alexander Graham Bell is widely credited with inventing the first practical telephone."
    return ""


def generate_general_llm_response(message, session, tokenizer, llm_model, language):
    language_instruction = "Respond in simple Hindi." if language == "Hindi" else "Respond in natural English."
    is_greeting = bool(re.search(r"\b(hello|hi|hey|good morning|good afternoon|good evening)\b", message.lower()))
    factual_context = ""
    if is_greeting:
        prompt = f"""
You are a friendly conversational assistant. {language_instruction}
Reply to the greeting warmly in one short sentence.
User: {message}
Assistant:
"""
    else:
        factual_question = is_general_factual_question(message)
        factual_context = retrieve_general_knowledge_context(message) if factual_question else ""
        direct_fact = extract_direct_general_fact(message, factual_context)
        if direct_fact:
            return direct_fact, ["wikipedia", "fact_extraction"]
        if factual_context:
            prompt = f"""
You are a precise general-knowledge question answering assistant.
{language_instruction}
Question: {message}

Give the standard accepted factual answer in one or two short sentences.
Read every word carefully and answer only from the reference context below.
For geography, distinguish an official capital from a largest or most famous city.
If the reference does not contain the answer, say that you could not verify it instead of guessing.
Do not force the answer to be about banking.

Reference context:
{factual_context}
Answer:
"""
        else:
            prompt = f"""
Answer this general-knowledge question accurately and briefly.
{language_instruction}
Give the standard widely accepted answer in one or two sentences.
Do not change the topic to banking. Do not repeat the question.

Question: {message}
Answer:
"""
    inputs = tokenizer(prompt, return_tensors="pt", max_length=768, truncation=True)
    outputs = llm_model.generate(**inputs, max_new_tokens=96, num_beams=3, do_sample=False)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    if not answer or answer.lower() == message.lower() or "clarify" in answer.lower():
        if is_greeting:
            return "Hello! How can I help you today?", ["llm"]
        return "I could not answer that question clearly. Please try rephrasing it.", ["llm"]
    methods = ["wikipedia", "llm"] if factual_context else ["llm"]
    return answer, methods


def rewrite_unclear_question(question, session, tokenizer, llm_model):
    current_topic = session.get("current_topic", "") or "none"
    history = build_recent_chat_history(session)
    prompt = f"""
You clean up banking questions.
Rewrite the user's message as one clear English banking question.
Use the current topic only when the message is a follow-up.
Do not answer the question.
If the message cannot reasonably be understood as a banking question, output exactly: CLARIFY

Current topic: {current_topic}
Recent conversation:
{history}
User message: {question}

Rewritten question:
"""
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    outputs = llm_model.generate(**inputs, max_new_tokens=48, num_beams=3, do_sample=False)
    rewritten = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    rewritten = re.sub(r"^(rewritten question|question)\s*:\s*", "", rewritten, flags=re.IGNORECASE).strip()
    if not rewritten or "clarify" in rewritten.lower():
        return ""
    if len(re.findall(r"[a-z0-9]+", rewritten.lower())) < 3:
        return ""
    original_terms = search_tokens(question)
    rewritten_terms = search_tokens(rewritten)
    meaningful_overlap = original_terms & rewritten_terms
    if not session.get("current_topic") and not meaningful_overlap and question_clarity_score(question, session) < 0.15:
        return ""
    return rewritten


def build_clarification_reply(session):
    topic = session.get("current_topic", "")
    if topic:
        return f"I could not clearly understand that. Are you asking about the documents, eligibility, fees, or application steps for {topic}?"
    return "I could not clearly understand that banking question. Please mention the product or service, for example savings account, home loan, FD, card, UPI, or net banking."


def search_tokens(text):
    stopwords = {
        "a", "an", "the", "is", "are", "was", "were", "be", "to", "of", "for",
        "and", "or", "in", "on", "at", "with", "my", "me", "i", "you", "your",
        "please", "tell", "about", "can", "could", "would", "do", "does",
        "how", "what", "which", "who", "when", "where", "why", "if", "it",
        "this", "that", "these", "those", "will", "should", "shall", "must",
    }
    return {
        normalize_search_token(token)
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if len(token) > 1 and token not in stopwords
    }


def normalized_text_for_similarity(text):
    tokens = re.findall(r"[a-z0-9]+", str(text or "").lower())
    return " ".join(normalize_search_token(token) for token in tokens if len(token) > 1)


def char_ngrams(text, size=3):
    compact = re.sub(r"[^a-z0-9]+", " ", normalized_text_for_similarity(text)).strip()
    if not compact:
        return set()
    padded = f" {compact} "
    if len(padded) <= size:
        return {padded}
    return {padded[index:index + size] for index in range(len(padded) - size + 1)}


def set_overlap_score(left, right):
    if not left or not right:
        return 0.0
    intersection = len(left & right)
    return intersection / max(len(left), len(right), 1)


def semantic_similarity(query, candidate):
    query_tokens = expand_query_terms(query)
    candidate_tokens = search_tokens(candidate)
    token_score = set_overlap_score(query_tokens, candidate_tokens)
    char_score = set_overlap_score(char_ngrams(query), char_ngrams(candidate))
    sequence_score = SequenceMatcher(
        None,
        normalized_text_for_similarity(query),
        normalized_text_for_similarity(candidate),
    ).ratio()
    return token_score * 0.45 + char_score * 0.35 + sequence_score * 0.20


def expand_query_terms(text):
    terms = set(search_tokens(text))
    q = str(text or "").lower()
    if ("card" in terms and (terms & {"lost", "stolen", "missing"})) or any(phrase in q for phrase in ["card is missing", "missing card"]):
        terms |= {"lost", "stolen", "missing", "block", "blocked", "blocking", "replacement", "replace", "report", "immediate", "steps"}
    if any(phrase in q for phrase in ["how long", "how many days", "when", "timeline", "take", "delay", "late", "wait"]):
        terms |= {"time", "timeline", "day", "working", "process", "processed", "processing", "review", "reviewed", "delay", "delayed"}
    if terms & {"cost", "costs", "price", "yearly", "charge", "charges"} or "every year" in q:
        terms |= {"fee", "annual", "joining", "charge", "charges", "cost", "price", "yearly", "waiver", "waived"}
    if terms & {"waiver", "waived", "waive"}:
        terms |= {"fee", "annual", "joining", "waiver", "waived", "milestone", "spend"}
    if any(phrase in q for phrase in ["what happens", "what if", "only pay", "after", "miss", "pay late", "late payment"]):
        terms |= {"happen", "result", "consequence", "impact", "charge", "interest", "fee", "forfeit", "forfeited", "redeem", "closure", "closed"}
    if terms & {"cashback", "reward", "rewards", "points"}:
        terms |= {"cashback", "reward", "rewards", "point", "points", "edge", "benefit", "benefits", "earn", "earned", "credited"}
    if terms & {"mile", "miles"}:
        terms |= {"mile", "miles", "edge", "travel", "flight", "hotel", "redeem", "redeemable", "reward", "currency"}
    if any(phrase in q for phrase in ["best for", "who should", "who can use", "should use", "right for", "meant for", "good for"]):
        terms |= {"best", "for", "suitable", "recommended", "goal", "use", "traveller", "travel", "spender", "shopper", "shopping", "food", "delivery"}
    if any(phrase in q for phrase in ["report phishing", "phishing", "vishing", "skimming", "what should i do next"]):
        terms |= {"report", "fraud", "phishing", "security", "dispute", "customer", "care", "block", "email", "call"}
    if terms & {"unauthorized", "permission", "strange", "suspicious", "fraudulent"} or any(phrase in q for phrase in ["without permission", "not done by me", "did not do", "not mine", "not me", "someone used", "someone made", "unknown transaction"]):
        terms |= {"unauthorized", "fraud", "fraudulent", "suspicious", "strange", "dispute", "transaction", "chargeback", "block", "report", "liability", "immediate", "steps", "minutes", "hours"}
    if terms & {"proof", "document", "documents", "evidence", "keep"}:
        terms |= {"proof", "document", "documents", "evidence", "record", "records", "communication", "reference", "date", "time", "merchant", "amount", "transaction"}
    if terms & {"paper", "papers"}:
        terms |= {"document", "documents", "proof", "kyc", "identity", "address", "income"}
    if terms & {"reject", "rejected", "deny", "denied"}:
        terms |= {"reject", "rejected", "chargeback", "temporary", "credit", "reversed", "debit"}
    if terms & {"complaint", "complain", "resolve", "unresolved"}:
        terms |= {"complaint", "complain", "resolve", "unresolved", "grievance", "ombudsman", "escalation", "escalate", "satisfactory", "responded"}
    if "rbi" in terms and (terms & {"day", "working", "time", "wait", "delayed", "timeline"}):
        terms |= {"complaint", "grievance", "ombudsman", "escalation", "resolve", "responded", "satisfactory", "unresolved"}
    if terms & {"approval", "approved"}:
        terms |= {"approval", "approved", "application", "processing", "review", "reviewed", "working", "day", "time", "process"}
    if terms & {"investigate", "investigation"}:
        terms |= {"investigate", "investigation", "chargeback", "dispute", "merchant", "response", "working", "day", "time"}
    if terms & {"merchant", "valid"}:
        terms |= {"merchant", "valid", "prove", "proof", "chargeback", "temporary", "credit", "reversed", "legitimate"}
    if terms & {"qualify", "qualified"} or "can i get" in q:
        terms |= {"eligible", "eligibility", "apply", "application", "resident", "age", "income", "employment"}
    return terms


def normalize_search_token(token):
    token = token.lower()
    synonyms = {
        "solve": "resolve",
        "solved": "resolve",
        "resolves": "resolve",
        "resolved": "resolve",
        "delay": "delayed",
        "late": "delayed",
        "cheaper": "lower",
        "lowest": "lower",
        "better": "best",
        "cost": "fee",
        "costs": "fee",
        "costly": "fee",
        "price": "fee",
        "yearly": "annual",
        "meant": "best",
        "missing": "lost",
        "papers": "document",
        "paper": "document",
        "travelling": "travel",
        "traveling": "travel",
        "traveller": "travel",
        "traveler": "travel",
        "investigate": "investigation",
        "investigated": "investigation",
        "proves": "prove",
        "proved": "prove",
        "waive": "waiver",
        "waived": "waiver",
        "charges": "charge",
        "unauthorised": "unauthorized",
        "suspicious": "strange",
        "fraudulent": "fraud",
    }
    if token in synonyms:
        return synonyms[token]
    if len(token) > 5 and token.endswith("ies"):
        return f"{token[:-3]}y"
    if len(token) > 5 and token.endswith("ing"):
        return token[:-3]
    if len(token) > 4 and token.endswith("ed"):
        return token[:-2]
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def split_answer_units(answer):
    units = []
    heading = ""
    paragraph = []
    table = []

    def flush_paragraph():
        nonlocal paragraph
        if paragraph:
            units.append({"heading": heading, "text": " ".join(paragraph).strip()})
            paragraph = []

    def flush_table():
        nonlocal table
        if table:
            units.append({"heading": heading, "text": "\n".join(table).strip()})
            table = []

    for raw_line in str(answer or "").splitlines():
        line = raw_line.strip()
        if not line:
            flush_paragraph()
            flush_table()
            continue
        if line.startswith("|") and line.endswith("|"):
            flush_paragraph()
            table.append(line)
            continue
        flush_table()
        if re.match(r"^(#{1,6}\s+|\*\*[^*]+:\*\*\s*$)", line):
            flush_paragraph()
            heading = line
            continue
        if re.match(r"^([-*]\s+|\d+\.\s+)", line):
            flush_paragraph()
            units.append({"heading": heading, "text": line})
            continue
        paragraph.append(line)

    flush_paragraph()
    flush_table()
    return [unit for unit in units if unit["text"]]


def looks_like_intro_unit(unit):
    text = str(unit.get("text", "")).strip()
    if not text:
        return False
    if text.startswith("|") or re.match(r"^([-*]\s+|\d+\.\s+)", text):
        return False
    words = re.findall(r"[a-z0-9]+", text.lower())
    if len(words) > 28:
        return False
    return bool(re.search(r"(:$|\b(include|includes|offer|offers|provide|provides|following|below|steps|methods|options|ways)\b)", text, re.IGNORECASE))


def expand_selected_answer_unit_indexes(units, ranked_units, selected_indexes):
    expanded = set(selected_indexes)
    if not units:
        return []

    for _, index, unit in ranked_units[:3]:
        if index not in expanded:
            continue
        should_include_following = looks_like_intro_unit(unit) or len(search_tokens(unit.get("text", ""))) <= 5
        if not should_include_following:
            continue
        for next_index in range(index + 1, min(len(units), index + 5)):
            next_text = str(units[next_index].get("text", "")).strip()
            if not next_text:
                continue
            expanded.add(next_index)
            if len(expanded) >= 6:
                break

    if len(expanded) <= 2:
        snippet_words = sum(len(re.findall(r"[a-z0-9]+", units[index].get("text", ""))) for index in expanded)
        if snippet_words < 22:
            first = min(expanded) if expanded else ranked_units[0][1]
            for next_index in range(first + 1, min(len(units), first + 5)):
                expanded.add(next_index)
                if len(expanded) >= 5:
                    break

    return sorted(expanded)


def build_previous_answer_candidate(current_query, previous_source):
    answer = str(previous_source.get("answer", ""))
    query_terms = expand_query_terms(current_query)
    if not answer or not query_terms:
        return None
    previous_terms = search_tokens(f"{previous_source.get('question', '')} {answer}")
    if query_terms & {"investigation", "chargeback"} and not (previous_terms & {"investigation", "chargeback", "provisional", "temporary"}):
        return None
    if query_terms & {"merchant", "valid", "prove"} and not (previous_terms & {"chargeback", "legitimate", "reversed", "temporary", "provisional"}):
        return None

    ranked_units = []
    temporal_query = bool(re.search(r"\b(how long|how many days|when|timeline|take|delay|late)\b", str(current_query or "").lower()))
    temporal_terms = {"time", "timeline", "day", "working", "process", "processed", "processing", "review", "reviewed", "delay", "delayed", "approval", "approved"}
    required_temporal_terms = expand_query_terms(current_query) & {"investigation", "approval", "closure", "complaint", "dispute", "chargeback"}
    if expand_query_terms(current_query) & {"investigation", "chargeback"}:
        required_temporal_terms = {"investigation", "chargeback", "merchant", "provisional"}
    action_followup = bool(re.search(r"\b(next|what should i do|what do i do)\b", str(current_query or "").lower()))
    action_terms = {"report", "call", "email", "visit", "file", "block", "contact", "request", "submit", "redeem"}
    units = split_answer_units(answer)
    for index, unit in enumerate(units):
        unit_text = unit["text"]
        if "see chunk" in unit_text.lower():
            continue
        unit_terms = search_tokens(f"{unit.get('heading', '')} {unit_text}")
        unit_text_terms = search_tokens(unit_text)
        if not unit_terms:
            continue
        hits = query_terms & unit_terms
        if temporal_query and not (unit_text_terms & temporal_terms):
            continue
        if temporal_query and required_temporal_terms and not (unit_text_terms & required_temporal_terms):
            continue
        if action_followup and not (unit_text_terms & action_terms):
            continue
        if not hits:
            if action_followup and (unit_text_terms & action_terms):
                hits = unit_text_terms & action_terms
            else:
                continue
        coverage = len(hits) / len(query_terms)
        density = len(hits) / max(len(unit_terms), 1)
        order_bonus = max(0.0, 1.0 - index * 0.03)
        similarity = semantic_similarity(current_query, f"{unit.get('heading', '')} {unit_text}")
        score = coverage * 24.0 + density * 10.0 + len(hits) * 2.0 + similarity * 35.0 + order_bonus
        ranked_units.append((score, index, unit))

    if not ranked_units:
        return None
    ranked_units.sort(key=lambda item: item[0], reverse=True)
    best_score = ranked_units[0][0]
    min_score = 4.0 if temporal_query or is_product_attribute_followup(current_query) else 9.0
    if best_score < min_score:
        return None

    selected_indexes = sorted(index for _, index, _ in ranked_units[:4])
    selected_indexes = expand_selected_answer_unit_indexes(units, ranked_units, selected_indexes)
    selected = [units[index] for index in selected_indexes]
    lines = []
    last_heading = None
    for unit in selected:
        heading = unit.get("heading", "")
        if heading and heading != last_heading:
            lines.append(heading)
            last_heading = heading
        lines.append(unit["text"])
    snippet = "\n".join(lines).strip()
    if not snippet:
        return None

    context_bonus = 60.0 if temporal_query else 20.0 if len(query_terms) == 1 else 45.0
    if query_terms & {"record", "records", "evidence", "proof", "keep"}:
        context_bonus += 35.0
    return {
        "score": round(best_score + context_bonus, 4),
        "metadata": {
            "section": previous_source.get("section", ""),
            "question": previous_source.get("question", ""),
            "answer": snippet,
            "context_answer": answer,
            "source": previous_source.get("source", "Official knowledge base"),
            "source_file": previous_source.get("source_file", ""),
            "dataset": "official_kb",
            "search_methods": ["official_kb", "previous_answer_context"],
            "hybrid_score": round(best_score + context_bonus, 4),
        },
    }


def is_pronoun_reference(question):
    return bool(re.search(r"\b(it|that|this|they|them|those|these)\b", str(question or "").lower()))


def candidate_key(metadata):
    return f"{metadata.get('question', '').strip().lower()}|{metadata.get('answer', '').strip().lower()}"


def normalize_match_text(text):
    return " ".join(re.findall(r"[a-z0-9]+", str(text or "").lower()))


def get_cached_banking_metadatas(banking_collection):
    count = banking_collection.count()
    if BANKING_METADATA_CACHE["count"] != count:
        results = banking_collection.get(include=["metadatas"])
        BANKING_METADATA_CACHE["count"] = count
        BANKING_METADATA_CACHE["metadatas"] = [
            metadata for metadata in results.get("metadatas", []) if metadata
        ]
    return BANKING_METADATA_CACHE["metadatas"]


def collection_has_official_kb(banking_collection):
    return any(
        metadata.get("dataset") == "official_kb"
        for metadata in get_cached_banking_metadatas(banking_collection)
    )


def lexical_candidate_score(query, metadata, topic="", intent="general"):
    question = str(metadata.get("question", ""))
    answer = str(metadata.get("answer", ""))
    section = str(metadata.get("section", ""))
    candidate_text = f"{question} {answer} {section}".lower()
    query_text = query.lower().strip()
    query_terms = expand_query_terms(query_text)
    candidate_terms = search_tokens(candidate_text)
    question_terms = search_tokens(question.lower())
    if not query_terms:
        return 0.0
    overlap = len(query_terms & candidate_terms) / len(query_terms)
    question_hits = query_terms & question_terms
    question_overlap = len(question_hits) / len(query_terms)
    question_ratio = SequenceMatcher(None, query_text, question.lower()).ratio()
    phrase_bonus = 5.0 if query_text and query_text in candidate_text else 0.0
    exact_bonus = 100.0 if normalize_match_text(query_text) == normalize_match_text(question) else 0.0
    topic_bonus = score_topic_match(topic, candidate_text)
    intent_bonus = score_intent_match(intent, candidate_text)
    query_bonus = score_query_specific_match(query_text, question.lower(), candidate_text)
    semantic_bonus = semantic_similarity(query_text, f"{question} {section}") * 18.0
    body_semantic_bonus = semantic_similarity(query_text, candidate_text) * 8.0
    return exact_bonus + overlap * 5.0 + question_overlap * 20.0 + len(question_hits) * 6.0 + question_ratio * 3.0 + phrase_bonus + topic_bonus + intent_bonus + query_bonus + semantic_bonus + body_semantic_bonus


def score_query_specific_match(query_text, question_text, candidate_text):
    query_terms = search_tokens(query_text)
    question_terms = search_tokens(question_text)
    candidate_terms = search_tokens(candidate_text)
    if not query_terms:
        return 0.0
    title_hits = query_terms & question_terms
    body_hits = query_terms & candidate_terms
    title_coverage = len(title_hits) / len(query_terms)
    body_coverage = len(body_hits) / len(query_terms)
    score = title_coverage * 10.0 + body_coverage * 3.0
    if "mile" in question_terms and "mile" not in query_terms:
        score -= 45.0
    if query_terms & {"lost", "stolen", "missing", "block", "blocking"}:
        if question_terms & {"lost", "stolen", "block"} or candidate_terms & {"block", "blocked", "blocking", "replacement"}:
            score += 30.0
        if question_terms & {"miss", "late", "payment"} and not (query_terms & {"late", "delayed"}):
            score -= 26.0
    if query_terms & {"record", "records", "evidence", "proof", "keep"}:
        if candidate_terms & {"communication", "reference", "record", "records", "merchant", "amount", "date", "time", "transaction"}:
            score += 16.0
        if question_terms & {"secure", "security"} and not (query_terms & {"secure", "security", "password", "otp", "pin"}):
            score -= 24.0
    unauthorized_action_terms = {"unauthorized", "permission", "fraud", "suspicious", "strange", "block", "report", "immediate"}
    if query_terms & unauthorized_action_terms:
        if question_terms & {"immediate", "what", "should"} or candidate_terms & {"immediate", "minutes", "hours", "block", "report"}:
            score += 22.0
        if question_terms & {"type", "types"} and not (query_terms & {"type", "types"}):
            score -= 14.0
    complaint_terms = {"complaint", "complain", "resolve", "unresolved", "grievance", "ombudsman", "escalation"}
    if query_terms & complaint_terms:
        if question_terms & complaint_terms:
            score += 28.0
        elif not (candidate_terms & complaint_terms):
            score -= 10.0
    if "rbi" in query_terms and query_terms & {"day", "time", "wait", "delayed", "timeline"}:
        if question_terms & {"complaint", "resolve", "unresolved", "ombudsman", "grievance"}:
            score += 26.0
        if question_terms & {"interest"} and not (question_terms & {"complaint", "resolve", "unresolved", "ombudsman"}):
            score -= 18.0
    if query_terms & {"eligible", "eligibility", "qualify", "qualified"}:
        if question_terms & {"eligible", "eligibility", "apply"} or candidate_terms & {"eligible", "eligibility", "age", "income", "employment"}:
            score += 24.0
        if question_terms & {"waiver", "waived"}:
            score -= 28.0
    if query_terms & {"apply", "application"} and query_terms & {"website", "online"}:
        if question_terms & {"apply", "online"} or candidate_terms & {"website", "apply", "online", "application"}:
            score += 30.0
        if question_terms & {"document", "documents"}:
            score -= 16.0
    if query_terms & {"approval", "approved"} and query_terms & {"day", "time", "working"}:
        if question_terms & {"apply", "application", "approved", "approval"} or candidate_terms & {"processing", "reviewed", "approval", "application", "working"}:
            score += 34.0
        if question_terms & {"late", "payment"}:
            score -= 24.0
    if query_terms & {"investigation", "chargeback"} and query_terms & {"day", "time", "working"}:
        if question_terms & {"chargeback", "investigation"} or candidate_terms & {"chargeback", "investigation", "merchant", "provisional"}:
            score += 48.0
        if question_terms & {"fee", "late", "payment"}:
            score -= 26.0
    if query_terms & {"merchant", "valid", "prove"}:
        if question_terms & {"chargeback", "investigation"} or candidate_terms & {"prove", "legitimate", "rejected", "reversed", "response"}:
            score += 82.0
        if question_terms & {"fee", "late", "payment"}:
            score -= 24.0
        if question_terms & {"interest"} and not (question_terms & {"chargeback", "investigation"}):
            score -= 32.0
        if question_terms & {"immediate"} and not (question_terms & {"chargeback", "investigation"}):
            score -= 35.0
    return score


def retrieve_hybrid_banking_context(
    raw_query,
    expanded_query,
    topic,
    intent,
    embedding_model,
    banking_collection,
    top_k=40,
):
    fused = {}
    query_variants = []
    for query in [raw_query, expanded_query, f"{topic} {intent}".strip()]:
        query = " ".join(str(query or "").split())
        if query and query.lower() not in {item.lower() for item in query_variants}:
            query_variants.append(query)

    collection_count = banking_collection.count()
    vector_limit = min(20, collection_count) if collection_count else 0
    for variant_index, query in enumerate(query_variants):
        if not vector_limit:
            break
        query_embedding = embedding_model.encode(query).tolist()
        results = banking_collection.query(query_embeddings=[query_embedding], n_results=vector_limit)
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        for rank, metadata in enumerate(metadatas):
            key = candidate_key(metadata)
            item = fused.setdefault(key, {"metadata": metadata, "score": 0.0, "distance": None, "methods": set()})
            item["score"] += (4.0 - min(variant_index, 2) * 0.6) / (rank + 1)
            if rank < len(distances):
                distance = float(distances[rank])
                item["distance"] = distance if item["distance"] is None else min(item["distance"], distance)
                item["score"] += max(0.0, 1.5 - distance)
            item["methods"].add("vector")

    lexical_ranked = []
    for metadata in get_cached_banking_metadatas(banking_collection):
        score = max(
            lexical_candidate_score(query, metadata, topic, intent)
            for query in query_variants
        )
        if score > 0:
            lexical_ranked.append((score, metadata))
    lexical_ranked.sort(key=lambda item: item[0], reverse=True)
    for rank, (score, metadata) in enumerate(lexical_ranked[:30]):
        key = candidate_key(metadata)
        item = fused.setdefault(key, {"metadata": metadata, "score": 0.0, "distance": None, "methods": set()})
        item["score"] += score + 3.0 / (rank + 1)
        item["methods"].add("lexical")
        question = str(metadata.get("question", "")).lower()
        if any(normalize_match_text(query) == normalize_match_text(question) for query in query_variants):
            item["methods"].add("exact")
            item["score"] += 50.0
        elif any(query.lower() in question for query in query_variants if len(query) > 4):
            item["methods"].add("phrase")
        elif any(SequenceMatcher(None, query.lower(), question).ratio() >= 0.72 for query in query_variants):
            item["methods"].add("fuzzy")

    structured = build_structured_product_candidate(topic, intent)
    if structured and not collection_has_official_kb(banking_collection):
        key = candidate_key(structured)
        fused[key] = {
            "metadata": structured,
            "score": 100.0,
            "distance": 0.0,
            "methods": {"structured"},
        }

    ranked = sorted(fused.values(), key=lambda item: item["score"], reverse=True)[:top_k]
    context_parts = []
    sources = []
    scores = []
    for index, item in enumerate(ranked, 1):
        metadata = item["metadata"]
        source = {**metadata, "search_methods": sorted(item["methods"]), "hybrid_score": round(item["score"], 4)}
        sources.append(source)
        scores.append(item["score"])
        context_parts.append(
            f"Result {index}:\n"
            f"Section: {metadata.get('section', '')}\n"
            f"Question: {metadata.get('question', '')}\n"
            f"Answer: {metadata.get('answer', '')}\n"
            f"Search methods: {', '.join(sorted(item['methods']))}\n"
            f"Hybrid score: {item['score']:.4f}\n"
        )
    return "\n".join(context_parts), sources, scores


def retrieve_lightweight_banking_answer(query, topic, intent, banking_collection):
    structured = build_structured_product_candidate(topic, intent)
    if structured and not collection_has_official_kb(banking_collection):
        return structured["answer"], [{
            **structured,
            "search_methods": ["structured"],
            "hybrid_score": 100.0,
        }]

    ranked = []
    for metadata in get_cached_banking_metadatas(banking_collection):
        candidate_text = f"{metadata.get('question', '')} {metadata.get('answer', '')}"
        if topic and not candidate_matches_topic(topic, candidate_text):
            continue
        score = lexical_candidate_score(query, metadata, topic, intent)
        if score > 0:
            ranked.append((score, metadata))
    ranked.sort(key=lambda item: item[0], reverse=True)

    if not ranked:
        return (
            "I could not find a verified answer. Please mention the banking product and what you want to know about it.",
            [],
        )

    score, metadata = ranked[0]
    stored_question = str(metadata.get("question", "")).lower()
    methods = ["lexical"]
    if normalize_match_text(query) == normalize_match_text(stored_question):
        methods.append("exact")
    elif query.lower().strip() in stored_question:
        methods.append("phrase")
    elif SequenceMatcher(None, query.lower().strip(), stored_question).ratio() >= 0.72:
        methods.append("fuzzy")
    source = {
        **metadata,
        "search_methods": methods,
        "hybrid_score": round(score, 4),
    }
    return str(metadata.get("answer", "")).strip(), [source]


def retrieve_lightweight_official_answer(query, topic="", intent="general", session=None, follow_up=False, current_query=None):
    documents = load_official_kb_documents()
    ranked = []
    session = session or {}
    current_query = current_query or query
    previous_source = session.get("last_kb_source") or {}
    entity_source = session.get("last_entity_kb_source") or {}
    product_attribute_context = (
        bool(entity_source)
        and is_product_attribute_followup(current_query)
        and not question_mentions_product_entity(current_query)
    )
    if product_attribute_context:
        previous_source = entity_source
    contextual_query = build_contextual_retrieval_query(current_query, session) if (follow_up or product_attribute_context) else current_query
    query_terms = search_tokens(current_query)
    logger.info(
        "retrieve.official.start query=%s current_query=%s contextual_query=%s topic=%s intent=%s follow_up=%s documents=%s previous_section=%s previous_question=%s previous_source_file=%s",
        log_text(query),
        log_text(current_query),
        log_text(contextual_query),
        topic or "",
        intent,
        follow_up,
        len(documents),
        previous_source.get("section", ""),
        log_text(previous_source.get("question", "")),
        previous_source.get("source_file", ""),
    )
    for metadata in documents:
        score = lexical_candidate_score(current_query, metadata, topic, intent)
        if follow_up:
            if query != current_query:
                score += lexical_candidate_score(query, metadata, topic, intent) * 0.05
            if contextual_query not in {query, current_query}:
                score += lexical_candidate_score(contextual_query, metadata, topic, intent) * 0.05
            question_terms = search_tokens(metadata.get("question", ""))
            direct_title_hits = query_terms & question_terms
            if direct_title_hits:
                score += len(direct_title_hits) * 8.0
            if previous_source.get("source_file") and metadata.get("source_file") == previous_source.get("source_file"):
                score += 4.0
            elif previous_source.get("section") and metadata.get("section") == previous_source.get("section"):
                score += 2.0
        if (
            product_attribute_context
            and previous_source.get("question")
            and normalize_match_text(previous_source.get("question", "")) == normalize_match_text(metadata.get("question", ""))
            and normalize_match_text(current_query) != normalize_match_text(metadata.get("question", ""))
        ):
            score += 70.0
        if (
            follow_up
            and previous_source.get("question")
            and semantic_similarity(current_query, previous_source.get("answer_preview", "")) > 0.18
            and normalize_match_text(previous_source.get("question", "")) == normalize_match_text(metadata.get("question", ""))
        ):
            score += 10.0
        if (
            follow_up
            and query_terms & {"record", "records", "evidence", "proof", "keep"}
            and previous_source.get("question")
            and normalize_match_text(previous_source.get("question", "")) == normalize_match_text(metadata.get("question", ""))
            and search_tokens(previous_source.get("answer", "")) & {"record", "records", "communication", "reference"}
        ):
            score += 35.0
        if (
            follow_up
            and previous_source.get("question")
            and normalize_match_text(previous_source.get("question", "")) == normalize_match_text(metadata.get("question", ""))
            and normalize_match_text(current_query) != normalize_match_text(metadata.get("question", ""))
            and not direct_title_hits
        ):
            score -= 18.0
        if normalize_match_text(current_query) == normalize_match_text(metadata.get("question", "")):
            score += 75.0
        if score > 0:
            ranked.append((score, metadata))

    previous_answer_candidates = []
    if follow_up or product_attribute_context:
        recent_sources = [previous_source] if is_pronoun_reference(current_query) and previous_source else list(session.get("kb_source_history") or [])
        if previous_source:
            source_key = (previous_source.get("source_file", ""), previous_source.get("question", ""))
            if not any((item.get("source_file", ""), item.get("question", "")) == source_key for item in recent_sources):
                recent_sources.append(previous_source)
        for source_index, source in enumerate(reversed(recent_sources[-5:])):
            candidate = build_previous_answer_candidate(current_query, source)
            if not candidate:
                continue
            recency_penalty = source_index * 4.0
            if entity_source and source.get("question") == entity_source.get("question") and is_product_attribute_followup(current_query):
                recency_penalty -= 42.0
            candidate["score"] = max(0.0, candidate["score"] - recency_penalty)
            candidate["metadata"]["hybrid_score"] = round(candidate["score"], 4)
            previous_answer_candidates.append(candidate)
    for previous_answer_candidate in previous_answer_candidates:
        ranked.append((previous_answer_candidate["score"], previous_answer_candidate["metadata"]))
        logger.info(
            "retrieve.official.previous_answer_candidate score=%.4f section=%s question=%s answer=%s",
            previous_answer_candidate["score"],
            previous_answer_candidate["metadata"].get("section", ""),
            log_text(previous_answer_candidate["metadata"].get("question", "")),
            log_text(previous_answer_candidate["metadata"].get("answer", "")),
        )
    ranked.sort(key=lambda item: item[0], reverse=True)
    for rank, (score, metadata) in enumerate(ranked[:5], 1):
        logger.info(
            "retrieve.official.candidate rank=%s score=%.4f section=%s question=%s source_file=%s",
            rank,
            score,
            metadata.get("section", ""),
            log_text(metadata.get("question", "")),
            metadata.get("source_file", ""),
        )

    if not ranked:
        logger.warning("retrieve.official.no_match query=%s topic=%s intent=%s", log_text(query), topic or "", intent)
        return (
            "I could not find that in the official banking knowledge base. Please ask about Axis Bank credit cards, fees, rewards, billing, fraud, RBI rules, or account management.",
            [],
        )

    score, metadata = ranked[0]
    methods = list(metadata.get("search_methods") or ["official_kb", "lexical"])
    if normalize_match_text(current_query) == normalize_match_text(metadata.get("question", "")) and "exact" not in methods:
        methods.append("exact")
    elif (
        SequenceMatcher(None, normalize_match_text(current_query), normalize_match_text(metadata.get("question", ""))).ratio() >= 0.72
        and "fuzzy" not in methods
    ):
        methods.append("fuzzy")
    source = {
        **metadata,
        "dataset": "official_kb",
        "search_methods": methods,
        "hybrid_score": round(score, 4),
    }
    logger.info(
        "retrieve.official.selected score=%.4f methods=%s section=%s question=%s source_file=%s",
        score,
        methods,
        metadata.get("section", ""),
        log_text(metadata.get("question", "")),
        metadata.get("source_file", ""),
    )
    return str(metadata.get("answer", "")).strip(), [source]


def score_intent_match(intent, candidate_text):
    text = candidate_text.lower()
    positive = {
        "opening": ["open", "opening", "apply", "register", "walk into", "nearest branch", "submit", "carry", "documents", "account opening"],
        "documents": ["documents", "proof", "kyc", "identity proof", "address proof", "photograph", "required", "carry"],
        "process": ["steps", "process", "procedure", "follow", "submit", "visit", "log in", "click", "fill"],
        "dispute": ["dispute", "chargeback", "fraud", "fraudulent", "unauthorized", "unauthorised", "complaint", "ombudsman", "block", "lost", "stolen", "suspicious", "report fraud", "transaction not initiated"],
        "fees": ["fee", "fees", "charge", "charges", "cost", "minimum balance", "penalty"],
        "interest": ["interest", "interest rate", "rate of interest", "returns", "earned"],
        "tenure": ["tenure", "duration", "maturity", "period", "months", "years"],
        "limits": ["limit", "maximum", "minimum", "per day", "daily", "amount"],
        "eligibility": ["eligible", "eligibility", "who can", "can open", "allowed", "resident", "individual"],
        "definition": ["is a", "means", "refers to", "defined", "product where"],
    }
    negative = {
        "opening": ["what is", "meaning", "basic bank account", "is a type", "is an account", "allows you to"],
        "documents": ["what is", "meaning", "is a type"],
        "process": ["what is", "meaning", "is a type"],
        "dispute": ["emi conversion", "apply for", "eligible to apply", "annual fee waived", "reward points", "cashback", "increase or decrease", "credit limit"],
        "definition": ["steps", "submit", "visit", "documents required", "opening"],
    }
    score = 0.0
    for phrase in positive.get(intent, []):
        if phrase in text:
            score += 0.9
    for phrase in negative.get(intent, []):
        if phrase in text:
            score -= 1.4
    return score


def score_topic_match(topic, candidate_text):
    if not topic:
        return 0.0
    text = candidate_text.lower()
    topic_text = topic.lower().strip()
    topic_tokens = [token for token in re.findall(r"[a-z0-9]+", topic_text) if len(token) > 2]
    domain_nouns = {
        "account", "card", "loan", "deposit", "facility", "service", "cheque",
        "payment", "banking", "insurance", "investment", "fund", "statement",
        "password", "pin", "branch", "netbanking", "transaction",
    }
    modifiers = [token for token in topic_tokens if token not in domain_nouns]
    score = 0.0
    if topic_text and topic_text in text:
        score += 4.0
    for token in topic_tokens:
        if re.search(rf"\b{re.escape(token)}\b", text):
            score += 0.8
    for modifier in modifiers:
        if not re.search(rf"\b{re.escape(modifier)}\b", text):
            score -= 2.5
    if modifiers and any(noun in topic_tokens for noun in domain_nouns):
        for noun in domain_nouns:
            if noun in topic_tokens and re.search(rf"\b{re.escape(noun)}\b", text) and not any(re.search(rf"\b{re.escape(mod)}\b", text) for mod in modifiers):
                score -= 2.0
                break
    return score


def candidate_matches_topic(topic, candidate_text):
    if not topic:
        return True
    text = candidate_text.lower()
    topic_tokens = [token for token in re.findall(r"[a-z0-9]+", topic.lower()) if len(token) > 2]
    if not topic_tokens:
        return True
    return all(re.search(rf"\b{re.escape(token)}\b", text) for token in topic_tokens)


def candidate_matches_intent(intent, candidate_text):
    if intent in {"general", "definition"}:
        return True
    return score_intent_match(intent, candidate_text) > 0


def choose_best_answer(user_question, banking_context, reranker_model, intent="general", rerank_query=None, topic=""):
    candidates = []
    for result in banking_context.split("Result "):
        if "Answer:" not in result:
            continue
        section_match = re.search(r"(?m)^Section:\s*(.*?)\s*$", result)
        question_match = re.search(r"(?m)^Question:\s*(.*?)\s*$", result)
        answer_match = re.search(r"(?s)Answer:\s*(.*?)(?:\nSearch methods:|\nHybrid score:|\Z)", result)
        section = section_match.group(1).strip() if section_match else ""
        question = question_match.group(1).strip() if question_match else ""
        answer = answer_match.group(1).strip() if answer_match else ""
        if answer:
            text = f"{question} {answer}"
            candidates.append({"section": section, "question": question, "answer": answer, "text": text})
    if not candidates:
        return "I found relevant banking information, but could not extract a clear answer."

    exact_candidates = [
        candidate for candidate in candidates
        if normalize_match_text(candidate["question"]) == normalize_match_text(rerank_query or user_question)
    ]
    if exact_candidates:
        return exact_candidates[0]["answer"]

    topic_candidates = [candidate for candidate in candidates if candidate_matches_topic(topic, candidate["text"])]
    if topic_candidates:
        candidates = topic_candidates
    elif topic:
        return f"I could not find a verified answer for {topic}. Please ask about its definition, eligibility, documents, fees, interest, or application process."

    structured_candidates = [
        candidate for candidate in candidates
        if candidate["section"].lower() == "structured product reference"
    ]
    if structured_candidates and intent in {"definition", "general"}:
        return structured_candidates[0]["answer"]

    intent_candidates = [candidate for candidate in candidates if candidate_matches_intent(intent, candidate["text"])]
    if intent_candidates:
        candidates = intent_candidates

    ranking_question = rerank_query or user_question
    scores = reranker_model.predict([[ranking_question, candidate["text"]] for candidate in candidates])
    adjusted_scores = [
        float(scores[i])
        + score_intent_match(intent, candidates[i]["text"])
        + score_topic_match(topic, candidates[i]["text"])
        for i in range(len(scores))
    ]
    return candidates[max(range(len(adjusted_scores)), key=lambda i: adjusted_scores[i])]["answer"]


def translate_question_for_search(question, language):
    if language == "English":
        return question
    try:
        return GoogleTranslator(source="auto", target="en").translate(question)
    except Exception:
        return question


def translate_answer(answer, language):
    if language == "English":
        return answer
    try:
        return GoogleTranslator(source="auto", target="hi").translate(answer)
    except Exception:
        return answer


def translate_suggested_questions(questions, language):
    if language == "English":
        return questions
    return [translate_answer(question, language) for question in questions]


PRODUCT_ALIASES = {
    "ace": "ACE Credit Card",
    "axis ace": "ACE Credit Card",
    "axis bank ace": "ACE Credit Card",
    "axis bank ace credit card": "ACE Credit Card",
    "flipkart": "Flipkart Axis Bank Credit Card",
    "flipkart axis": "Flipkart Axis Bank Credit Card",
    "flipkart axis bank credit card": "Flipkart Axis Bank Credit Card",
    "atlas": "Atlas Credit Card",
    "axis atlas": "Atlas Credit Card",
    "axis bank atlas": "Atlas Credit Card",
    "axis bank atlas credit card": "Atlas Credit Card",
    "airtel": "Airtel Axis Bank Credit Card",
    "airtel axis": "Airtel Axis Bank Credit Card",
    "airtel axis card": "Airtel Axis Bank Credit Card",
    "airtel axis bank credit card": "Airtel Axis Bank Credit Card",
    "fd": "Fixed Deposit",
    "fixed deposit": "Fixed Deposit",
    "rd": "Recurring Deposit",
    "recurring deposit": "Recurring Deposit",
    "savings account": "Savings Account",
    "saving account": "Savings Account",
    "current account": "Current Account",
    "credit card": "Credit Card",
    "debit card": "Debit Card",
    "home loan": "Home Loan",
    "personal loan": "Personal Loan",
}

PRODUCT_FEATURES = {
    "Fixed Deposit": {
        "Deposit": "One-time lump sum",
        "Interest": "Fixed for the chosen tenure",
        "Best for": "Parking a fixed amount safely",
        "Flexibility": "Low before maturity",
        "Typical use": "Short or medium-term savings goal",
    },
    "Recurring Deposit": {
        "Deposit": "Monthly fixed instalment",
        "Interest": "Fixed for the chosen tenure",
        "Best for": "Building savings every month",
        "Flexibility": "Low before maturity",
        "Typical use": "Disciplined monthly saving",
    },
    "Savings Account": {
        "Deposit": "Any amount anytime",
        "Interest": "Variable and usually lower",
        "Best for": "Daily banking and UPI payments",
        "Flexibility": "High",
        "Typical use": "Salary, UPI, withdrawals, bill payments",
    },
    "Current Account": {
        "Deposit": "Frequent business deposits",
        "Interest": "Usually no interest",
        "Best for": "Business transactions",
        "Flexibility": "High",
        "Typical use": "Shop, company, merchant payments",
    },
    "Credit Card": {
        "Deposit": "No deposit; bank gives credit limit",
        "Interest": "Charged if dues are unpaid",
        "Best for": "Short-term credit and rewards",
        "Flexibility": "High, but needs discipline",
        "Typical use": "Purchases, bills, online payments",
    },
    "Airtel Axis Bank Credit Card": {
        "Fee": "Rs. 500 + GST",
        "Rewards": "25% cashback on Airtel services, 10% on selected food/grocery apps, 1% on other spends",
        "Best for": "Airtel users and people who spend on recharges, broadband, DTH, food delivery, and groceries",
        "Flexibility": "Cashback is credited to the card account",
        "Typical use": "Airtel bills, recharges, broadband, DTH, Swiggy, BigBasket, Zomato",
    },
    "Debit Card": {
        "Deposit": "Uses money from bank account",
        "Interest": "No card interest",
        "Best for": "Spending own account balance",
        "Flexibility": "High",
        "Typical use": "ATM withdrawal, POS, online payment",
    },
    "Home Loan": {
        "Deposit": "Down payment plus EMI",
        "Interest": "Fixed or floating loan interest",
        "Best for": "Buying or building property",
        "Flexibility": "Medium",
        "Typical use": "House purchase, construction, renovation",
    },
    "Personal Loan": {
        "Deposit": "No deposit; EMI repayment",
        "Interest": "Usually higher than secured loans",
        "Best for": "Urgent personal expenses",
        "Flexibility": "Medium",
        "Typical use": "Medical, travel, education, emergency needs",
    },
    "ACE Credit Card": {
        "Fee": "Rs. 499 + GST",
        "Rewards": "5% utility bill cashback via Google Pay; 4% on Swiggy, Zomato, Ola; 2% on other spends",
        "Best for": "Everyday spenders who pay utility bills digitally and order food",
        "Flexibility": "Cashback is credited directly to the card account",
        "Typical use": "Utility bills, food delivery, app-based daily spending",
    },
    "Flipkart Axis Bank Credit Card": {
        "Fee": "Rs. 500 + GST",
        "Rewards": "5% cashback on Flipkart and Myntra; 4% on preferred partners; 1.5% on other spends",
        "Best for": "Regular Flipkart, Myntra, and partner-brand shoppers",
        "Flexibility": "Cashback is credited to the card account",
        "Typical use": "Online shopping and partner-brand purchases",
    },
    "Atlas Credit Card": {
        "Fee": "Rs. 5,000 + GST",
        "Rewards": "EDGE Miles for flight and hotel redemptions; milestone reward of 2,500 EDGE Miles",
        "Best for": "Frequent travellers who redeem rewards for flights and hotels",
        "Flexibility": "Tiered travel benefits based on annual spend",
        "Typical use": "Travel bookings, hotel stays, high annual card spends",
    },
}


def build_structured_product_candidate(topic, intent):
    if intent not in {"definition", "general"} or not topic:
        return None
    product_name = next(
        (name for name in PRODUCT_FEATURES if name.lower() == topic.lower()),
        None,
    )
    if not product_name:
        return None
    features = PRODUCT_FEATURES[product_name]
    best_for = features.get("Best for", "banking needs").rstrip(".")
    typical_use = features.get("Typical use", "").rstrip(".")
    interest = features.get("Interest", "").rstrip(".")
    answer = f"A {product_name.lower()} is a banking product used for {best_for.lower()}."
    if typical_use:
        answer += f" It is commonly used for {typical_use.lower()}."
    if interest:
        answer += f" Its interest arrangement is: {interest.lower()}."
    return {
        "section": "structured product reference",
        "question": f"What is a {product_name.lower()}?",
        "answer": answer,
    }


def detect_compare_products(question):
    q = question.lower()
    if not any(word in q for word in ["compare", "difference", "versus", " vs ", "better"]):
        return []
    matches = []
    for alias, product in sorted(PRODUCT_ALIASES.items(), key=lambda item: len(item[0]), reverse=True):
        match = re.search(rf"\b{re.escape(alias)}\b", q)
        if match:
            matches.append((match.start(), -len(alias), product))
    if any(product.endswith("Credit Card") and product != "Credit Card" for _, _, product in matches):
        matches = [item for item in matches if item[2] != "Credit Card"]
    ordered = []
    for _, _, product in sorted(matches):
        if product not in ordered:
            ordered.append(product)
    return ordered[:2] if len(ordered) >= 2 else []


def compare_products(question):
    products = detect_compare_products(question)
    if len(products) < 2:
        return None
    feature_order = ["Fee", "Rewards", "Deposit", "Interest", "Best for", "Flexibility", "Typical use"]
    features = [
        feature for feature in feature_order
        if any(PRODUCT_FEATURES.get(product, {}).get(feature) for product in products)
    ]
    rows = [
        {
            "feature": feature,
            products[0]: PRODUCT_FEATURES.get(products[0], {}).get(feature, "-"),
            products[1]: PRODUCT_FEATURES.get(products[1], {}).get(feature, "-"),
        }
        for feature in features
    ]
    reply = f"Here is a simple comparison of {products[0]} and {products[1]}:"
    return {
        "reply": reply,
        "comparison_table": {
            "title": f"{products[0]} vs {products[1]}",
            "columns": ["Feature", products[0], products[1]],
            "rows": rows,
        },
        "topic": f"{products[0]} vs {products[1]}",
    }


def source_entity_label(source):
    text = f"{source.get('question', '')} {source.get('answer_preview', '')}".lower()
    for alias, product in sorted(PRODUCT_ALIASES.items(), key=lambda item: len(item[0]), reverse=True):
        if product == "Credit Card":
            continue
        if product.endswith("Credit Card") and re.search(rf"\b{re.escape(alias)}\b", text):
            return product
    return ""


def question_mentions_product_entity(question):
    q = str(question or "").lower()
    return bool(source_entity_label({"question": q, "answer_preview": ""}))


def is_product_attribute_followup(question):
    terms = expand_query_terms(question)
    attribute_terms = {
        "fee", "annual", "joining", "cashback", "reward", "rewards", "point",
        "points", "mile", "miles", "best", "suitable", "use", "lounge",
        "benefit", "benefits", "limit", "interest", "cost", "price", "yearly",
    }
    return bool(terms & attribute_terms) or is_pronoun_reference(question)


def answer_comparison_followup(question, session):
    table = session.get("last_comparison") or {}
    if not table:
        return None
    q_terms = expand_query_terms(question)
    if not q_terms:
        return None
    q_text = str(question or "").lower()
    comparison_cues = {
        "which", "one", "both", "between", "better", "best", "less", "lower",
        "more", "compare", "comparison", "cost", "costs", "cheap", "cheaper",
        "gives", "offers", "prefer", "worth", "travel", "travelling",
    }
    comparison_phrases = [
        "which one", "which is", "which has", "which gives", "between them",
        "of these", "these two", "the first", "the second", "both cards",
    ]
    if not ((q_terms & comparison_cues) or any(phrase in q_text for phrase in comparison_phrases)):
        return None
    columns = table.get("columns") or []
    rows = table.get("rows") or []
    if len(columns) < 3 or not rows:
        return None

    ranked_rows = []
    for row in rows:
        feature = str(row.get("feature", ""))
        row_text = " ".join(str(value) for value in row.values())
        terms = expand_query_terms(f"{feature} {row_text}")
        hits = q_terms & terms
        if hits:
            ranked_rows.append((len(hits), feature, row))
    lower_words = {"lower", "less", "cheap", "cheaper", "lowest"}
    shopping_words = {"shopping", "shop", "online", "flipkart", "myntra"}
    cashback_words = {"cashback", "reward", "rewards"}
    travel_words = {"travel", "travelling", "flight", "hotel", "mile", "miles"}
    if not ranked_rows:
        fallback_feature = ""
        if q_terms & lower_words:
            fallback_feature = "Fee"
        elif q_terms & travel_words:
            fallback_feature = "Typical use"
        elif q_terms & cashback_words:
            fallback_feature = "Rewards"
        elif q_terms & shopping_words:
            fallback_feature = "Typical use"
        for row in rows:
            if str(row.get("feature", "")).lower() == fallback_feature.lower():
                ranked_rows.append((1, str(row.get("feature", "")), row))
                break
    if not ranked_rows:
        return None
    ranked_rows.sort(key=lambda item: item[0], reverse=True)
    _, feature, row = ranked_rows[0]
    left, right = columns[1], columns[2]
    left_value = str(row.get(left, "-"))
    right_value = str(row.get(right, "-"))

    if q_terms & lower_words:
        answer = f"For {feature.lower()}, {left} has: {left_value}. {right} has: {right_value}."
    elif q_terms & shopping_words:
        answer = f"For shopping, compare the '{feature}' row: {left}: {left_value}. {right}: {right_value}."
    elif q_terms & cashback_words:
        answer = f"For cashback/rewards, {left}: {left_value}. {right}: {right_value}."
    elif q_terms & travel_words:
        answer = f"For travel use, compare the '{feature}' row: {left}: {left_value}. {right}: {right_value}."
    else:
        answer = f"For {feature.lower()}, {left}: {left_value}. {right}: {right_value}."
    return answer


def is_account_recommendation_question(question):
    q = question.lower()
    phrases = ["which account", "what account", "account should i open", "recommend account", "recommend an account", "best account", "suitable account", "suggest account", "suggest an account", "open for me"]
    profile_words = [
        "i am", "i'm", "college", "student", "salary", "salaried", "business",
        "upi", "digital", "online", "retired", "senior", "nri", "minimum balance",
        "zero balance", "save money", "frequent transactions",
    ]
    return any(phrase in q for phrase in phrases) or ("account" in q and any(word in q for word in profile_words)) or ("i am" in q and any(word in q for word in profile_words))


def extract_account_profile(question):
    q = question.lower()
    signals = {
        "student": any(word in q for word in ["student", "college", "university", "study", "school"]),
        "digital": any(word in q for word in ["upi", "digital", "online", "mobile banking", "net banking", "qr", "payments"]),
        "business": any(word in q for word in ["business", "shop", "company", "startup", "merchant", "daily transactions", "frequent transactions"]),
        "salary": any(word in q for word in ["salary", "salaried", "employee", "job", "monthly income", "working"]),
        "senior": any(word in q for word in ["senior", "retired", "retirement", "pension", "old age"]),
        "nri": any(word in q for word in ["nri", "abroad", "foreign", "overseas", "outside india"]),
        "low_balance": any(word in q for word in ["zero balance", "no minimum balance", "low balance", "minimum balance", "low minimum"]),
        "saving": any(word in q for word in ["save money", "savings", "personal", "general", "normal", "basic"]),
    }
    return {name: value for name, value in signals.items() if value}


def recommend_account(question):
    signals = extract_account_profile(question)
    options = [
        {
            "account": "Student Savings Account",
            "weights": {"student": 6, "digital": 2, "low_balance": 2, "saving": 1},
            "why": "It fits a student profile and usually keeps banking simple with lower balance requirements and easy digital access.",
            "benefits": ["Low or relaxed minimum balance", "UPI, debit card, and mobile banking support", "Simple KYC and everyday spending features"],
        },
        {
            "account": "Current Account",
            "weights": {"business": 7, "digital": 1},
            "why": "It is designed for business users who need frequent deposits, withdrawals, and payments.",
            "benefits": ["Higher transaction limits", "Business payments and collections", "Cheque book and merchant-friendly banking features"],
        },
        {
            "account": "Salary Account",
            "weights": {"salary": 7, "digital": 1, "saving": 1},
            "why": "It is suitable when salary is credited every month and you need everyday banking features.",
            "benefits": ["Often zero-balance while salary is credited", "Debit card and digital banking", "Easy bill payments and transfers"],
        },
        {
            "account": "Senior Citizen Savings Account",
            "weights": {"senior": 7, "saving": 1},
            "why": "It is meant for retired or senior customers who want convenient personal banking.",
            "benefits": ["Senior-friendly banking services", "Savings and pension handling", "Branch and digital access depending on the bank"],
        },
        {
            "account": "NRI Account",
            "weights": {"nri": 8},
            "why": "It is suitable for Non-Resident Indians who need to manage Indian income or overseas funds.",
            "benefits": ["Manage India-linked funds", "NRE/NRO options depending on need", "International banking support"],
        },
        {
            "account": "Zero Balance Savings Account",
            "weights": {"low_balance": 6, "digital": 2, "student": 1, "saving": 1},
            "why": "It is suitable if you want basic banking without maintaining a high balance.",
            "benefits": ["No high minimum balance pressure", "UPI and basic digital banking", "Good for light everyday use"],
        },
        {
            "account": "Regular Savings Account",
            "weights": {"saving": 4, "digital": 1},
            "why": "It is a reliable default for everyday savings, payments, withdrawals, and personal banking.",
            "benefits": ["Savings and withdrawals", "Debit card, UPI, and net banking", "Useful for general personal banking"],
        },
    ]
    ranked = sorted(
        (
            {
                **option,
                "score": sum(weight for signal, weight in option["weights"].items() if signals.get(signal)),
            }
            for option in options
        ),
        key=lambda item: item["score"],
        reverse=True,
    )
    if ranked[0]["score"] == 0:
        ranked[0] = next(option for option in ranked if option["account"] == "Regular Savings Account")

    best = ranked[0]
    detected = [name.replace("_", " ") for name in signals]
    card = {
        "title": "Recommended account",
        "account": best["account"],
        "why": best["why"],
        "benefits": best["benefits"],
        "detected_profile": detected,
        "agent": "Smart account agent",
    }
    reply = (
        f"Based on your profile, a {best['account']} would be suitable.\n\n"
        f"Why recommended:\n{best['why']}\n\n"
        "Benefits:\n"
        + "\n".join(f"- {benefit}" for benefit in best["benefits"])
    )
    return reply, card


def is_generic_help_answer(answer):
    text = answer.lower()
    generic_phrases = [
        "i can help you with any kind of banking queries",
        "i can help you with banking queries",
        "how can i help",
    ]
    return any(phrase in text for phrase in generic_phrases)


def generate_llm_answer(user_question, banking_context, memory_context, recent_chat_history, reranker_model, tokenizer, llm_model, language, intent="general", rerank_query=None, topic=""):
    best_answer = choose_best_answer(user_question, banking_context, reranker_model, intent, rerank_query, topic)
    if "Section: structured product reference" in banking_context and intent in {"definition", "general"}:
        return best_answer
    instruction = "Answer in simple Hindi. Use common banking words that Indian users understand." if language == "Hindi" else "Answer in simple English."
    prompt = f"""
You are a banking assistant.
{instruction}

Rewrite the retrieved answer into a short, clear answer for the user.
Keep the user's requested intent: {intent}.
Keep the banking topic: {topic or 'unknown'}.
Use recent conversation only to resolve follow-up references.
Do not replace a requested process, document list, fee, or eligibility answer with a definition.
Do not add facts that are not supported by the retrieved answer.

Recent Conversation:
{recent_chat_history}

Relevant Past Memory:
{memory_context}

User Question:
{rerank_query or user_question}

Retrieved Answer:
{best_answer}

Final Answer:
"""
    inputs = tokenizer(prompt, return_tensors="pt", max_length=768, truncation=True)
    outputs = llm_model.generate(**inputs, max_new_tokens=80, num_beams=1, do_sample=False)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    if len(answer.split()) < 5 or "?" in answer or answer.lower() == user_question.lower() or is_generic_help_answer(answer):
        answer = best_answer
    return answer


def answer_message(message, language="English", session_id=None):
    message = message.strip()
    if not message:
        raise ValueError("Message cannot be empty.")
    if language not in {"English", "Hindi"}:
        language = "English"

    session_id, session = get_session(session_id)
    logger.info(
        "answer.start session_id=%s language=%s lightweight=%s message=%s history_count=%s current_topic=%s current_intent=%s",
        session_id,
        language,
        LIGHTWEIGHT_MODE,
        log_text(message),
        len(session.get("chat_history", [])),
        session.get("current_topic", ""),
        session.get("current_intent", "general"),
    )
    if is_sensitive_personal_question(message):
        logger.info("answer.route privacy_guard session_id=%s message=%s", session_id, log_text(message))
        reply = build_sensitive_personal_refusal(language)
        session["chat_history"].append({"role": "user", "message": message})
        session["chat_history"].append({"role": "bot", "message": reply})
        return {
            "session_id": session_id,
            "reply": reply,
            "history": session["chat_history"],
            "sources": [],
            "suggested_questions": translate_suggested_questions([
                "How do I reset my banking password?",
                "How do I report a lost card?",
                "How do I protect my bank account?",
            ], language),
            "topic": "privacy restriction",
            "restricted": True,
            "search_methods": ["privacy_guard"],
        }

    translated_question = translate_question_for_search(message, language)
    search_question = normalize_banking_spelling(translated_question)
    resolved_question = resolve_question_context(search_question, session)
    intent = detect_question_intent(resolved_question)
    if intent == "general" and is_follow_up_question(search_question):
        intent = session.get("current_intent", "general")
    topic = extract_topic_from_question(resolved_question)
    if not topic and is_follow_up_question(search_question):
        topic = session.get("current_topic", "")
    if not topic and session.get("current_topic") and intent in {"dispute", "documents", "fees", "interest", "tenure", "limits", "eligibility", "opening", "process"}:
        topic = session.get("current_topic", "")
        resolved_question = resolve_question_context(f"{search_question} about {topic}", session)
        logger.info(
            "answer.topic_fallback session_id=%s inherited_topic=%s resolved=%s intent=%s",
            session_id,
            topic,
            log_text(resolved_question),
            intent,
        )
    logger.info(
        "answer.analysis session_id=%s translated=%s search=%s resolved=%s follow_up=%s intent=%s topic=%s current_topic=%s topic_history=%s",
        session_id,
        log_text(translated_question),
        log_text(search_question),
        log_text(resolved_question),
        is_follow_up_question(search_question),
        intent,
        topic or "",
        session.get("current_topic", ""),
        session.get("topic_history", []),
    )
    session["chat_history"].append({"role": "user", "message": message})
    if is_credential_help_question(search_question):
        logger.info("answer.route credential_help session_id=%s topic=%s intent=%s", session_id, topic or "", intent)
        reply = build_credential_help_answer(search_question, language)
        remember_conversation_context(session, "banking password" if "password" in search_question else "banking PIN", "process", resolved_question)
        session["chat_history"].append({"role": "bot", "message": reply})
        return {
            "session_id": session_id,
            "reply": reply,
            "history": session["chat_history"],
            "sources": [],
            "suggested_questions": build_credential_help_suggestions(language),
            "topic": "banking password" if "password" in search_question else "banking PIN",
            "search_methods": ["credential_help"],
        }

    pending_flow_topic = (session.get("pending_flow") or {}).get("topic", "")
    suggested_questions = translate_suggested_questions(build_suggested_questions(topic, intent), language)

    comparison = compare_products(search_question)
    if comparison:
        logger.info("answer.route comparison session_id=%s topic=%s", session_id, comparison.get("topic", ""))
        reply = translate_answer(comparison["reply"], language)
        remember_conversation_context(session, comparison["topic"], "comparison", comparison["topic"])
        session["last_comparison"] = comparison["comparison_table"]
        session["chat_history"].append({"role": "bot", "message": reply})
        return {
            "session_id": session_id,
            "reply": reply,
            "history": session["chat_history"],
            "sources": [],
            "suggested_questions": translate_suggested_questions([
                f"Which is better: {comparison['comparison_table']['columns'][1]} or {comparison['comparison_table']['columns'][2]}?",
                f"What documents are needed for {comparison['comparison_table']['columns'][1]}?",
                f"What are the charges for {comparison['comparison_table']['columns'][2]}?",
            ], language),
            "comparison_table": comparison["comparison_table"],
            "topic": comparison["topic"],
        }

    if is_account_recommendation_question(search_question):
        logger.info("answer.route recommendation session_id=%s question=%s", session_id, log_text(search_question))
        reply, recommendation_card = recommend_account(search_question)
        reply = translate_answer(reply, language)
        recommendation_topic = recommendation_card.get("account", "account recommendation")
        remember_conversation_context(session, recommendation_topic, "recommendation", search_question)
        session["chat_history"].append({"role": "bot", "message": reply})
        return {
            "session_id": session_id,
            "reply": reply,
            "history": session["chat_history"],
            "sources": [],
            "suggested_questions": translate_suggested_questions([
                "How do I open the recommended account?",
                "Which documents are needed for the recommended account?",
                "What benefits does the recommended account offer?",
            ], language),
            "recommendation_card": recommendation_card,
            "topic": recommendation_topic,
        }

    comparison_followup = answer_comparison_followup(search_question, session)
    if comparison_followup:
        logger.info("answer.route comparison_followup session_id=%s question=%s", session_id, log_text(search_question))
        reply = translate_answer(comparison_followup, language)
        session["chat_history"].append({"role": "bot", "message": reply})
        return {
            "session_id": session_id,
            "reply": reply,
            "history": session["chat_history"],
            "sources": [],
            "suggested_questions": suggested_questions,
            "topic": session.get("current_topic", ""),
            "search_methods": ["comparison_context"],
        }

    pending_reply = handle_pending_flow(search_question, session)
    if pending_reply:
        logger.info("answer.route pending_flow session_id=%s response_topic=%s intent=%s", session_id, pending_flow_topic or topic, intent)
        response_topic = pending_flow_topic or topic
        response_suggestions = translate_suggested_questions(build_suggested_questions(response_topic, intent), language)
        reply = translate_answer(pending_reply, language)
        remember_conversation_context(session, response_topic, intent, resolved_question)
        session["chat_history"].append({"role": "bot", "message": reply})
        return {
            "session_id": session_id,
            "reply": reply,
            "history": session["chat_history"],
            "sources": [],
            "suggested_questions": response_suggestions,
            "topic": response_topic,
        }

    banking_related = is_banking_related_question(search_question, session)

    if LIGHTWEIGHT_MODE and banking_related:
        logger.info("answer.route lightweight_official session_id=%s topic=%s intent=%s resolved=%s", session_id, topic or "", intent, log_text(resolved_question))
        reply, sources = retrieve_lightweight_official_answer(
            resolved_question,
            topic,
            intent,
            session=session,
            follow_up=is_follow_up_question(search_question) or bool(session.get("last_kb_source") and topic == session.get("current_topic")),
            current_query=search_question,
        )
        reply = translate_answer(reply, language)
        remember_conversation_context(session, topic, intent, resolved_question)
        if sources:
            remember_kb_source_context(session, sources[0])
        response_suggestions = translate_suggested_questions(
            build_contextual_suggested_questions(sources[0] if sources else None, topic, intent),
            language,
        )
        session["chat_history"].append({"role": "bot", "message": reply})
        return {
            "session_id": session_id,
            "reply": reply,
            "history": session["chat_history"],
            "sources": sources[:3],
            "suggested_questions": response_suggestions,
            "recommendation_card": None,
            "topic": topic,
            "rewritten_question": None,
            "search_methods": sorted({
                method
                for source in sources
                for method in source.get("search_methods", [])
            } | {"lightweight_official_kb"}),
        }

    clarifying_reply = build_clarifying_question(search_question, topic, intent, session)
    if clarifying_reply:
        logger.info("answer.route clarifying_question session_id=%s topic=%s intent=%s", session_id, topic or "", intent)
        reply = translate_answer(clarifying_reply, language)
        remember_conversation_context(session, topic, intent, resolved_question)
        session["chat_history"].append({"role": "bot", "message": reply})
        return {
            "session_id": session_id,
            "reply": reply,
            "history": session["chat_history"],
            "sources": [],
            "suggested_questions": suggested_questions,
            "topic": topic,
        }

    if should_use_form_assistant(resolved_question, topic, intent):
        logger.info("answer.route form_assistant session_id=%s topic=%s intent=%s resolved=%s", session_id, topic or "", intent, log_text(resolved_question))
        reply = build_form_assistant_answer(topic, intent, search_question)
        reply = translate_answer(reply, language)
        remember_conversation_context(session, topic, intent, resolved_question)
        session["chat_history"].append({"role": "bot", "message": reply})
        return {
            "session_id": session_id,
            "reply": reply,
            "history": session["chat_history"],
            "sources": [],
            "suggested_questions": suggested_questions,
            "topic": topic,
        }

    if not banking_related:
        if LIGHTWEIGHT_MODE:
            logger.info("answer.route lightweight_general session_id=%s question=%s", session_id, log_text(message))
            reply, general_methods = build_general_reference_answer(message, language)
            session["chat_history"].append({"role": "bot", "message": reply})
            return {
                "session_id": session_id,
                "reply": reply,
                "history": session["chat_history"],
                "sources": [],
                "suggested_questions": translate_suggested_questions(build_suggested_questions(), language),
                "topic": "general conversation",
                "general_chat": True,
                "search_methods": general_methods,
            }
        ready_models = get_ready_models()
        logger.info("answer.route general session_id=%s ready_models=%s", session_id, ready_models is not None)
        if ready_models is not None:
            try:
                _, _, tokenizer, llm_model = ready_models
                reply, general_methods = generate_general_llm_response(
                    message,
                    session,
                    tokenizer,
                    llm_model,
                    language,
                )
            except Exception:
                reply, general_methods = build_general_reference_answer(message, language)
        else:
            reply, general_methods = build_general_reference_answer(message, language)
        session["chat_history"].append({"role": "bot", "message": reply})
        return {
            "session_id": session_id,
            "reply": reply,
            "history": session["chat_history"],
            "sources": [],
            "suggested_questions": translate_suggested_questions(build_suggested_questions(), language),
            "topic": "general conversation",
            "general_chat": True,
            "search_methods": general_methods,
        }

    if LIGHTWEIGHT_MODE:
        logger.info("answer.route lightweight_official session_id=%s topic=%s intent=%s resolved=%s", session_id, topic or "", intent, log_text(resolved_question))
        reply, sources = retrieve_lightweight_official_answer(
            resolved_question,
            topic,
            intent,
            session=session,
            follow_up=is_follow_up_question(search_question) or bool(session.get("last_kb_source") and topic == session.get("current_topic")),
            current_query=search_question,
        )
        reply = translate_answer(reply, language)
        remember_conversation_context(session, topic, intent, resolved_question)
        if sources:
            remember_kb_source_context(session, sources[0])
        response_suggestions = translate_suggested_questions(
            build_contextual_suggested_questions(sources[0] if sources else None, topic, intent),
            language,
        )
        session["chat_history"].append({"role": "bot", "message": reply})
        return {
            "session_id": session_id,
            "reply": reply,
            "history": session["chat_history"],
            "sources": sources[:3],
            "suggested_questions": response_suggestions,
            "recommendation_card": None,
            "topic": topic,
            "rewritten_question": None,
            "search_methods": sorted({
                method
                for source in sources
                for method in source.get("search_methods", [])
            } | {"lightweight_official_kb"}),
        }

    banking_collection = load_vector_db()
    ready_models = get_ready_models()
    if ready_models is None:
        logger.info("answer.route chroma_lightweight_fallback session_id=%s topic=%s intent=%s", session_id, topic or "", intent)
        reply, sources = retrieve_lightweight_banking_answer(
            resolved_question,
            topic,
            intent,
            banking_collection,
        )
        reply = translate_answer(reply, language)
        remember_conversation_context(session, topic, intent, resolved_question)
        session["chat_history"].append({"role": "bot", "message": reply})
        return {
            "session_id": session_id,
            "reply": reply,
            "history": session["chat_history"],
            "sources": sources[:3],
            "suggested_questions": suggested_questions,
            "recommendation_card": None,
            "topic": topic,
            "rewritten_question": None,
            "search_methods": sorted({
                method
                for source in sources
                for method in source.get("search_methods", [])
            } | {"lightweight_fallback"}),
        }

    embedding_model, reranker_model, tokenizer, llm_model = ready_models
    rewritten_question = ""
    if question_needs_llm_rewrite(resolved_question, session):
        logger.info("answer.rewrite.start session_id=%s resolved=%s current_topic=%s", session_id, log_text(resolved_question), session.get("current_topic", ""))
        rewritten_question = rewrite_unclear_question(resolved_question, session, tokenizer, llm_model)
        if not rewritten_question:
            logger.info("answer.route clarification_after_rewrite session_id=%s resolved=%s", session_id, log_text(resolved_question))
            reply = translate_answer(build_clarification_reply(session), language)
            session["chat_history"].append({"role": "bot", "message": reply})
            return {
                "session_id": session_id,
                "reply": reply,
                "history": session["chat_history"],
                "sources": [],
                "suggested_questions": translate_suggested_questions(build_suggested_questions(session.get("current_topic", "")), language),
                "topic": session.get("current_topic", ""),
                "needs_clarification": True,
            }
        resolved_question = resolve_question_context(rewritten_question, session)
        intent = detect_question_intent(resolved_question)
        if intent == "general" and is_follow_up_question(rewritten_question):
            intent = session.get("current_intent", "general")
        topic = extract_topic_from_question(resolved_question) or session.get("current_topic", "")
        logger.info(
            "answer.rewrite.done session_id=%s rewritten=%s resolved=%s intent=%s topic=%s",
            session_id,
            log_text(rewritten_question),
            log_text(resolved_question),
            intent,
            topic or "",
        )
        suggested_questions = translate_suggested_questions(build_suggested_questions(topic, intent), language)

    search_query = build_intent_search_query(resolved_question, intent)

    recommendation_card = None
    banking_context, sources, _ = retrieve_hybrid_banking_context(
        resolved_question,
        search_query,
        topic,
        intent,
        embedding_model,
        banking_collection,
    )
    exact_official_source = next(
        (
            source for source in sources
            if source.get("dataset") == "official_kb"
            and "exact" in source.get("search_methods", [])
            and normalize_match_text(source.get("question", "")) == normalize_match_text(resolved_question)
        ),
        None,
    )
    # Chroma chat memory is shared across users, so it must not influence answers.
    # Session-scoped recent history provides context without cross-account leakage.
    memory_context = ""
    recent_history = build_recent_chat_history(session)
    if exact_official_source:
        logger.info("answer.route exact_official_source session_id=%s question=%s", session_id, log_text(exact_official_source.get("question", "")))
        reply = str(exact_official_source.get("answer", "")).strip()
    else:
        logger.info("answer.route llm_rag session_id=%s topic=%s intent=%s sources=%s", session_id, topic or "", intent, [log_text(source.get("question", "")) for source in sources[:3]])
        reply = generate_llm_answer(message, banking_context, memory_context, recent_history, reranker_model, tokenizer, llm_model, language, intent, search_query, topic)
    reply = translate_answer(reply, language)

    remember_conversation_context(session, topic, intent, resolved_question)
    if sources:
        remember_kb_source_context(session, sources[0])
    response_suggestions = translate_suggested_questions(
        build_contextual_suggested_questions(sources[0] if sources else None, topic, intent),
        language,
    )
    session["chat_history"].append({"role": "bot", "message": reply})
    used_search_methods = sorted({
        method
        for source in sources
        for method in source.get("search_methods", [])
    } | {"cross_encoder"})

    return {
        "session_id": session_id,
        "reply": reply,
        "history": session["chat_history"],
        "sources": sources[:3] if isinstance(sources, list) else [],
        "suggested_questions": response_suggestions,
        "recommendation_card": recommendation_card,
        "topic": topic,
        "rewritten_question": rewritten_question or None,
        "search_methods": used_search_methods,
    }
