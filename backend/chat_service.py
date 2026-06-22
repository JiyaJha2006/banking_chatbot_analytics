from __future__ import annotations

import uuid
import re
from difflib import SequenceMatcher, get_close_matches
from functools import lru_cache
from pathlib import Path

import chromadb
from deep_translator import GoogleTranslator
from sentence_transformers import CrossEncoder, SentenceTransformer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DB_PATH = str(DATA_DIR / "vector_db")
BANKING_COLLECTION_NAME = "banking_knowledge_base"
MEMORY_COLLECTION_NAME = "chat_memory"
SESSIONS = {}
BANKING_METADATA_CACHE = {"count": -1, "metadatas": []}

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
    "deposit", "document", "eligibility", "fee", "interest", "loan", "money",
    "nominee", "password", "payment", "pin", "statement", "transaction",
    "transfer", "withdrawal", "apply", "open", "activate", "documents",
    "required", "salaried", "self-employed", "compare", "benefits", "fees",
    "charges", "steps",
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


@lru_cache(maxsize=1)
def load_models():
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    reranker_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    llm_model_name = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
    llm_model = AutoModelForSeq2SeqLM.from_pretrained(llm_model_name)
    return embedding_model, reranker_model, tokenizer, llm_model


@lru_cache(maxsize=1)
def load_vector_db():
    client = chromadb.PersistentClient(path=DB_PATH)
    return (
        client.get_or_create_collection(name=BANKING_COLLECTION_NAME),
        client.get_or_create_collection(name=MEMORY_COLLECTION_NAME),
    )


def get_session(session_id=None):
    if not session_id:
        session_id = str(uuid.uuid4())
    SESSIONS.setdefault(session_id, {
        "chat_history": [],
        "current_topic": "",
        "current_intent": "general",
        "topic_history": [],
        "last_resolved_question": "",
    })
    return session_id, SESSIONS[session_id]


def is_follow_up_question(user_question):
    q = user_question.lower().strip()
    phrases = [
        "it", "that", "this", "they", "them", "those", "these", "for it",
        "about it", "for that", "about that", "how much", "what documents",
        "documents needed", "required documents", "explain more", "tell me more",
        "what about", "does it", "can it", "is it", "which is better",
        "better one", "how long", "what next", "then what",
    ]
    if any(re.search(rf"\b{re.escape(phrase)}\b", q) for phrase in phrases):
        return True
    intent = detect_question_intent(q)
    return len(q.split()) <= 7 and intent in {
        "documents", "fees", "interest", "tenure", "limits", "eligibility", "opening", "process"
    }


def build_recent_chat_history(session):
    text = ""
    for chat in session["chat_history"][-6:]:
        role = "User" if chat["role"] == "user" else "Assistant"
        text += f"{role}: {chat['message']}\n"
    return text


def build_search_query(user_question, session):
    q = user_question.lower().strip()
    current_topic = str(session.get("current_topic", "")).strip()
    if current_topic and is_follow_up_question(user_question):
        rewritten = f" {user_question} "
        for pronoun in ["it", "this", "that", "they", "them", "those", "these"]:
            rewritten = rewritten.lower().replace(f" {pronoun} ", f" {current_topic} ")
        rewritten = rewritten.strip()
        return f"{user_question} about {current_topic}" if rewritten == q else rewritten
    return user_question


def resolve_question_context(user_question, session):
    current_topic = str(session.get("current_topic", "")).strip()
    explicit_topic = extract_topic_from_question(user_question)
    if explicit_topic or not current_topic or not is_follow_up_question(user_question):
        return user_question

    resolved = f" {user_question.strip()} "
    for pronoun in ["it", "this", "that", "they", "them", "those", "these"]:
        resolved = re.sub(rf"\b{pronoun}\b", current_topic, resolved, flags=re.IGNORECASE)
    resolved = " ".join(resolved.split())
    if current_topic.lower() not in resolved.lower():
        resolved = f"{resolved} about {current_topic}"
    return resolved


def remember_conversation_context(session, topic="", intent="general", resolved_question=""):
    topic = normalize_topic_label(topic)
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


def detect_question_intent(question):
    q = question.lower().strip()
    intent_rules = [
        ("documents", ["document", "documents", "proof", "kyc", "required", "requirement", "need to carry", "needed"]),
        ("fees", ["fee", "fees", "charge", "charges", "cost", "minimum balance", "penalty"]),
        ("interest", ["interest rate", "rate of interest", "returns", "interest earned", "how much interest"]),
        ("tenure", ["how long", "tenure", "duration", "maturity period", "term period"]),
        ("limits", ["limit", "limits", "maximum amount", "minimum amount", "transaction limit", "withdrawal limit"]),
        ("eligibility", ["eligible", "eligibility", "who can", "can i", "allowed", "qualify"]),
        ("opening", ["open", "opening", "make", "create", "start", "set up", "setup", "apply", "register", "get an account", "get a card"]),
        ("process", ["how can", "how do", "how to", "steps", "procedure", "process", "way to", "ways to", "help me"]),
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
        return False
    q = str(question or "").lower()
    action_words = ["apply", "open", "make", "create", "start", "documents", "required", "needed", "how do", "how can", "steps", "process"]
    return intent in {"opening", "documents", "process"} or any(word in q for word in action_words)


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


def retrieve_banking_context(search_query, embedding_model, banking_collection, top_k=40):
    query_embedding = embedding_model.encode(search_query).tolist()
    results = banking_collection.query(query_embeddings=[query_embedding], n_results=top_k)
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]
    context = ""
    for i in range(len(documents)):
        metadata = metadatas[i]
        context += f"""
Result {i + 1}:
Section: {metadata.get('section', '')}
Question: {metadata.get('question', '')}
Answer: {metadata.get('answer', '')}
Distance: {distances[i] if i < len(distances) else ''}
"""
    return context, metadatas, distances


def search_tokens(text):
    stopwords = {
        "a", "an", "the", "is", "are", "was", "were", "be", "to", "of", "for",
        "and", "or", "in", "on", "at", "with", "my", "me", "i", "you", "your",
        "please", "tell", "about", "can", "could", "would", "do", "does",
    }
    return {
        token for token in re.findall(r"[a-z0-9]+", text.lower())
        if len(token) > 1 and token not in stopwords
    }


def candidate_key(metadata):
    return f"{metadata.get('question', '').strip().lower()}|{metadata.get('answer', '').strip().lower()}"


def get_cached_banking_metadatas(banking_collection):
    count = banking_collection.count()
    if BANKING_METADATA_CACHE["count"] != count:
        results = banking_collection.get(include=["metadatas"])
        BANKING_METADATA_CACHE["count"] = count
        BANKING_METADATA_CACHE["metadatas"] = [
            metadata for metadata in results.get("metadatas", []) if metadata
        ]
    return BANKING_METADATA_CACHE["metadatas"]


def lexical_candidate_score(query, metadata, topic="", intent="general"):
    question = str(metadata.get("question", ""))
    answer = str(metadata.get("answer", ""))
    section = str(metadata.get("section", ""))
    candidate_text = f"{question} {answer} {section}".lower()
    query_text = query.lower().strip()
    query_terms = search_tokens(query_text)
    candidate_terms = search_tokens(candidate_text)
    if not query_terms:
        return 0.0
    overlap = len(query_terms & candidate_terms) / len(query_terms)
    question_ratio = SequenceMatcher(None, query_text, question.lower()).ratio()
    phrase_bonus = 5.0 if query_text and query_text in candidate_text else 0.0
    topic_bonus = score_topic_match(topic, candidate_text)
    intent_bonus = score_intent_match(intent, candidate_text)
    return overlap * 7.0 + question_ratio * 3.0 + phrase_bonus + topic_bonus + intent_bonus


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
        if any(query.lower() in question for query in query_variants if len(query) > 4):
            item["methods"].add("exact")
        elif any(SequenceMatcher(None, query.lower(), question).ratio() >= 0.72 for query in query_variants):
            item["methods"].add("fuzzy")

    structured = build_structured_product_candidate(topic, intent)
    if structured:
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


def retrieve_chat_memory(search_query, embedding_model, memory_collection, top_k=1):
    if memory_collection.count() == 0:
        return ""
    query_embedding = embedding_model.encode(search_query).tolist()
    results = memory_collection.query(query_embeddings=[query_embedding], n_results=top_k)
    docs = results.get("documents", [[]])[0]
    return "".join(f"\nPast Conversation {i + 1}:\n{doc}\n" for i, doc in enumerate(docs))


def save_chat_memory(user_question, bot_answer, embedding_model, memory_collection):
    memory_text = f"\nUser: {user_question}\nAssistant: {bot_answer}\n"
    memory_collection.add(
        ids=[str(uuid.uuid4())],
        embeddings=[embedding_model.encode(memory_text).tolist()],
        documents=[memory_text],
        metadatas=[{"user_question": user_question, "bot_answer": bot_answer}],
    )


def score_intent_match(intent, candidate_text):
    text = candidate_text.lower()
    positive = {
        "opening": ["open", "opening", "apply", "register", "walk into", "nearest branch", "submit", "carry", "documents", "account opening"],
        "documents": ["documents", "proof", "kyc", "identity proof", "address proof", "photograph", "required", "carry"],
        "process": ["steps", "process", "procedure", "follow", "submit", "visit", "log in", "click", "fill"],
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
        section = ""
        question = ""
        answer = ""
        for line in result.splitlines():
            line = line.strip()
            if line.startswith("Section:"):
                section = line.replace("Section:", "").strip()
            elif line.startswith("Question:"):
                question = line.replace("Question:", "").strip()
            elif line.startswith("Answer:"):
                answer = line.replace("Answer:", "").strip()
        if answer:
            text = f"{question} {answer}"
            candidates.append({"section": section, "question": question, "answer": answer, "text": text})
    if not candidates:
        return "I found relevant banking information, but could not extract a clear answer."

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
        if re.search(rf"\b{re.escape(alias)}\b", q) and product not in matches:
            matches.append(product)
    return matches[:2] if len(matches) >= 2 else []


def compare_products(question):
    products = detect_compare_products(question)
    if len(products) < 2:
        return None
    features = ["Deposit", "Interest", "Best for", "Flexibility", "Typical use"]
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


def is_account_recommendation_question(question):
    q = question.lower()
    phrases = ["which account", "what account", "account should i open", "recommend account", "best account", "suitable account", "suggest account", "open for me"]
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
    translated_question = translate_question_for_search(message, language)
    search_question = normalize_banking_spelling(translated_question)
    resolved_question = resolve_question_context(search_question, session)
    intent = detect_question_intent(resolved_question)
    if intent == "general" and is_follow_up_question(search_question):
        intent = session.get("current_intent", "general")
    topic = extract_topic_from_question(resolved_question)
    if not topic and is_follow_up_question(search_question):
        topic = session.get("current_topic", "")
    session["chat_history"].append({"role": "user", "message": message})
    pending_flow_topic = (session.get("pending_flow") or {}).get("topic", "")
    suggested_questions = translate_suggested_questions(build_suggested_questions(topic, intent), language)

    comparison = compare_products(search_question)
    if comparison:
        reply = translate_answer(comparison["reply"], language)
        remember_conversation_context(session, comparison["topic"], "comparison", comparison["topic"])
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

    pending_reply = handle_pending_flow(search_question, session)
    if pending_reply:
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

    clarifying_reply = build_clarifying_question(search_question, topic, intent, session)
    if clarifying_reply:
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

    if is_account_recommendation_question(search_question):
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

    embedding_model, reranker_model, tokenizer, llm_model = load_models()
    banking_collection, _ = load_vector_db()

    rewritten_question = ""
    if question_needs_llm_rewrite(resolved_question, session):
        rewritten_question = rewrite_unclear_question(resolved_question, session, tokenizer, llm_model)
        if not rewritten_question:
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
    # Chroma chat memory is shared across users, so it must not influence answers.
    # Session-scoped recent history provides context without cross-account leakage.
    memory_context = ""
    recent_history = build_recent_chat_history(session)
    reply = generate_llm_answer(message, banking_context, memory_context, recent_history, reranker_model, tokenizer, llm_model, language, intent, search_query, topic)
    reply = translate_answer(reply, language)

    remember_conversation_context(session, topic, intent, resolved_question)
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
        "suggested_questions": suggested_questions,
        "recommendation_card": recommendation_card,
        "topic": topic,
        "rewritten_question": rewritten_question or None,
        "search_methods": used_search_methods,
    }
