import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer
import os
import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DB_PATH = str(DATA_DIR / "vector_db")
BANK_FAQS_PATH = DATA_DIR / "BankFAQs.csv"
COLLECTION_NAME = "banking_knowledge_base"


def clean_text(value):
    if pd.isna(value):
        return ""

    value = str(value)
    value = value.replace("\n", " ")
    value = value.replace("\r", " ")
    value = value.replace("â€™", "'")
    value = value.replace("â€œ", '"')
    value = value.replace("â€", '"')
    value = value.replace("â€“", "-")
    value = value.replace("Ã¢â¬â¢", "'")
    value = value.replace("Â", "")
    value = value.replace("\xa0", " ")
    value = value.replace("View more", "")
    value = value.replace("view more", "")
    value = value.replace("Read more", "")
    value = value.replace("read more", "")
    value = " ".join(value.split())

    return value.strip()


print("Loading dataset...")
try:
    df_main = pd.read_csv(BANK_FAQS_PATH, encoding="utf-8")
except UnicodeDecodeError:
    df_main = pd.read_csv(BANK_FAQS_PATH, encoding="latin1")

core_data = [
    {
        "Question": "What is a savings account?",
        "Answer": "A savings account is a basic bank account used to safely deposit money, earn interest, withdraw cash, make payments, and manage everyday personal banking needs.",
        "Class": "accounts"
    },
    {
        "Question": "What is a current account?",
        "Answer": "A current account is a bank account mainly used by businesses, shops, companies, and professionals for frequent deposits, withdrawals, and transactions.",
        "Class": "accounts"
    },
    {
        "Question": "What is a fixed deposit?",
        "Answer": "A fixed deposit is a banking product where money is deposited for a fixed period at a fixed interest rate. It usually gives higher interest than a savings account.",
        "Class": "accounts"
    },
    {
        "Question": "What is a recurring deposit?",
        "Answer": "A recurring deposit is a deposit account where a customer saves a fixed amount every month for a fixed period and earns interest on it.",
        "Class": "accounts"
    },
    {
        "Question": "What is a salary account?",
        "Answer": "A salary account is a savings account opened by an employer for employees to receive monthly salary payments.",
        "Class": "accounts"
    },
    {
        "Question": "What is an NRI account?",
        "Answer": "An NRI account is a bank account for Non-Resident Indians to manage income, savings, or foreign earnings in India.",
        "Class": "accounts"
    },
    {
        "Question": "What is a zero balance account?",
        "Answer": "A zero balance account is a savings account that does not require the customer to maintain a minimum balance.",
        "Class": "accounts"
    },
    {
        "Question": "What is a debit card?",
        "Answer": "A debit card is a payment card linked to a bank account that allows customers to withdraw cash from ATMs and make payments using available account balance.",
        "Class": "cards"
    },
    {
        "Question": "What is a credit card?",
        "Answer": "A credit card allows customers to borrow money up to a fixed limit for purchases and repay it later.",
        "Class": "cards"
    },
    {
        "Question": "What is KYC?",
        "Answer": "KYC means Know Your Customer. It is a process where banks verify a customer's identity and address using official documents like Aadhaar, PAN, passport, or voter ID.",
        "Class": "kyc"
    }
]

df_core = pd.DataFrame(core_data)

df = pd.concat([df_core, df_main], ignore_index=True)

print("Original dataset shape:", df.shape)
print("Original columns:", df.columns.tolist())

df.columns = [col.strip().lower() for col in df.columns]

possible_question_cols = [
    "question",
    "faq",
    "query",
    "title"
]

possible_answer_cols = [
    "answer",
    "response",
    "reply",
    "content",
    "text"
]

possible_section_cols = [
    "class",
    "section",
    "category",
    "topic"
]

question_col = None
answer_col = None
section_col = None

for col in df.columns:
    if col in possible_question_cols:
        question_col = col

    if col in possible_answer_cols:
        answer_col = col

    if col in possible_section_cols:
        section_col = col

if question_col is None:
    df["question"] = df[answer_col].apply(lambda x: clean_text(x)[:120])
    question_col = "question"

if answer_col is None:
    raise ValueError("No answer column found")

if section_col is None:
    df["section"] = "general"
    section_col = "section"

df = df[[question_col, answer_col, section_col]]
bad_answers = ["n/a", "na", "none", "null", "-", "--", "not available"]

df = df[~df["answer"].str.lower().isin(bad_answers)]

df = df.rename(
    columns={
        question_col: "question",
        answer_col: "answer",
        section_col: "section"
    }
)

df["question"] = df["question"].apply(clean_text)
df["answer"] = df["answer"].apply(clean_text)
df["section"] = df["section"].apply(clean_text)

df = df.dropna()
df = df[df["question"] != ""]
df = df[df["answer"] != ""]
df = df[df["question"].str.len() > 5]
df = df[df["answer"].str.len() > 15]
df = df.drop_duplicates(subset=["question", "answer"])
df = df.reset_index(drop=True)

print("Cleaned dataset shape:", df.shape)
print("Final columns:", df.columns.tolist())
print("Sample cleaned rows:")
print(df[["section", "question", "answer"]].head(10))

if os.path.exists(DB_PATH):
    print("Deleting old vector database...")
    shutil.rmtree(DB_PATH)

print("Loading embedding model...")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.PersistentClient(path=DB_PATH)

collection = client.get_or_create_collection(
    name=COLLECTION_NAME
)

ids = []
documents = []
metadatas = []
embeddings = []

print("Preparing documents...")

for index, row in df.iterrows():
    section = row["section"]
    question = row["question"]
    answer = row["answer"]

    document_text = f"""
Section: {section}
Question: {question}
Answer: {answer}
"""

    embedding = embedding_model.encode(document_text).tolist()

    ids.append(str(index))
    documents.append(document_text)
    embeddings.append(embedding)
    metadatas.append(
        {
            "section": section,
            "question": question,
            "answer": answer
        }
    )

print("Adding documents to ChromaDB...")

collection.add(
    ids=ids,
    documents=documents,
    embeddings=embeddings,
    metadatas=metadatas
)

print("Vector database created successfully!")
print("Total documents stored:", collection.count())