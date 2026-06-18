from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer
from db import get_mysql_connection


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DB_PATH = str(DATA_DIR / "vector_db")
COLLECTION_NAME = "banking_knowledge_base"


print("Loading embedding model...")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

print("Connecting to ChromaDB...")
chroma_client = chromadb.PersistentClient(path=DB_PATH)

collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME
)

print("Connecting to MySQL...")
conn = get_mysql_connection()
cursor = conn.cursor(dictionary=True)

cursor.execute(
    """
    SELECT id, section, question, answer
    FROM banking_faqs
    WHERE needs_embedding = TRUE
    """
)

rows = cursor.fetchall()

print("Rows needing embeddings:", len(rows))

if len(rows) == 0:
    print("No new embeddings needed.")
else:
    ids = []
    documents = []
    metadatas = []
    embeddings = []

    for row in rows:
        mysql_id = row["id"]
        section = row["section"]
        question = row["question"]
        answer = row["answer"]

        document_text = f"""
Section: {section}
Question: {question}
Answer: {answer}
"""

        embedding = embedding_model.encode(document_text).tolist()

        chroma_id = f"mysql_{mysql_id}"

        ids.append(chroma_id)
        documents.append(document_text)
        embeddings.append(embedding)
        metadatas.append(
            {
                "mysql_id": mysql_id,
                "section": section,
                "question": question,
                "answer": answer
            }
        )

    print("Adding/updating embeddings in ChromaDB...")

    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas
    )

    print("Marking rows as embedded in MySQL...")

    row_ids = [row["id"] for row in rows]

    for row_id in row_ids:
        cursor.execute(
            """
            UPDATE banking_faqs
            SET needs_embedding = FALSE,
                embedding_created = TRUE
            WHERE id = %s
            """,
            (row_id,)
        )

    conn.commit()

    print("Embedding sync completed.")
    print("Total documents in ChromaDB:", collection.count())

cursor.close()
conn.close()