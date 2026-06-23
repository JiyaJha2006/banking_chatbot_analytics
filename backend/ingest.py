from __future__ import annotations

import os
import shutil
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

try:
    from .official_kb import load_official_kb_documents
except ImportError:
    from official_kb import load_official_kb_documents


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DB_PATH = str(DATA_DIR / "vector_db")
COLLECTION_NAME = "banking_knowledge_base"


def build_vector_database(reset=True):
    documents = load_official_kb_documents()
    if not documents:
        raise ValueError("No official knowledge-base documents were found in data/official_kb.")

    if reset and os.path.exists(DB_PATH):
        print("Deleting old vector database...")
        shutil.rmtree(DB_PATH)

    print("Loading embedding model...")
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    print("Connecting to ChromaDB...")
    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    ids = []
    chroma_documents = []
    metadatas = []
    embeddings = []

    print(f"Preparing {len(documents)} official knowledge-base chunks...")
    for item in documents:
        document_text = (
            f"Section: {item['section']}\n"
            f"Question: {item['question']}\n"
            f"Answer: {item['answer']}\n"
            f"Source: {item['source']}"
        )
        ids.append(item["id"])
        chroma_documents.append(document_text)
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

    print("Adding official documents to ChromaDB...")
    collection.upsert(
        ids=ids,
        documents=chroma_documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    print("Vector database created successfully!")
    print("Total documents stored:", collection.count())
    return collection


if __name__ == "__main__":
    build_vector_database(reset=True)
