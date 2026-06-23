from __future__ import annotations

try:
    from .db import get_mysql_connection
    from .official_kb import load_official_kb_documents
except ImportError:
    from db import get_mysql_connection
    from official_kb import load_official_kb_documents


documents = load_official_kb_documents()
print("Official KB rows to insert:", len(documents))

if not documents:
    raise ValueError("No official knowledge-base documents were found in data/official_kb.")

conn = get_mysql_connection()
cursor = conn.cursor()

cursor.execute("DELETE FROM banking_faqs")

for item in documents:
    cursor.execute(
        """
        INSERT INTO banking_faqs
        (section, question, answer, source)
        VALUES (%s, %s, %s, %s)
        """,
        (
            item["section"],
            item["question"],
            item["answer"],
            f"official_kb:{item['source_file']}",
        )
    )

conn.commit()
cursor.close()
conn.close()

print("Official KB inserted into MySQL successfully.")
print("All rows are marked needs_embedding = TRUE by trigger/default.")
