from pathlib import Path
import pandas as pd
from db import get_mysql_connection

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
BANK_FAQS_PATH = DATA_DIR / "BankFAQs.csv"


def clean_text(value):
    if pd.isna(value):
        return ""

    value = str(value)

    replacements = {
        "\n": " ",
        "\r": " ",
        "\t": " ",
        "â€™": "'",
        "Ã¢â¬â¢": "'",
        "â€œ": '"',
        "â€": '"',
        "â€“": "-",
        "Â": "",
        "\xa0": " ",
        "View more": "",
        "view more": ""
    }

    for old, new in replacements.items():
        value = value.replace(old, new)

    return " ".join(value.split()).strip()


print("Loading BankFAQs.csv...")

try:
    df = pd.read_csv(BANK_FAQS_PATH, encoding="utf-8")
except UnicodeDecodeError:
    df = pd.read_csv(BANK_FAQS_PATH, encoding="latin1")

df.columns = [col.strip().lower() for col in df.columns]

df = df.rename(
    columns={
        "class": "section"
    }
)

df["question"] = df["question"].apply(clean_text)
df["answer"] = df["answer"].apply(clean_text)
df["section"] = df["section"].apply(clean_text)

df = df[df["question"] != ""]
df = df[df["answer"] != ""]
df = df.drop_duplicates(subset=["question", "answer"])
df = df.reset_index(drop=True)

print("Rows to insert:", len(df))

conn = get_mysql_connection()
cursor = conn.cursor()

cursor.execute("DELETE FROM banking_faqs")

for _, row in df.iterrows():
    cursor.execute(
        """
        INSERT INTO banking_faqs
        (section, question, answer, source)
        VALUES (%s, %s, %s, %s)
        """,
        (
            row["section"],
            row["question"],
            row["answer"],
            "BankFAQs"
        )
    )

conn.commit()
cursor.close()
conn.close()

print("Data inserted into MySQL successfully.")
print("All rows are marked needs_embedding = TRUE by trigger/default.")