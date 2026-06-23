from __future__ import annotations

import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OFFICIAL_KB_DIR = DATA_DIR / "official_kb"


TEXT_REPLACEMENTS = {
    "\r": "\n",
    "\t": " ",
    "â€”": "-",
    "â€“": "-",
    "â†’": "->",
    "â‚¹": "Rs.",
    "â€™": "'",
    "â€œ": '"',
    "â€": '"',
    "Â": "",
    "\xa0": " ",
}


def clean_text(value):
    text = str(value or "")
    for old, new in TEXT_REPLACEMENTS.items():
        text = text.replace(old, new)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ ]{2,}", " ", text)
    return text.strip()


def slug_from_filename(path):
    name = path.stem
    name = re.sub(r"^kb_\d+_", "", name)
    return name.replace("_", " ").strip().title()


def parse_front_matter(text):
    if not text.startswith("---"):
        return {}, text
    match = re.match(r"^---\n(.*?)\n---\n?", text, flags=re.DOTALL)
    if not match:
        return {}, text
    metadata = {}
    for line in match.group(1).splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        metadata[key.strip()] = value.strip().strip('"')
    return metadata, text[match.end():]


def remove_bot_instruction(text):
    lines = []
    for line in text.splitlines():
        if "Bot instruction:" in line:
            continue
        lines.append(line)
    return "\n".join(lines)


def split_chunks(body):
    parts = re.split(r"(?m)^##\s+CHUNK\s+\d+\s*[-—â€”]?.*$", body)
    chunks = [part.strip() for part in parts if part.strip()]
    if len(chunks) > 1:
        return chunks[1:] if parts[0].lstrip().startswith("#") else chunks
    sections = re.split(r"(?m)^###\s+", body)
    if len(sections) > 1:
        return [f"### {section.strip()}" for section in sections[1:] if section.strip()]
    return [body.strip()] if body.strip() else []


def question_from_chunk(chunk, fallback):
    match = re.search(r"(?m)^###\s+(.+?)\s*$", chunk)
    if match:
        return clean_text(match.group(1))
    first_line = next((line.strip("# ").strip() for line in chunk.splitlines() if line.strip()), "")
    return clean_text(first_line or fallback)


def answer_from_chunk(chunk, question):
    answer = re.sub(r"(?m)^###\s+.+?\s*$", "", chunk, count=1).strip()
    answer = re.sub(r"(?m)^---\s*$", "", answer)
    return clean_text(answer or question)


def load_official_kb_documents(kb_dir=OFFICIAL_KB_DIR):
    documents = []
    for path in sorted(Path(kb_dir).glob("*.md")):
        raw_text = path.read_text(encoding="utf-8")
        metadata, body = parse_front_matter(raw_text)
        body = remove_bot_instruction(body)
        section = clean_text(metadata.get("topic") or slug_from_filename(path))
        source = clean_text(metadata.get("source") or "Official knowledge base")
        source_id = clean_text(metadata.get("id") or path.stem)

        for index, chunk in enumerate(split_chunks(body), 1):
            question = question_from_chunk(chunk, section)
            answer = answer_from_chunk(chunk, question)
            if len(question) < 5 or len(answer) < 20:
                continue
            documents.append(
                {
                    "id": f"{source_id}_{index}",
                    "section": section,
                    "question": question,
                    "answer": answer,
                    "source": source,
                    "source_file": path.name,
                }
            )
    return documents
