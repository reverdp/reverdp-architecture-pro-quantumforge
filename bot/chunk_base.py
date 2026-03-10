import json
import uuid
from pathlib import Path
from datetime import datetime

from langchain_text_splitters import RecursiveCharacterTextSplitter


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "knowledge_base"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
CHUNKS_FILE = ARTIFACTS_DIR / "chunks.jsonl"
STATS_FILE = ARTIFACTS_DIR / "chunk_stats.txt"

MIN_WORDS = 100
MAX_WORDS = 300
OVERLAP_WORDS = 30

ROUGH_CHUNK_SIZE = 4000
ROUGH_CHUNK_OVERLAP = 0


def split_by_words(text: str, min_words: int = MIN_WORDS, max_words: int = MAX_WORDS, overlap_words: int = OVERLAP_WORDS):
    words = text.split()
    if not words:
        return []

    chunks = []
    step = max_words - overlap_words
    if step <= 0:
        step = max_words

    start = 0
    while start < len(words):
        end = min(start + max_words, len(words))
        chunk_words = words[start:end]

        # Если остался хвост и уже есть чанки - добавим к последнему
        if len(chunk_words) < min_words and chunks:
            chunks[-1] = chunks[-1] + " " + " ".join(chunk_words)
            break

        chunks.append(" ".join(chunk_words))

        if end >= len(words):
            break
        start += step

    return chunks


def chunk_text(text: str):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=ROUGH_CHUNK_SIZE,
        chunk_overlap=ROUGH_CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    rough_chunks = splitter.split_text(text)

    final_chunks = []
    for rc in rough_chunks:
        rc = rc.strip()
        if not rc:
            continue

        words_count = len(rc.split())
        if words_count <= MAX_WORDS:
            final_chunks.append(rc)
        else:
            final_chunks.extend(split_by_words(rc))

    cleaned = []
    for ch in final_chunks:
        ch = " ".join(ch.split())
        if ch:
            cleaned.append(ch)

    return cleaned


def main():
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Data directory not found: {DATA_DIR}")

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    txt_files = sorted(DATA_DIR.rglob("*.txt"))
    print(f"Found txt files: {len(txt_files)}")

    total_chunks = 0
    total_files = 0
    total_words = 0
    generated_at = datetime.utcnow().isoformat(timespec="seconds") + "Z"

    with CHUNKS_FILE.open("w", encoding="utf-8") as out:
        for file_path in txt_files:
            try:
                text = file_path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                text = file_path.read_text(encoding="utf-8", errors="ignore")

            chunks = chunk_text(text)
            total_files += 1
            total_chunks += len(chunks)

            print(f"\nFile: {file_path.name}")
            print(f"  Chunks created: {len(chunks)}")

            for idx, chunk_text_value in enumerate(chunks):
                words_count = len(chunk_text_value.split())
                total_words += words_count

                record = {
                    "chunk_id": str(uuid.uuid4()),
                    "source_path": str(file_path),
                    "file_name": file_path.name,
                    "doc_title": file_path.stem,
                    "doc_type": "txt",
                    "chunk_index": idx,
                    "words_count": words_count,
                    "text": chunk_text_value,
                }

                out.write(json.dumps(record, ensure_ascii=False) + "\n")

                # Превью первых 2 чанков каждого файла
                if idx < 2:
                    preview = chunk_text_value[:180].replace("\n", " ")
                    print(f"  - chunk_index={idx} | words={words_count} | preview={preview}...")

    avg_words = (total_words / total_chunks) if total_chunks else 0

    # Статистика
    with STATS_FILE.open("w", encoding="utf-8") as f:
        f.write(f"files_total={total_files}\n")
        f.write(f"chunks_total={total_chunks}\n")
        f.write(f"avg_words_per_chunk={avg_words:.2f}\n")

    print("\n=== DONE ===")
    print(f"Chunks saved to: {CHUNKS_FILE}")
    print(f"Stats saved to:  {STATS_FILE}")
    print(f"Total files:     {total_files}")
    print(f"Total chunks:    {total_chunks}")
    print(f"Avg words/chunk: {avg_words:.2f}")


if __name__ == "__main__":
    main()