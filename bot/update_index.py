import os
import json
import time
import hashlib
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter


# -----------------------------
# CONFIG
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent

# источник новых документов (могут лежать в подпапках)
SOURCE_DIR = BASE_DIR / "source_docs"

# база знаний
KB_DIR = BASE_DIR / "knowledge_base"

ARTIFACTS_DIR = BASE_DIR / "artifacts"

CHROMA_DIR = ARTIFACTS_DIR / "chroma_db"
STATE_FILE = ARTIFACTS_DIR / "source_state.json"
LOG_FILE = ARTIFACTS_DIR / "update_log.jsonl"

COLLECTION_NAME = os.getenv("COLLECTION_NAME", "quantumforge_kb")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-base-en-v1.5")

MIN_WORDS = 100
MAX_WORDS = 300
OVERLAP_WORDS = 30

ROUGH_CHUNK_SIZE = 4000
ROUGH_CHUNK_OVERLAP = 0

# Настройки батчинга
EMBED_BATCH_SIZE = 32

# берем только файлы txt из source_docs
GLOBS = ["**/*.txt"]


def utc_now() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8", errors="ignore")


def split_by_words(
    text: str,
    min_words: int = MIN_WORDS,
    max_words: int = MAX_WORDS,
    overlap_words: int = OVERLAP_WORDS
) -> List[str]:
    words = text.split()
    if not words:
        return []

    chunks: List[str] = []
    step = max_words - overlap_words
    if step <= 0:
        step = max_words

    start = 0
    while start < len(words):
        end = min(start + max_words, len(words))
        chunk_words = words[start:end]

        if len(chunk_words) < min_words and chunks:
            chunks[-1] = chunks[-1] + " " + " ".join(chunk_words)
            break

        chunks.append(" ".join(chunk_words))

        if end >= len(words):
            break
        start += step

    return chunks


def chunk_text(text: str) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=ROUGH_CHUNK_SIZE,
        chunk_overlap=ROUGH_CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    rough = splitter.split_text(text)

    final: List[str] = []
    for rc in rough:
        rc = rc.strip()
        if not rc:
            continue
        wc = len(rc.split())
        if wc <= MAX_WORDS:
            final.append(rc)
        else:
            final.extend(split_by_words(rc))

    cleaned = []
    for ch in final:
        ch = " ".join(ch.split())
        if ch:
            cleaned.append(ch)
    return cleaned


def stable_chunk_id(kb_rel_path: str, file_hash: str, chunk_index: int) -> str:
    return f"{kb_rel_path}:{file_hash}:{chunk_index}"


def load_state() -> Dict[str, Any]:
    if not STATE_FILE.exists():
        return {"files": {}}
    with STATE_FILE.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_state(state: Dict[str, Any]) -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    with STATE_FILE.open("w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def log_event(event: Dict[str, Any]) -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")


def scan_source_files() -> List[Path]:
    files: List[Path] = []
    if not SOURCE_DIR.exists():
        return files
    for g in GLOBS:
        files.extend(SOURCE_DIR.glob(g))
    files = [p for p in files if p.is_file()]
    files.sort()
    return files


# Копирует файл из source_docs в knowledge_base, сохраняя структуру подпапок
def copy_to_knowledge_base(src_path: Path) -> Path:

    rel = src_path.relative_to(SOURCE_DIR)
    dst_path = KB_DIR / rel
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_path, dst_path)
    return dst_path


def get_collection() -> chromadb.Collection:
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False),
    )

    names = [c.name for c in client.list_collections()]
    if COLLECTION_NAME not in names:
        return client.create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    return client.get_collection(COLLECTION_NAME)


def embed_texts(model: SentenceTransformer, texts: List[str]) -> List[List[float]]:
    embs = model.encode(
        texts,
        batch_size=EMBED_BATCH_SIZE,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return embs.tolist()


def add_file_chunks_append_only(
    collection: chromadb.Collection,
    model: SentenceTransformer,
    kb_path: Path,
    file_hash: str,
) -> Tuple[int, List[str]]:
    
    text = read_text(kb_path)
    chunks = chunk_text(text)
    if not chunks:
        return 0, []

    kb_rel = str(kb_path.relative_to(KB_DIR)).replace("\\", "/")

    ids: List[str] = []
    docs: List[str] = []
    metas: List[Dict[str, Any]] = []

    ingested_at = utc_now()
    file_name = kb_path.name
    doc_title = kb_path.stem
    doc_type = kb_path.suffix.lower().lstrip(".") or "txt"

    for i, ch in enumerate(chunks):
        cid = stable_chunk_id(kb_rel, file_hash, i)
        ids.append(cid)
        docs.append(ch)
        metas.append({
            "source_path": str(kb_path),
            "kb_rel_path": kb_rel,
            "file_name": file_name,
            "doc_title": doc_title,
            "doc_type": doc_type,
            "chunk_index": i,
            "words_count": len(ch.split()),
            "ingested_at": ingested_at,
            "file_hash": file_hash,
            "source_kind": "knowledge_base",
        })

    embeddings = embed_texts(model, docs)

    if hasattr(collection, "upsert"):
        collection.upsert(ids=ids, documents=docs, metadatas=metas, embeddings=embeddings)
    else:
        try:
            collection.add(ids=ids, documents=docs, metadatas=metas, embeddings=embeddings)
        except Exception:
            collection.delete(ids=ids)
            collection.add(ids=ids, documents=docs, metadatas=metas, embeddings=embeddings)

    return len(ids), ids


def main():
    start_ts = time.time()
    run_started = utc_now()

    state = load_state()
    prev_files: Dict[str, Any] = state.get("files", {})

    files = scan_source_files()

    to_ingest: List[Tuple[str, Path, str]] = []
    for src_path in files:
        key = str(src_path)
        h = sha256_file(src_path)
        prev = prev_files.get(key)
        if (prev is None) or (prev.get("hash") != h):
            to_ingest.append((key, src_path, h))

    start_event = {
        "event": "update_start",
        "started_at": run_started,
        "source_dir": str(SOURCE_DIR),
        "knowledge_base_dir": str(KB_DIR),
        "files_total_in_source": len(files),
        "files_new_or_changed": len(to_ingest),
        "mode": "copy_to_kb_and_append_only_index",
    }
    log_event(start_event)
    print(json.dumps(start_event, ensure_ascii=False, indent=2))

    collection = get_collection()
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    errors: List[str] = []
    files_ingested = 0
    chunks_added = 0

    for key, src_path, h in to_ingest:
        try:
            # копируем в knowledge_base с той же структурой папок
            kb_path = copy_to_knowledge_base(src_path)

            # индексируем файл в knowledge_base
            added, chunk_ids = add_file_chunks_append_only(collection, model, kb_path, h)

            chunks_added += added
            files_ingested += 1

            prev_files[key] = {
                "hash": h,
                "mtime": src_path.stat().st_mtime,
                "kb_path": str(kb_path),
                "chunk_count": added,
                "updated_at": utc_now(),
                "last_chunk_id_sample": chunk_ids[:3],
            }
        except Exception as e:
            errors.append(f"{key}: {e}")

    state["files"] = prev_files
    save_state(state)

    run_finished = utc_now()
    elapsed = time.time() - start_ts

    finish_event = {
        "event": "update_finish",
        "started_at": run_started,
        "finished_at": run_finished,
        "elapsed_sec": round(elapsed, 2),
        "files_ingested": files_ingested,
        "chunks_added": chunks_added,
        "errors_count": len(errors),
        "errors": errors[:10],
        "index_path": str(CHROMA_DIR),
        "state_path": str(STATE_FILE),
        "model": EMBEDDING_MODEL_NAME,
        "collection": COLLECTION_NAME,
        "mode": "copy_to_kb_and_append_only_index",
        "note": "No deletions from index. Old versions remain",
    }
    log_event(finish_event)
    print(json.dumps(finish_event, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()