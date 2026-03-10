import json
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
CHUNKS_FILE = ARTIFACTS_DIR / "chunks.jsonl"
CHROMA_DIR = ARTIFACTS_DIR / "chroma_db"
INDEX_STATS_FILE = ARTIFACTS_DIR / "index_stats.txt"

COLLECTION_NAME = "quantumforge_kb"
EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5"

# Настройки батчинга
EMBED_BATCH_SIZE = 32

# query instruction для поисковых запросов
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def load_chunks_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Chunks file not found: {path}")

    chunks = []
    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_num}: {e}") from e

            required_fields = ["chunk_id", "text"]
            for field in required_fields:
                if field not in obj:
                    raise ValueError(f"Missing field '{field}' in line {line_num}")

            chunks.append(obj)

    return chunks


def chunked(iterable: List[Any], size: int):
    for i in range(0, len(iterable), size):
        yield iterable[i:i + size]


def build_embeddings(model: SentenceTransformer, texts: List[str]) -> List[List[float]]:
    embeddings = model.encode(
        texts,
        batch_size=EMBED_BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return embeddings.tolist()


def prepare_chroma_collection() -> chromadb.Collection:
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False),
    )

    existing = [c.name for c in client.list_collections()]
    if COLLECTION_NAME in existing:
        client.delete_collection(COLLECTION_NAME)

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    return collection


def add_chunks_to_chroma(collection: chromadb.Collection, chunks: List[Dict[str, Any]], model: SentenceTransformer):
    total = len(chunks)
    print(f"Adding {total} chunks to Chroma...")

    for batch_idx, batch in enumerate(chunked(chunks, EMBED_BATCH_SIZE), start=1):
        ids = []
        documents = []
        metadatas = []

        for item in batch:
            ids.append(str(item["chunk_id"]))
            documents.append(item["text"])

            meta = {
                "source_path": str(item.get("source_path", "")),
                "file_name": str(item.get("file_name", "")),
                "doc_title": str(item.get("doc_title", "")),
                "doc_type": str(item.get("doc_type", "")),
                "chunk_index": int(item.get("chunk_index", 0)),
                "words_count": int(item.get("words_count", 0)),
            }
            metadatas.append(meta)

        embeddings = build_embeddings(model, documents)

        collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )

        print(f"  Batch {batch_idx}: +{len(batch)} chunks")

    print("Chroma indexing completed.")


def save_index_stats(
    chunks_count: int,
    unique_files_count: int,
    elapsed_sec: float,
    model_name: str = EMBEDDING_MODEL_NAME,
):
    INDEX_STATS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with INDEX_STATS_FILE.open("w", encoding="utf-8") as f:
        f.write(f"built_at_utc={datetime.utcnow().isoformat(timespec='seconds')}Z\n")
        f.write(f"embedding_model={model_name}\n")
        f.write("embedding_dim=768\n")
        f.write(f"vector_db=ChromaDB\n")
        f.write(f"collection_name={COLLECTION_NAME}\n")
        f.write(f"chunks_total={chunks_count}\n")
        f.write(f"files_total={unique_files_count}\n")
        f.write(f"embed_batch_size={EMBED_BATCH_SIZE}\n")
        f.write(f"elapsed_sec={elapsed_sec:.2f}\n")
        f.write(f"chunks_source_file={CHUNKS_FILE}\n")
        f.write(f"chroma_path={CHROMA_DIR}\n")


def main():
    start = time.time()

    if not CHUNKS_FILE.exists():
        raise FileNotFoundError(
            f"Chunks file not found: {CHUNKS_FILE}\n"
            f"Run chunk_base.py first."
        )

    print("[1/5] Loading chunks.jsonl...")
    chunks = load_chunks_jsonl(CHUNKS_FILE)
    print(f"Loaded chunks: {len(chunks)}")

    unique_files = {c.get("source_path", "") for c in chunks}
    print(f"Unique source files: {len(unique_files)}")

    print("[2/5] Loading embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    print(f"Model loaded: {EMBEDDING_MODEL_NAME}")

    print("[3/5] Preparing Chroma collection...")
    collection = prepare_chroma_collection()
    print(f"Chroma path: {CHROMA_DIR}")
    print(f"Collection: {COLLECTION_NAME}")

    print("[4/5] Generating embeddings and building index...")
    add_chunks_to_chroma(collection, chunks, model)

    elapsed = time.time() - start
    save_index_stats(
        chunks_count=len(chunks),
        unique_files_count=len(unique_files),
        elapsed_sec=elapsed,
    )

    print("\n=== DONE ===")
    print(f"Index saved to: {CHROMA_DIR}")
    print(f"Stats saved to: {INDEX_STATS_FILE}")
    print(f"Total chunks: {len(chunks)}")
    print(f"Elapsed time: {elapsed:.2f} sec")


if __name__ == "__main__":
    main()