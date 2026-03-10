from pathlib import Path

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent
CHROMA_DIR = BASE_DIR / "artifacts" / "chroma_db"

COLLECTION_NAME = "quantumforge_kb"
EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5"
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def main():
    # 1 открыть существующий индекс
    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False),
    )
    collection = client.get_collection(COLLECTION_NAME)

    # 2 модель для embedding запроса
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    # 3 запрос
    query = "Where was Draxen Vornek born?"
    query_embedding = model.encode(
        [BGE_QUERY_PREFIX + query],
        normalize_embeddings=True
    )[0].tolist()

    # 4 получить top-k чанков
    k = 3
    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )

    print("=== QUERY ===")
    print(query)

    print("\n=== TOP RESULTS ===")
    ids = result["ids"][0]
    docs = result["documents"][0]
    metas = result["metadatas"][0]
    dists = result["distances"][0]

    for rank, (rid, doc, meta, dist) in enumerate(zip(ids, docs, metas, dists), start=1):
        preview = " ".join(doc.split())[:350]
        print(f"\n--- Result #{rank} ---")
        print(f"id: {rid}")
        print(f"distance: {dist:.4f}")
        print(f"source_path: {meta.get('source_path')}")
        print(f"file_name: {meta.get('file_name')}")
        print(f"doc_title: {meta.get('doc_title')}")
        print(f"chunk_index: {meta.get('chunk_index')}")
        print(f"words_count: {meta.get('words_count')}")
        print(f"text_preview: {preview}...")


if __name__ == "__main__":
    main()