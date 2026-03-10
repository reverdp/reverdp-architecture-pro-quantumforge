import os
import re
import json
import time
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import ollama


# -----------------------------
# CONFIG
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
CHROMA_DIR = BASE_DIR / "artifacts" / "chroma_db"

GOLDEN_FILE = Path(os.getenv("GOLDEN_FILE", str(BASE_DIR / "golden_questions.jsonl")))
OUT_FILE = Path(os.getenv("EVAL_OUT_FILE", str(BASE_DIR / "artifacts" / "eval_results.jsonl")))

COLLECTION_NAME = os.getenv("COLLECTION_NAME", "quantumforge_kb")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-base-en-v1.5")

os.environ.setdefault("OLLAMA_HOST", "http://localhost:11434")

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

TOP_K = int(os.getenv("TOP_K", "5"))
FINAL_K = int(os.getenv("FINAL_K", "3"))
MAX_CHUNK_CHARS_IN_PROMPT = int(os.getenv("MAX_CHUNK_CHARS_IN_PROMPT", "1500"))
I_DONT_KNOW_DISTANCE = float(os.getenv("I_DONT_KNOW_DISTANCE", "0.52"))

FEW_SHOT_QUESTIONS = [
    "Where was Draxen Vornek born?",
    "What rank did Draxen Vornek achieve in the Republic Navy?",
]


@dataclass
class RetrievedChunk:
    chunk_id: str
    distance: float
    text: str
    meta: Dict[str, Any]


def utc_now() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def load_golden_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Golden set not found: {path}")
    items = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def get_collection() -> chromadb.Collection:
    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False),
    )
    return client.get_collection(COLLECTION_NAME)


def embed_query(embedder: SentenceTransformer, query: str) -> List[float]:
    emb = embedder.encode([BGE_QUERY_PREFIX + query], normalize_embeddings=True)[0]
    return emb.tolist()


def retrieve(collection: chromadb.Collection, embedder: SentenceTransformer, query: str, k: int = TOP_K) -> List[RetrievedChunk]:
    qemb = embed_query(embedder, query)
    res = collection.query(
        query_embeddings=[qemb],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )

    ids = res.get("ids", [[]])[0]
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]

    chunks: List[RetrievedChunk] = []
    for cid, doc, meta, dist in zip(ids, docs, metas, dists):
        chunks.append(RetrievedChunk(
            chunk_id=str(cid),
            distance=float(dist),
            text=str(doc),
            meta=dict(meta or {}),
        ))

    chunks.sort(key=lambda c: c.distance)
    return chunks


def answer_signal_filter(question: str, chunk_text: str) -> bool:
    q = question.lower()
    t = chunk_text.lower()
    if "born" in q or "birth" in q:
        return ("was born" in t) or (" born " in f" {t} ") or (" birth " in f" {t} ")
    return True


def pick_best_chunks(question: str, chunks: List[RetrievedChunk], k: int = FINAL_K) -> List[RetrievedChunk]:
    filtered = [c for c in chunks if answer_signal_filter(question, c.text)]
    if filtered:
        filtered.sort(key=lambda c: c.distance)
        return filtered[:k]
    return chunks[:k]


def build_context(chunks: List[RetrievedChunk]) -> str:
    blocks = []
    for i, c in enumerate(chunks, start=1):
        text = " ".join(c.text.split())[:MAX_CHUNK_CHARS_IN_PROMPT]
        blocks.append(f"### Source [{i}]\n{text}\n")
    return "\n".join(blocks)


def format_sources(chunks: List[RetrievedChunk]) -> str:
    lines = []
    for i, c in enumerate(chunks, start=1):
        src = c.meta.get("source_path") or c.meta.get("file_name") or "unknown"
        chunk_idx = c.meta.get("chunk_index", "?")
        lines.append(f"[{i}] {src} (chunk {chunk_idx})")
    return "\n".join(lines)


def extract_short_answer(question: str, chunks: List[RetrievedChunk]) -> str:
    q = question.lower()
    if "born" in q:
        for c in chunks:
            m = re.search(r"([^.]*\bwas born\b[^.]*\.)", c.text, flags=re.IGNORECASE)
            if m:
                return m.group(1).strip()
    if chunks:
        sents = re.split(r"(?<=[.!?])\s+", " ".join(chunks[0].text.split()))
        return " ".join(sents[:2]).strip()
    return "I don't know."


def build_few_shot_examples(collection: chromadb.Collection, embedder: SentenceTransformer) -> str:
    examples = []
    for q in FEW_SHOT_QUESTIONS[:2]:
        raw = retrieve(collection, embedder, q, k=TOP_K)
        best = pick_best_chunks(q, raw, k=3)
        a = extract_short_answer(q, best)
        srcs = format_sources(best)
        examples.append(
            "Q: " + q + "\n"
            "A: " + a + "\n"
            "Sources:\n" + srcs + "\n"
        )
    return "\n".join(examples)


def should_say_idk(best_chunks: List[RetrievedChunk]) -> bool:
    if not best_chunks:
        return True
    return best_chunks[0].distance > I_DONT_KNOW_DISTANCE


def build_prompts(user_query: str, context: str, few_shot: str, sources: str) -> tuple[str, str]:
    system_prompt = (
        "You are a helper who thinks first and then responds. Always write down your steps.\n"
        "Rules:\n"
        "1) Use ONLY the provided sources\n"
        "2) If the answer is not in sources, say: \"I don't know\".\n"
        "3) Write exactly 2-4 steps, briefly, without unnecessary fluff.\n"
        "4) After the steps, write the line: \"Answer: ...\"\n"
    )

    user_prompt = (
        "Few-shot examples (from the same knowledge base):\n"
        f"{few_shot}\n\n"
        f"User question: {user_query}\n\n"
        "Retrieved sources:\n"
        f"{context}\n"
        "Source list:\n"
        f"{sources}\n\n"
        "Output format:\n"
        "Step 1\n"
        "Step 2\n"
        "Step 3\n"
        "Answer: <final answer>\n"
        "- ...\n"
        "- ...\n"
    )
    return system_prompt, user_prompt


def call_ollama(system_prompt: str, user_prompt: str) -> str:
    resp = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        options={"temperature": 0.2},
    )
    return resp["message"]["content"].strip()


def extract_answer_only(full_text: str) -> str:
    m = re.search(r"(?im)^\s*answer\s*:\s*(.+?)\s*$", full_text)
    if m:
        return f"Answer: {m.group(1).strip()}"

    m2 = re.search(r"(?is)answer\s*:\s*(.+)", full_text)
    if m2:
        ans = m2.group(1).strip().splitlines()[0].strip()
        return f"Answer: {ans}"
    return full_text.strip()


def is_idk(answer_only: str) -> bool:
    return "i don't know" in answer_only.lower()


def keyword_ok(answer_only: str, keywords: List[str]) -> bool:
    if not keywords:
        return True
    low = answer_only.lower()
    return all(k.lower() in low for k in keywords)

def main():
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    golden = load_golden_jsonl(GOLDEN_FILE)
    collection = get_collection()
    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)

    total = 0
    correct = 0
    fp = 0
    fn = 0

    with OUT_FILE.open("w", encoding="utf-8") as out:
        for item in golden:
            total += 1
            qid = item["id"]
            query = item["query"]
            expect_has_answer = bool(item.get("expect_has_answer", True))
            must = item.get("must_contain_keywords", [])

            raw = retrieve(collection, embedder, query, k=TOP_K)
            best = pick_best_chunks(query, raw, k=FINAL_K)

            sources_meta = []
            for c in best:
                sources_meta.append({
                    "source_path": c.meta.get("source_path") or c.meta.get("file_name"),
                    "chunk_index": c.meta.get("chunk_index"),
                    "distance": c.distance,
                    "chunk_id": c.chunk_id,
                })

            if should_say_idk(best):
                full_answer = "Answer: I don't know."
            else:
                ctx = build_context(best)
                srcs = format_sources(best)
                few = build_few_shot_examples(collection, embedder)
                system_prompt, user_prompt = build_prompts(query, ctx, few, srcs)
                full_answer = call_ollama(system_prompt, user_prompt)

            answer_only = extract_answer_only(full_answer)

            predicted_has_answer = not is_idk(answer_only)
            kw_ok = keyword_ok(answer_only, must)

            if expect_has_answer:
                is_correct = predicted_has_answer and kw_ok
            else:
                is_correct = (not predicted_has_answer)

            if is_correct:
                correct += 1
            else:
                if expect_has_answer and (not predicted_has_answer):
                    fn += 1
                if (not expect_has_answer) and predicted_has_answer:
                    fp += 1

            record = {
                "timestamp": utc_now(),
                "id": qid,
                "query": query,
                "expect_has_answer": expect_has_answer,
                "predicted_has_answer": predicted_has_answer,
                "keyword_ok": kw_ok,
                "chunks_found": len(raw),
                "top_distance": best[0].distance if best else None,
                "sources": sources_meta,
                "full_answer": full_answer,
                "answer_only": answer_only,
                "answer_length": len(answer_only),
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")

    acc = (correct / total) if total else 0.0
    summary = {
        "total": total,
        "correct": correct,
        "accuracy": round(acc, 3),
        "false_positives": fp,
        "false_negatives": fn,
        "golden_file": str(GOLDEN_FILE),
        "out_file": str(OUT_FILE),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()