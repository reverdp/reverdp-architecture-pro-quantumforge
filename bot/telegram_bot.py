import os
import re
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any
import json
from datetime import datetime

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import ollama

from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters


# -----------------------------
# CONFIG
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
CHROMA_DIR = BASE_DIR / "artifacts" / "chroma_db"

COLLECTION_NAME = os.getenv("COLLECTION_NAME", "quantumforge_kb")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-base-en-v1.5")

LOGS_DIR = BASE_DIR / "artifacts"
QUERY_LOG_FILE = LOGS_DIR / "query_logs.jsonl"

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


def get_collection() -> chromadb.Collection:
    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False),
    )
    return client.get_collection(COLLECTION_NAME)


print("Loading Chroma collection...")
COLLECTION = get_collection()

print("Loading embedding model...")
EMBEDDER = SentenceTransformer(EMBEDDING_MODEL_NAME)

print("Telegram bot config:",
      f"\n- OLLAMA_HOST={os.environ.get('OLLAMA_HOST')}",
      f"\n- OLLAMA_MODEL={OLLAMA_MODEL}",
      f"\n- CHROMA_DIR={CHROMA_DIR}")


# -----------------------------
# RAG helpers
# -----------------------------
def embed_query(query: str) -> List[float]:
    emb = EMBEDDER.encode([BGE_QUERY_PREFIX + query], normalize_embeddings=True)[0]
    return emb.tolist()


def retrieve(query: str, k: int = TOP_K) -> List[RetrievedChunk]:
    qemb = embed_query(query)
    res = COLLECTION.query(
        query_embeddings=[qemb],
        n_results=k,
        include=["documents", "metadatas", "distances"],  # NOTE: no "ids" in include
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


def format_sources(chunks: List[RetrievedChunk]) -> str:
    lines = []
    for i, c in enumerate(chunks, start=1):
        src = c.meta.get("source_path") or c.meta.get("file_name") or "unknown"
        chunk_idx = c.meta.get("chunk_index", "?")
        lines.append(f"[{i}] {src} (chunk {chunk_idx})")
    return "\n".join(lines)


def build_context(chunks: List[RetrievedChunk]) -> str:
    blocks = []
    for i, c in enumerate(chunks, start=1):
        text = " ".join(c.text.split())[:MAX_CHUNK_CHARS_IN_PROMPT]
        blocks.append(f"### Source [{i}]\n{text}\n")
    return "\n".join(blocks)


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


def build_few_shot_examples() -> str:
    examples = []
    for q in FEW_SHOT_QUESTIONS[:2]:
        raw = retrieve(q, k=TOP_K)
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


def call_ollama(system_prompt: str, user_prompt: str) -> str:
    resp = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        options={
        "temperature": 0.2,
    },
    )
    return resp["message"]["content"].strip()


def build_prompts(user_query: str, context: str, few_shot: str, sources: str):
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

def utc_now():
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"

def append_query_log(record: Dict[str, Any]):
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    with QUERY_LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

def is_successful_answer(answer: str) -> bool:
    return "i don't know" not in answer.lower()

def rag_answer(query: str) -> str:
    raw = retrieve(query, k=TOP_K)
    best = pick_best_chunks(query, raw, k=FINAL_K)

    if should_say_idk(best):
        return (
            "Answer: I don't know.\n"
        )

    context = build_context(best)
    sources = format_sources(best)
    few_shot = build_few_shot_examples()

    system_prompt, user_prompt = build_prompts(query, context, few_shot, sources)
    return call_ollama(system_prompt, user_prompt)

def rag_answer(query: str):
    raw = retrieve(query, k=TOP_K)
    best = pick_best_chunks(query, raw, k=FINAL_K)

    if should_say_idk(best):
        return "Answer: I don't know.\n", best, len(raw)

    context = build_context(best)
    sources = format_sources(best)
    few_shot = build_few_shot_examples()

    system_prompt, user_prompt = build_prompts(query, context, few_shot, sources)
    answer = call_ollama(system_prompt, user_prompt)
    return answer, best, len(raw)

# -----------------------------
# Telegram handlers
# -----------------------------
async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Hi! Send me a question and I'll answer using the knowledge base.\n"
        "Commands:\n"
        "/help - tips"
    )


async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Tips:\n"
        "- Ask specific questions.\n"
        "- If the info isn't in the sources, I'll answer: I don't know.\n"
        "Example:\n"
        "Where was Draxen Vornek born?"
    )


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = (update.message.text or "").strip()
    if not query:
        return

    # show typing...
    await update.message.chat.send_action(action=ChatAction.TYPING)

    try:
        answer, best_chunks, chunks_found_total = rag_answer(query)
        sources_for_log = []
        for c in best_chunks:
            sources_for_log.append({
                "source_path": c.meta.get("source_path") or c.meta.get("file_name"),
                "chunk_index": c.meta.get("chunk_index"),
                "distance": c.distance,
                "chunk_id": c.chunk_id,
            })

        record = {
            "timestamp": utc_now(),
            "query": query,
            "chunks_found": chunks_found_total,
            "top_distance": best_chunks[0].distance if best_chunks else None,
            "answer_length": len(answer),
            "success": is_successful_answer(answer),
            "sources": sources_for_log,
        }

        append_query_log(record)

        # Telegram has message length limits
        if len(answer) > 3500:
            answer = answer[:3500] + "\n\n(Truncated)"
        await update.message.reply_text(answer)
    except Exception as e:
        await update.message.reply_text(f"[ERROR] {e}")


def main():
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        token = "8708530778:AAHL-H8-0ON2ruGX234TyabT2IK2C35MfoQ"

    app = Application.builder().token(token).build()

    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    print("Telegram bot started. Press Ctrl+C to stop")
    app.run_polling()


if __name__ == "__main__":
    main()