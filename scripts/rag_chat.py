import json
import os
import re
import sys
from functools import lru_cache

from dotenv import load_dotenv
from groq import Groq

try:
    import faiss
except ImportError:
    faiss = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.dirname(SCRIPT_DIR)
PROJECT_DIR = os.path.dirname(BACKEND_DIR)

load_dotenv(os.path.join(PROJECT_DIR, ".env"), override=True)

api_key = os.getenv("GROQ_API_KEY")
client = None
embedding_model = None

# Change this line:
CHAT_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
REWRITE_MODEL = os.getenv("GROQ_REWRITE_MODEL", CHAT_MODEL)
MAX_HISTORY_MESSAGES = int(os.getenv("ARTEFACT_MAX_HISTORY_MESSAGES", "12"))
MAX_HISTORY_CHARS = int(os.getenv("ARTEFACT_MAX_HISTORY_CHARS", "9000"))
MAX_CONTEXT_CHARS = int(os.getenv("ARTEFACT_MAX_CONTEXT_CHARS", "14000"))
RETRIEVAL_K = int(os.getenv("ARTEFACT_RETRIEVAL_K", "5"))
USE_EMBEDDINGS = os.getenv("ARTEFACT_USE_EMBEDDINGS", "false").lower() == "true"

FOLLOW_UP_HINTS = (
    " he ",
    " she ",
    " it ",
    " they ",
    " this ",
    " that ",
    " these ",
    " those ",
    " his ",
    " her ",
    " their ",
    " them ",
    " then ",
    " next ",
    " previous ",
    " earlier ",
    " continue ",
    " elaborate ",
    " explain more ",
)


def _get_groq_client():
    global client
    if client is None:
        if not api_key:
            raise RuntimeError("GROQ_API_KEY is not configured.")
        client = Groq(api_key=api_key)
    return client


def _get_embedding_model():
    global embedding_model
    if SentenceTransformer is None:
        raise RuntimeError("sentence-transformers is not installed.")
    if embedding_model is None:
        embedding_model = SentenceTransformer("all-MiniLM-L6-v2", local_files_only=True)
    return embedding_model


def _tokenize(text):
    stop_words = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "how",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "the",
        "to",
        "was",
        "what",
        "when",
        "where",
        "who",
        "why",
        "with",
    }
    return {
        word
        for word in re.findall(r"[a-z0-9]+", text.lower())
        if len(word) > 2 and word not in stop_words
    }


def _fallback_retrieved_context(chunks, retrieval_query):
    query_terms = _tokenize(retrieval_query)
    if not query_terms:
        return _clip_text("\n\n".join(chunk.get("text", "") for chunk in chunks[:RETRIEVAL_K]), MAX_CONTEXT_CHARS)

    scored_chunks = []
    for position, chunk in enumerate(chunks):
        text = chunk.get("text", "").strip()
        if not text:
            continue
        chunk_terms = _tokenize(text)
        score = len(query_terms & chunk_terms)
        if score:
            scored_chunks.append((score, position, text))

    if not scored_chunks:
        return "No matching passages were retrieved from the selected book."

    scored_chunks.sort(key=lambda item: (-item[0], item[1]))
    context = "\n\n".join(text for _, _, text in scored_chunks[:RETRIEVAL_K])
    return _clip_text(context, MAX_CONTEXT_CHARS)


def _clip_text(text, limit):
    if not text:
        return ""
    text = " ".join(str(text).split())
    if len(text) <= limit:
        return text
    return f"{text[: limit - 3].rstrip()}..."


def _normalize_history(history):
    normalized = []
    for item in history or []:
        sender = item.get("sender") or item.get("role") or "user"
        text = _clip_text(item.get("text") or item.get("content") or "", 1200)
        if not text:
            continue
        role = "assistant" if sender in {"assistant", "bot"} else "user"
        normalized.append({"role": role, "text": text})
    return normalized


def _looks_like_follow_up(question):
    normalized = f" {question.strip().lower()} "
    word_count = len(re.findall(r"\w+", normalized))
    starts_with_follow_up = normalized.strip().startswith(
        ("and ", "what about", "how about", "why", "how", "then", "after that", "continue")
    )
    return word_count <= 10 or starts_with_follow_up or any(hint in normalized for hint in FOLLOW_UP_HINTS)


def _format_history(history):
    if not history:
        return "No earlier messages in this conversation."

    recent_history = history[-MAX_HISTORY_MESSAGES:]
    lines = []
    total_chars = 0

    for message in reversed(recent_history):
        line = f"{message['role'].title()}: {message['text']}"
        if total_chars + len(line) > MAX_HISTORY_CHARS:
            break
        lines.append(line)
        total_chars += len(line)

    return "\n".join(reversed(lines)) if lines else "No earlier messages in this conversation."


def _rewrite_follow_up_question(civilization, book, history, question):
    if not history or not _looks_like_follow_up(question):
        return question

    memory = _format_history(history)
    rewrite_prompt = f"""
Rewrite the user's latest question into a standalone query using only the conversation memory below.
Do not answer the question.
Do not add new facts.
Keep names, places, and events consistent with the conversation.
If the latest question is already standalone, return it unchanged.

Conversation scope:
- Civilization: {civilization}
- Book: {book}

Conversation memory:
{memory}

Latest user question:
{question}
""".strip()

    try:
        response = _get_groq_client().chat.completions.create(
            model=REWRITE_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You rewrite ambiguous follow-up questions into standalone queries without inventing details.",
                },
                {"role": "user", "content": rewrite_prompt},
            ],
            temperature=0.0,
        )
        rewritten = response.choices[0].message.content.strip()
        return rewritten or question
    except Exception:
        return question


@lru_cache(maxsize=16)
def _load_book_data(civilization, book):
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    chunks_path = os.path.join(base_path, "indexes", civilization, book, "chunks.jsonl")
    index_path = os.path.join(base_path, "indexes", civilization, book, "faiss.index")

    if not os.path.exists(chunks_path):
        return [], None

    with open(chunks_path, "r", encoding="utf-8") as file:
        chunks = [json.loads(line) for line in file]

    if not USE_EMBEDDINGS:
        return chunks, None

    if faiss is None or not os.path.exists(index_path):
        return chunks, None

    return chunks, faiss.read_index(index_path)


def _get_retrieved_context(civilization, book, retrieval_query):
    chunks, index = _load_book_data(civilization, book)
    if not chunks:
        return "No local documents found for this specific book."

    if not USE_EMBEDDINGS or index is None:
        return _fallback_retrieved_context(chunks, retrieval_query)

    try:
        query_vector = _get_embedding_model().encode([retrieval_query]).astype("float32")
        k = min(RETRIEVAL_K, len(chunks))
        _, indices = index.search(query_vector, k)
    except Exception:
        return _fallback_retrieved_context(chunks, retrieval_query)

    selected_chunks = []
    for idx in indices[0]:
        if idx < 0 or idx >= len(chunks):
            continue
        text = chunks[idx].get("text", "").strip()
        if text:
            selected_chunks.append(text)

    context = "\n\n".join(selected_chunks)
    return _clip_text(context, MAX_CONTEXT_CHARS) or "No matching passages were retrieved from the selected book."


def _build_prompt(civilization, book, question, history=None):
    normalized_history = _normalize_history(history)
    standalone_question = _rewrite_follow_up_question(civilization, book, normalized_history, question)
    retrieval_query = standalone_question
    memory = _format_history(normalized_history)
    context = _get_retrieved_context(civilization, book, retrieval_query)

    return f"""
You are Artefact, the mythology guide for a single active conversation.

Conversation rules:
1. Treat the supplied conversation memory as the only memory for this answer.
2. Use the retrieved book context as the highest-priority source for claims about the selected civilization or book.
3. If the latest question refers to "it", "that", "him", "what happened next", or similar follow-up wording, resolve it from the conversation memory before answering.
4. Earlier assistant replies are helpful context but are not authoritative if they conflict with the retrieved book context.
5. If the evidence is weak or missing, say that plainly and ask one short clarifying question instead of guessing.
6. Do not invent characters, plot points, verses, citations, or timelines that are not supported by the memory or retrieved context.
7. Keep the answer focused on the user's latest question.

Selected scope:
- Civilization: {civilization}
- Book: {book}

Conversation memory:
{memory}

Standalone interpretation of the latest question:
{standalone_question}

Retrieved book context:
{context}

Latest user question:
{question}
""".strip()


def _get_chat_messages(prompt):
    return [
        {
            "role": "system",
            "content": "You are Artefact, a grounded assistant for mythology and literature. Prefer retrieved evidence over speculation.",
        },
        {"role": "user", "content": prompt},
    ]


def get_rag_response(civilization, book, question, history=None):
    prompt = _build_prompt(civilization, book, question, history)

    response = _get_groq_client().chat.completions.create(
        model=CHAT_MODEL,
        messages=_get_chat_messages(prompt),
        temperature=0.2,
    )

    return response.choices[0].message.content


def stream_rag_response(civilization, book, question, history=None):
    prompt = _build_prompt(civilization, book, question, history)

    stream = _get_groq_client().chat.completions.create(
        model=CHAT_MODEL,
        messages=_get_chat_messages(prompt),
        temperature=0.2,
        stream=True,
    )

    for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        if delta and delta.content:
            yield delta.content

if __name__ == "__main__":
    if len(sys.argv) >= 4:
        civ = sys.argv[1]
        bk = sys.argv[2]
        qs = " ".join(sys.argv[3:])
        print(get_rag_response(civ, bk, qs))
