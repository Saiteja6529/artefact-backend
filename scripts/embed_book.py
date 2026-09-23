import sys
import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# ---- Arguments ----
if len(sys.argv) != 3:
    print("Usage: python embed_book.py <civilization> <book>")
    sys.exit(1)

civilization = sys.argv[1]
book = sys.argv[2]

# ---- Base Path ----
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---- Paths ----
chunks_path = os.path.join(
    BASE_PATH,
    "indexes",
    civilization,
    book,
    "chunks.jsonl"
)

index_output_path = os.path.join(
    BASE_PATH,
    "indexes",
    civilization,
    book,
    "faiss.index"
)

# ---- Check file exists ----
if not os.path.exists(chunks_path):
    print(f"ERROR: chunks file not found at:\n{chunks_path}")
    sys.exit(1)

# ---- Load JSONL properly ----
chunks = []
with open(chunks_path, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            chunks.append(json.loads(line))

if not chunks:
    print("ERROR: chunks file is empty.")
    sys.exit(1)

texts = [c["text"] for c in chunks]

print(f"Embedding {civilization} - {book}")
print(f"Total chunks: {len(texts)}")

# ---- Load model ----
model = SentenceTransformer("all-MiniLM-L6-v2")

# ---- Embed ----
embeddings = model.encode(
    texts,
    batch_size=32,
    show_progress_bar=True,
    convert_to_numpy=True
)

embeddings = embeddings.astype("float32")

# ---- Create FAISS index ----
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# ---- Save index ----
faiss.write_index(index, index_output_path)

print("Embedding completed successfully.")
print(f"Index saved at: {index_output_path}")
