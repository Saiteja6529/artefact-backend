import sys
import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

if len(sys.argv) < 4:
    print("Usage: python search_book.py <civilization> <book> <query>")
    sys.exit(1)

civilization = sys.argv[1]
book = sys.argv[2]
query = " ".join(sys.argv[3:])

BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

chunks_path = os.path.join(
    BASE_PATH,
    "indexes",
    civilization,
    book,
    "chunks.jsonl"
)

index_path = os.path.join(
    BASE_PATH,
    "indexes",
    civilization,
    book,
    "faiss.index"
)

# Load chunks
with open(chunks_path, "r", encoding="utf-8") as f:
    chunks = [json.loads(line) for line in f]

# Load index
index = faiss.read_index(index_path)

# Load model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Embed query
query_vector = model.encode([query]).astype("float32")

# Search
k = 5
distances, indices = index.search(query_vector, k)

print("\nTop Results:\n")

for idx in indices[0]:
    print("-" * 80)
    print(chunks[idx]["text"])
    print()
