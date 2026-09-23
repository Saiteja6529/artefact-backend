import os
import json

BASE_DIR = "../knowledge_base"
OUTPUT_DIR = "../indexes"

CHUNK_SIZE = 800
OVERLAP = 150

def chunk_text(text):
    tokens = re.findall(r"\S+\s*", text)
    chunks = []
    start = 0

    while start < len(tokens):
        chunk = ""
        i = start
        while i < len(tokens) and len(chunk) + len(tokens[i]) <= CHUNK_SIZE:
            chunk += tokens[i]
            i += 1

        if i == start and tokens[i]:
            chunk = tokens[i]
            i += 1

        chunks.append(chunk)

        if i >= len(tokens):
            break

        overlap = ""
        j = i
        while j > start and len(overlap) + len(tokens[j - 1]) <= OVERLAP:
            j -= 1
            overlap = tokens[j] + overlap

        start = j

    return chunks

for root, dirs, files in os.walk(BASE_DIR):
    if "cleaned" in root:
        parts = root.split(os.sep)

        civilization = parts[-3]
        book = parts[-2]

        output_path = os.path.join(OUTPUT_DIR, civilization, book)
        os.makedirs(output_path, exist_ok=True)

        all_chunks = []

        for file in files:
            if file.endswith(".txt"):
                path = os.path.join(root, file)

                with open(path, "r", encoding="utf-8") as f:
                    text = f.read()

                chunks = chunk_text(text)

                for i, chunk in enumerate(chunks):
                    record = {
                        "civilization": civilization,
                        "book": book,
                        "source_file": file,
                        "chunk_id": f"{book}_{i}",
                        "text": chunk
                    }
                    all_chunks.append(record)

        with open(os.path.join(output_path, "chunks.jsonl"), "w", encoding="utf-8") as f:
            for item in all_chunks:
                f.write(json.dumps(item) + "\n")

        print("Chunked:", civilization, book)

print("All books chunked.")
