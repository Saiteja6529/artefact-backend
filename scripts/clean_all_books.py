import os
import re

BASE_DIR = "../knowledge_base"

def clean_text(text):
    text = text.replace("\ufeff", "")
    text = text.replace("\r\n", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()

for root, dirs, files in os.walk(BASE_DIR):
    if "raw" in root:
        cleaned_path = root.replace("raw", "cleaned")
        os.makedirs(cleaned_path, exist_ok=True)

        for file in files:
            if file.endswith(".txt"):
                raw_file = os.path.join(root, file)
                cleaned_file = os.path.join(cleaned_path, file)

                with open(raw_file, "r", encoding="utf-8") as f:
                    text = f.read()

                cleaned_text = clean_text(text)

                with open(cleaned_file, "w", encoding="utf-8") as f:
                    f.write(cleaned_text)

                print("Cleaned:", raw_file)

print("All books cleaned.")
