<<<<<<< HEAD
# artefact-backend
=======
# Artefact AI – Backend

This is the backend service for the **Artefact AI RAG Chatbot**.  
It handles document retrieval, vector search, and LLM response generation.

---

## Architecture

User Query
↓
Flask API
↓
RAG Pipeline
↓
FAISS Vector Search
↓
Context Retrieval
↓
Groq LLM (Llama 3.1)
↓
Final Answer

---

## Technologies Used

- Python
- Flask
- FAISS (Vector Search)
- Sentence Transformers
- Groq API (Llama 3.1)
- dotenv

---

## Project Structure

```
backend
│
├── scripts
│   ├── clean_all_books.py
│   ├── chunk_all_books.py
│   ├── embed_books.py
│   ├── rag_chat.py
│   └── search_book.py
│
├── indexes
│
├── knowledge_base
│
├── app.py
├── requirements.txt
└── .env.example
```

---

## Installation

Clone the repository:

```
git clone https://github.com/yourname/artefact-backend.git
cd artefact-backend
```

Create virtual environment:

```
python -m venv .venv
```

Activate:

Windows

```
.venv\Scripts\activate
```

Install dependencies:

```
pip install -r requirements.txt
```

---

## Environment Variables

Create `.env` file:

```
GROQ_API_KEY=your_api_key_here
```

---

## Running the Server

```
python app.py
```

Server will start at:

```
http://127.0.0.1:5000
```

---

## API Endpoint

POST `/chat`

Example request:

```
{
 "query": "Who is Achilles?",
 "category": "greek",
 "book": "iliad"
}
```

Response:

```
{
 "answer": "Achilles is the greatest warrior in the Greek army..."
}
```

---

## RAG Pipeline

1. Clean documents
2. Chunk text
3. Generate embeddings
4. Store vectors in FAISS
5. Retrieve relevant chunks
6. Send context to LLM
7. Generate answer

---

## Author
>>>>>>> bc72f83 (Initial backend commit)
