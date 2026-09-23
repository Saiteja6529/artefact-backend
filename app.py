from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS
import sys
import os
from dotenv import load_dotenv
try:
    from deepgram import DeepgramClient
except ImportError:
    DeepgramClient = None

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BACKEND_DIR)

# Load the project-level .env no matter where python app.py is launched from.
load_dotenv(os.path.join(PROJECT_DIR, ".env"), override=True)

# Add scripts folder to Python path
sys.path.append(os.path.join(BACKEND_DIR, 'scripts'))

# Import your working RAG function
try:
    from rag_chat import CHAT_MODEL, get_rag_response, stream_rag_response
except ImportError:
    CHAT_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
    # Fallback if import fails (for testing)
    def get_rag_response(category, book, query, history=None):
        return f"RAG response for: category={category}, book={book}, query={query}, history={history}"

    def stream_rag_response(category, book, query, history=None):
        yield get_rag_response(category, book, query, history=history)

app = Flask(__name__)
CORS(app)


def get_json_payload():
    return request.get_json(silent=True) or {}


def error_response(message, status=500):
    return jsonify({"error": message}), status


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "ok",
        "groq_model": CHAT_MODEL,
    })

# ------------------------------------------------------------
# Endpoint 1: Return text answer (original)
# ------------------------------------------------------------
@app.route('/chat', methods=['POST'])
def chat():
    data = get_json_payload()
    user_query = (data.get('query') or '').strip()
    category = data.get('category', 'greek')
    book = data.get('book', 'iliad')
    history = data.get('history', [])

    if not user_query:
        return error_response("Query is required.", 400)

    try:
        answer = get_rag_response(category, book, user_query, history=history)
        return jsonify({"answer": answer})
    except Exception as exc:
        app.logger.exception("Chat error")
        return error_response(f"Chat failed: {exc}", 500)


@app.route('/chat/stream', methods=['POST'])
def chat_stream():
    data = get_json_payload()
    user_query = (data.get('query') or '').strip()
    category = data.get('category', 'greek')
    book = data.get('book', 'iliad')
    history = data.get('history', [])

    if not user_query:
        return Response("Query is required.", status=400, mimetype="text/plain; charset=utf-8")

    def generate():
        try:
            yield from stream_rag_response(category, book, user_query, history=history)
        except Exception as exc:
            app.logger.exception("Streaming chat error")
            yield f"Chat failed: {exc}"

    return Response(
        stream_with_context(generate()),
        mimetype="text/plain; charset=utf-8",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )

# ------------------------------------------------------------
# Endpoint 2: Return voice audio (UPDATED for SDK 6.x)
# ------------------------------------------------------------
@app.route('/synthesize', methods=['POST'])
def synthesize():
    data = get_json_payload()
    text_to_speak = data.get('text', '')
    if not text_to_speak:
        return Response("No text provided", status=400)
    if DeepgramClient is None:
        return Response("Voice synthesis dependency is not installed.", status=500)
    if not os.getenv("DEEPGRAM_API_KEY"):
        return Response("DEEPGRAM_API_KEY is not configured.", status=500)

    try:
        deepgram = DeepgramClient(api_key=os.getenv("DEEPGRAM_API_KEY"))
        
        # The generate method returns a generator that yields audio chunks
        audio_generator = deepgram.speak.v1.audio.generate(
            text=text_to_speak,
           model="aura-zeus-en"
        )
        
        # Collect all audio chunks into a single bytes object
        audio_bytes = b''.join(chunk for chunk in audio_generator)
        
        return Response(audio_bytes, mimetype="audio/mpeg")
    except Exception as e:
        print(f"Deepgram error: {e}")
        return Response(f"Voice synthesis failed: {str(e)}", status=500)

# ------------------------------------------------------------
# Run the server
# ------------------------------------------------------------
if __name__ == '__main__':
    print("ARTEFACT backend running on http://127.0.0.1:5000")
    print(f" - GROQ_MODEL     -> {CHAT_MODEL}")
    print(" - /chat          -> returns text answer")
    print(" - /synthesize    -> returns MP3 audio of any text")
    app.run(port=5000, debug=True)
