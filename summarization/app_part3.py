"""
Flask API — Part 3
Receives clean text from Part 2 (or directly), returns summary + structured JSON.
"""

from flask import Flask, request, jsonify
from summarizer import process_clean_text

app = Flask(__name__)


@app.route("/")
def home():
    return jsonify({"message": "Part 3 — Summarization API is running ✅"})


@app.route("/summarize", methods=["POST"])
def summarize():
    """
    POST /summarize
    Body (JSON): { "text": "your cleaned text here" }
    Returns: structured JSON with summary, dates, title, etc.
    """
    data = request.get_json()

    if not data or "text" not in data:
        return jsonify({"error": "Send JSON with a 'text' field"}), 400

    clean_text = data["text"].strip()

    if len(clean_text) < 20:
        return jsonify({"error": "Text is too short to summarize"}), 400

    result = process_clean_text(clean_text)
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, port=5001)
