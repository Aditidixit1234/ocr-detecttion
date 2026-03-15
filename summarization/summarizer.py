"""
Part 3 — Summarization & Structured Output
Receives clean text from Part 2, summarizes it, and returns structured JSON.
"""

import re
import json
from transformers import pipeline

# Load BART summarization model once (takes ~30s first time, cached after)
print("Loading summarization model...")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
print("Model ready.")


# ─────────────────────────────────────────────
# STEP 1 — Summarize the clean text
# ─────────────────────────────────────────────

def summarize_text(clean_text: str) -> str:
    """
    Takes a long cleaned string and returns a shorter summary.
    BART works best on text between 60–1024 tokens.
    For very long text, we chunk it first.
    """
    # BART has a max input limit — chunk if text is too long
    MAX_CHARS = 1000
    chunks = [clean_text[i:i+MAX_CHARS] for i in range(0, len(clean_text), MAX_CHARS)]

    summaries = []
    for chunk in chunks:
        if len(chunk.strip()) < 30:   # skip tiny leftover chunks
            continue
        result = summarizer(
            chunk,
            max_length=120,
            min_length=30,
            do_sample=False
        )
        summaries.append(result[0]["summary_text"])

    return " ".join(summaries)


# ─────────────────────────────────────────────
# STEP 2 — Extract structured info from text
# ─────────────────────────────────────────────

def extract_structure(clean_text: str, summary: str) -> dict:
    """
    Tries to pull out common key fields from the text using regex.
    Works well for meeting notes, study notes, assignments.
    You can extend the patterns for your specific use case.
    """

    # Look for dates like: 20 Nov, Nov 20, 2024-11-20, 20/11/2024
    date_pattern = r'\b(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}|\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*(?:\s+\d{4})?|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2}(?:\s*,?\s*\d{4})?)\b'
    dates = re.findall(date_pattern, clean_text, re.IGNORECASE)

    # Look for numbers that might indicate dataset size, count, etc.
    number_pattern = r'\b(\d+)\s*(images?|samples?|records?|files?|items?|pages?|words?)\b'
    quantities = re.findall(number_pattern, clean_text, re.IGNORECASE)
    quantities_list = [{"count": int(n), "unit": u} for n, u in quantities]

    # Extract likely title: first non-empty line of the text
    lines = [l.strip() for l in clean_text.split("\n") if l.strip()]
    title = lines[0] if lines else "Untitled"
    # Clean up if title is too long
    if len(title) > 60:
        title = title[:57] + "..."

    # Look for deadline keywords
    deadline_pattern = r'(?:deadline|due|submit(?:ted)?|by)\s*[:\-]?\s*([^\.\n,]{3,30})'
    deadline_match = re.search(deadline_pattern, clean_text, re.IGNORECASE)
    deadline = deadline_match.group(1).strip() if deadline_match else None

    # Word count of original
    word_count = len(clean_text.split())

    return {
        "title": title,
        "word_count": word_count,
        "summary": summary,
        "dates_found": dates if dates else [],
        "deadline": deadline,
        "quantities": quantities_list if quantities_list else [],
        "raw_text_preview": clean_text[:200] + ("..." if len(clean_text) > 200 else "")
    }


# ─────────────────────────────────────────────
# STEP 3 — Main pipeline function
# ─────────────────────────────────────────────

def process_clean_text(clean_text: str) -> dict:
    """
    Full Part 3 pipeline:
      clean text → summary → structured JSON
    Returns a dict ready to be sent as JSON response.
    """
    print("Summarizing...")
    summary = summarize_text(clean_text)

    print("Extracting structure...")
    structured = extract_structure(clean_text, summary)

    return structured


# ─────────────────────────────────────────────
# Quick test 
# ─────────────────────────────────────────────

if __name__ == "__main__":
    sample_text = """
    Meeting notes: Discussed project deadlines for the handwriting recognition system.
    The dataset includes 200 images of handwritten English text collected from students.
    The team decided to use EasyOCR for the OCR stage and SymSpell for correction.
    Next meeting scheduled on 20 Nov 2024 to review progress.
    Final submission deadline is 30 Nov 2024.
    Each team member should complete their assigned module by 25 Nov.
    """

    result = process_clean_text(sample_text.strip())
    print("\n── Structured Output ──")
    print(json.dumps(result, indent=2))


