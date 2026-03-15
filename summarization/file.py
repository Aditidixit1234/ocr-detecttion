import re
import json
import math

from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

from keybert import KeyBERT
import networkx as nx
import nltk
from nltk.tokenize import sent_tokenize

nltk.download('punkt')

print("Loading models...")

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

kw_model = KeyBERT()

print("Models ready.")


# ─────────────────────────────────
# Split long text
# ─────────────────────────────────

def split_into_chunks(text, max_words=400):

    sentences = re.split(r'(?<=[.!?]) +', text)

    chunks = []
    current_chunk = []
    word_count = 0

    for sentence in sentences:

        words = sentence.split()

        if word_count + len(words) > max_words:

            chunks.append(" ".join(current_chunk))
            current_chunk = []
            word_count = 0

        current_chunk.append(sentence)
        word_count += len(words)

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


# ─────────────────────────────────
# Summarization
# ─────────────────────────────────

def summarize_text(text):

    chunks = split_into_chunks(text)

    summaries = []

    for chunk in chunks:

        result = summarizer(
            chunk,
            max_length=120,
            min_length=30,
            do_sample=False
        )

        summaries.append(result[0]["summary_text"])

    return " ".join(summaries)


# ─────────────────────────────────
# Multi-length summary
# ─────────────────────────────────

def multi_length_summary(text):

    short = summarizer(text, max_length=40, min_length=10)[0]["summary_text"]

    medium = summarizer(text, max_length=80, min_length=20)[0]["summary_text"]

    long = summarizer(text, max_length=120, min_length=40)[0]["summary_text"]

    return {
        "short": short,
        "medium": medium,
        "detailed": long
    }


# ─────────────────────────────────
# Keywords
# ─────────────────────────────────

def extract_keywords(text):

    vectorizer = TfidfVectorizer(stop_words="english", max_features=1000)

    X = vectorizer.fit_transform([text])

    scores = zip(vectorizer.get_feature_names_out(), X.toarray()[0])

    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)

    keywords = [w for w, s in sorted_scores[:10]]

    return keywords


# ─────────────────────────────────
# Topics
# ─────────────────────────────────

def detect_topics(text):

    vectorizer = TfidfVectorizer(stop_words="english", max_features=500)

    X = vectorizer.fit_transform([text])

    scores = zip(vectorizer.get_feature_names_out(), X.toarray()[0])

    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)

    return [w for w, s in sorted_scores[:5]]


# ─────────────────────────────────
# Key Points
# ─────────────────────────────────

def extract_key_points(summary):

    sentences = re.split(r'(?<=[.!?]) +', summary)

    return [s.strip() for s in sentences if len(s.strip()) > 20]


# ─────────────────────────────────
# Reading Time
# ─────────────────────────────────

def reading_time(text):

    words = len(text.split())

    minutes = words / 200

    return round(minutes, 2)


# ─────────────────────────────────
# Heading generation
# ─────────────────────────────────

def generate_heading(summary):

    words = summary.split()

    heading = " ".join(words[:6])

    return heading.capitalize()


# ─────────────────────────────────
# Flashcards
# ─────────────────────────────────

def generate_flashcards(text):

    sentences = sent_tokenize(text)

    cards = []

    for s in sentences[:5]:

        words = s.split()

        if len(words) > 6:

            q = "What is meant by " + " ".join(words[:3]) + "?"

            cards.append({
                "question": q,
                "answer": s
            })

    return cards


# ─────────────────────────────────
# Concept extraction
# ─────────────────────────────────

def extract_concepts(text):

    keywords = kw_model.extract_keywords(text, top_n=6)

    return [k[0] for k in keywords]


# ─────────────────────────────────
# Difficulty detection
# ─────────────────────────────────

def detect_difficulty(text):

    sentences = sent_tokenize(text)

    words = len(text.split())

    avg = words / len(sentences)

    if avg < 12:
        return "Beginner"

    elif avg < 20:
        return "Intermediate"

    else:
        return "Advanced"


# ─────────────────────────────────
# Concept graph
# ─────────────────────────────────

def concept_graph(concepts):

    G = nx.Graph()

    for c in concepts:
        G.add_node(c)

    for i in range(len(concepts) - 1):
        G.add_edge(concepts[i], concepts[i + 1])

    return list(G.edges())


# ─────────────────────────────────
# Question generator
# ─────────────────────────────────

def generate_questions(text):

    sentences = sent_tokenize(text)

    questions = []

    for s in sentences[:5]:

        words = s.split()

        if len(words) > 5:

            q = "Explain: " + " ".join(words[:5]) + "?"

            questions.append(q)

    return questions


# ─────────────────────────────────
# Main pipeline
# ─────────────────────────────────

def process_text(text):

    summary = summarize_text(text)

    keywords = extract_keywords(text)

    topics = detect_topics(text)

    key_points = extract_key_points(summary)

    reading = reading_time(text)

    heading = generate_heading(summary)

    flashcards = generate_flashcards(text)

    concepts = extract_concepts(text)

    difficulty = detect_difficulty(text)

    graph = concept_graph(concepts)

    multi_summary = multi_length_summary(text)

    questions = generate_questions(text)

    result = {

        "generated_heading": heading,

        "difficulty_level": difficulty,

        "reading_time_minutes": reading,

        "summary": summary,

        "multi_length_summary": multi_summary,

        "keywords": keywords,

        "topics": topics,

        "key_points": key_points,

        "concepts": concepts,

        "concept_graph_edges": graph,

        "flashcards": flashcards,

        "generated_questions": questions
    }

    return result


# ─────────────────────────────────
# Test
# ─────────────────────────────────

if __name__ == "__main__":

    sample_text = """
    Machine learning helps computers learn patterns from data.
    It is widely used in image recognition, speech processing, and recommendation systems.
    Students collected a dataset of 200 handwritten images.
    The project uses EasyOCR for optical character recognition.
    The team will review progress in the next meeting on 20 Nov 2024.
    """

    output = process_text(sample_text)

    print(json.dumps(output, indent=2))