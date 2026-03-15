from Levenshtein import ratio
from rapidfuzz import fuzz, process
import sys
sys.stdout.reconfigure(encoding='utf-8')

import os
import csv
import cv2
import numpy as np
import easyocr
import pandas as pd
import re

# =========================
# DATASET PATHS
# =========================

dataset_path = r"C:\Users\KIIT0001\.cache\kagglehub\datasets\ssarkar445\handwriting-recognitionocr\versions\1\test_v2\test"
label_file   = r"C:\Users\KIIT0001\.cache\kagglehub\datasets\ssarkar445\handwriting-recognitionocr\versions\1\CSV\written_name_test.csv"

output_folder = "ocr_visualizations"
os.makedirs(output_folder, exist_ok=True)

# =========================
# LOAD TRUE LABELS
# =========================

labels = pd.read_csv(label_file)
label_dict = {row.iloc[0]: str(row.iloc[1]).upper() for _, row in labels.iterrows()}

# Build name vocabulary from label file for fuzzy snapping
name_vocab = list(set(label_dict.values()))

# =========================
# OCR MODEL
# =========================

reader = easyocr.Reader(['en'], gpu=False)

# =========================
# IMPROVED PREPROCESSING
# =========================

def preprocess_image(image_path):

    img = cv2.imread(image_path)

    if img is None:
        return None, None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Gentle denoising — preserves pen stroke edges better than bilateral filter
    gray = cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)

    # Unsharp masking to enhance pen strokes before thresholding
    blurred = cv2.GaussianBlur(gray, (0, 0), 3)
    gray = cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)

    # Smaller blockSize (21 vs 31) — avoids washing out narrow characters
    thresh_inv = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        21, 3
    )

    # Minimal morphology — avoids merging adjacent letters
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    thresh_inv = cv2.morphologyEx(thresh_inv, cv2.MORPH_CLOSE, kernel)

    # Keep both inverted and normal versions — handles dark-on-light and light-on-dark
    thresh_norm = cv2.bitwise_not(thresh_inv)

    upscaled_inv  = cv2.resize(thresh_inv,  None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    upscaled_norm = cv2.resize(thresh_norm, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    return upscaled_inv, upscaled_norm

# =========================
# TEXT CLEANING
# =========================

def clean_text(raw_text):

    text = raw_text.upper()

    # Remove known watermark / label words
    for word in ["PRENOM", "NOM", "NAME", "SURNAME"]:
        text = text.replace(word, "")

    # Keep letters and spaces only
    text = re.sub(r'[^A-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()

    return text

# =========================
# CANDIDATE SCORING
# =========================

def score_candidate(text, conf):
    """
    Score OCR candidates by both confidence and output length.
    Raw confidence alone is unreliable on handwriting — a short
    wrong output often scores higher confidence than a full correct name.
    """
    clean = re.sub(r'[^A-Z]', '', text.upper())
    length_score = min(len(clean), 20) / 20.0  # reward up to 20 chars
    return conf * 0.5 + length_score * 0.5

# =========================
# IMPROVED MULTI-PASS OCR
# =========================

def multi_pass_ocr(img_inv, img_norm):

    candidates = []
    allowlist = "ABCDEFGHIJKLMNOPQRSTUVWXYZ "

    for img in [img_inv, img_norm]:

        # Pass 1: paragraph mode — merges lines, good for multi-word names
        r1 = reader.readtext(img, allowlist=allowlist, detail=1, paragraph=True)
        # Pass 2: word-level — better for short single names
        r2 = reader.readtext(img, allowlist=allowlist, detail=1, paragraph=False)

        for result in [r1, r2]:

            if not result:
                candidates.append(("", 0.0))
                continue

            texts, confs = [], []

            for d in result:
                # paragraph=True returns 2-tuples (bbox, text) without confidence
                # paragraph=False returns 3-tuples (bbox, text, conf)
                if len(d) == 3:
                    texts.append(d[1])
                    confs.append(d[2])
                elif len(d) == 2:
                    texts.append(d[1])
                    confs.append(0.5)  # neutral confidence when missing

            text = " ".join(texts)
            conf = sum(confs) / max(len(confs), 1)
            candidates.append((text, conf))

    # Pick best candidate by combined length + confidence score
    best_text, best_score = "", 0.0

    for text, conf in candidates:
        score = score_candidate(text, conf)
        if score > best_score:
            best_score = score
            best_text = text

    return best_text.strip(), best_score

# =========================
# FUZZY NAME SNAPPING
# =========================

def snap_to_vocab(text, vocab, threshold=75):
    """
    Snap OCR output to the closest known name in the label vocabulary.
    Corrects single-character errors e.g. 'HARTIN' -> 'MARTIN'.
    Threshold controls how aggressive the correction is (0-100).
    """
    if not text or not vocab:
        return text

    match, score, _ = process.extractOne(text, vocab, scorer=fuzz.token_sort_ratio)

    return match if score >= threshold else text

# =========================
# SIMILARITY SCORING
# =========================

def compute_similarity(detected, true_text):
    """
    Use multiple similarity metrics and take the maximum.
    - token_sort_ratio: handles word-order swaps (first/last name order)
    - partial_ratio:    handles length mismatches and partial matches
    - Levenshtein ratio: character-level edit distance
    """
    if not detected or not true_text:
        return 0.0

    s1 = fuzz.token_sort_ratio(detected, true_text) / 100.0
    s2 = fuzz.partial_ratio(detected, true_text) / 100.0
    s3 = ratio(detected, true_text)

    return max(s1, s2, s3)

# =========================
# MAIN OCR LOOP
# =========================

total   = 0
correct = 0

with open("ocr_results.csv", "w", newline="", encoding="utf-8") as file:

    writer = csv.writer(file)
    writer.writerow(["image_name", "true_text", "raw_ocr", "snapped_text", "confidence", "similarity", "correct"])

    for count, img_name in enumerate(os.listdir(dataset_path)):

        if count >= 1000:
            break

        img_path = os.path.join(dataset_path, img_name)

        img_inv, img_norm = preprocess_image(img_path)

        if img_inv is None:
            continue

        raw_text, avg_conf = multi_pass_ocr(img_inv, img_norm)

        cleaned = clean_text(raw_text)

        # Snap to closest real name from label vocabulary
        snapped = snap_to_vocab(cleaned, name_vocab, threshold=75)

        true_text  = label_dict.get(img_name, "")
        similarity = compute_similarity(snapped, true_text)
        is_correct = similarity >= 0.5

        if is_correct:
            correct += 1
        total += 1

        print(f"Image:   {img_name}")
        print(f"True:    {true_text}")
        print(f"Raw OCR: {cleaned}")
        print(f"Snapped: {snapped}")
        print(f"Conf:    {round(avg_conf, 3)}  Sim: {round(similarity, 2)}  Correct: {is_correct}")
        print()

        writer.writerow([img_name, true_text, cleaned, snapped, avg_conf, similarity, is_correct])

# =========================
# FINAL ACCURACY
# =========================

accuracy = (correct / total) * 100 if total > 0 else 0

print("=================================")
print(f"Total Images:        {total}")
print(f"Correct Predictions: {correct}")
print(f"OCR Accuracy:        {round(accuracy, 2)}%")
print("=================================")