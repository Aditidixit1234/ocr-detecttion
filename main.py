import os
import json
import argparse
import shutil

# Your existing modules
from src.ocr import OCRSystem
from src.nlp_correct import TextCorrector
from src.summarizer import InsightsExtractor

# FastAPI imports
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware


# -------------------------------
# CORE FUNCTION (MODIFIED)
# -------------------------------
def process_handwriting(image_path):
    # Initialize Pipeline
    ocr = OCRSystem()
    nlp = TextCorrector()
    summ = InsightsExtractor()

    # Step 1: OCR
    raw_text, confidence_scores = ocr.process_image(image_path)

    # Calculate confidence
    avg_conf = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0

    # Step 2: NLP Correction
    cleaned_text = nlp.neural_correct(raw_text, confidence_scores)

    # Step 3: Summarization
    results = summ.extract(cleaned_text)

    # ✅ IMPORTANT: RETURN (not just print)
    return {
        "raw_text": raw_text,
        "cleaned_text": cleaned_text,
        "insights": results,
        "confidence": avg_conf
    }


# -------------------------------
# FASTAPI BACKEND
# -------------------------------
app = FastAPI()

# Allow frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/process")
async def process_image(file: UploadFile = File(...)):
    file_path = "temp.png"

    # Save uploaded image
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Run your pipeline
    result = process_handwriting(file_path)

    return result


# -------------------------------
# CLI MODE (OPTIONAL)
# -------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Handwriting OCR Pipeline")
    parser.add_argument("--image", type=str, help="Path to the handwritten image")

    args = parser.parse_args()

    if args.image:
        if os.path.exists(args.image):
            result = process_handwriting(args.image)

            print("\n" + "="*40)
            print("FINAL RESULTS")
            print("="*40)
            print(f"Raw OCR Output: {result['raw_text']}")
            print(f"Cleaned Text:   {result['cleaned_text']}")
            print("\nInsights (JSON):")
            print(json.dumps(result['insights'], indent=2))
            print(f"\nConfidence: {result['confidence']:.4f}")
            print("="*40)

        else:
            print(f"Error: Image file {args.image} not found.")
    else:
        print("Please provide an image path using --image <path>")