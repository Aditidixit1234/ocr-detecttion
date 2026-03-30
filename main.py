import os
import json
import argparse
import shutil
import uuid
import logging
from contextlib import asynccontextmanager

# Your existing modules
from src.ocr import OCRSystem
from src.nlp_correct import TextCorrector
from src.summarizer import InsightsExtractor

# FastAPI imports
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware


# -------------------------------
# GLOBAL INSTANCES (SINGLETONS)
# -------------------------------
ocr = None
nlp = None
summ = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global ocr, nlp, summ
    print("Initializing models...")
    ocr = OCRSystem()
    nlp = TextCorrector()
    summ = InsightsExtractor()
    yield
    print("Shutting down...")

# -------------------------------
# CORE FUNCTION (SHARED)
# -------------------------------
def process_handwriting(image_path, verbose=False):
    # Use global singletons if they exist, otherwise instantiate (for CLI)
    global ocr, nlp, summ
    if ocr is None:
        ocr = OCRSystem()
    if nlp is None:
        nlp = TextCorrector()
    if summ is None:
        summ = InsightsExtractor()

    if verbose:
        print(f"\n[1/3] Processing OCR for: {image_path}")

    # Step 1: OCR
    raw_text, confidence_scores = ocr.process_image(image_path)

    avg_conf = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0

    if verbose:
        print(f"OCR Complete. Average Confidence: {avg_conf:.4f}")
        print(f"[2/3] Applying Neural Spell Correction...")

    # Step 2: NLP Correction
    cleaned_text = nlp.neural_correct(raw_text, confidence_scores)

    if verbose:
        print(f"[3/3] Generating summary and extracting insights...")

    # Step 3: Summarization
    results = summ.extract(cleaned_text)

    output = {
        "raw_text": raw_text,
        "cleaned_text": cleaned_text,
        "insights": results.get("summary", results),
        "confidence": avg_conf
    }

    if verbose:
        print("\n" + "="*40)
        print("FINAL RESULTS")
        print("="*40)
        print(f"Raw OCR Output: {raw_text}")
        print(f"Cleaned Text:   {cleaned_text}")
        print("\nInsights (JSON):")
        print(json.dumps(results, indent=2))
        print(f"\nConfidence: {avg_conf:.4f}")
        print("="*40)

    return output


# -------------------------------
# FASTAPI BACKEND
# -------------------------------
app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False, # Changed to False to fix CORS conflict
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/process")
async def process_image(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail=f"File {file.filename} is not an image.")

    file_path = f"temp_{uuid.uuid4().hex}.png"

    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        result = process_handwriting(file_path, verbose=False)
        return result
    except Exception as e:
        logging.error(f"Error processing image: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)


# -------------------------------
# CLI MODE
# -------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Handwriting OCR Pipeline")
    parser.add_argument("--image", type=str, help="Path to the handwritten image")

    args = parser.parse_args()

    if args.image:
        if os.path.exists(args.image):
            process_handwriting(args.image, verbose=True)
        else:
            print(f"Error: Image file {args.image} not found.")
    else:
        # Default demo image
        demo_img = os.path.join("data", "iam", "words", "a01", "a01-000u", "a01-000u-00-00.png")

        if os.path.exists(demo_img):
            print("No image provided. Running demo...")
            process_handwriting(demo_img, verbose=True)
        else:
            print("Please provide an image path using --image <path>")
