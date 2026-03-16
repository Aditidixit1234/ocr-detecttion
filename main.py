import os
import json
import argparse
from src.ocr import OCRSystem
from src.nlp_correct import TextCorrector
from src.summarizer import InsightsExtractor

def process_handwriting(image_path):
    # Initialize Pipeline
    ocr = OCRSystem()
    nlp = TextCorrector()
    summ = InsightsExtractor()

    # Execution
    print(f"\n[1/3] Processing OCR for: {image_path}")
    raw_text = ocr.process_image(image_path)
    
    print(f"[2/3] Cleaning and correcting text...")
    cleaned_text = nlp.clean_text(raw_text)
    
    print(f"[3/3] Generating summary and extracting insights...")
    results = summ.extract(cleaned_text)
    
    # Final Output
    print("\n" + "="*40)
    print("FINAL RESULTS")
    print("="*40)
    print(f"Raw OCR Output: {raw_text}")
    print(f"Cleaned Text:   {cleaned_text}")
    print("\nInsights (JSON):")
    print(json.dumps(results, indent=2))
    print("="*40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Handwriting OCR Pipeline")
    parser.add_argument("--image", type=str, help="Path to the handwritten image")
    
    args = parser.parse_args()
    
    if args.image:
        if os.path.exists(args.image):
            process_handwriting(args.image)
        else:
            print(f"Error: Image file {args.image} not found.")
    else:
        # Default demo image
        demo_img = r"C:\Users\Harsh\.cache\kagglehub\datasets\ssarkar445\handwriting-recognitionocr\versions\1\test_v2\test\TEST_0008.jpg"
        if os.path.exists(demo_img):
            print("No image provided. Running demo on sample image...")
            process_handwriting(demo_img)
        else:
            print("Please provide an image path using --image <path>")
