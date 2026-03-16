import torch
import re
from transformers import pipeline

class InsightsExtractor:
    def __init__(self, model_name="facebook/bart-large-cnn"):
        print(f"Loading Summarizer: {model_name}...")
        device = 0 if torch.cuda.is_available() else -1
        self.summarizer = pipeline("summarization", model=model_name, device=device)

    def extract(self, text):
        """Extracts dates, titles, and a summary from the corrected text."""
        # Simple date regex
        date_pattern = r'\b(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}|\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*)\b'
        dates = re.findall(date_pattern, text, re.IGNORECASE)
        
        summary = text
        if len(text.split()) > 15:
            # BART has max length constraints
            result = self.summarizer(text, max_length=50, min_length=10, do_sample=False)
            summary = result[0]["summary_text"]

        return {
            "cleaned_text": text,
            "summary": summary,
            "dates_found": dates,
            "word_count": len(text.split())
        }
