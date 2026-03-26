import torch
import re
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class InsightsExtractor:
    def __init__(self, model_name="facebook/bart-large-cnn"):
        print(f"Loading Summarizer: {model_name}...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)

    def extract(self, text):
        """Extracts dates, titles, and a summary from the corrected text."""
        # Simple date regex
        date_pattern = r'\b(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}|\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*)\b'
        dates = re.findall(date_pattern, text, re.IGNORECASE)
        
        summary = text
        if len(text.split()) > 15:
            # Manual generation for summarization
            inputs = self.tokenizer(text, max_length=1024, return_tensors="pt", truncation=True).to(self.device)
            summary_ids = self.model.generate(
                inputs["input_ids"], 
                num_beams=4, 
                max_length=50, 
                min_length=10,
                early_stopping=True,
                no_repeat_ngram_size=3
            )
            summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        return {
            "cleaned_text": text,
            "summary": summary,
            "dates_found": dates,
            "word_count": len(text.split())
        }
