import os
import re
import torch
from symspellpy import SymSpell, Verbosity
from transformers import pipeline

class TextCorrector:
    def __init__(self, dict_path="data/frequency_dictionary_en_82_765.txt"):
        print("Initializing Text Corrector...")
        self.sym_spell = SymSpell(max_dictionary_edit_distance=2)
        if os.path.exists(dict_path):
            self.sym_spell.load_dictionary(dict_path, term_index=0, count_index=1)
        else:
            print(f"Warning: Spelling dictionary not found at {dict_path}. Running without SymSpell.")

        print("Loading Neural Context Corrector (RoBERTa)...")
        self.device = 0 if torch.cuda.is_available() else -1
        self.unmasker = pipeline("fill-mask", model="roberta-base", device=self.device)

    def clean_text(self, text):
        """Standard SymSpell cleaning."""
        if not text: return ""
        text = re.sub(r'\s+', ' ', text).strip()
        try:
            suggestions = self.sym_spell.lookup_compound(text, max_edit_distance=2)
            return suggestions[0].term if suggestions else text
        except:
            return text

    def neural_correct(self, text, line_scores, threshold=0.99):
        """
        Uses RoBERTa to fix low-confidence words or suspicious short-word patterns.
        """
        if not text: return ""
        
        words = text.split()
        corrected_words = list(words)
        
        # In this PoC, we'll scan for 1-2 letter words that are often OCR noise
        for i in range(len(words)):
            word = words[i]
            
            # TRIGGER: Word is very short (1-2 chars) AND we have context
            if len(word) <= 2 and i > 0 and i < len(words) - 1:
                # Construct context
                start = max(0, i-5)
                end = min(len(words), i+6)
                
                context_before = words[start:i]
                context_after = words[i+1:end]
                
                # RoBERTa uses <mask> instead of [MASK]
                masked_sentence = " ".join(context_before + ["<mask>"] + context_after)
                
                try:
                    predictions = self.unmasker(masked_sentence)
                    top_prediction = predictions[0]['token_str'].strip()
                    score = predictions[0]['score']
                    
                    # If RoBERTa is confident and the word is different/better
                    if score > 0.15 and top_prediction.lower() != word.lower():
                        if len(top_prediction) >= len(word) and top_prediction.isalpha():
                            corrected_words[i] = top_prediction
                except Exception:
                    continue

        return self.clean_text(" ".join(corrected_words))
