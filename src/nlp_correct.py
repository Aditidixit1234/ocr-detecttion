import os
import re
from symspellpy import SymSpell, Verbosity

class TextCorrector:
    def __init__(self, dict_path="data/frequency_dictionary_en_82_765.txt"):
        print("Initializing Text Corrector...")
        self.sym_spell = SymSpell(max_dictionary_edit_distance=2)
        if os.path.exists(dict_path):
            self.sym_spell.load_dictionary(dict_path, term_index=0, count_index=1)
        else:
            # Fallback to a default dictionary if possible or just log warning
            print(f"Warning: Spelling dictionary not found at {dict_path}. Running without correction.")

    def clean_text(self, text):
        """Cleans and corrects recognized text."""
        if not text:
            return ""
            
        # Basic cleaning: remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Word segmentation (useful for joined words in OCR)
        # and spelling correction
        try:
            # SymSpell's lookup_compound is great for sentences with spelling errors and joined words
            suggestions = self.sym_spell.lookup_compound(text, max_edit_distance=2)
            if suggestions:
                corrected_text = suggestions[0].term
            else:
                corrected_text = text
        except Exception as e:
            print(f"Correction error: {e}")
            corrected_text = text
            
        return corrected_text
