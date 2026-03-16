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
            print(f"Warning: Spelling dictionary not found at {dict_path}. Running without correction.")

    def clean_text(self, text):
        """Cleans and corrects recognized text."""
        # Simple cleanup
        text = text.strip()
        
        words = text.split()
        corrected_words = []
        for word in words:
            # Skip numbers or very short words
            if len(word) < 2 or not word.isalpha():
                corrected_words.append(word)
                continue
            
            suggestions = self.sym_spell.lookup(word, Verbosity.CLOSEST, max_edit_distance=2)
            corrected_words.append(suggestions[0].term if suggestions else word)
        
        return " ".join(corrected_words)
