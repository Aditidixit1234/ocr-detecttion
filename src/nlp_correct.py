import os
import re
import logging
from symspellpy import SymSpell

class TextCorrector:
    def __init__(self, dict_path="data/frequency_dictionary_en_82_765.txt"):
        print("Initializing Text Corrector (SymSpell only)...")
        self.sym_spell = SymSpell(max_dictionary_edit_distance=2)
        if os.path.exists(dict_path):
            self.sym_spell.load_dictionary(dict_path, term_index=0, count_index=1)
        else:
            logging.warning(f"Spelling dictionary not found at {dict_path}. Running without SymSpell.")

    def clean_text(self, text):
        """Standard SymSpell cleaning."""
        if not text: return ""
        text = re.sub(r'\s+', ' ', text).strip()
        try:
            suggestions = self.sym_spell.lookup_compound(text, max_edit_distance=2)
            return suggestions[0].term if suggestions else text
        except Exception as e:
            logging.error(f"SymSpell error: {e}")
            return text

    def neural_correct(self, text, line_scores=None, threshold=0.99):
        """
        Fallback method that now only performs standard cleaning since RoBERTa was removed.
        Maintains the signature for compatibility with main.py.
        """
        return self.clean_text(text)
