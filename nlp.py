import re
import spacy
from symspellpy import SymSpell

ocr_text = "Meetng nots: Discusd projct deadlns and datset of 200 ims"

clean_text = re.sub(r'[^a-zA-Z0-9\s]', '', ocr_text).lower()

sym_spell = SymSpell(max_dictionary_edit_distance=2)

sym_spell.load_dictionary(
    "frequency_dictionary_en_82_765.txt",
    0,
    1
)

suggestions = sym_spell.lookup_compound(clean_text, max_edit_distance=2)

corrected_text = suggestions[0].term

print("Corrected text:")
print(corrected_text)