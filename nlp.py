import re
import spacy
from symspellpy import SymSpell, Verbosity

# OCR output
ocr_text = "Meetng nots: Discusd projct deadlns and datset of 200 ims"

# Step 1 clean text
clean_text = re.sub(r'[^a-zA-Z0-9\s]', '', ocr_text)

# Step 2 tokenize
nlp = spacy.load("en_core_web_sm")
doc = nlp(clean_text)

tokens = [token.text for token in doc]

# Step 3 initialize symspell
sym_spell = SymSpell(max_dictionary_edit_distance=2)

sym_spell.load_dictionary(
    "frequency_dictionary_en_82_765.txt",
    term_index=0,
    count_index=1
)

# Step 4 correct words
corrected_words = []

for word in tokens:
    suggestions = sym_spell.lookup(word, Verbosity.CLOSEST, max_edit_distance=2)

    if suggestions:
        corrected_words.append(suggestions[0].term)
    else:
        corrected_words.append(word)

# Step 5 rebuild sentence
final_text = " ".join(corrected_words)

print("Corrected text:")
print(final_text)