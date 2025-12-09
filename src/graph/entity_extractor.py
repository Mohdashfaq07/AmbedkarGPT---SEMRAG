"""
entity_extractor.py
Extracts named entities from chunk text using spaCy
"""

import spacy

# Load once globally
try:
    nlp = spacy.load("en_core_web_sm")
except:
    import os
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")


def extract_entities(text: str):
    doc = nlp(text)
    ents = []
    for e in doc.ents:
        if e.label_ in ["PERSON", "ORG", "GPE", "NORP", "EVENT", "WORK_OF_ART"]:
            ents.append({"text": e.text.strip(), "label": e.label_})
    return ents


if __name__ == "__main__":
    test = "Dr. B. R. Ambedkar was born in Maharashtra and wrote the Indian Constitution."
    print(extract_entities(test))
