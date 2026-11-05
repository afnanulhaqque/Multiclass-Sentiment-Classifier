import re
import nltk
from nltk.tokenize import word_tokenize

def ensure_nltk_resources():
    resources = [
        ("tokenizers/punkt_tab", "punkt_tab"),
    ]
    for path, package in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(package, quiet=True)

def preprocess_text(text):
    """Clean text and tokenize it while preserving negation words."""
    if not isinstance(text, str):
        return []
    ensure_nltk_resources()
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^a-z\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = word_tokenize(text)
    return tokens
