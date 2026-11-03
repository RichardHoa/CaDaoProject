"""
Every data path the project uses, resolved from the repo root.
Import from here instead of hard-coding paths, so scripts work from any cwd.
"""

import os


def ensure_dir(path):
    """Create a data directory if it is missing and return it."""
    os.makedirs(path, exist_ok=True)
    return path


ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")

# Scanned dictionary PDFs (raw and split into single pages)
SOURCE_DIR = os.path.join(DATA_DIR, "source")
SPLIT_DEBUG_DIR = os.path.join(SOURCE_DIR, "debug_output")


def source_pdf(name):
    """Path of a PDF in data/source/; the name is reduced to its basename."""
    return os.path.join(SOURCE_DIR, os.path.basename(name))


# OCR editor output
OCR_DIR = os.path.join(DATA_DIR, "ocr")
OCR_ENTRIES_CSV = os.path.join(OCR_DIR, "entries-extracted.csv")
OCR_SCRATCH_DIR = os.path.join(OCR_DIR, "scratch")

# Cropped scan image of each entry, named <id>.png
IMAGES_DIR = os.path.join(DATA_DIR, "images")

# Human review
REVIEW_DIR = os.path.join(DATA_DIR, "review")
REVIEW_INPUT_CSV = os.path.join(REVIEW_DIR, "input.csv")
REVIEW_OUTPUT_CSV = os.path.join(REVIEW_DIR, "reviewed.csv")
REVIEW_STATE_JSON = os.path.join(REVIEW_DIR, "review_state.json")

# LLM enrichment
ENRICHMENT_DIR = os.path.join(DATA_DIR, "enrichment")
KEYWORDS_JSON = os.path.join(ENRICHMENT_DIR, "keywords.json")
KEYWORD_EMBEDDINGS_JSON = os.path.join(ENRICHMENT_DIR, ".keywords_embeddings.json")

# Master dataset
WIKI_DIR = os.path.join(DATA_DIR, "wiki")
WIKI_CSV = os.path.join(WIKI_DIR, "wiki.csv")
WIKI_DB = os.path.join(WIKI_DIR, "wiki.db")

# Search and Q&A indexes
INDEXES_DIR = os.path.join(DATA_DIR, "indexes")
EMBEDDINGS_PKL = os.path.join(INDEXES_DIR, "embeddings.pkl")
KEYWORDS_PKL = os.path.join(INDEXES_DIR, "keywords.pkl")
ADVICE_INDEX_PKL = os.path.join(INDEXES_DIR, "advice_index.pkl")

# Q&A source
ADVICE_DIR = os.path.join(DATA_DIR, "advice")
ADVICE_CSV = os.path.join(ADVICE_DIR, "advice.csv")

# Learning game
LEARNING_DIR = os.path.join(DATA_DIR, "learning")
LEARNING_JSON = os.path.join(LEARNING_DIR, "learning-data.json")
LEARNING_CSV = os.path.join(LEARNING_DIR, "learning-data.csv")
LEARNING_TXT = os.path.join(LEARNING_DIR, "learning-data.txt")
INTERPRETATIONS_TXT = os.path.join(LEARNING_DIR, "user_interpretations.txt")
