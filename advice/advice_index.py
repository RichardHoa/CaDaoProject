#!/usr/bin/env python3
"""
Create advice embedding database.
Indexes searchable_explanation from advice.csv and saves as advice_index.pkl at root.
"""

import os
import csv
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

# Paths
ADVICE_CSV = os.path.join(os.path.dirname(__file__), "advice.csv")
OUTPUT_PKL = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "advice_index.pkl"))

def main():
    print("Loading SentenceTransformer model 'AITeamVN/Vietnamese_Embedding'...")
    model = SentenceTransformer("AITeamVN/Vietnamese_Embedding")

    print(f"Reading advice data from {ADVICE_CSV}...")
    if not os.path.exists(ADVICE_CSV):
        print(f"Error: {ADVICE_CSV} not found!")
        return

    poems = []
    texts_to_embed = []

    with open(ADVICE_CSV, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            poem_text = row.get("poem", "").strip()
            explanation = row.get("searchable_explanation", "").strip()
            if not poem_text or not explanation:
                continue
            poems.append({
                "id": row.get("id", ""),
                "poem": poem_text,
                "category": row.get("category", ""),
                "searchable_explanation": explanation,
                "keywords": row.get("keywords", "")
            })
            texts_to_embed.append(explanation)

    print(f"Found {len(poems)} valid advice entries.")
    if not texts_to_embed:
        print("No texts to embed. Exiting.")
        return

    print("Generating embeddings...")
    embeddings = model.encode(texts_to_embed, show_progress_bar=True, convert_to_numpy=True)
    
    # L2 normalize embeddings using numpy
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    embeddings = embeddings / norms

    print(f"Saving index to {OUTPUT_PKL}...")
    with open(OUTPUT_PKL, "wb") as f:
        pickle.dump({
            "poems": poems,
            "embeddings": embeddings
        }, f)
    print("Indexing completed successfully.")

if __name__ == "__main__":
    main()
