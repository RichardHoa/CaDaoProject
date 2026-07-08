#!/usr/bin/env python3
"""
Create advice embedding database.
Indexes searchable_explanation, poem text, and keywords from advice.csv
using a multi-vector approach and saves as advice_index.pkl at root.

Multi-vector approach: Each poem gets 3 vectors (explanation, poem, keywords).
At retrieval time, the max score across all 3 vectors is used per poem.
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
    vector_to_poem_map = []  # Maps vector index -> poem index

    with open(ADVICE_CSV, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            poem_text = row.get("poem", "").strip()
            explanation = row.get("searchable_explanation", "").strip()
            keywords = row.get("keywords", "").strip()
            if not poem_text or not explanation:
                continue

            poem_idx = len(poems)
            poems.append({
                "id": row.get("id", ""),
                "poem": poem_text,
                "category": row.get("category", ""),
                "searchable_explanation": explanation,
                "keywords": keywords
            })

            # Vector 0: explanation (searchable_explanation)
            texts_to_embed.append(explanation)
            vector_to_poem_map.append(poem_idx)

            # Vector 1: poem text
            texts_to_embed.append(poem_text)
            vector_to_poem_map.append(poem_idx)

            # Vector 2: keywords
            if keywords:
                texts_to_embed.append(keywords)
                vector_to_poem_map.append(poem_idx)

    print(f"Found {len(poems)} valid advice entries.")
    print(f"Total vectors to embed: {len(texts_to_embed)} (multi-vector: explanation + poem + keywords)")
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
            "embeddings": embeddings,
            "vector_to_poem_map": vector_to_poem_map
        }, f)
    print(f"Indexing completed successfully.")
    print(f"  - {len(poems)} poems")
    print(f"  - {len(texts_to_embed)} vectors ({len(texts_to_embed)/len(poems):.1f} avg per poem)")
    print(f"  - Embedding dimension: {embeddings.shape[1]}")

if __name__ == "__main__":
    main()
