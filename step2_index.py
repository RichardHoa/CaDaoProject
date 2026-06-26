#!/usr/bin/env python3
"""
Step 2: Build embedding databases.
- embeddings.pkl: Poem embeddings + poem data (poem, meaning, keywords)
- keywords.pkl: All unique keyword embeddings + keyword list
"""

import pickle
import time
import pandas as pd
import numpy as np
import faiss
import os
from dotenv import load_dotenv
from sklearn.preprocessing import normalize
from sentence_transformers import SentenceTransformer

# Initialize SentenceTransformer Model
print("Loading SentenceTransformer model 'AITeamVN/Vietnamese_Embedding'...")
embedding_model = SentenceTransformer("AITeamVN/Vietnamese_Embedding")

INPUT_CSV = "extractions/wiki.csv"
EMBEDDINGS_FILE = "embeddings.pkl"
KEYWORDS_FILE = "keywords.pkl"


def create_embeddings(texts, label="items"):
    print(f"Creating embeddings for {len(texts)} {label}...")
    
    if not texts:
        # Return an empty array with 2 dimensions if no texts provided
        return np.array([], dtype=np.float32).reshape(0, 0)

    # Generate embeddings in batch
    embeddings = embedding_model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    return np.array(embeddings, dtype=np.float32)


def main():
    print(f"--- Bước 2: Xây dựng cơ sở dữ liệu embedding ---")

    # Load extracted data
    print(f"Loading {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV, dtype=str).fillna("")

    # Prepare poems data
    poems_data = []
    all_keywords = set()

    for index, row in df.iterrows():
        poem = str(row.get("poem", "")).strip()
        meaning = str(row.get("searchable_explanation", "")).strip()
        keywords_str = str(row.get("keywords", "")).strip()
        explanation = str(row.get("explanation", "")).strip()
        category = str(row.get("category", "")).strip()

        # Skip poems with missing components (meaning or keywords)
        if not poem or not meaning or not keywords_str:
            continue

        poems_data.append({
            "poem": poem,
            "meaning": meaning,
            "explanation": explanation,
            "category": category,
            "keywords": keywords_str
        })

        # Collect unique keywords
        if keywords_str:
            for kw in keywords_str.split(","):
                kw = kw.strip()
                if kw:
                    all_keywords.add(kw)

    print(f"Found {len(poems_data)} poems and {len(all_keywords)} unique keywords")

    # 2. Prepare Multi-Vector Embeddings
    print("\n[1/2] Creating poem embeddings (Multi-Vector approach)...")
    texts_to_embed = []
    vector_to_poem_map = []  # This maps the vector index to the poem index

    for i, p in enumerate(poems_data):
        texts_to_embed.append(p['poem'])
        vector_to_poem_map.append(i)

        texts_to_embed.append(p['meaning'])
        vector_to_poem_map.append(i)

        texts_to_embed.append(p['keywords'])
        vector_to_poem_map.append(i)

    # Generate and Normalize
    poem_embeddings = create_embeddings(texts_to_embed, "multi-vector poems")
    poem_embeddings = normalize(poem_embeddings, norm="l2")

    # 3. Build FAISS index
    poem_index = faiss.IndexFlatIP(poem_embeddings.shape[1])
    poem_index.add(poem_embeddings)

    # 4. Save (IMPORTANT: We save the map now)
    with open(EMBEDDINGS_FILE, "wb") as f:
        pickle.dump(
            {
                "poems": poems_data,
                "embeddings": poem_embeddings,
                "index": poem_index,
                "vector_map": vector_to_poem_map,  # <-- Added this
            },
            f,
        )

    # Step 2: Create keyword embeddings
    print("\n[2/2] Creating keyword embeddings...")
    keywords_list = sorted(list(all_keywords))
    keywords_to_embed = [kw for kw in keywords_list]
    keyword_embeddings = create_embeddings(keywords_to_embed, "keywords")
    keyword_embeddings = normalize(keyword_embeddings, norm="l2")

    # Build FAISS index for keywords
    print("Building FAISS index for keywords...")
    keyword_index = faiss.IndexFlatIP(keyword_embeddings.shape[1])
    keyword_index.add(keyword_embeddings)

    # Save keyword embeddings
    with open(KEYWORDS_FILE, "wb") as f:
        pickle.dump(
            {
                "keywords": keywords_list,
                "embeddings": keyword_embeddings,
                "index": keyword_index,
            },
            f,
        )
    print(f"Saved keyword embeddings to {KEYWORDS_FILE}")

    print("\n--- Hoàn thành Bước 2 ---")
    print(f"  - {EMBEDDINGS_FILE}: {len(poems_data)} poems")
    print(f"  - {KEYWORDS_FILE}: {len(keywords_list)} keywords")
    print(f"Embedding dimension: {poem_embeddings.shape[1]}")


if __name__ == "__main__":
    main()
