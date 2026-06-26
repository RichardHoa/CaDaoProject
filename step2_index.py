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
import ollama
import os
from dotenv import load_dotenv
from sklearn.preprocessing import normalize

# Load API key
load_dotenv()
APOLLO_API_KEY = os.getenv("APOLLO_API_KEY")
APOLLO_URL = "https://apollo.quocanmeomeo.io.vn"

# Initialize Ollama Client
client = ollama.Client(host=APOLLO_URL, headers={'Authorization': f'Bearer {APOLLO_API_KEY}'})

INPUT_CSV = "extractions/wiki.csv"
EMBEDDINGS_FILE = "embeddings.pkl"
KEYWORDS_FILE = "keywords.pkl"
EMBEDDING_MODEL = "qwen3-embedding:8b"


def create_embeddings(texts, label="items"):
    print(f"Creating embeddings for {len(texts)} {label}...")
    embeddings = []
    
    if not texts:
        # Return an empty array with 2 dimensions if no texts provided
        return np.array([], dtype=np.float32).reshape(0, 0)

    for i, text in enumerate(texts):
        if i % 50 == 0:
            print(f"  [{i}/{len(texts)}] {label}...")

        # Skip empty text just in case (should be filtered by caller)
        if not text or not text.strip():
            continue

        # Infinite retry loop
        while True:
            try:
                response = client.embeddings(model=EMBEDDING_MODEL, prompt=text)
                vector = response["embedding"]
                
                # Check for inconsistent dimensions
                if embeddings and len(vector) != len(embeddings[0]):
                    raise ValueError(f"Inconsistent embedding dimension at index {i}. Expected {len(embeddings[0])}, got {len(vector)}. This will break indices.")

                embeddings.append(vector)
                break  # Success! Exit the while loop and move to the next text
            except Exception as e:
                print(
                    f"  [WARN] Failed to embed '{text[:30]}': {e}. Retrying in 2 seconds..."
                )
                time.sleep(2)  # Wait before trying again

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
