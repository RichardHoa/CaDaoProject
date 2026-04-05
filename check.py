import pickle
import numpy as np


def inspect_file(filename):
    print(f"\n{'='*20} INSPECTING: {filename} {'='*20}")
    with open(filename, "rb") as f:
        data = pickle.load(f)

    # 1. Basic Structure Check
    print(f"Keys found: {list(data.keys())}")

    embs = data["embeddings"]
    print(
        f"Embedding Shape: {embs.shape} (Count: {embs.shape[0]}, Dim: {embs.shape[1]})"
    )

    # 2. Check for Zero Vectors (The "Ollama Failed" bug)
    zero_vector_count = np.sum(np.all(embs == 0, axis=1))
    if zero_vector_count > 0:
        print(f"❌ CRITICAL: Found {zero_vector_count} vectors that are ALL ZEROS!")
    else:
        print("✅ SUCCESS: No zero-vectors found.")

    # 3. Check for Duplicate Vectors (The "1.0 Similarity" bug)
    # We use unique on rows to see how many distinct vectors actually exist
    unique_vectors = np.unique(embs, axis=0)
    duplicate_count = embs.shape[0] - unique_vectors.shape[0]

    if duplicate_count > 0:
        print(
            f"⚠️ WARNING: Found {duplicate_count} duplicate vectors. (This might be normal for short keywords, but suspicious for poems)."
        )
    else:
        print("✅ SUCCESS: All vectors are unique.")

    # 4. Check Normalization (Should be ~1.0 since you used L2 normalization)
    norms = np.linalg.norm(embs, axis=1)
    avg_norm = np.mean(norms)
    print(f"Average Vector Norm: {avg_norm:.4f} (Should be 1.0000)")

    # 5. Data Integrity Check (Updated for Multi-Vector support)
    data_list_key = "poems" if "poems" in data else "keywords"
    items_count = len(data[data_list_key])
    embs_count = embs.shape[0]
    
    if "vector_map" in data:
        map_count = len(data["vector_map"])
        if map_count == embs_count:
            print(f"✅ SUCCESS: Multi-vector map ({map_count}) matches embeddings.")
            print(f"   (Detected {items_count} items with ~{embs_count/items_count:.1f} vectors each)")
        else:
            print(f"❌ ERROR: Vector map mismatch! {map_count} map entries vs {embs_count} embeddings.")
    elif items_count == embs_count:
        print(f"✅ SUCCESS: 1:1 Mapping - {items_count} {data_list_key} matches count of embeddings.")
    else:
        print(f"❌ ERROR: Mismatch! {items_count} items vs {embs_count} embeddings.")


if __name__ == "__main__":
    try:
        inspect_file("embeddings.pkl")
        inspect_file("keywords.pkl")
    except FileNotFoundError as e:
        print(
            f"Error: {e}. Make sure to run this in the same folder as your .pkl files."
        )


# import pickle
# from collections import defaultdict

# with open("keywords.pkl", "rb") as f:
#     data = pickle.load(f)
#     kws = data["keywords"]
#     embs = data["embeddings"]

# # Map vector (as a tuple) to the words that produced it
# vector_map = defaultdict(list)
# for i, vector in enumerate(embs):
#     vector_map[tuple(vector.tolist())].append(kws[i])

# # Print the "clusters"
# print("--- Identical Vector Clusters ---")
# for vec, words in vector_map.items():
#     if len(words) > 1:
#         print(f"These {len(words)} words have the same vector: {words[:5]}...")
