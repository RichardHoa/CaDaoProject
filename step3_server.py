#!/usr/bin/env python3
"""
Step 3: Semantic search server for Vietnamese folk poetry.

Search flow:
1. Semantic Match: Direct vector matching against original query
2. Deep Semantic: AI-driven query expansion for better alignment
3. Keyword Expansion: Similar keyword search with literal lookup fallback
4. No literal-first search on user query

Frontend displays: poem, meaning, keywords, matched keyword, accuracy %
"""

import json
import pickle
import sqlite3
import numpy as np
import faiss
import time
import os
import csv
import random
from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template
from sklearn.preprocessing import normalize
from sentence_transformers import SentenceTransformer

# Load environment
load_dotenv()

from i18n import TRANSLATIONS

app = Flask(__name__)

@app.context_processor
def inject_translations():
    lang = request.cookies.get('lang', 'vi')
    if lang not in TRANSLATIONS:
        lang = 'vi'
    return dict(lang=lang, t=TRANSLATIONS[lang])

@app.after_request
def log_request(response):
    """Log HTTP requests (Waitress is silent by default)."""
    print(f"{request.remote_addr} - - [{time.strftime('%Y-%m-%d %H:%M:%S')}] \"{request.method} {request.path}\" {response.status_code}", flush=True)
    return response

# Config
EMBEDDINGS_FILE = "embeddings.pkl"
KEYWORDS_FILE = "keywords.pkl"
SIMILARITY_THRESHOLD = 0.45
DECOMPOSED_SIMILARITY_THRESHOLD = 0.55
KEYWORD_SIMILARITY_THRESHOLD = 0.70
TARGET_RESULTS = 20
TOP_K_KEYWORDS = 10
LEARNING_DATA_FILE = "data/learning_data.json"
INTERPRETATIONS_FILE = "data/user_interpretations.txt"
ADVICE_FILE = "advice.csv"

# Initialize SentenceTransformer Model
print("Loading SentenceTransformer model 'AITeamVN/Vietnamese_Embedding'...")
embedding_model = SentenceTransformer("AITeamVN/Vietnamese_Embedding")

# Global data
DATA_LOADED = False
poems_data = []
poem_embeddings = None
poem_index = None
keywords_list = []
keyword_embeddings = None
keyword_index = None
vector_to_poem_map = []
keyword_to_poem_map = {}
advice_data = []


def load_data():
    """Load all data files if they exist."""
    global poems_data, poem_embeddings, poem_index
    global keywords_list, keyword_embeddings, keyword_index
    global vector_to_poem_map, keyword_to_poem_map, advice_data
    global DATA_LOADED

    if not os.path.exists(EMBEDDINGS_FILE) or not os.path.exists(KEYWORDS_FILE):
        print(f"WARNING: Embedding files ({EMBEDDINGS_FILE} or {KEYWORDS_FILE}) not found.")
        print("Search functionality will be disabled.")
        DATA_LOADED = False
        return

    try:
        print("Loading poem embeddings...")
        with open(EMBEDDINGS_FILE, "rb") as f:
            data = pickle.load(f)
            poems_data = data["poems"]
            poem_embeddings = data["embeddings"]
            poem_index = data["index"]
            vector_to_poem_map = data.get("vector_map", [])

        print("Loading keyword embeddings...")
        with open(KEYWORDS_FILE, "rb") as f:
            data = pickle.load(f)
            keywords_list = data["keywords"]
            keyword_embeddings = data["embeddings"]
            keyword_index = data["index"]

        # Build literal keyword-to-poem map for fast O(1) lookup
        keyword_to_poem_map = {}
        for i, p in enumerate(poems_data):
            kws = [k.strip().lower() for k in p.get("keywords", "").split(",")]
            for kw in kws:
                if kw:
                    if kw not in keyword_to_poem_map:
                        keyword_to_poem_map[kw] = []
                    keyword_to_poem_map[kw].append(i)

        print(f"Loaded {len(poems_data)} poems, {len(keywords_list)} keywords")

        # Load advice data
        print(f"Loading advice from {ADVICE_FILE}...")
        if os.path.exists(ADVICE_FILE):
            with open(ADVICE_FILE, mode='r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                advice_data = list(reader)
            print(f"Loaded {len(advice_data)} advice entries")
        else:
            print(f"WARNING: advice file {ADVICE_FILE} not found.")

        DATA_LOADED = True
    except Exception as e:
        print(f"ERROR: Failed to load embedding files: {e}")
        DATA_LOADED = False


def embed_text(text):
    """Create embedding for text using local SentenceTransformer model.
    
    Note: AITeamVN/Vietnamese_Embedding (based on BGE-M3) does NOT require
    any instruction prefixes (no 'query:' or 'passage:' needed).
    """
    emb = embedding_model.encode([text], convert_to_numpy=True)
    return normalize(emb, norm="l2")





def search_keywords(query_embedding, top_k=TOP_K_KEYWORDS):
    """
    Search for most similar keywords.
    Returns list of (keyword, score) tuples.
    """
    scores, indices = keyword_index.search(query_embedding, top_k)

    results = []
    for i, idx in enumerate(indices[0]):
        if idx < len(keywords_list):
            results.append((keywords_list[idx], float(scores[0][i])))

    return results


# Vietnamese stop words for query decomposition
VIETNAMESE_STOP_WORDS = {
    'của', 'và', 'là', 'các', 'có', 'được', 'trong', 'cho', 'với',
    'này', 'đã', 'từ', 'một', 'những', 'để', 'về', 'như', 'không',
    'khi', 'thì', 'sẽ', 'còn', 'đến', 'cũng', 'theo', 'trên',
    'sau', 'nếu', 'tại', 'bị', 'nên', 'vì', 'hay', 'đây', 'do',
    'lại', 'mà', 'ra', 'rất', 'đó', 'vào', 'sự', 'nhiều', 'qua',
    'ai', 'gì', 'nào', 'đều', 'mỗi', 'hơn', 'rồi', 'lên', 'xuống',
}


def decompose_query_for_keywords(query):
    """
    Break a multi-word query into meaningful segments for keyword matching.
    Returns a list of segments (unigrams and bigrams) plus the full query.
    
    For example: "sự tích cực trong nghịch cảnh"
    -> ['tích cực', 'nghịch cảnh', 'tích', 'cực', 'nghịch', 'cảnh',
        'tích cực trong nghịch cảnh']
    """
    words = query.lower().strip().split()
    # Filter stop words and single-character words
    meaningful = [w for w in words if w not in VIETNAMESE_STOP_WORDS and len(w) > 1]

    segments = set()
    # Add individual words
    for w in meaningful:
        segments.add(w)

    # Add bigrams from meaningful words (Vietnamese compound words are often 2 syllables)
    for i in range(len(meaningful) - 1):
        segments.add(f"{meaningful[i]} {meaningful[i+1]}")

    # Always include the full query as one segment too
    segments.add(query.lower().strip())

    return list(segments)


def search_keywords_decomposed(query, top_k=TOP_K_KEYWORDS):
    """
    Search keywords using query decomposition for better matching.
    
    Instead of embedding the full query as one piece and comparing against
    short keywords (which gives weak scores due to semantic scale mismatch),
    this splits the query into meaningful segments and searches keywords
    with each segment. This produces keyword-to-keyword comparisons with
    much higher and more accurate similarity scores.
    
    Returns deduplicated list of (keyword, best_score) sorted by score desc.
    """
    segments = decompose_query_for_keywords(query)

    keyword_scores = {}  # keyword -> best score across all segments

    for segment in segments:
        seg_embedding = embed_text(segment)
        results = search_keywords(seg_embedding, top_k)
        for kw, score in results:
            if kw not in keyword_scores or score > keyword_scores[kw]:
                keyword_scores[kw] = score

    # Sort by score descending and return top_k
    sorted_results = sorted(keyword_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_results[:top_k]


def search_poems_by_embedding(
    query_embedding, threshold=SIMILARITY_THRESHOLD, limit=TARGET_RESULTS
):
    """
    Search poems semantically using the multi-vector index.
    Returns list of (poem_id, score, match_type) tuples.
    """
    scores, indices = poem_index.search(query_embedding, limit * 3)

    results = []
    seen_ids = set()

    # Match types mapping based on Step 2 ordering:
    # 0: Poem, 1: Meaning, 2: Keywords
    MATCH_TYPES = {
        0: "Khớp theo lời thơ",
        1: "Khớp theo ý nghĩa",
        2: "Khớp theo từ khóa"
    }

    for i, idx in enumerate(indices[0]):
        if idx != -1 and scores[0][i] >= threshold:
            poem_id = vector_to_poem_map[idx]
            match_type_id = idx % 3
                
            match_description = MATCH_TYPES.get(match_type_id, "Khớp ngữ nghĩa")

            if poem_id not in seen_ids:
                results.append((poem_id, float(scores[0][i]), match_description))
                seen_ids.add(poem_id)

        if len(results) >= limit:
            break

    return results


def search_poems_literal(query, limit=TARGET_RESULTS):
    """
    Search poems by literal substring matching in:
    1. Poem text
    2. Meaning (searchable explanation)
    3. Explanation (original explanation)
    4. Keywords
    Returns list of (poem_id, score) where score is 1.0 for literal matches.
    """
    query_lower = query.lower().strip()
    results = []
    
    for i, p in enumerate(poems_data):
        poem_text = p.get("poem", "").lower()
        meaning = p.get("meaning", "").lower()
        explanation = p.get("explanation", "").lower()
        keywords = p.get("keywords", "").lower()
        
        # Check for literal match
        if query_lower in poem_text or query_lower in meaning or query_lower in explanation or query_lower in keywords:
            results.append((i, 1.0))
            
        if len(results) >= limit:
            break
            
    return results


def search_poems_by_keyword_literal(
    keyword, limit=TARGET_RESULTS
):
    """
    Search poems by literal keyword match in the keywords string.
    Uses the pre-built keyword_to_poem_map for O(1) lookup.
    """
    kw_lower = keyword.lower().strip()
    indices = keyword_to_poem_map.get(kw_lower, [])
    
    results = []
    for i in indices:
        results.append((i, 1.0, f"Khớp từ khóa: {keyword}"))
        if len(results) >= limit:
            break
            
    return results


def search_poems(query, top_n=TARGET_RESULTS):
    """
    Main search function follows a semantic-first pipeline:
    1. Semantic Match (Original Query) - Direct vector matching
    2. Decomposed Semantic Search - Search poems with each sub-term directly
    3. Keyword Expansion + Literal Lookup (Strict threshold, high-precision)
    4. Adaptive Fallback - For zero-result cases
    5. Literal Search - Absolute last resort
    """
    query_lower = query.lower().strip()
    seen_poem_indices = set()
    results = []

    print(f"\n[SEARCH] Query: '{query}'")

    # Step 1: Semantic Match (Original Query)
    if len(results) < 15:
        print(f"  [1] Searching poems semantically with '{query_lower}'...")
        query_embedding = embed_text(query_lower)
        poem_matches = search_poems_by_embedding(
            query_embedding, SIMILARITY_THRESHOLD, 15 - len(results)
        )

        for poem_idx, score, match_type in poem_matches:
            if poem_idx not in seen_poem_indices:
                seen_poem_indices.add(poem_idx)
                poem = poems_data[poem_idx]
                results.append(
                    {
                        "poem": poem["poem"],
                        "meaning": poem.get("explanation", ""),
                        "category": poem.get("category", ""),
                        "keywords": poem.get("keywords", ""),
                        "matched_keyword": match_type,
                        "score": score,
                    }
                )

        print(f"  [RESULT] Total: {len(results)} poems after original semantic search")

    # Step 2: Decomposed Semantic Search
    # Instead of searching a small keyword vocabulary (bottleneck),
    # decompose the query into sub-terms and search the poem index directly
    # with each one. This finds poems matching each sub-concept.
    if len(results) < 15:
        segments = decompose_query_for_keywords(query_lower)
        # Remove the full query since Step 1 already searched with it
        segments = [s for s in segments if s != query_lower]

        if segments:
            print(f"  [2] Decomposed semantic search with {len(segments)} sub-terms: {segments}")

            for segment in segments:
                if len(results) >= 15:
                    break
                seg_embedding = embed_text(segment)
                poem_matches = search_poems_by_embedding(
                    seg_embedding, DECOMPOSED_SIMILARITY_THRESHOLD, 15 - len(results)
                )

                added = 0
                for poem_idx, score, match_type in poem_matches:
                    if poem_idx not in seen_poem_indices:
                        seen_poem_indices.add(poem_idx)
                        poem = poems_data[poem_idx]
                        results.append(
                            {
                                "poem": poem["poem"],
                                "meaning": poem.get("explanation", ""),
                                "category": poem.get("category", ""),
                                "keywords": poem.get("keywords", ""),
                                "matched_keyword": f"{match_type} ({segment})",
                                "score": score,
                            }
                        )
                        added += 1

                if added > 0:
                    print(f"    - '{segment}' added {added} poems")

            print(f"  [RESULT] Total: {len(results)} poems after decomposed semantic search")

    # Step 3: Keyword Expansion + Literal Lookup (Strict threshold)
    # Only accept keywords with high confidence (>=0.70) to avoid noise
    similar_keywords = []  # Store for potential use in fallback
    if len(results) < 15:
        print(f"  [3] Searching similar keywords (strict threshold 0.70)...")
        similar_keywords = search_keywords_decomposed(query_lower, TOP_K_KEYWORDS)
        print(f"  [KEYWORDS] Found {len(similar_keywords)} similar keywords:")
        for kw, kw_score in similar_keywords:
            print(f"    - '{kw}' (similarity: {kw_score:.4f})")

        for kw, kw_score in similar_keywords:
            if len(results) >= 15:
                break

            if kw_score < KEYWORD_SIMILARITY_THRESHOLD:
                print(f"  [3.x] Skipping keyword '{kw}' (similarity: {kw_score:.2f} < {KEYWORD_SIMILARITY_THRESHOLD})")
                continue

            print(f"  [3.x] Searching with keyword '{kw}' (similarity: {kw_score:.2f})...")
            poem_matches = search_poems_by_keyword_literal(kw, 15 - len(results))

            for poem_idx, score, match_type in poem_matches:
                if poem_idx not in seen_poem_indices:
                    seen_poem_indices.add(poem_idx)
                    poem = poems_data[poem_idx]
                    results.append(
                        {
                            "poem": poem["poem"],
                            "meaning": poem.get("explanation", ""),
                            "category": poem.get("category", ""),
                            "keywords": poem.get("keywords", ""),
                            "matched_keyword": match_type,
                            "score": score,
                        }
                    )

        print(f"  [RESULT] Total: {len(results)} poems after keyword expansion")

    # Step 4: Adaptive fallback - if still 0 results, relax thresholds
    if len(results) == 0:
        print(f"  [4] Zero results! Applying adaptive fallback...")

        # 4a. Relax semantic threshold
        relaxed_matches = search_poems_by_embedding(
            query_embedding, threshold=0.3, limit=5
        )
        for poem_idx, score, match_type in relaxed_matches:
            if poem_idx not in seen_poem_indices:
                seen_poem_indices.add(poem_idx)
                poem = poems_data[poem_idx]
                results.append(
                    {
                        "poem": poem["poem"],
                        "meaning": poem.get("explanation", ""),
                        "category": poem.get("category", ""),
                        "keywords": poem.get("keywords", ""),
                        "matched_keyword": f"{match_type} (gợi ý)",
                        "score": score,
                    }
                )

        print(f"  [RESULT] Total: {len(results)} poems after relaxed semantic search")

        # 4b. Force-accept top keyword matches if still empty
        if len(results) == 0 and similar_keywords:
            print(f"  [4b] Still zero, force-accepting top keywords...")
            for kw, kw_score in similar_keywords[:3]:
                poem_matches = search_poems_by_keyword_literal(kw, 5 - len(results))
                for poem_idx, score, match_type in poem_matches:
                    if poem_idx not in seen_poem_indices:
                        seen_poem_indices.add(poem_idx)
                        poem = poems_data[poem_idx]
                        results.append(
                            {
                                "poem": poem["poem"],
                                "meaning": poem.get("explanation", ""),
                                "category": poem.get("category", ""),
                                "keywords": poem.get("keywords", ""),
                                "matched_keyword": f"{match_type} (gợi ý)",
                                "score": kw_score,
                            }
                        )
                if len(results) >= 5:
                    break

            print(f"  [RESULT] Total: {len(results)} poems after forced keyword fallback")

    # Step 5: Literal search as absolute last resort
    if len(results) == 0:
        print(f"  [5] Absolute fallback: literal substring search...")
        literal_matches = search_poems_literal(query_lower, 5)
        for poem_idx, score in literal_matches:
            if poem_idx not in seen_poem_indices:
                seen_poem_indices.add(poem_idx)
                poem = poems_data[poem_idx]
                results.append(
                    {
                        "poem": poem["poem"],
                        "meaning": poem.get("explanation", ""),
                        "category": poem.get("category", ""),
                        "keywords": poem.get("keywords", ""),
                        "matched_keyword": "Tìm theo văn bản",
                        "score": score,
                    }
                )
        print(f"  [RESULT] Total: {len(results)} poems after literal search")

    print(f"\n[FINAL] Returning {len(results)} poems")
    return results[:top_n]


def strip_markdown(text):
    """Remove common markdown artifacts from the text."""
    import re
    # Remove bold/italic
    text = re.sub(r'[\*_]{1,3}', '', text)
    # Remove headers
    text = re.sub(r'^#+\s+', '', text, flags=re.MULTILINE)
    # Remove links [text](url) -> text
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    # Remove list bullet points
    text = re.sub(r'^\s*[\-\*\+]\s+', '', text, flags=re.MULTILINE)
    # Remove numbered lists
    text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)
    return text.strip()


@app.route("/")
def index():
    """Serve the intro page"""
    return render_template("index.html")


@app.route("/search-page")
def search_page():
    """Serve the search interface"""
    return render_template("search.html")


@app.route("/learning-page")
def learning_page():
    """Serve the learning interface"""
    return render_template("learning.html")


@app.route("/qa-page")
def qa_page():
    """Serve the QA interface"""
    return render_template("qa.html")


@app.route("/search")
def search():
    """Search endpoint - returns up to 20 poems"""
    query = request.args.get("q", "")
    top_n = request.args.get("k", TARGET_RESULTS, type=int)

    if not query:
        return jsonify({"results": []})

    if not DATA_LOADED:
        print(f"  [WARN] Search requested but DATA_LOADED is False. Returning 'no data'.")
        return jsonify({"results": [], "info": "Semantic search data not found."})

    results = search_poems(query, top_n)
    return jsonify({"results": results})


@app.route("/api/qa", methods=["POST"])
def api_qa():
    """
    QA endpoint - Pick a random poem and return static advice.
    Returns JSON { "poem": ..., "advice": ..., "source": ..., "attempts": ... }
    """
    data = request.json
    concern = data.get("concern", "")

    if not concern:
        return jsonify({"error": "No concern provided"}), 400

    if not advice_data:
        return jsonify({"error": "Advice data not loaded"}), 500

    print(f"\n[QA] User concern: '{concern}'")
    
    selected_poem = random.choice(advice_data)
    static_advice = selected_poem.get("Advice", "")

    return jsonify({
        "poem": selected_poem["Poem"],
        "advice": static_advice,
        "source": selected_poem.get("Source", ""),
        "attempts": 1
    })


@app.route("/api/learning/data")
def learning_data():
    """Serve the learning data from private directory"""
    if not os.path.exists(LEARNING_DATA_FILE):
        return jsonify({"error": "Data file not found"}), 404
    with open(LEARNING_DATA_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    return jsonify(data)


@app.route("/api/learning/interpretations/<int:poem_id>")
def get_interpretations(poem_id):
    """Retrieve user interpretations for a specific poem"""
    results = []
    if os.path.exists(INTERPRETATIONS_FILE):
        with open(INTERPRETATIONS_FILE, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(" | ")
                if len(parts) >= 4:
                    try:
                        pid = int(parts[0])
                        if pid == poem_id:
                            results.append({
                                "username": parts[1],
                                "status": parts[2],
                                "interpretation": parts[3]
                            })
                    except ValueError:
                        continue
    return jsonify({"results": results})


@app.route("/api/learning/feedback", methods=["POST"])
def post_feedback():
    """Record user like/dislike and interpretation"""
    data = request.json
    poem_id = data.get("poem_id")
    status = data.get("status")  # 'like' or 'dislike'
    username = data.get("username") or "Ẩn danh"
    interpretation = data.get("interpretation", "").replace("\n", " ").replace("|", " ")

    if poem_id is None or status is None:
        return jsonify({"error": "Missing required fields"}), 400

    # Ensure data directory exists
    os.makedirs(os.path.dirname(INTERPRETATIONS_FILE), exist_ok=True)

    # Append to file: POEM_ID | USERNAME | STATUS | INTERPRETATION
    with open(INTERPRETATIONS_FILE, "a", encoding="utf-8") as f:
        f.write(f"{poem_id} | {username} | {status} | {interpretation}\n")

    return jsonify({"status": "success"})


# --- Wiki Configurations & Helper Functions ---
WIKI_CSV_FILE = "extractions/wiki.csv"
WIKI_DB_FILE = "data/wiki.db"
WIKI_IMAGES_DIR = "extractions/output-folder"

def get_base_alphabet(letter):
    if not letter:
        return 'A'
    l = letter.strip().upper()[0]
    
    # Map Vietnamese accented characters to their base letters
    if l in ('A', 'Ă', 'Â', 'Á', 'À', 'Ả', 'Ã', 'Ạ', 'Ấ', 'Ầ', 'Ẩ', 'Ẫ', 'Ậ', 'Ắ', 'Ằ', 'Ẳ', 'Ẵ', 'Ặ'):
        return 'A'
    if l in ('E', 'Ê', 'É', 'È', 'Ẻ', 'Ẽ', 'Ẹ', 'Ế', 'Ề', 'Ể', 'Ễ', 'Ệ'):
        return 'E'
    if l in ('I', 'Í', 'Ì', 'Ỉ', 'Ĩ', 'Ị'):
        return 'I'
    if l in ('O', 'Ô', 'Ơ', 'Ó', 'Ò', 'Ỏ', 'Õ', 'Ọ', 'Ố', 'Ồ', 'Ổ', 'Ỗ', 'Ộ', 'Ớ', 'Ờ', 'Ở', 'Ỡ', 'Ợ'):
        return 'O'
    if l in ('U', 'Ư', 'Ú', 'Ù', 'Ủ', 'Ũ', 'Ụ', 'Ứ', 'Ừ', 'Ử', 'Ữ', 'Ự'):
        return 'U'
    if l in ('Y', 'Ý', 'Ỳ', 'Ỷ', 'Ỹ'):
        return 'Y'
    if l in ('D'):
        return 'D'
    if l in ('Đ'):
        return 'Đ'
    return l

def init_wiki_db():
    """Initialize SQLite database for Wiki if it does not exist."""
    db_dir = os.path.dirname(WIKI_DB_FILE)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)
        
    conn = sqlite3.connect(WIKI_DB_FILE)
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS wiki (
            id TEXT PRIMARY KEY,
            Alphabet TEXT,
            base_alphabet TEXT,
            poem TEXT,
            category TEXT,
            explanation TEXT,
            pages TEXT
        )
    """)
    conn.commit()
    
    cursor.execute("SELECT COUNT(*) FROM wiki")
    count = cursor.fetchone()[0]
    
    if count == 0 and os.path.exists(WIKI_CSV_FILE):
        print(f"Initializing wiki database from {WIKI_CSV_FILE}...")
        try:
            with open(WIKI_CSV_FILE, mode='r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                to_insert = []
                for row in reader:
                    row_id = row.get("id", "")
                    alphabet = row.get("Alphabet", "")
                    base_alphabet = get_base_alphabet(alphabet)
                    poem = row.get("poem", "")
                    category = row.get("category", "")
                    explanation = row.get("explanation", "")
                    pages = row.get("pages", "")
                    
                    to_insert.append((
                        row_id, alphabet, base_alphabet, poem, category, explanation, pages
                    ))
                
                cursor.executemany("""
                    INSERT INTO wiki (id, Alphabet, base_alphabet, poem, category, explanation, pages)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, to_insert)
                
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_wiki_base_alphabet ON wiki (base_alphabet)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_wiki_category ON wiki (category)")
                
                conn.commit()
                
                cursor.execute("SELECT COUNT(*) FROM wiki")
                new_count = cursor.fetchone()[0]
                print(f"Wiki database initialized with {new_count} entries.")
        except Exception as e:
            print(f"ERROR: Failed to import wiki.csv to database: {e}")
            conn.rollback()
            
    conn.close()

# --- Wiki Routes ---

@app.route("/wiki-page")
def wiki_page():
    """Serve the wiki interface"""
    return render_template("wiki.html")

@app.route("/wiki/image/<image_id>")
def get_wiki_image(image_id):
    """Serve cropped scan images safely from extractions/output-folder."""
    safe_id = os.path.basename(image_id)
    if not safe_id.endswith(".png"):
        safe_id += ".png"
        
    img_path = os.path.join("extractions", "output-folder", safe_id)
    if not os.path.exists(img_path):
        return "Image not found", 404
        
    from flask import send_file
    return send_file(img_path, mimetype='image/png')

@app.route("/api/wiki")
def api_wiki():
    """Retrieve paginated, filtered, and searched wiki entries from SQLite."""
    alphabet = request.args.get("alphabet", "A").strip().upper()
    category = request.args.get("category", "").strip()
    search_query = request.args.get("search", "").strip()
    page = request.args.get("page", 1, type=int)
    per_page = request.args.get("per_page", 20, type=int)
    
    if page < 1:
        page = 1
    if per_page < 1:
        per_page = 20
        
    offset = (page - 1) * per_page
    
    params = {}
    sql = "SELECT id, Alphabet, base_alphabet, poem, category, explanation, pages FROM wiki WHERE 1=1"
    count_sql = "SELECT COUNT(*) FROM wiki WHERE 1=1"
    
    if search_query:
        sql += " AND LOWER(poem) LIKE :search"
        count_sql += " AND LOWER(poem) LIKE :search"
        params["search"] = f"%{search_query.lower()}%"
    else:
        sql += " AND base_alphabet = :alphabet"
        count_sql += " AND base_alphabet = :alphabet"
        params["alphabet"] = alphabet
        
    if category and category != "All":
        sql += " AND category = :category"
        count_sql += " AND category = :category"
        params["category"] = category
        
    sql += " ORDER BY Alphabet ASC, poem ASC LIMIT :limit OFFSET :offset"
    params["limit"] = per_page
    params["offset"] = offset
    
    try:
        conn = sqlite3.connect(WIKI_DB_FILE)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute(count_sql, {k: v for k, v in params.items() if k not in ("limit", "offset")})
        total_count = cursor.fetchone()[0]
        
        cursor.execute(sql, params)
        rows = cursor.fetchall()
        
        entries = []
        for r in rows:
            entries.append({
                "id": r["id"],
                "Alphabet": r["Alphabet"],
                "base_alphabet": r["base_alphabet"],
                "poem": r["poem"],
                "category": r["category"],
                "explanation": r["explanation"],
                "pages": r["pages"]
            })
            
        conn.close()
        
        import math
        total_pages = math.ceil(total_count / per_page)
        
        return jsonify({
            "entries": entries,
            "total_count": total_count,
            "total_pages": total_pages,
            "current_page": page,
            "per_page": per_page
        })
    except Exception as e:
        print(f"ERROR in api_wiki: {e}")
        return jsonify({"error": "Failed to query database", "details": str(e)}), 500


# Initialize data on module load so WSGI servers (Waitress/Gunicorn) can access it
init_wiki_db()
load_data()

def main():
    """Initialize and run the server"""
    app.run(host="0.0.0.0", port=4000, debug=True)

if __name__ == "__main__":
    main()
