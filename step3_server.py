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
import time
import os
import csv
import random
from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template, redirect, url_for

# Heavy ML dependencies are optional — allows the server to run
# without transformers/faiss/numpy on lightweight dev machines.
try:
    import numpy as np
    import faiss
    from sklearn.preprocessing import normalize
    from sentence_transformers import SentenceTransformer, CrossEncoder
    SEARCH_AVAILABLE = True
except ImportError as _import_err:
    print(f"WARNING: ML dependencies not available ({_import_err}).")
    print("Search functionality will be disabled. All other features remain active.")
    SEARCH_AVAILABLE = False

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
SIMILARITY_THRESHOLD = 0.6
DECOMPOSED_SIMILARITY_THRESHOLD = 0.45
KEYWORD_SIMILARITY_THRESHOLD = 0.60
TARGET_RESULTS = 20
TOP_K_KEYWORDS = 10
LEARNING_DATA_FILE = "data/learning_data.json"
INTERPRETATIONS_FILE = "data/user_interpretations.txt"
ADVICE_FILE = "advice.csv"
ADVICE_INDEX_FILE = "advice_index.pkl"

# Initialize SentenceTransformer Model (only when ML libs are available)
embedding_model = None
reranker_model = None
if SEARCH_AVAILABLE:
    print("Loading SentenceTransformer model 'AITeamVN/Vietnamese_Embedding'...")
    embedding_model = SentenceTransformer("AITeamVN/Vietnamese_Embedding")
    print("Loading CrossEncoder reranker 'AITeamVN/Vietnamese_Reranker'...")
    reranker_model = CrossEncoder("AITeamVN/Vietnamese_Reranker")
else:
    print("Skipping model load — search is disabled.")

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
advice_index_data = None


def load_data():
    """Load all data files if they exist."""
    global poems_data, poem_embeddings, poem_index
    global keywords_list, keyword_embeddings, keyword_index
    global vector_to_poem_map, keyword_to_poem_map, advice_data
    global advice_index_data
    global DATA_LOADED

    # Always load advice data (needed for QA, no ML dependency)
    # Check both root and advice/ folder for advice.csv
    advice_path = ADVICE_FILE
    if not os.path.exists(advice_path) and os.path.exists(os.path.join("advice", ADVICE_FILE)):
        advice_path = os.path.join("advice", ADVICE_FILE)

    print(f"Loading advice from {advice_path}...")
    if os.path.exists(advice_path):
        with open(advice_path, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            advice_data = list(reader)
        print(f"Loaded {len(advice_data)} advice entries")
    else:
        print(f"WARNING: advice file {ADVICE_FILE} not found.")

    # Load advice index from pickle if it exists
    print(f"Loading advice index from {ADVICE_INDEX_FILE}...")
    if os.path.exists(ADVICE_INDEX_FILE):
        try:
            with open(ADVICE_INDEX_FILE, "rb") as f:
                advice_index_data = pickle.load(f)
            print(f"Loaded advice index with {len(advice_index_data['poems'])} poems")
        except Exception as e:
            print(f"ERROR: Failed to load advice index: {e}")

    if not SEARCH_AVAILABLE:
        print("Search dependencies not available — skipping embedding data load.")
        DATA_LOADED = False
        return

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


def get_poem_id_from_db(poem_text):
    """Query the SQLite database to find the ID corresponding to a given poem."""
    if not poem_text:
        return None
    try:
        conn = sqlite3.connect(WIKI_DB_FILE)
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM wiki WHERE poem = ? LIMIT 1", (poem_text.strip(),))
        row = cursor.fetchone()
        conn.close()
        if row:
            return row[0]
    except Exception as e:
        print(f"Error looking up poem ID: {e}")
    return None


def get_poem_explanation_from_db(poem_text):
    """Query the SQLite database to find the original explanation corresponding to a given poem."""
    if not poem_text:
        return ""
    try:
        conn = sqlite3.connect(WIKI_DB_FILE)
        cursor = conn.cursor()
        cursor.execute("SELECT explanation FROM wiki WHERE poem = ? LIMIT 1", (poem_text.strip(),))
        row = cursor.fetchone()
        conn.close()
        if row:
            return row[0]
    except Exception as e:
        print(f"Error looking up poem explanation: {e}")
    return ""


def clean_category_string(c):
    return c.replace('(', '').replace(')', '').strip().lower()


def search_poems(query, category_filter=None, top_n=TARGET_RESULTS):
    """
    Main search function follows a semantic-first pipeline:
    1. Semantic Match (Original Query) - Direct vector matching
    2. Decomposed Semantic Search - Search poems with each sub-term directly
    3. Keyword Expansion + Literal Lookup
    4. Adaptive Fallback - For zero-result cases
    5. Literal Search - Absolute last resort
    """
    query_lower = query.lower().strip()
    candidate_poems = {} # poem_idx -> {"score": float, "match_type": str}

    def add_candidate(poem_idx, score, match_type):
        if poem_idx not in candidate_poems:
            candidate_poems[poem_idx] = {"score": score, "match_type": match_type}
        else:
            if score > candidate_poems[poem_idx]["score"]:
                candidate_poems[poem_idx]["score"] = score
                candidate_poems[poem_idx]["match_type"] = match_type

    print(f"\n[SEARCH] Query: '{query}' (Filter: '{category_filter}')")

    query_embedding = embed_text(query_lower)

    # Step 1: Semantic Match (Original Query)
    print(f"  [1] Searching poems semantically with '{query_lower}'...")
    poem_matches = search_poems_by_embedding(
        query_embedding, SIMILARITY_THRESHOLD, limit=50
    )
    for poem_idx, score, match_type in poem_matches:
        add_candidate(poem_idx, score, match_type)

    print(f"  [RESULT] Found {len(candidate_poems)} candidate poems after original semantic search")

    # Step 2: Keyword Expansion + Literal Lookup
    print(f"  [2] Searching similar keywords (strict threshold {KEYWORD_SIMILARITY_THRESHOLD})...")
    similar_keywords = search_keywords(query_embedding, TOP_K_KEYWORDS)
    print(f"  [KEYWORDS] Found {len(similar_keywords)} similar keywords:")
    for kw, kw_score in similar_keywords:
        print(f"    - '{kw}' (similarity: {kw_score:.4f})")

    for kw, kw_score in similar_keywords:
        if kw_score < KEYWORD_SIMILARITY_THRESHOLD:
            print(f"  [2.x] Skipping keyword '{kw}' (similarity: {kw_score:.2f} < {KEYWORD_SIMILARITY_THRESHOLD})")
            continue

        print(f"  [2.x] Searching with keyword '{kw}' (similarity: {kw_score:.2f})...")
        kw_matches = search_poems_by_keyword_literal(kw, 20)
        for poem_idx, score, match_type in kw_matches:
            add_candidate(poem_idx, kw_score, match_type)

    print(f"  [RESULT] Total unique candidates: {len(candidate_poems)} after keyword expansion")

    # Step 3: Adaptive fallback - if still 0 results, relax thresholds
    if len(candidate_poems) == 0:
        print(f"  [3] Zero results! Applying adaptive fallback...")

        # 3a. Relax semantic threshold
        relaxed_matches = search_poems_by_embedding(
            query_embedding, threshold=0.25, limit=10
        )
        for poem_idx, score, match_type in relaxed_matches:
            add_candidate(poem_idx, score, f"{match_type} (gợi ý)")

        print(f"  [RESULT] Total unique candidates: {len(candidate_poems)} after relaxed semantic search")

        # 3b. Force-accept top keyword matches if still empty
        if len(candidate_poems) == 0 and similar_keywords:
            print(f"  [3b] Still zero, force-accepting top keywords...")
            for kw, kw_score in similar_keywords[:3]:
                kw_matches = search_poems_by_keyword_literal(kw, 5)
                for poem_idx, score, match_type in kw_matches:
                    add_candidate(poem_idx, kw_score, f"{match_type} (gợi ý)")

            print(f"  [RESULT] Total unique candidates: {len(candidate_poems)} after forced keyword fallback")

    # Step 4: Literal search as absolute last resort
    if len(candidate_poems) == 0:
        print(f"  [4] Absolute fallback: literal substring search...")
        literal_matches = search_poems_literal(query_lower, 10)
        for poem_idx, score in literal_matches:
            add_candidate(poem_idx, score, "Tìm theo văn bản")
        print(f"  [RESULT] Total unique candidates: {len(candidate_poems)} after literal search")

    # Sort all candidate poems by score descending
    sorted_candidates = sorted(candidate_poems.items(), key=lambda x: x[1]["score"], reverse=True)

    results = []
    for poem_idx, data in sorted_candidates:
        poem = poems_data[poem_idx]
        poem_category = poem.get("category", "")
        
        # Apply category filter if set
        if category_filter and clean_category_string(category_filter) not in ("all", ""):
            if clean_category_string(category_filter) != clean_category_string(poem_category):
                continue
                
        poem_text = poem["poem"]
        poem_id = get_poem_id_from_db(poem_text)
        
        results.append(
            {
                "id": poem_id,
                "poem": poem_text,
                "meaning": poem.get("explanation", ""),
                "category": poem_category,
                "keywords": poem.get("keywords", ""),
                "matched_keyword": data["match_type"],
                "score": data["score"],
            }
        )
        
        if len(results) >= top_n:
            break

    print(f"\n[FINAL] Returning {len(results)} poems")
    return results


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
    """Redirect search-page to home page"""
    return redirect(url_for("index"))


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
    category = request.args.get("cat", "")
    top_n = request.args.get("k", TARGET_RESULTS, type=int)

    if not query:
        return jsonify({"results": []})

    if not SEARCH_AVAILABLE:
        return jsonify({"results": [], "info": "Search is disabled — ML dependencies not installed."})

    if not DATA_LOADED:
        print(f"  [WARN] Search requested but DATA_LOADED is False. Returning 'no data'.")
        return jsonify({"results": [], "info": "Semantic search data not found."})

    results = search_poems(query, category, top_n)
    return jsonify({"results": results})


@app.route("/api/qa", methods=["POST"])
def api_qa():
    """
    QA endpoint - Match user concern using AITeamVN/Vietnamese_Embedding embeddings,
    retrieve top 3 poems, and use gpt-5-mini with CA_DAO_VAN_DAP_KEY to select
    the best poem and generate an elegant answer.
    """
    data = request.json
    concern = data.get("concern", "")

    if not concern:
        return jsonify({"error": "No concern provided"}), 400

    print(f"\n[QA] User concern: '{concern}'")

    # Try embedding-based search + LLM generation first if search & index are available
    if advice_index_data and SEARCH_AVAILABLE and embedding_model:
        try:
            # 1. Embed user concern
            q_emb = embedding_model.encode([concern], convert_to_numpy=True)
            # Normalize query
            q_norm = np.linalg.norm(q_emb, axis=1, keepdims=True)
            q_norm[q_norm == 0] = 1.0
            q_emb = q_emb / q_norm

            # 2. Cosine similarity against all vectors (multi-vector)
            scores = np.dot(advice_index_data["embeddings"], q_emb.T).flatten()

            # 3. Aggregate: max score per poem across its vectors
            vector_to_poem_map = advice_index_data.get("vector_to_poem_map", None)
            poems_list = advice_index_data["poems"]

            if vector_to_poem_map is not None:
                # Multi-vector index: aggregate max score per poem
                poem_max_scores = {}  # poem_idx -> max_score
                for vec_idx, score in enumerate(scores):
                    poem_idx = vector_to_poem_map[vec_idx]
                    if poem_idx not in poem_max_scores or score > poem_max_scores[poem_idx]:
                        poem_max_scores[poem_idx] = score

                # Sort poems by max score descending, take top 20 for reranking
                sorted_poems = sorted(poem_max_scores.items(), key=lambda x: x[1], reverse=True)
                top_n_for_rerank = 20
                candidate_indices = [idx for idx, _ in sorted_poems[:top_n_for_rerank]]
                candidate_scores = {idx: sc for idx, sc in sorted_poems[:top_n_for_rerank]}
            else:
                # Legacy single-vector index (backwards compatibility)
                top_n_for_rerank = 20
                top_vec_indices = np.argsort(scores)[::-1][:top_n_for_rerank]
                candidate_indices = list(top_vec_indices)
                candidate_scores = {int(idx): float(scores[idx]) for idx in top_vec_indices}

            print(f"[QA] Multi-vector retrieval: {len(candidate_indices)} candidates (top bi-encoder scores)")
            for i, idx in enumerate(candidate_indices):
                print(f"  [{i}] BiEnc Score: {candidate_scores[idx]:.4f} | Poem: {repr(poems_list[idx]['poem'][:80])}")

            # 4. Cross-encoder reranking
            if reranker_model and len(candidate_indices) > 0:
                print(f"[QA] Reranking {len(candidate_indices)} candidates with CrossEncoder...")
                rerank_pairs = []
                for idx in candidate_indices:
                    # Feed concern + explanation to the reranker
                    doc_text = poems_list[idx]["searchable_explanation"]
                    rerank_pairs.append([concern, doc_text])

                rerank_scores = reranker_model.predict(rerank_pairs)

                # Combine: sort by reranker score
                reranked = sorted(
                    zip(candidate_indices, rerank_scores),
                    key=lambda x: x[1],
                    reverse=True
                )

                print(f"[QA] Reranked Top {len(reranked)}:")
                for i, (idx, rscore) in enumerate(reranked):
                    print(f"  [{i}] Rerank: {rscore:.4f} | BiEnc: {candidate_scores[idx]:.4f} | Poem: {repr(poems_list[idx]['poem'][:80])}")

                # Take top 3 after reranking
                top_indices = [idx for idx, _ in reranked[:3]]
            else:
                # No reranker available, use bi-encoder scores directly
                top_indices = candidate_indices[:3]

            top_poems = [poems_list[idx] for idx in top_indices]

            # Print final selection
            print("[QA] Final Top 3 candidate poems:")
            for i, p in enumerate(top_poems):
                print(f"  [{i}] Poem: {repr(p['poem'])}")
                print(f"      Explanation: {p['searchable_explanation']}")

            # 5. Use gpt-5 with CA_DAO_VAN_DAP_KEY
            from openai import OpenAI
            api_key = os.getenv("CA_DAO_VAN_DAP_KEY")
            if not api_key:
                print("WARNING: CA_DAO_VAN_DAP_KEY environment variable is not set. Using fallback.")
                # Fallback to top poem with standard explanation if key is missing
                selected_poem = top_poems[0]
                return jsonify({
                    "poem": selected_poem["poem"],
                    "advice": selected_poem["searchable_explanation"],
                    "source": "",
                    "attempts": 1
                })

            client = OpenAI(api_key=api_key)
            system_prompt = (
                "Bạn là một chuyên gia về văn hóa dân gian Việt Nam và tâm lý học đời sống.\n"
                "Nhiệm vụ của bạn là lắng nghe nỗi lòng/câu hỏi của người dùng và chọn ra triết lý phù hợp nhất để khuyên giải họ từ 3 gợi ý.\n\n"
                "QUY TẮC CHỌN LỰA:\n"
                "1. Hãy phân tích kỹ tâm tư của người dùng. Họ đang lo lắng về sự nghiệp, tình yêu, việc học tập, gia đình, hay sự kiên trì?\n"
                "2. Đối chiếu ý nghĩa thực tế (thông qua phần giải thích gốc và tóm tắt) với nỗi lòng của người dùng để chọn ra nội dung phù hợp nhất với hoàn cảnh của họ.\n"
                "3. Nếu cả 3 gợi ý đều không hoàn toàn khớp, hãy chọn nội dung có triết lý sống gần gũi nhất.\n\n"
                "QUY TẮC VIẾT CÂU TRẢ LỜI:\n"
                "1. Bạn PHẢI xưng hô với người dùng là 'bạn' và xưng mình là 'tôi'.\n"
                "2. Tuyệt đối KHÔNG đề cập đến các từ 'ca dao', 'tục ngữ', 'bài ca dao này/kia', 'câu ca dao' hay bất kỳ từ ngữ nào ám chỉ đây là một bài ca dao/tục ngữ trong câu trả lời.\n"
                "3. Hãy giải thích ngắn gọn trong 1-2 câu về nghĩa đen, nghĩa thực tế trực tiếp của các câu thơ.\n"
                "4. Sau phần giải thích nghĩa đen trực tiếp đó, hãy đưa ra những lời khuyên, khuyến nghị (recommendation) hành động thiết thực, tích cực tiếp theo dành cho họ để áp dụng vào hoàn cảnh hiện tại.\n"
                "5. Viết câu trả lời thật thanh tao, tinh tế, giàu tình cảm bằng tiếng Việt.\n\n"
                "Bạn PHẢI trả về kết quả dưới dạng JSON với định dạng sau:\n"
                "{\n"
                "  \"selected_index\": <chỉ số được chọn, là 0, 1 hoặc 2>,\n"
                "  \"elegant_answer\": \"<câu trả lời bằng tiếng Việt, xưng hô 'bạn'>\"\n"
                "}"
            )
            
            user_prompt = f"Nỗi lòng/Câu hỏi của người dùng:\n\"{concern}\"\n\n"
            user_prompt += "Dưới đây là 3 bài ca dao ứng viên:\n"
            for idx, p in enumerate(top_poems):
                orig_explanation = get_poem_explanation_from_db(p["poem"])
                user_prompt += f"Bài ca dao [{idx}]:\n"
                user_prompt += f"- Lời thơ: {p['poem']}\n"
                user_prompt += f"- Giải thích chi tiết (Gốc): {orig_explanation}\n"
                user_prompt += f"- Giải thích tóm tắt (Tìm kiếm): {p['searchable_explanation']}\n\n"
            user_prompt += "Hãy chọn bài phù hợp nhất (0, 1 hoặc 2) và viết câu trả lời."

            response = client.chat.completions.create(
                model="gpt-5",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"}
            )

            result_json = json.loads(response.choices[0].message.content)
            selected_idx = int(result_json.get("selected_index", 0))
            if selected_idx not in [0, 1, 2]:
                selected_idx = 0
            elegant_answer = result_json.get("elegant_answer", "")
            selected_poem = top_poems[selected_idx]

            return jsonify({
                "poem": selected_poem["poem"],
                "advice": elegant_answer,
                "source": "",
                "attempts": 1
            })

        except Exception as e:
            print(f"Error in embedding/LLM processing: {e}")
            import traceback
            traceback.print_exc()
            # Fallback if processing fails but we have retrieved candidates
            if 'top_poems' in locals() and top_poems:
                selected_poem = top_poems[0]
                return jsonify({
                    "poem": selected_poem["poem"],
                    "advice": selected_poem["searchable_explanation"],
                    "source": "",
                    "attempts": 1
                })

    # Legacy static/random fallback if advice_index is not loaded or search is disabled
    if not advice_data:
        return jsonify({"error": "Advice data not loaded"}), 500

    selected_poem = random.choice(advice_data)
    poem_text = selected_poem.get("poem", selected_poem.get("Poem", ""))
    static_advice = selected_poem.get("searchable_explanation", selected_poem.get("Advice", ""))

    return jsonify({
        "poem": poem_text,
        "advice": static_advice,
        "source": source_text,
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
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_wiki_base_alphabet ON wiki (base_alphabet)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_wiki_category ON wiki (category)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_wiki_poem ON wiki (poem)")
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
