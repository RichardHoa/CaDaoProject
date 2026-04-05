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
import numpy as np
import faiss
import time
import os
import csv
import random
import ollama
from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template
from sklearn.preprocessing import normalize

# Load API key
load_dotenv()
APOLLO_API_KEY = os.getenv("APOLLO_API_KEY")
APOLLO_URL = "https://apollo.quocanmeomeo.io.vn"

# Initialize Ollama Client
client = ollama.Client(host=APOLLO_URL, headers={'Authorization': f'Bearer {APOLLO_API_KEY}'})

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
TARGET_RESULTS = 20
TOP_K_KEYWORDS = 5
LEARNING_DATA_FILE = "data/learning_data.json"
INTERPRETATIONS_FILE = "data/user_interpretations.txt"
ADVICE_FILE = "advice.csv"

# Qwen3 Instruction-Aware Embedding
TASK_INSTRUCTION = "Given a search query for Vietnamese folk poetry, retrieve relevant poems or passages that accurately match the user's intent."

def get_detailed_instruct(query: str) -> str:
    """Prepend the task instruction as per Qwen3 documentation."""
    return f'Instruct: {TASK_INSTRUCTION}\nQuery:{query}'

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


def embed_text(text, use_instruction=True):
    """Create embedding for text using Ollama with a fallback/retry."""
    import time

    while True:
        try:
            # Apply Qwen3 instruction-aware prompt only if requested (asymmetric)
            # Use raw text for keyword-to-keyword matching (symmetric)
            instructed_query = get_detailed_instruct(text) if use_instruction else text
            
            response = client.embeddings(
                model="qwen3-embedding:8b", prompt=instructed_query
            )
            emb = np.array([response["embedding"]], dtype=np.float32)
            # Ensure the query matches the index dimension
            return normalize(emb, norm="l2")
        except Exception as e:
            print(f"Ollama embedding error in Step 3: {e}. Retrying...")
            time.sleep(1)


def expand_query_with_ai(query):
    """
    Use an LLM (sailor2:20b) to expand a short query into a descriptive paragraph.
    This helps match the semantic density of poem meanings.
    """
    try:
        print(f"  [AI] Expanding query '{query}' with AI...")
        system_prompt = (
            "Bạn là một engine phân tích ngữ nghĩa cho Ca Dao Vietnamese. "
            "Mục tiêu là tạo ra một đoạn mô tả ngắn gọn, súc tích về ý nghĩa tâm lý của chủ đề: '{query}'.\n"
            "LUẬT NGHIÊM NGẶT ĐỂ TRÁNH HALLUCINATION:\n"
            "1. BẮT ĐẦU NGAY LẬP TỨC bằng một động từ (ví dụ: 'Thể hiện...', 'Khám phá...', 'Ca ngợi...').\n"
            "2. TUYỆT ĐỐI KHÔNG lặp lại từ khóa, không thêm tiêu đề, không ghi 'Kết quả:' hay 'Phân tích:'.\n"
            "3. TUYỆT ĐỐI KHÔNG dùng markdown (không dùng dấu **, không dùng dấu #).\n"
            "4. TUYỆT ĐỐI KHÔNG bịa ra hình ảnh cụ thể (con vật, cây cối, địa danh) nếu không có trong từ khóa.\n"
            "5. Độ dài: Tối đa 2-3 câu, gộp thành 1 đoạn duy nhất, không xuống dòng.\n\n"
            "VÍ DỤ MẪU:\n"
            "Chủ đề: 'nỗi nhớ'\n"
            "Kết quả: Thể hiện tâm trạng da diết và sự hoài niệm sâu sắc về những kỷ niệm đã qua. Khám phá sự trống trải trong tâm hồn và khao khát được gặp lại người thương trong không gian tĩnh lặng.\n"
        )
        user_prompt = f"Hãy tạo một đoạn phân tích semantic (chỉ bao gồm nội dung phân tích, không có gì khác) cho từ khóa: '{query}'"
        
        response = client.chat(
            model="sailor2:20b",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            options={
                "temperature": 0.2, 
                "top_p": 0.9,
                "presence_penalty": 0.0,
                "frequency_penalty": 0.0
            }
        )
        expansion = response.message.content.strip()
        print(f"  [AI] Expansion: {expansion}")
        return expansion
    except Exception as e:
        print(f"  [WARN] AI Expansion failed: {e}")
        return None


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
        1: "Khớp theo ý nghĩa"
    }

    for i, idx in enumerate(indices[0]):
        if idx != -1 and scores[0][i] >= threshold:
            poem_id = vector_to_poem_map[idx]
            match_type_id = idx % 3
            
            # Skip keyword vector matches in Step 1/1.5 (we prefer literal matches in Step 2)
            if match_type_id == 2:
                continue
                
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
    2. Meaning
    3. Keywords
    Returns list of (poem_id, score) where score is 1.0 for literal matches.
    """
    query_lower = query.lower().strip()
    results = []
    
    for i, p in enumerate(poems_data):
        poem_text = p.get("poem", "").lower()
        meaning = p.get("meaning", "").lower()
        keywords = p.get("keywords", "").lower()
        
        # Check for literal match
        if query_lower in poem_text or query_lower in meaning or query_lower in keywords:
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
    2. AI Expansion Search (Deep Semantic) - Using LLM to expand query
    3. Keyword Expansion + Literal Lookup (High-precision fallback)
    """
    query_lower = query.lower().strip()
    seen_poem_indices = set()
    results = []

    print(f"\n[SEARCH] Query: '{query}'")

    # Step 1: Semantic Match (Original Query)
    if len(results) < 15:
        print(f"  [1] Searching poems semantically with '{query_lower}'...")
        query_embedding = embed_text(query_lower)
        # Cap at 15 as requested
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
                        "meaning": poem["meaning"],
                        "keywords": poem.get("keywords", ""),
                        "matched_keyword": match_type,
                        "score": score,
                    }
                )

        print(f"  [RESULT] Total: {len(results)} poems after original semantic search")

    # Step 2: Keyword Expansion + Literal Lookup (Symmetric Matching)
    if len(results) < 15:
        print(f"  [2] Need more, searching similar keywords (symmetric, cap at 15)...")
        # For Tier 2, we use a RAW embedding (no instruction) to match the keyword index
        raw_query_embedding = embed_text(query_lower, use_instruction=False)
        similar_keywords = search_keywords(raw_query_embedding, TOP_K_KEYWORDS)
        print(f"  [KEYWORDS] Found {len(similar_keywords)} similar keywords")

        for kw, kw_score in similar_keywords:
            if len(results) >= 15:
                break

            print(f"  [2.x] Searching with keyword '{kw}' (similarity: {kw_score:.2f})...")
            # USE LITERAL LOOKUP FOR KEYWORDS
            poem_matches = search_poems_by_keyword_literal(kw, 15 - len(results))

            for poem_idx, score, match_type in poem_matches:
                if poem_idx not in seen_poem_indices:
                    seen_poem_indices.add(poem_idx)
                    poem = poems_data[poem_idx]
                    results.append(
                        {
                            "poem": poem["poem"],
                            "meaning": poem["meaning"],
                            "keywords": poem.get("keywords", ""),
                            "matched_keyword": match_type,
                            "score": score,
                        }
                    )

        print(f"  [RESULT] Total: {len(results)} poems after keyword expansion")

    # Step 3: AI Expansion Search (Deep Semantic)
    if len(results) < top_n:
        print(f"  [3] Deep searching with AI Analysis to reach {top_n}...")
        ai_expansion = expand_query_with_ai(query_lower)
        if ai_expansion:
            expansion_embedding = embed_text(ai_expansion)
            # We use a slightly lower threshold for expansion as it's more descriptive
            expansion_matches = search_poems_by_embedding(
                expansion_embedding, 0.5, top_n - len(results)
            )

            for poem_idx, score, match_type in expansion_matches:
                if poem_idx not in seen_poem_indices:
                    seen_poem_indices.add(poem_idx)
                    poem = poems_data[poem_idx]
                    results.append(
                        {
                            "poem": poem["poem"],
                            "meaning": poem["meaning"],
                            "keywords": poem.get("keywords", ""),
                            "matched_keyword": "Khớp theo phân tích AI",
                            "score": score,
                        }
                    )
            print(f"  [RESULT] Total: {len(results)} poems after AI analysis search")

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


def evaluate_poem_suitability(concern, poem_row):
    """
    Ask sailor2:20b if the given poem is suitable for the user's concern.
    Returns True if suitable, False otherwise.
    """
    try:
        system_prompt = (
            "Bạn là một máy kiểm tra logic. "
            "Nhiệm vụ: Kiểm tra xem câu ca dao có phù hợp chặt chẽ với vấn đề của người dùng không.\n"
            "QUY TẮC TỐI THƯỢNG:\n"
            "1. CHỈ ĐƯỢC TRẢ LỜI ĐÚNG 2 TỪ: 'PHÙ HỢP' HOẶC 'KHÔNG PHÙ HỢP'.\n"
            "2. TUYỆT ĐỐI KHÔNG GIẢI THÍCH, KHÔNG NÓI THÊM BẤT CỨ ĐIỀU GÌ.\n"
            "3. 'PHÙ HỢP' chỉ khi có logic cực kỳ mạnh mẽ. Tránh các liên hệ lỏng lẻo (ví dụ: đừng dùng 'bắt cá hai tay' cho câu hỏi về trường đại học trừ khi người dùng nói họ đang làm 2 việc).\n"
        )
        user_prompt = (
            f"Vấn đề: '{concern}'\n"
            f"Ca dao: '{poem_row['Poem']}'\n"
            f"Ý nghĩa: '{poem_row['Advice']}'\n"
            "Trả lời PHÙ HỢP hoặc KHÔNG PHÙ HỢP:"
        )
        
        response = client.chat(
            model="sailor2:20b",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            options={"temperature": 0.0}
        )
        # Extract only the first few words to avoid hallucinations
        answer_raw = response.message.content.strip().upper()
        print(f"  [QA Match] Suitability raw: {answer_raw}")
        
        if "KHÔNG PHÙ HỢP" in answer_raw or "KHONG PHU HOP" in answer_raw:
            return False
        if "PHÙ HỢP" in answer_raw or "PHU HOP" in answer_raw:
            return True
        return False
    except Exception as e:
        print(f"  [WARN] Suitability check failed: {e}")
        return True # Fallback to True if AI fails


def generate_final_advice(concern, poem_row):
    """
    Use sailor2:20b to generate a personalized advice based on the poem.
    """
    try:
        system_prompt = (
            "BẠN LÀ MỘT BẬC TÚ NHO (NHÀ NHO CŨ) ĐANG TƯ VẤN CHO HẬU THẾ. "
            "Hãy dùng ngôn từ thâm trầm, cổ kính. Tuyệt đối không dùng giọng điệu của chuyên gia hiện đại.\n\n"
            "PHONG CÁCH (BẮT BUỘC):\n"
            "- Xưng hô: Dùng 'Thưa bạn', 'Bằng hữu' hoặc ẩn chủ ngữ. TUYỆT ĐỐI KHÔNG dùng từ 'Ngươi'.\n"
            "- Cấu trúc: Đoạn 1 nhắc nhẹ về câu ca dao, không viết lại nó 100% và ý nghĩa sâu xa của nó. Đoạn 2 tập trung hoàn toàn vào lời khuyên thực tế cho vấn đề của người hỏi.\n"
            "- Từ vựng: Dùng 'chí hướng', 'vốn liếng', 'bằng hữu', 'vinh hoa', 'khốn khó'. Tránh từ hiện đại như 'chiến lược', 'kỹ năng', 'phát triển'.\n\n"
            "Ví dụ phong cách:\n"
            "- Trong các quan niệm về hôn phối của tiền nhân, ta thấy hiện lên một tư tưởng thật thanh cao và đầy tính nhân bản: ấy là lối hôn nhân vị luyến ái.\n"
            "- Những câu hát huê tình, đôi khi pha chút dí dỏm, trào lộng như thế, xin bạn chớ lầm hiểu là sự khinh nhờn đối với phận má đào. Trong tâm thức của người dân quê mình, người con gái không hề cam chịu nép mình trong bóng tối, mà trái lại, vẫn có quyền cất lên tiếng nói riêng, bày tỏ ý nhị mà sắc sảo những suy nghĩ của mình, chẳng chút e sợ những định kiến khắt khe của thế gian.\n"
            "CẤM (PHẠM QUY SẼ BỊ PHẠT):\n"
            "1. TUYỆT ĐỐI KHÔNG trích dẫn lại câu ca dao (không viết lại nguyên văn bất kỳ phần nào của câu thơ).\n"
            "2. CẤM dùng từ 'Ngươi'.\n"
            "3. CẤM dùng tiêu đề, không in đậm, không dùng markdown (#, **).\n"
            "4. CẤM viết thơ, vè. CHỈ VIẾT VĂN XUÔI.\n"
            "5. CẤM lặp lại cấu trúc 'Ý nghĩa:', 'Lời khuyên:'.\n\n"
            "ĐỊNH DẠNG: Viết đúng 2 đoạn văn thuần túy, phân tách bằng dòng trống. Bắt đầu ngay vào nội dung."
        )
        user_prompt = (
            f"Vấn đề: '{concern}'\n"
            f"Ca dao: '{poem_row['Poem']}'\n"
            f"Ý nghĩa: '{poem_row['Advice']}'\n"
            "Hãy nhắc về ý nghĩa của câu ca dao trên và đưa ra lời khuyên sâu sắc gồm đúng 2 đoạn văn văn xuôi (TUYỆT ĐỐI KHÔNG trích dẫn lại câu ca dao trên)."
        )
        
        import re
        
        response = client.chat(
            model="sailor2:20b",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            options={
                "temperature": 0.6,
                "top_p": 0.9
            }
        )
        raw_advice = response.message.content.strip()
        
        # AGGRESSIVE POST-PROCESSING
        raw_paragraphs = raw_advice.split('\n\n')
        processed_paragraphs = []
        
        for rp in raw_paragraphs:
            # Clean each paragraph
            p_lines = [l.strip() for l in rp.split('\n') if l.strip() and not l.strip().startswith('#')]
            p_text = ' '.join(p_lines)
            p_text = re.sub(r'^[^\n:]{2,40}:\s*', '', p_text)
            p_text = re.sub(r'[\*_#]', '', p_text)
            p_text = re.sub(r'^[\-\*\+\d\.\s]+', '', p_text) # Remove bullets
            if p_text:
                processed_paragraphs.append(p_text)
        
        # If the AI failed to give 2 paragraphs, we try to split by sentences
        if len(processed_paragraphs) == 1:
            all_text = processed_paragraphs[0]
            sentences = re.split(r'(?<=[.!?])\s+', all_text)
            if len(sentences) >= 2:
                # Split in half
                n = len(sentences)
                p1 = ' '.join(sentences[:n//2])
                p2 = ' '.join(sentences[n//2:])
                processed_paragraphs = [p1, p2]
        
        return '\n\n'.join(processed_paragraphs[:2])
    except Exception as e:
        print(f"  [ERROR] Final advice generation failed: {e}")
        return f"Dựa trên câu ca dao: '{poem_row['Poem']}', tôi khuyên bạn: {poem_row['Advice']}"


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
    QA endpoint - Iteratively find a suitable poem and return AI advice.
    Returns JSON { "poem": ..., "advice": ..., "source": ..., "attempts": ... }
    """
    data = request.json
    concern = data.get("concern", "")

    if not concern:
        return jsonify({"error": "No concern provided"}), 400

    if not advice_data:
        return jsonify({"error": "Advice data not loaded"}), 500

    print(f"\n[QA] User concern: '{concern}'")
    
    selected_poem = None
    attempts = 0
    max_attempts = 20

    # Shuffle to ensure randomness without repeating the same sequence too often
    # though with small data it doesn't matter much
    available_indices = list(range(len(advice_data)))
    random.shuffle(available_indices)

    for i in range(min(max_attempts, len(available_indices))):
        attempts += 1
        idx = available_indices[i]
        poem_row = advice_data[idx]
        
        print(f"  [QA] Attempt {attempts}: Testing poem '{poem_row['Poem'][:50]}...'")
        
        # Check suitability
        if evaluate_poem_suitability(concern, poem_row):
            selected_poem = poem_row
            print(f"  [QA] Found suitable poem on attempt {attempts}!")
            break

    # If no suitable poem found after 5 attempts, use the last one tested
    if not selected_poem:
        selected_poem = advice_data[available_indices[min(max_attempts, len(available_indices)) - 1]]
        print(f"  [QA] No perfect match found. Using best fallback.")

    # Generate final advice
    ai_advice = generate_final_advice(concern, selected_poem)

    return jsonify({
        "poem": selected_poem["Poem"],
        "advice": ai_advice,
        "source": selected_poem.get("Source", ""),
        "attempts": attempts
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


# Initialize data on module load so WSGI servers (Waitress/Gunicorn) can access it
load_data()

def main():
    """Initialize and run the server"""
    print("Starting server at http://localhost:4000")
    app.run(host="0.0.0.0", port=4001, debug=True)

if __name__ == "__main__":
    main()
