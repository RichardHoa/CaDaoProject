import os
import json
import argparse
import threading
import concurrent.futures
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("CHATGPT_KEY"))

# Use a separate hidden file for embeddings so keywords.json remains clean
EMBEDDINGS_FILE = ".keywords_embeddings.json"

def get_embedding(text):
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def resolve_keyword(keyword, keywords_list, embeddings_dict, keyword_lock, threshold=0.85):
    if not isinstance(keyword, str) or not keyword.strip():
        return ""
        
    keyword = keyword.strip()
    
    with keyword_lock:
        # exact match
        for k in keywords_list:
            if keyword.lower() == k.lower():
                return k
    
    # Check similarity
    emb = get_embedding(keyword)
    
    with keyword_lock:
        # Re-check exact match in case it was added by another thread while waiting for embedding
        for k in keywords_list:
            if keyword.lower() == k.lower():
                return k
                
        best_match = None
        best_sim = -1
        
        for k, v_emb in embeddings_dict.items():
            sim = cosine_similarity(emb, v_emb)
            if sim > best_sim:
                best_sim = sim
                best_match = k
                
        if best_sim >= threshold:
            return best_match
        
        # If no match > threshold, it's a new keyword
        keywords_list.append(keyword)
        embeddings_dict[keyword] = emb
        return keyword

def process_row(poem, explanation, keywords_list, embeddings_dict, keyword_lock, max_retries=3):
    system_prompt = """Bạn là một chuyên gia văn học phân tích Thành Ngữ, Tục Ngữ, Ca Dao Việt Nam. Nhiệm vụ của bạn là đọc kỹ câu/bài và phần giải thích, sau đó cung cấp một định dạng JSON với 2 trường dữ liệu sau:

1. "searchable_explanation": Một lời giải thích ngắn gọn, chuẩn SEO (khoảng 2-3 câu). Ý nghĩa của câu/bài sẽ được nhúng (embedding) dựa trên trường này để người dùng tìm kiếm theo ngữ nghĩa, do đó nó cần truyền tải súc tích ý nghĩa cốt lõi, dễ hiểu, tối ưu hóa cho tìm kiếm vector.
2. "keywords": Mảng (array) chứa các từ khoá mô tả chính xác. Bạn PHẢI tuân thủ các yêu cầu sau:
   - Cần ít nhất 1 từ khoá về MỤC ĐÍCH của câu (ví dụ: "Phê phán", "Lời khuyên", "Kinh nghiệm sống", "Trào phúng", "Châm chọc", v.v.). Đọc kỹ phần giải thích để xác định đúng mục đích, không phải câu nào cũng là phê phán!
   - Cần ít nhất 1 từ khoá về CẢM XÚC mà câu mang lại (ví dụ: "Buồn bã", "Vui tươi", "Đau khổ", "Xót xa", "Hài hước", v.v.).
   - Cần ít nhất 2 từ khoá về CHỦ ĐỀ của câu (ví dụ: "Tình yêu", "Khoảng cách giàu nghèo", "Gia đình", "Lao động", v.v.).
   - Hãy chọn các từ khóa trong danh sách có sẵn (nếu phù hợp). Chỉ đề xuất từ khóa mới nếu chủ đề, cảm xúc hoặc mục đích hoàn toàn chưa có trong danh sách.

TẤT CẢ PHẢI BẰNG TIẾNG VIỆT. ĐỊNH DẠNG TRẢ VỀ PHẢI LÀ JSON CHUẨN. KHÔNG CÓ TRƯỜNG NÀO BẰNG TIẾNG ANH.

Danh sách từ khóa có sẵn:
"""
    user_prompt = f"Thành ngữ / Tục ngữ / Ca dao:\n{poem}\n\nGiải thích:\n{explanation}"
    
    for attempt in range(max_retries):
        try:
            with keyword_lock:
                current_keywords_json = json.dumps(keywords_list, ensure_ascii=False)
                
            response = client.chat.completions.create(
                model="gpt-5-mini",
                messages=[
                    {"role": "system", "content": system_prompt + current_keywords_json},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={ "type": "json_object" }
            )
            
            result = json.loads(response.choices[0].message.content)
            
            search_expl = result.get("searchable_explanation", "").strip()
            raw_keywords = result.get("keywords", [])
            
            if not search_expl or not raw_keywords:
                print(f"Attempt {attempt+1} returned empty results. Retrying...")
                continue
            
            # Deduplicate using embeddings
            final_keywords = []
            for kw in raw_keywords:
                resolved_kw = resolve_keyword(kw, keywords_list, embeddings_dict, keyword_lock)
                if resolved_kw and resolved_kw not in final_keywords:
                    final_keywords.append(resolved_kw)
                    
            if not final_keywords:
                print(f"Attempt {attempt+1} resulted in empty valid keywords. Retrying...")
                continue
                
            return search_expl, final_keywords
            
        except Exception as e:
            print(f"Error processing row on attempt {attempt+1}: {e}")
            
    print("Max retries reached. Returning empty results.")
    return "", []

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="wiki_with_metadata_fixed.csv", help="Input CSV file")
    parser.add_argument("--output", type=str, default="wiki_with_metadata_fixed.csv", help="Output CSV file")
    parser.add_argument("--workers", type=int, default=20, help="Number of concurrent workers")
    args = parser.parse_args()
    
    input_csv = args.input
    output_csv = args.output
    keywords_file = "keywords.json"
    
    print(f"Loading {input_csv}...")
    try:
        df = pd.read_csv(input_csv)
    except FileNotFoundError:
        print(f"Error: {input_csv} not found.")
        return
        
    # Check if necessary columns exist
    if "searchable_explanation" not in df.columns:
        df["searchable_explanation"] = ""
    if "keywords" not in df.columns:
        df["keywords"] = ""
        
    # Load JSONs
    if os.path.exists(keywords_file):
        with open(keywords_file, "r", encoding="utf-8") as f:
            keywords_list = json.load(f)
    else:
        keywords_list = []
        
    if os.path.exists(EMBEDDINGS_FILE):
        with open(EMBEDDINGS_FILE, "r", encoding="utf-8") as f:
            embeddings_dict = json.load(f)
    else:
        embeddings_dict = {}
        
    keyword_lock = threading.Lock()
    
    # Find rows that need processing
    def needs_processing(row):
        return pd.isna(row.get("searchable_explanation")) or str(row.get("searchable_explanation")).strip() == "" or \
               pd.isna(row.get("keywords")) or str(row.get("keywords")).strip() == ""

    indices_to_process = [i for i, row in df.iterrows() if needs_processing(row)]
    
    if not indices_to_process:
        print("No missing metadata found. All rows are complete!")
        return
        
    print(f"Found {len(indices_to_process)} rows missing metadata.")
    
    processed_count = 0
    df_lock = threading.Lock() # To safely update the main DataFrame
    
    def worker(idx):
        row = df.loc[idx].copy()
        poem = row.get("poem")
        explanation = row.get("explanation")
        
        if pd.notna(poem) and pd.notna(explanation) and str(poem).strip() and str(explanation).strip():
            expl, kws = process_row(poem, explanation, keywords_list, embeddings_dict, keyword_lock)
            if expl and kws:
                with df_lock:
                    df.at[idx, "searchable_explanation"] = expl
                    df.at[idx, "keywords"] = ", ".join(kws)
        return True
        
    print(f"Starting parallel processing with {args.workers} workers...")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        # submit all tasks
        futures = {executor.submit(worker, idx): idx for idx in indices_to_process}
        
        for future in concurrent.futures.as_completed(futures):
            processed_count += 1
            
            if processed_count % 10 == 0:
                print(f"Processed {processed_count}/{len(indices_to_process)} rows. Saving progress...")
                with df_lock:
                    # Save temporary then rename to be safe
                    temp_out = output_csv + ".tmp"
                    df.to_csv(temp_out, index=False)
                    os.replace(temp_out, output_csv)
                    
                # Save state thread-safely
                with keyword_lock:
                    with open(keywords_file, "w", encoding="utf-8") as f:
                        json.dump(keywords_list, f, ensure_ascii=False, indent=2)
                    with open(EMBEDDINGS_FILE, "w", encoding="utf-8") as f:
                        json.dump(embeddings_dict, f, ensure_ascii=False)
                        
    # Final save
    print("Final save...")
    with df_lock:
        temp_out = output_csv + ".tmp"
        df.to_csv(temp_out, index=False)
        os.replace(temp_out, output_csv)
    with keyword_lock:
        with open(keywords_file, "w", encoding="utf-8") as f:
            json.dump(keywords_list, f, ensure_ascii=False, indent=2)
        with open(EMBEDDINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(embeddings_dict, f, ensure_ascii=False)
            
    print("Done!")

if __name__ == "__main__":
    main()
