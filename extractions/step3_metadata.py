import os
import json
import argparse
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

def resolve_keyword(keyword, keywords_list, embeddings_dict, threshold=0.85):
    if not isinstance(keyword, str) or not keyword.strip():
        return ""
        
    keyword = keyword.strip()
    
    # exact match
    for k in keywords_list:
        if keyword.lower() == k.lower():
            return k
    
    # Check similarity
    emb = get_embedding(keyword)
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

def process_row(poem, explanation, keywords_list, embeddings_dict, max_retries=3):
    system_prompt = """Bạn là một chuyên gia văn học phân tích ca dao Việt Nam. Nhiệm vụ của bạn là đọc bài ca dao và phần giải thích, sau đó cung cấp một định dạng JSON với 2 trường dữ liệu sau:

1. "searchable_explanation": Một lời giải thích ngắn gọn, chuẩn SEO (khoảng 2-3 câu). Ý nghĩa của bài ca dao sẽ được nhúng (embedding) dựa trên trường này để người dùng tìm kiếm theo ngữ nghĩa, do đó nó cần truyền tải súc tích ý nghĩa cốt lõi, dễ hiểu, tối ưu hóa cho tìm kiếm vector.
2. "keywords": Một mảng danh sách các từ khóa chủ đề (ví dụ: ["Tình cảm gia đình", "Phê phán xã hội"]). Hãy chọn các từ khóa trong danh sách có sẵn (nếu phù hợp). Chỉ đề xuất từ khóa mới nếu chủ đề hoàn toàn chưa có trong danh sách.

TẤT CẢ PHẢI BẰNG TIẾNG VIỆT. ĐỊNH DẠNG TRẢ VỀ PHẢI LÀ JSON CHUẨN. KHÔNG CÓ TRƯỜNG NÀO BẰNG TIẾNG ANH.

Danh sách từ khóa có sẵn:
""" + json.dumps(keywords_list, ensure_ascii=False)

    user_prompt = f"Bài ca dao:\n{poem}\n\nGiải thích:\n{explanation}"
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-5-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
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
                resolved_kw = resolve_keyword(kw, keywords_list, embeddings_dict)
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
    parser.add_argument("--start", type=int, default=0, help="Start index")
    parser.add_argument("--stop", type=int, default=10, help="Stop index")
    args = parser.parse_args()
    
    input_csv = "wiki.csv"
    output_csv = "wiki_with_metadata.csv"
    keywords_file = "keywords.json"
    
    start = max(0, args.start)
    stop = max(start, args.stop)
    num_rows = stop - start
    
    if num_rows <= 0:
        print("Start index must be less than Stop index.")
        return
        
    print(f"Loading {num_rows} rows (from {start} to {stop})...")
    # skiprows=range(1, start + 1) skips from row 1 to row `start`, preserving row 0 (header)
    df = pd.read_csv(input_csv, skiprows=range(1, start + 1), nrows=num_rows)
    
    if "searchable_explanation" not in df.columns:
        df["searchable_explanation"] = ""
    if "keywords" not in df.columns:
        df["keywords"] = ""
    
    # Load keywords list
    if os.path.exists(keywords_file):
        with open(keywords_file, "r", encoding="utf-8") as f:
            keywords_list = json.load(f)
    else:
        keywords_list = []
        
    # Load embeddings cache
    if os.path.exists(EMBEDDINGS_FILE):
        with open(EMBEDDINGS_FILE, "r", encoding="utf-8") as f:
            embeddings_dict = json.load(f)
    else:
        embeddings_dict = {}
        
    processed_count = 0
    df_chunk_to_save = []

    for idx in range(len(df)):
        real_row_num = start + idx
        print(f"Processing row {real_row_num}...")
        poem = df.at[idx, "poem"]
        explanation = df.at[idx, "explanation"]
        
        if pd.notna(poem) and pd.notna(explanation):
            expl, kws = process_row(poem, explanation, keywords_list, embeddings_dict)
            df.at[idx, "searchable_explanation"] = expl
            df.at[idx, "keywords"] = ", ".join(kws)
            
        df_chunk_to_save.append(df.iloc[[idx]])
        processed_count += 1
        
        # Save every 10 rows
        if processed_count % 10 == 0:
            print("Saving progress...")
            save_df = pd.concat(df_chunk_to_save)
            header = not os.path.exists(output_csv)
            save_df.to_csv(output_csv, mode='a', header=header, index=False)
            df_chunk_to_save = [] # Clear the saved chunk
            
            with open(keywords_file, "w", encoding="utf-8") as f:
                json.dump(keywords_list, f, ensure_ascii=False, indent=2)
            with open(EMBEDDINGS_FILE, "w", encoding="utf-8") as f:
                json.dump(embeddings_dict, f, ensure_ascii=False)
                
    # Save remaining rows
    if df_chunk_to_save:
        print("Final save...")
        save_df = pd.concat(df_chunk_to_save)
        header = not os.path.exists(output_csv)
        save_df.to_csv(output_csv, mode='a', header=header, index=False)
        
        with open(keywords_file, "w", encoding="utf-8") as f:
            json.dump(keywords_list, f, ensure_ascii=False, indent=2)
        with open(EMBEDDINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(embeddings_dict, f, ensure_ascii=False)

    print("Done!")

if __name__ == "__main__":
    main()
