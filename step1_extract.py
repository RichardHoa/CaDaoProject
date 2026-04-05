#!/usr/bin/env python3
"""
Step 1: Extract meanings and keywords from Vietnamese folk poetry.
Uses local Ollama with sailor2:20b to analyze poems and extract semantic keywords.

NOTE: Remote Apollo API code is saved below (commented out) for future use.
"""

import ollama
import pandas as pd
import time
import os
from dotenv import load_dotenv
from collections import defaultdict

# Load API key
load_dotenv()
APOLLO_API_KEY = os.getenv("APOLLO_API_KEY")
APOLLO_URL = "https://apollo.quocanmeomeo.io.vn"

# Initialize Ollama Client
client = ollama.Client(host=APOLLO_URL, headers={'Authorization': f'Bearer {APOLLO_API_KEY}'})

# Local model config
LOCAL_MODEL = "sailor2:20b"
INPUT_CSV = "input.csv"
OUTPUT_CSV = "output.csv"


def validate_meaning(poem, meaning):
    """
    Self-correcting validator: Checks if the meaning output meets requirements.
    Returns (is_valid, feedback_message)
    """
    if not meaning or not meaning.strip():
        return False, "Phản hồi trống"

    meaning = meaning.strip()

    # Check for bullet points or markdown
    if any(c in meaning for c in ['*', '-', '•', '[', ']', '#']):
        return False, "KHÔNG DÙNG MARKDOWN: Bỏ *, -, •, [], #"

    # Check for required keyword section
    has_tu_khoa = "Keywords:" in meaning or "Từ khóa:" in meaning

    if not has_tu_khoa:
        return False, "THIẾU TỪ KHÓA: Phải có 'Keywords:' ở cuối"

    # Check minimum length
    if len(meaning) < 50:
        return False, f"QUÁ NGẮN: Cần ≥50 ký tự, hiện có {len(meaning)}"

    # Check for Vietnamese characters
    vietnamese_chars = set('ơưăêôâđáàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđ')
    if not any(c in vietnamese_chars for c in meaning.lower()):
        return False, "THIẾU TIẾNG VIỆT"

    # Check structure: analysis + keywords
    parts = meaning.split("Keywords:") if "Keywords:" in meaning else meaning.split("Từ khóa:")

    if len(parts) < 2 or not parts[0].strip():
        return False, "THIẾU PHÂN TÍCH: Cần đoạn văn trước phần từ khóa"

    keywords_part = parts[1].strip()
    if not keywords_part:
        return False, "TỪ KHÓA TRỐNG"

    # Check keywords is a simple comma-separated list (no brackets, bullets)
    if any(c in keywords_part for c in ['[', ']', '•', '-']):
        return False, "TỪ KHÓA SAI ĐỊNH DẠNG: Chỉ dùng dấu phẩy ngăn cách"

    keywords = [k.strip() for k in keywords_part.split(',') if k.strip()]
    if len(keywords) > 5:
        return False, f"QUÁ NHIỀU TỪ KHÓA: Tối đa 5, hiện có {len(keywords)}"

    return True, ""


def parse_keywords(meaning):
    """Extract keywords from the meaning string."""
    if "Keywords:" in meaning:
        parts = meaning.split("Keywords:")
    elif "Từ khóa:" in meaning:
        parts = meaning.split("Từ khóa:")
    else:
        return []

    keywords_part = parts[1].strip()
    keywords = [k.strip() for k in keywords_part.split(',') if k.strip()]
    return keywords


def get_meaning(poem, previous_feedback=None):
    if not poem or pd.isna(poem):
        return "", []

    # Build user message: feedback FIRST, then the task
    if previous_feedback:
        user_content = (
            f"**SỬA LỖI:** {previous_feedback}\n\n"
            f"**YÊU CẦU:** Viết lại phân tích cho bài ca dao sau, ĐÚNG ĐỊNH DẠNG ví dụ mẫu.\n\n"
            f'Bài ca dao: "{poem}"'
        )
    else:
        user_content = f'Bài ca dao: "{poem}"'

    system_content = (
        "Bạn là chuyên gia phê bình văn học dân gian Việt Nam.\n\n"
        "**NHIỆM VỤ:**\n"
        "1. Phân tích ý nghĩa ẩn sâu bài ca dao (tâm lý, ý đồ người nói, triết lý nhân sinh)\n"
        "2. Liệt kê TỐI ĐA 5 từ khóa mô tả CHỦ ĐỀ CHÍNH thay vì liệt kê nội dung bề mặt\n\n"
        "**ĐỊNH DẠNG BẮT BUỘC:**\n"
        "[2-3 câu phân tích]. Keywords: từ1, từ2, từ3, từ4, từ5\n\n"
        "**QUY TẮC:**\n"
        "- Viết tiếng Việt có dấu\n"
        "- KHÔNG dùng *, **, #, bullet points (-, •), markdown\n"
        "- Phải có 'Keywords:' ở cuối\n"
        "- Từ khóa: tập trung vào bản chất (ví dụ: 'vòng lặp nhân quả', 'đạo hiếu', 'nghịch lý xã hội')\n"
        "- Tối đa 5 từ khóa, cách nhau bởi dấu phẩy"
    )

    # Build few-shot examples
    few_shot_messages = [
        {'role': 'user', 'content': 'Bài: "Công cha như núi Thái Sơn / Nghĩa mẹ như nước trong nguồn chảy ra"'},
        {'role': 'assistant', 'content': 'Tôn vinh đạo hiếu và lòng biết ơn vô hạn đối với công ơn sinh thành. Khẳng định giá trị vĩnh cửu của tình thân và trách nhiệm của con cái với cội nguồn. Keywords: đạo hiếu, tình thâm, biết ơn, nghĩa vụ, cội nguồn'},
        {'role': 'user', 'content': 'Bài: "Con dao vàng rọc lá trầu vàng / Mắt anh anh liếc, mắt nàng nàng đưa."'},
        {'role': 'assistant', 'content': 'Ca ngợi sự môn đăng hộ đối và vẻ đẹp của sự tương xứng trong tình yêu (trai tài gái sắc). Thể hiện nghệ thuật thả thính tinh tế và sự chủ động giao cảm giữa đôi lứa. Keywords: tình yêu, tương xứng, làm quen, tâm đầu ý hợp, văn hóa giao duyên'},
        {'role': 'user', 'content': 'Bài: "Gió đưa bụi chuối sau hè / Anh mê vợ bé bỏ bè con thơ."'},
        {'role': 'assistant', 'content': 'Phê phán sự thiếu trách nhiệm và thói trăng hoa phá vỡ hạnh phúc gia đình. Nhấn mạnh nỗi đau bị phản bội và bi kịch tan vỡ do sự thay lòng đổi dạ của người đàn ông. Keywords: ngoại tình, phản bội, gia đình tan vỡ, vô trách nhiệm, hận thù tình cảm'},
        {'role': 'user', 'content': 'Bài: "Cô kia khăn trắng tang ai / Nhất tang cha mẹ, thứ hai tang chồng..."'},
        {'role': 'assistant', 'content': 'Mượn bối cảnh trang nghiêm để thực hiện lời hát ghẹo táo bạo nhằm phá vỡ khoảng cách. Thể hiện sự lém lỉnh, khao khát kết thân và nghệ thuật làm quen lãng mạn của nam giới ngày xưa. Keywords: hát ghẹo, tán tỉnh, lém lỉnh, chủ động, giao duyên'},
        {'role': 'user', 'content': user_content}
    ]

    messages = [{'role': 'system', 'content': system_content}] + few_shot_messages

    try:
        print(f"  [DEBUG] Sending request with poem: {poem[:80]}...")
        if previous_feedback:
            print(f"  [DEBUG] With feedback: {previous_feedback[:100]}...")

        response = client.chat(
            model=LOCAL_MODEL,
            messages=messages,
            options={"temperature": 0.7}
        )

        result = response.message.content
        print(f"  [DEBUG] Raw content: {repr(result[:150])}")

        if not result or not result.strip():
            print(f"  [WARN] Empty response from model")
            return "", []

        result = result.strip()
        result = result.replace("*", "")

        # Split analysis and keywords
        analysis = result
        keywords = []
        if "Keywords:" in result:
            parts = result.split("Keywords:")
            analysis = parts[0].strip()
            keywords = [k.strip() for k in parts[1].split(',') if k.strip()]
        elif "Từ khóa:" in result:
            parts = result.split("Từ khóa:")
            analysis = parts[0].strip()
            keywords = [k.strip() for k in parts[1].split(',') if k.strip()]

        print(f"  [DEBUG] Keywords ({len(keywords)}): {keywords}")
        return analysis, keywords

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return "Error", []


def main():
    print(f"--- Bước 1: Trích xuất ý nghĩa ca dao (sailor2:20b) ---")
    df = pd.read_csv(INPUT_CSV, dtype={'Meaning': str, 'Keywords': str})

    if 'Meaning' not in df.columns:
        df['Meaning'] = ""
    if 'Keywords' not in df.columns:
        df['Keywords'] = ""

    df['Meaning'] = df['Meaning'].fillna("")
    df['Keywords'] = df['Keywords'].fillna("")

    for index, row in df.iterrows():
        if row['Meaning'].strip() and row['Keywords'].strip():
            continue

        print(f"Xử lý {index + 1}/{len(df)}...")

        meaning = ""
        keywords = []
        retry_count = 0
        last_feedback = None
        max_retries = 20

        while retry_count < max_retries:
            analysis, keywords = get_meaning(row['Poem'], previous_feedback=last_feedback)

            if analysis and analysis != "Error" and keywords:
                # For validation, we pass the full string to match the current validator logic
                full_result = f"{analysis}. Keywords: {', '.join(keywords)}"
                is_valid, feedback = validate_meaning(row['Poem'], full_result)
                if is_valid:
                    meaning = analysis
                    break
                else:
                    last_feedback = feedback
                    retry_count += 1
                    time.sleep(0.5)
            else:
                last_feedback = "Phản hồi lỗi hoặc trống."
                retry_count += 1
                time.sleep(0.5)

        if retry_count >= max_retries:
            print(f"  [DEBUG] Failed after {max_retries} retries. Poem: {row['Poem']}")

        df.at[index, 'Meaning'] = meaning
        df.at[index, 'Keywords'] = ', '.join(keywords) if keywords else ""

        # Save after each row
        df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
        print(f"  [SAVED] Row {index + 1}")

    print(f"--- Hoàn thành! Kết quả tại {OUTPUT_CSV} ---")

main()

# ============================================================================
# REMOTE APOLLO API CODE (Saved for future use)
# ============================================================================
"""
To use remote Apollo API instead, replace the imports and get_meaning() function:

import socket
import requests
import json
from dotenv import load_dotenv
import os

# Load API key from .env
load_dotenv()
APOLLO_API_KEY = os.getenv("APOLLO_API_KEY")

# Force IPv4 to prevent tunnel timeout
socket.getaddrinfo = lambda *args, **kwargs: [(socket.AF_INET, socket.SOCK_STREAM, 6, '', (args[0], args[1]))]

APOLLO_URL = "https://apollo.quocanmeomeo.io.vn/v1/chat/completions"
APOLLO_MODEL = "qwen3.5:35b"

def get_meaning(poem, previous_feedback=None):
    if not poem or pd.isna(poem):
        return "", []

    if previous_feedback:
        user_content = (
            f"**SỬA LỖI:** {previous_feedback}\n\n"
            f"**YÊU CẦU:** Viết lại phân tích cho bài ca dao sau, ĐÚNG ĐỊNH DẠNG ví dụ mẫu.\n\n"
            f'Bài ca dao: "{poem}"'
        )
    else:
        user_content = f'Bài ca dao: "{poem}"'

    system_content = (
        "Bạn là chuyên gia phê bình văn học dân gian Việt Nam.\n\n"
        "**NHIỆM VỤ:**\n"
        "1. Phân tích ý nghĩa ẩn sau bài ca dao (tâm lý, ý đồ người nói)\n"
        "2. Liệt kê từ khóa người dùng hay tìm kiếm\n\n"
        "**ĐỊNH DẠNG BẮT BUỘC:**\n"
        "[2-3 câu phân tích]. Từ khóa: từ1, từ2, từ3\n\n"
        "**QUY TẮC:**\n"
        "- Viết tiếng Việt có dấu\n"
        "- KHÔNG dùng *, **, #, bullet points (-, •), markdown\n"
        "- Phải có 'Từ khóa:' ở cuối\n"
        "- Từ khóa: danh sách từ đơn giản, cách nhau bởi dấu phẩy, KHÔNG ngoặc vuông"
    )

    few_shot_messages = [
        {'role': 'user', 'content': 'Bài: "Công cha như núi Thái Sơn / Nghĩa mẹ như nước trong nguồn chảy ra"'},
        {'role': 'assistant', 'content': 'Tôn vinh đạo hiếu và lòng biết ơn vô hạn đối với công ơn sinh thành. Khẳng định giá trị vĩnh cửu của tình thân và trách nhiệm của con cái với cội nguồn. Từ khóa: đạo hiếu, cha mẹ, biết ơn, nghĩa vụ, lòng hiếu thảo, cội nguồn, gia đình'},
        {'role': 'user', 'content': 'Bài: "Con dao vàng rọc lá trầu vàng / Mắt anh anh liếc, mắt nàng nàng đưa."'},
        {'role': 'assistant', 'content': 'Ca ngợi sự môn đăng hộ đối và vẻ đẹp của sự tương xứng trong tình yêu (trai tài gái sắc). Thể hiện nghệ thuật thả thính tinh tế và sự chủ động giao cảm giữa đôi lứa. Từ khóa: tình yêu, tương xứng, trai tài gái sắc, môn đăng hộ đối, thả thính, làm quen, tâm đầu ý hợp'},
        {'role': 'user', 'content': 'Bài: "Gió đưa bụi chuối sau hè / Anh mê vợ bé bỏ bè con thơ."'},
        {'role': 'assistant', 'content': 'Phê phán sự thiếu trách nhiệm và thói trăng hoa phá vỡ hạnh phúc gia đình. Nhấn mạnh nỗi đau bị phản bội và bi kịch tan vỡ do sự thay lòng đổi dạ của người đàn ông. Từ khóa: ngoại tình, phản bội, vợ bé, trà xanh, gia đình tan vỡ, vô trách nhiệm, hận thù tình cảm'},
        {'role': 'user', 'content': 'Bài: "Cô kia khăn trắng tang ai / Nhất tang cha mẹ, thứ hai tang chồng..."'},
        {'role': 'assistant', 'content': 'Mượn bối cảnh trang nghiêm để thực hiện lời hát ghẹo táo bạo nhằm phá vỡ khoảng cách. Thể hiện sự lém lỉnh, khao khát kết thân và nghệ thuật làm quen lãng mạn của nam giới ngày xưa. Từ khóa: hát ghẹo, tán tỉnh, lém lỉnh, chủ động, làm quen, giao duyên, tỏ tình'},
        {'role': 'user', 'content': user_content}
    ]

    messages = [{'role': 'system', 'content': system_content}] + few_shot_messages

    try:
        print(f"  [DEBUG] Sending request with poem: {poem[:80]}...")
        if previous_feedback:
            print(f"  [DEBUG] With feedback: {previous_feedback[:100]}...")

        payload = {
            "model": APOLLO_MODEL,
            "messages": messages,
            "stream": True
        }

        headers = {
            "Authorization": f"Bearer {APOLLO_API_KEY}",
            "Content-Type": "application/json"
        }

        response = requests.post(APOLLO_URL, headers=headers, json=payload, timeout=120, stream=True)
        response.raise_for_status()

        result = ""
        for line in response.iter_lines():
            if line:
                decoded_line = line.decode('utf-8').replace('data: ', '')
                if decoded_line != "[DONE]":
                    try:
                        chunk = json.loads(decoded_line)
                        content = chunk['choices'][0]['delta'].get('content', '')
                        if content:
                            result += content
                    except json.JSONDecodeError:
                        continue

        print(f"  [DEBUG] Raw content: {repr(result[:150])}")

        if not result or not result.strip():
            print(f"  [WARN] Empty response from model")
            return "", []

        result = result.strip()
        result = result.replace("*", "")

        keywords = parse_keywords(result)

        print(f"  [DEBUG] Keywords: {keywords}")
        return result, keywords

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return "Error", []
"""
