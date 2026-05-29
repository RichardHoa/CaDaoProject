#!/usr/bin/env python3
import os
import csv
import json
import sys
from flask import Flask, request, jsonify, render_template, send_from_directory

# Configure template and static folders relative to this script
script_dir = os.path.dirname(os.path.abspath(__file__))
app = Flask(
    __name__,
    template_folder=os.path.join(script_dir, 'templates'),
    static_folder=os.path.join(script_dir, 'static')
)

INPUT_CSV = os.path.join(script_dir, 'input.csv')
OUTPUT_CSV = os.path.join(script_dir, 'filter-dataset.csv')
STATE_JSON = os.path.join(script_dir, 'review_state.json')

# Cache of byte offsets in the input.csv file
ROW_OFFSETS = []

DEFAULT_TYPES = [
    "Khuyên bảo",
    "Châm biếm",
    "Trào lộng",
    "Tình cảm",
    "Thiên nhiên",
    "Lao động",
    "Lịch sử"
]

DEFAULT_KEYWORDS = [
    "tình yêu",
    "vợ chồng",
    "cha mẹ",
    "bạn bè",
    "con cái",
    "gia đình",
    "quê hương",
    "thân phận",
    "lao động",
    "học hành",
    "đạo đức",
    "nhân nghĩa",
    "châm biếm",
    "trào lộng",
    "thiên nhiên",
    "đất nước",
    "lịch sử",
    "tôn giáo",
    "triết lý"
]

def build_row_offsets():
    global ROW_OFFSETS
    ROW_OFFSETS = []
    if not os.path.exists(INPUT_CSV):
        print(f"Error: {INPUT_CSV} not found!")
        return
        
    print("Scanning CSV and building row offsets (O(1) seek index)...")
    with open(INPUT_CSV, 'r', encoding='utf-8', newline='') as f:
        # Skip header line
        header = f.readline()
        current_offset = f.tell()
        ROW_OFFSETS.append(current_offset)
        
        in_quotes = False
        while True:
            char = f.read(1)
            if not char:
                break
            if char == '"':
                in_quotes = not in_quotes
            elif char == '\n' and not in_quotes:
                next_offset = f.tell()
                peek = f.read(1)
                if peek:
                    f.seek(next_offset)
                    ROW_OFFSETS.append(next_offset)
                else:
                    break
    print(f"Loaded offset map for {len(ROW_OFFSETS)} rows. Scanning completed!")

def get_row_by_id(poem_id):
    if poem_id < 1 or poem_id > len(ROW_OFFSETS):
        return None
    offset = ROW_OFFSETS[poem_id - 1]
    with open(INPUT_CSV, 'r', encoding='utf-8', newline='') as f:
        f.seek(offset)
        reader = csv.reader(f)
        try:
            return next(reader)
        except StopIteration:
            return None

def load_state():
    state = {
        "current_id": 1,
        "edits": {},
        "deleted": [],
        "types": DEFAULT_TYPES,
        "keywords": DEFAULT_KEYWORDS
    }
    
    if os.path.exists(STATE_JSON):
        with open(STATE_JSON, 'r', encoding='utf-8') as f:
            try:
                loaded = json.load(f)
                if isinstance(loaded, dict):
                    state.update(loaded)
            except Exception as e:
                print(f"Error reading review_state.json: {e}")
                
    # Ensure nested elements exist
    if 'edits' not in state:
        state['edits'] = {}
    if 'deleted' not in state:
        state['deleted'] = []
    if 'types' not in state or not state['types']:
        state['types'] = DEFAULT_TYPES
    if 'keywords' not in state or not state['keywords']:
        state['keywords'] = DEFAULT_KEYWORDS
        
    return state

def save_state(state):
    with open(STATE_JSON, 'w', encoding='utf-8') as f:
        json.dump(state, f, ensure_ascii=False, indent=2)

def get_max_id(state):
    original_max = len(ROW_OFFSETS)
    max_id = original_max
    for k in state['edits'].keys():
        try:
            val = int(k)
            if val > max_id:
                max_id = val
        except ValueError:
            pass
    return max_id

def sync_to_filter_dataset(state):
    """
    Stream updates from input.csv to filter-dataset.csv without keeping the entire
    table in memory. Safe overwrite using temp file write + rename pattern.
    Output headers: ['id', 'poem', 'category', 'explanation', 'Quyển', 'keyword', 'type']
    """
    if not os.path.exists(INPUT_CSV):
        return
        
    temp_output = OUTPUT_CSV + '.tmp'
    original_count = len(ROW_OFFSETS)
    deleted_ids = set(str(d) for d in state.get('deleted', []))
    
    with open(INPUT_CSV, 'r', encoding='utf-8', newline='') as infile, \
         open(temp_output, 'w', encoding='utf-8', newline='') as outfile:
        
        reader = csv.reader(infile)
        writer = csv.writer(outfile)
        
        # Write updated header
        header = next(reader)
        writer.writerow(['id', 'poem', 'category', 'explanation', 'Quyển', 'keyword', 'type'])
        
        # 1. Stream original rows (ID 1 to original_count)
        for row in reader:
            row_id_str = row[0]
            if row_id_str in deleted_ids:
                continue # Skip deleted row
                
            poem_val = row[1]
            cat_val = row[2]
            exp_val = row[3]
            quyen_val = row[4]
            keyword_val = ""
            type_val = ""
            
            if row_id_str in state['edits']:
                edited = state['edits'][row_id_str]
                poem_val = edited.get('poem', poem_val)
                cat_val = edited.get('category', cat_val)
                exp_val = edited.get('explanation', exp_val)
                quyen_val = edited.get('book', quyen_val)
                keyword_val = edited.get('keyword', '')
                type_val = edited.get('type', '')
                
            writer.writerow([row_id_str, poem_val, cat_val, exp_val, quyen_val, keyword_val, type_val])
            
        # 2. Append new custom-added rows (ID > original_count)
        new_ids = []
        for k in state['edits'].keys():
            try:
                val = int(k)
                if val > original_count and k not in deleted_ids:
                    new_ids.append(val)
            except ValueError:
                pass
                
        new_ids.sort()
        for nid in new_ids:
            nid_str = str(nid)
            edited = state['edits'][nid_str]
            writer.writerow([
                nid_str,
                edited.get('poem', ''),
                edited.get('category', '(CA DAO)'),
                edited.get('explanation', ''),
                edited.get('book', 'TỪ ĐIỂN TỤC NGỮ THÀNH NGỮ CA DAO VIỆT NAM - Quyển Thượng'),
                edited.get('keyword', ''),
                edited.get('type', '')
            ])
            
    # Swap file
    if os.path.exists(OUTPUT_CSV):
        os.remove(OUTPUT_CSV)
    os.rename(temp_output, OUTPUT_CSV)

# Routes
@app.route('/')
def index():
    return render_template('review.html')

@app.route('/api/progress')
def get_progress():
    state = load_state()
    max_id = get_max_id(state)
    deleted_set = set(str(d) for d in state.get('deleted', []))
    
    # Reviewed count is the number of edits that are not deleted
    reviewed_count = 0
    for k in state['edits'].keys():
        if k not in deleted_set:
            reviewed_count += 1
            
    return jsonify({
        "total": max_id,
        "reviewed": reviewed_count,
        "current_id": state.get("current_id", 1)
    })

@app.route('/api/types', methods=['GET', 'POST'])
def handle_types():
    state = load_state()
    if request.method == 'POST':
        data = request.json
        new_type = data.get('type', '').strip()
        if not new_type:
            return jsonify({"error": "Tên thể loại phụ không hợp lệ!"}), 400
            
        if new_type not in state['types']:
            state['types'].append(new_type)
            save_state(state)
            
    return jsonify(state['types'])

@app.route('/api/keywords', methods=['GET', 'POST'])
def handle_keywords():
    state = load_state()
    if request.method == 'POST':
        data = request.json
        new_kw = data.get('keyword', '').strip().lower()
        if not new_kw:
            return jsonify({"error": "Tên từ khóa không hợp lệ!"}), 400
            
        if new_kw not in state['keywords']:
            state['keywords'].append(new_kw)
            save_state(state)
            
    return jsonify(state['keywords'])

@app.route('/api/poem/<int:poem_id>')
def get_poem(poem_id):
    state = load_state()
    poem_id_str = str(poem_id)
    
    # If deleted, report back deleted state
    if poem_id_str in state.get('deleted', []):
        orig_row = get_row_by_id(poem_id) if poem_id <= len(ROW_OFFSETS) else None
        p_val = orig_row[1] if orig_row else ""
        c_val = orig_row[2] if orig_row else "(CA DAO)"
        e_val = orig_row[3] if orig_row else ""
        b_val = orig_row[4] if orig_row else "TỪ ĐIỂN TỤC NGỮ THÀNH NGỮ CA DAO VIỆT NAM - Quyển Thượng"
        k_val = ""
        t_val = ""
        
        if poem_id_str in state['edits']:
            ed = state['edits'][poem_id_str]
            p_val = ed.get('poem', p_val)
            c_val = ed.get('category', c_val)
            e_val = ed.get('explanation', e_val)
            b_val = ed.get('book', b_val)
            k_val = ed.get('keyword', k_val)
            t_val = ed.get('type', t_val)
            
        return jsonify({
            "id": poem_id,
            "poem": p_val,
            "category": c_val,
            "book": b_val,
            "explanation": e_val,
            "keyword": k_val,
            "type": t_val,
            "is_deleted": True
        })
        
    # Check if there is an edited version in state
    if poem_id_str in state['edits']:
        edited = state['edits'][poem_id_str]
        return jsonify({
            "id": poem_id,
            "poem": edited.get('poem', ''),
            "category": edited.get('category', '(CA DAO)'),
            "book": edited.get('book', 'TỪ ĐIỂN TỤC NGỮ THÀNH NGỮ CA DAO VIỆT NAM - Quyển Thượng'),
            "explanation": edited.get('explanation', ''),
            "keyword": edited.get('keyword', ''),
            "type": edited.get('type', ''),
            "is_reviewed": True,
            "is_deleted": False
        })
        
    # Read original row from CSV via offset
    row = get_row_by_id(poem_id)
    if not row:
        return jsonify({"error": "Poem not found"}), 404
        
    # CSV columns: id, poem, category, explanation, Quyển
    return jsonify({
        "id": int(row[0]),
        "poem": row[1],
        "category": row[2],
        "book": row[4],
        "explanation": row[3],
        "keyword": "",
        "type": "",
        "is_reviewed": False,
        "is_deleted": False
    })

@app.route('/api/poem/<int:poem_id>', methods=['POST'])
def save_poem(poem_id):
    state = load_state()
    poem_id_str = str(poem_id)
    
    # Block saves if it is marked as deleted
    if poem_id_str in state.get('deleted', []):
        return jsonify({"error": "Bài thơ này đã bị xóa!"}), 400
        
    data = request.json
    poem = data.get('poem', '')
    category = data.get('category', '')
    explanation = data.get('explanation', '')
    book = data.get('book', 'TỪ ĐIỂN TỤC NGỮ THÀNH NGỮ CA DAO VIỆT NAM - Quyển Thượng')
    keyword = data.get('keyword', '').strip()
    poem_type = data.get('type', '').strip()
    
    # Validate keywords are part of the master system chips
    kws = [k.strip().lower() for k in keyword.split(',') if k.strip()]
    for kw in kws:
        if kw not in state['keywords']:
            return jsonify({"error": f"Từ khóa '{kw}' không có trong danh sách chip cho phép! Vui lòng thêm từ khóa vào hệ thống trước."}), 400
            
    # Enforce: poem type must be in the allowed types list
    if not poem_type or poem_type not in state['types']:
        return jsonify({"error": "Thể loại phụ phải thuộc danh sách được cho phép!"}), 400
        
    # Update edits in state
    state['edits'][poem_id_str] = {
        "poem": poem,
        "category": category,
        "explanation": explanation,
        "book": book,
        "keyword": ", ".join(kws), # standardized format
        "type": poem_type
    }
    state['current_id'] = poem_id
    
    save_state(state)
    
    # Sync edits to filter-dataset.csv
    sync_to_filter_dataset(state)
    
    return jsonify({"status": "success", "id": poem_id})

@app.route('/api/poem/<int:poem_id>/delete', methods=['POST'])
def delete_poem(poem_id):
    state = load_state()
    if 'deleted' not in state:
        state['deleted'] = []
        
    poem_id_str = str(poem_id)
    if poem_id_str not in state['deleted']:
        state['deleted'].append(poem_id_str)
        
    # Find next active (non-deleted) ID to switch to
    max_id = get_max_id(state)
    deleted_set = set(state['deleted'])
    
    # Try next ID
    next_id = poem_id + 1
    while next_id <= max_id and str(next_id) in deleted_set:
        next_id += 1
        
    if next_id > max_id:
        # Try going backwards
        next_id = poem_id - 1
        while next_id >= 1 and str(next_id) in deleted_set:
            next_id -= 1
            
    if next_id < 1:
        next_id = 1
        
    state['current_id'] = next_id
    save_state(state)
    
    # Sync to output CSV (this will skip the deleted poem)
    sync_to_filter_dataset(state)
    
    return jsonify({"status": "success", "next_id": next_id})

@app.route('/api/poem/<int:poem_id>/restore', methods=['POST'])
def restore_poem(poem_id):
    state = load_state()
    poem_id_str = str(poem_id)
    
    if 'deleted' in state and poem_id_str in state['deleted']:
        state['deleted'].remove(poem_id_str)
        state['current_id'] = poem_id
        save_state(state)
        sync_to_filter_dataset(state)
        return jsonify({"status": "success", "id": poem_id})
        
    return jsonify({"error": "Poem is not deleted"}), 400

@app.route('/api/poem/new', methods=['POST'])
def add_new_poem():
    state = load_state()
    max_id = get_max_id(state)
    new_id = max_id + 1
    new_id_str = str(new_id)
    
    # Create empty record with default drop-down settings
    default_type = state['types'][0] if state['types'] else ""
    state['edits'][new_id_str] = {
        "poem": "",
        "category": "(CA DAO)",
        "explanation": "",
        "book": "TỪ ĐIỂN TỤC NGỮ THÀNH NGỮ CA DAO VIỆT NAM - Quyển Thượng",
        "keyword": "",
        "type": default_type
    }
    state['current_id'] = new_id
    save_state(state)
    
    # Sync to output
    sync_to_filter_dataset(state)
    
    return jsonify({
        "status": "success",
        "id": new_id,
        "poem": "",
        "category": "(CA DAO)",
        "book": "TỪ ĐIỂN TỤC NGỮ THÀNH NGỮ CA DAO VIỆT NAM - Quyển Thượng",
        "keyword": "",
        "type": default_type,
        "is_reviewed": True
    })

def main():
    # Build the offset map before starting the server
    build_row_offsets()
    
    # Pre-populate the output CSV on first run if it doesn't exist yet
    if not os.path.exists(OUTPUT_CSV):
        print("Initializing filter-dataset.csv from input.csv...")
        state = load_state()
        sync_to_filter_dataset(state)
        print("Initialization completed.")
        
    print("\n" + "="*50)
    print(" HỒN VIỆT - INTERACTIVE REVIEW SERVER STARTED ")
    print(" => URL: http://localhost:4005")
    print(" => Press Ctrl+C to stop the server")
    print("="*50 + "\n")
    
    app.run(host="0.0.0.0", port=4005, debug=True)

if __name__ == '__main__':
    main()
