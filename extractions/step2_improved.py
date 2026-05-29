import os
import re
import csv
import io
import json as py_json
import tempfile
from flask import Flask, request, jsonify, render_template, send_file
import fitz  # PyMuPDF
import objc
from Foundation import NSURL

# Load macOS Vision framework via PyObjC
try:
    objc.loadBundle('Vision', bundle_path='/System/Library/Frameworks/Vision.framework', module_globals=globals())
except Exception as e:
    print(f"Error: Failed to load macOS Vision framework: {e}")
    pass

# Initialize Flask locally inside the extractions/ folder
app = Flask(__name__, template_folder='templates_editor', static_folder='static')

# In-memory OCR cache to prevent redundant processing
OCR_CACHE = {}

# Session dictionary to save user page states (boxes, text, stitching markers, OCR lines)
# Keyed by (pdf_name, page_idx)
PAGE_STATES = {}

def is_category(text):
    """Checks if a text line represents a category anchor."""
    text_clean = text.strip()
    match = re.match(r'^\(([^)]+)\)', text_clean)
    if match:
        content = match.group(1).strip()
        if content.isupper() and 3 <= len(content) <= 25:
            return True
    return False

def run_apple_ocr(page, page_idx):
    """Executes Apple Vision OCR on a rendered page pixmap and returns text lines with coordinates."""
    temp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scratch")
    os.makedirs(temp_dir, exist_ok=True)
    
    with tempfile.NamedTemporaryFile(suffix=".png", dir=temp_dir, delete=False) as temp_file:
        temp_img_path = temp_file.name
        
    try:
        pix = page.get_pixmap(dpi=150)
        pix.save(temp_img_path)
        
        url = NSURL.fileURLWithPath_(temp_img_path)
        handler = VNImageRequestHandler.alloc().initWithURL_options_(url, None)
        request = VNRecognizeTextRequest.alloc().init()
        request.setRecognitionLanguages_(["vi-VN", "en-US"])
        request.setUsesLanguageCorrection_(True)
        
        success = handler.performRequests_error_([request], None)
        lines = []
        if success:
            results = request.results()
            for obs in results:
                box = obs.boundingBox()
                x = box.origin.x
                y = 1.0 - box.origin.y - box.size.height
                w = box.size.width
                h = box.size.height
                
                candidates = obs.topCandidates_(1)
                text = candidates[0].string() if candidates else ""
                
                # Split inline categories if present (e.g. (CA DAO) Chi tiết...)
                text_clean = text.strip()
                match = re.match(r'^\(([^)]+)\)', text_clean)
                if match:
                    content = match.group(1).strip()
                    if content.isupper() and 3 <= len(content) <= 25:
                        category_text = f"({content})"
                        rest = text_clean[match.end():].strip()
                        if rest.startswith('.') or rest.startswith(':'):
                            rest = rest[1:].strip()
                        
                        if rest:
                            lines.append({
                                'text': category_text,
                                'x': x,
                                'y': y,
                                'w': w,
                                'h': h * 0.4
                            })
                            lines.append({
                                'text': rest,
                                'x': x,
                                'y': y + h * 0.5,
                                'w': w,
                                'h': h * 0.5
                            })
                            continue
                
                lines.append({
                    'text': text,
                    'x': x,
                    'y': y,
                    'w': w,
                    'h': h
                })
    finally:
        if os.path.exists(temp_img_path):
            os.remove(temp_img_path)
            
    # Sort top-to-bottom, left-to-right
    lines.sort(key=lambda l: (l['y'], l['x']))
    return lines

def get_bbox_of_lines(lines_list):
    """Calculates standard bounding box enclosing all lines in list."""
    if not lines_list:
        return None
    x0 = min(l['x'] for l in lines_list)
    y0 = min(l['y'] for l in lines_list)
    x1 = max(l['x'] + l['w'] for l in lines_list)
    y1 = max(l['y'] + l['h'] for l in lines_list)
    return {
        'x': max(0.0, x0 - 0.01),
        'y': max(0.0, y0 - 0.01),
        'w': min(1.0, x1 - x0 + 0.02),
        'h': min(1.0, y1 - y0 + 0.02)
    }

def get_initial_boxes(lines):
    """Calculates initial layout bounding boxes for all entries (Poem, Category, and Explanation) on the page."""
    cat_indices = []
    for idx, line in enumerate(lines):
        if is_category(line['text']):
            cat_indices.append(idx)
            
    poem_start_indices = {}
    for j, cat_idx in enumerate(cat_indices):
        curr = cat_idx - 1
        lower_limit = cat_indices[j-1] + 1 if j > 0 else 0
        
        poem_start = curr
        for i in range(curr, lower_limit - 1, -1):
            line_text = lines[i]['text'].strip()
            
            if i < curr:
                # Heuristic stop 1: Vertical Gap (gap >= 0.012)
                gap = lines[i+1]['y'] - (lines[i]['y'] + lines[i]['h'])
                if gap >= 0.012:
                    break
                    
                # Heuristic stop 2: Lowercase starting character
                first_alpha = None
                for char in line_text:
                    if char.isalpha():
                        first_alpha = char
                        break
                if first_alpha and first_alpha.islower():
                    break
                
            poem_start = i
            
        poem_start_indices[cat_idx] = poem_start
        
    entries = []
    orphan_box = None
    if cat_indices:
        first_poem_start = poem_start_indices[cat_indices[0]]
        orphan_lines = lines[0:first_poem_start]
        
        for j, cat_idx in enumerate(cat_indices):
            poem_start = poem_start_indices[cat_idx]
            poem_lines = lines[poem_start:cat_idx]
            category_line = lines[cat_idx]
            
            exp_start = cat_idx + 1
            if j < len(cat_indices) - 1:
                next_cat_idx = cat_indices[j+1]
                next_poem_start = poem_start_indices[next_cat_idx]
                explanation_lines = lines[exp_start:next_poem_start]
            else:
                explanation_lines = lines[exp_start:]
                
            poem_box = get_bbox_of_lines(poem_lines)
            
            category_box = {
                'x': max(0.0, category_line['x'] - 0.01),
                'y': max(0.0, category_line['y'] - 0.01),
                'w': min(1.0, category_line['w'] + 0.02),
                'h': min(1.0, category_line['h'] + 0.02)
            }
            
            explanation_box = get_bbox_of_lines(explanation_lines)
            
            entries.append({
                'entry_idx': j,
                'boxes': {
                    'poem': poem_box,
                    'category': category_box,
                    'explanation': explanation_box
                },
                'category': category_line['text'].strip(),
                'poem_original': "\n".join(l['text'].strip() for l in sorted(poem_lines, key=lambda l: (l['y'], l['x']))),
                'poem_corrected': "\n".join(l['text'].strip() for l in sorted(poem_lines, key=lambda l: (l['y'], l['x']))),
                'explanation_original': join_explanation_lines_with_spacing(explanation_lines),
                'explanation_corrected': join_explanation_lines_with_spacing(explanation_lines)
            })
    else:
        orphan_lines = lines
        
    orphan_text = join_explanation_lines_with_spacing(orphan_lines)
    orphan_box = get_bbox_of_lines(orphan_lines) if orphan_lines else None
    return entries, orphan_text, orphan_box

def join_explanation_lines_with_spacing(lines_list):
    """Joins explanation lines, using vertical spacing gaps to preserve paragraph bounds."""
    if not lines_list:
        return ""
    
    lines_sorted = sorted(lines_list, key=lambda l: (l['y'], l['x']))
    
    text_parts = [lines_sorted[0]['text'].strip()]
    for i in range(1, len(lines_sorted)):
        prev = lines_sorted[i-1]
        curr = lines_sorted[i]
        
        gap = curr['y'] - (prev['y'] + prev['h'])
        if gap >= 0.003:
            text_parts.append("\n" + curr['text'].strip())
        else:
            text_parts.append(curr['text'].strip())
            
    result = ""
    for part in text_parts:
        if not result:
            result = part
        elif part.startswith('\n'):
            result += part
        else:
            result += " " + part
    return result


# --- ROUTES ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/pdfs')
def list_pdfs():
    """Lists all split PDF files in the extractions/ folder."""
    extractions_dir = os.path.dirname(os.path.abspath(__file__))
    pdfs = []
    if os.path.exists(extractions_dir):
        pdfs = [f for f in os.listdir(extractions_dir) if f.endswith('.pdf')]
    if "1-split.pdf" in pdfs:
        pdfs.remove("1-split.pdf")
        pdfs.insert(0, "1-split.pdf")
    return jsonify({'pdfs': pdfs})

@app.route('/api/page_image')
def page_image():
    """Renders a specific page of a PDF and returns it as a PNG image."""
    pdf_name = request.args.get('pdf')
    page_idx = int(request.args.get('page_idx', 0))
    
    pdf_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), pdf_name)
    if not os.path.exists(pdf_path):
        return "PDF not found", 404
        
    doc = fitz.open(pdf_path)
    if page_idx < 0 or page_idx >= len(doc):
        doc.close()
        return "Page index out of range", 400
        
    page = doc[page_idx]
    zoom = 2.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)
    
    img_data = pix.tobytes("png")
    doc.close()
    
    return send_file(io.BytesIO(img_data), mimetype='image/png')

@app.route('/api/progress')
def get_progress():
    """Returns the saved progress (next starting page) for a PDF."""
    pdf_name = request.args.get('pdf')
    if not pdf_name:
        return jsonify({'error': 'Missing PDF name'}), 400
        
    progress_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "user-progress.json")
    last_page = 1
    if os.path.exists(progress_path):
        try:
            with open(progress_path, 'r', encoding='utf-8') as f:
                data = py_json.load(f)
                if pdf_name in data:
                    last_page = data[pdf_name].get('last_processed_page', 0) + 1
        except Exception:
            pass
            
    return jsonify({'last_processed_page': last_page})

@app.route('/api/load_pages')
def load_pages():
    """Loads page layout details (page index, bounds) with orphan lookahead checking."""
    pdf_name = request.args.get('pdf')
    start_page = int(request.args.get('start_page', 1))  # 1-indexed page number
    batch_size = int(request.args.get('batch_size', 10))
    
    pdf_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), pdf_name)
    if not os.path.exists(pdf_path):
        return jsonify({'error': 'PDF not found'}), 404
        
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    
    start_idx = start_page - 1
    end_idx = min(start_idx + batch_size, total_pages)
    
    # 1. Determine batch indices, adding lookahead for orphans at the end
    page_indices = list(range(start_idx, end_idx))
    
    curr_idx = end_idx
    while curr_idx < total_pages:
        cache_key = (pdf_name, curr_idx)
        if cache_key in OCR_CACHE:
            lines = OCR_CACHE[cache_key]
        else:
            page = doc[curr_idx]
            lines = run_apple_ocr(page, curr_idx)
            OCR_CACHE[cache_key] = lines
            
        entries, orphan_text, orphan_box = get_initial_boxes(lines)
        has_cat = len(entries) > 0
        has_text = len(lines) > 0
        
        # If the page has text but no category anchors, it is an orphan continuation page.
        # We must include it in the batch.
        if has_text and not has_cat:
            page_indices.append(curr_idx)
            curr_idx += 1
        else:
            break
            
    pages = []
    for idx in page_indices:
        state_key = (pdf_name, idx)
        
        # If page already has state in session
        if state_key in PAGE_STATES:
            state = PAGE_STATES[state_key]
            pages.append({
                'page_idx': state['page_idx'],
                'page_num': state['page_num'],
                'stitch_poem': state.get('stitch_poem', False),
                'stitch_explanation': state.get('stitch_explanation', False),
                'orphan_explanation': state.get('orphan_explanation', ''),
                'orphan_box': state.get('orphan_box', None),
                'entries': state.get('entries', [])
            })
        else:
            cache_key = (pdf_name, idx)
            if cache_key in OCR_CACHE:
                lines = OCR_CACHE[cache_key]
            else:
                page = doc[idx]
                lines = run_apple_ocr(page, idx)
                OCR_CACHE[cache_key] = lines
                
            entries, orphan_text, orphan_box = get_initial_boxes(lines)
            has_cat = len(entries) > 0
            
            initial_state = {
                'page_idx': idx,
                'page_num': idx + 1,
                'stitch_poem': False,
                'stitch_explanation': not has_cat,
                'orphan_explanation': orphan_text,
                'orphan_box': orphan_box,
                'entries': entries,
                'lines': lines  # Cache OCR lines for extraction later
            }
            
            PAGE_STATES[state_key] = initial_state
            
            pages.append({
                'page_idx': idx,
                'page_num': idx + 1,
                'stitch_poem': False,
                'stitch_explanation': not has_cat,
                'orphan_explanation': orphan_text,
                'orphan_box': orphan_box,
                'entries': entries
            })
            
    doc.close()
    return jsonify({'pages': pages})

@app.route('/api/process_batch', methods=['POST'])
def process_batch():
    """
    Extracts text from the final adjusted bounding boxes, saves raw texts to JSON,
    and returns text for review.
    """
    data = request.json
    pdf_name = data.get('pdf')
    pages_data = data.get('pages', [])
    
    if not pdf_name or not pages_data:
        return jsonify({'error': 'Missing required parameters'}), 400
        
    processed_pages = []
    
    # 1. Update session PAGE_STATES with the user's adjusted boxes and stitch flags
    for p_data in pages_data:
        page_idx = int(p_data.get('page_idx'))
        state_key = (pdf_name, page_idx)
        
        if state_key in PAGE_STATES:
            PAGE_STATES[state_key]['entries'] = p_data.get('entries', [])
            PAGE_STATES[state_key]['orphan_box'] = p_data.get('orphan_box')
            PAGE_STATES[state_key]['stitch_poem'] = p_data.get('stitch_poem', False)
            PAGE_STATES[state_key]['stitch_explanation'] = p_data.get('stitch_explanation', False)
            
    # 2. Extract raw text from final boxes for all pages in this batch
    for p_data in pages_data:
        page_idx = int(p_data.get('page_idx'))
        state_key = (pdf_name, page_idx)
        
        state = PAGE_STATES[state_key]
        lines = state.get('lines', [])
        
        if not lines:
            cache_key = (pdf_name, page_idx)
            if cache_key in OCR_CACHE:
                lines = OCR_CACHE[cache_key]
            else:
                pdf_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), pdf_name)
                doc = fitz.open(pdf_path)
                page = doc[page_idx]
                lines = run_apple_ocr(page, page_idx)
                OCR_CACHE[cache_key] = lines
                doc.close()
            state['lines'] = lines
            
        # Extract text for each entry using its respective bounding boxes
        for entry in state.get('entries', []):
            poem_box = entry['boxes'].get('poem')
            cat_box = entry['boxes'].get('category')
            exp_box = entry['boxes'].get('explanation')
            
            p_lines = []
            c_lines = []
            e_lines = []
            
            for line in lines:
                cx = line['x'] + line['w'] / 2.0
                cy = line['y'] + line['h'] / 2.0
                
                def in_box(x, y, box):
                    if not box: return False
                    return box['x'] <= x <= box['x'] + box['w'] and box['y'] <= y <= box['y'] + box['h']
                    
                if in_box(cx, cy, cat_box):
                    c_lines.append(line)
                elif in_box(cx, cy, poem_box):
                    p_lines.append(line)
                elif in_box(cx, cy, exp_box):
                    e_lines.append(line)
                    
            poem_orig = "\n".join(l['text'].strip() for l in sorted(p_lines, key=lambda l: (l['y'], l['x'])))
            category = " ".join(l['text'].strip() for l in sorted(c_lines, key=lambda l: (l['y'], l['x'])))
            explanation_orig = join_explanation_lines_with_spacing(e_lines)
            
            entry['poem_original'] = poem_orig
            entry['category'] = category
            entry['explanation_original'] = explanation_orig
            
            # Store directly in the corrected/edited properties
            entry['poem_corrected'] = poem_orig
            entry['explanation_corrected'] = explanation_orig

        # Calculate orphan_explanation (lines inside orphan_box, with fallback to lines above first poem)
        orphan_box = state.get('orphan_box')
        o_lines = []
        if orphan_box:
            for line in lines:
                cx = line['x'] + line['w'] / 2.0
                cy = line['y'] + line['h'] / 2.0
                if orphan_box['x'] <= cx <= orphan_box['x'] + orphan_box['w'] and orphan_box['y'] <= cy <= orphan_box['y'] + orphan_box['h']:
                    o_lines.append(line)
        else:
            first_entry = state['entries'][0] if state.get('entries') else None
            first_poem_box = first_entry['boxes'].get('poem') if first_entry else None
            for line in lines:
                cx = line['x'] + line['w'] / 2.0
                cy = line['y'] + line['h'] / 2.0
                
                inside_any = False
                for entry in state.get('entries', []):
                    for box_name in ['poem', 'category', 'explanation']:
                        box = entry['boxes'].get(box_name)
                        if box and box['x'] <= cx <= box['x'] + box['w'] and box['y'] <= cy <= box['y'] + box['h']:
                            inside_any = True
                            break
                    if inside_any:
                        break
                        
                if not inside_any and first_poem_box and cy < first_poem_box['y']:
                    o_lines.append(line)
                    
        state['orphan_explanation'] = join_explanation_lines_with_spacing(o_lines)

    # 3. Save extracted raw text data of all pages to a local JSON file inside the extractions folder
    json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "poem-extraction-data.json")
    try:
        json_data = []
        sorted_keys = sorted(PAGE_STATES.keys(), key=lambda k: (k[0], k[1]))
        for k in sorted_keys:
            st = PAGE_STATES[k]
            json_data.append({
                'pdf': k[0],
                'page_idx': st['page_idx'],
                'page_num': st['page_num'],
                'stitch_poem': st.get('stitch_poem', False),
                'stitch_explanation': st.get('stitch_explanation', False),
                'orphan_explanation': st.get('orphan_explanation', ''),
                'entries': [
                    {
                        'entry_idx': e['entry_idx'],
                        'category': e.get('category', ''),
                        'poem_original': e.get('poem_original', ''),
                        'explanation_original': e.get('explanation_original', '')
                    } for e in st.get('entries', [])
                ]
            })
        with open(json_path, 'w', encoding='utf-8') as f:
            py_json.dump(json_data, f, ensure_ascii=False, indent=2)
        print(f"Saved raw text data to {json_path}")
    except Exception as e:
        print(f"Error saving JSON file: {e}")
        
    # 4. Return pages data
    for p_data in pages_data:
        page_idx = int(p_data.get('page_idx'))
        state_key = (pdf_name, page_idx)
        state = PAGE_STATES[state_key]
        
        processed_pages.append({
            'page_idx': state['page_idx'],
            'page_num': state['page_num'],
            'stitch_poem': state.get('stitch_poem', False),
            'stitch_explanation': state.get('stitch_explanation', False),
            'orphan_explanation': state.get('orphan_explanation', ''),
            'orphan_box': state.get('orphan_box'),
            'entries': state.get('entries', [])
        })
        
    return jsonify({'pages': processed_pages})

@app.route('/api/extract_text', methods=['POST'])
def extract_text():
    """Extracts text within boxes (retained as fallback)."""
    data = request.json
    pdf_name = data.get('pdf')
    page_idx = int(data.get('page_idx'))
    entries_data = data.get('entries', [])
    
    cache_key = (pdf_name, page_idx)
    if cache_key in OCR_CACHE:
        lines = OCR_CACHE[cache_key]
    else:
        pdf_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), pdf_name)
        doc = fitz.open(pdf_path)
        page = doc[page_idx]
        lines = run_apple_ocr(page, page_idx)
        OCR_CACHE[cache_key] = lines
        doc.close()
        
    extracted_entries = []
    for entry in entries_data:
        poem_box = entry['boxes'].get('poem')
        cat_box = entry['boxes'].get('category')
        exp_box = entry['boxes'].get('explanation')
        
        p_lines = []
        c_lines = []
        e_lines = []
        
        for line in lines:
            cx = line['x'] + line['w'] / 2.0
            cy = line['y'] + line['h'] / 2.0
            
            def in_box(x, y, box):
                if not box: return False
                return box['x'] <= x <= box['x'] + box['w'] and box['y'] <= y <= box['y'] + box['h']
                
            if in_box(cx, cy, cat_box):
                c_lines.append(line)
            elif in_box(cx, cy, poem_box):
                p_lines.append(line)
            elif in_box(cx, cy, exp_box):
                e_lines.append(line)
                
        poem_orig = "\n".join(l['text'].strip() for l in sorted(p_lines, key=lambda l: (l['y'], l['x'])))
        category = " ".join(l['text'].strip() for l in sorted(c_lines, key=lambda l: (l['y'], l['x'])))
        explanation_orig = join_explanation_lines_with_spacing(e_lines)
        
        extracted_entries.append({
            'entry_idx': entry.get('entry_idx', 0),
            'category': category,
            'poem_original': poem_orig,
            'poem_corrected': poem_orig,
            'explanation_original': explanation_orig,
            'explanation_corrected': explanation_orig
        })
        
    return jsonify({'entries': extracted_entries})

@app.route('/api/save_page_state', methods=['POST'])
def save_page_state():
    """Saves user modifications for a page into the session dictionary while preserving cached OCR lines."""
    data = request.json
    pdf_name = data.get('pdf')
    page_idx = int(data.get('page_idx'))
    state = data.get('state')
    
    state_key = (pdf_name, page_idx)
    if state_key in PAGE_STATES:
        # Merge values to avoid deleting cached OCR lines
        for k, v in state.items():
            PAGE_STATES[state_key][k] = v
    else:
        PAGE_STATES[state_key] = state
        
    return jsonify({'success': True})

@app.route('/api/reset_boxes')
def reset_boxes():
    """Resets coordinates to the initial auto-detected boxes."""
    pdf_name = request.args.get('pdf')
    page_idx = int(request.args.get('page_idx'))
    
    cache_key = (pdf_name, page_idx)
    if cache_key in OCR_CACHE:
        lines = OCR_CACHE[cache_key]
    else:
        pdf_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), pdf_name)
        doc = fitz.open(pdf_path)
        page = doc[page_idx]
        lines = run_apple_ocr(page, page_idx)
        OCR_CACHE[cache_key] = lines
        doc.close()
        
    entries, orphan_text, orphan_box = get_initial_boxes(lines)
    
    # Reset PAGE_STATES for this page
    state_key = (pdf_name, page_idx)
    if state_key in PAGE_STATES:
        PAGE_STATES[state_key]['entries'] = entries
        PAGE_STATES[state_key]['orphan_explanation'] = orphan_text
        PAGE_STATES[state_key]['orphan_box'] = orphan_box
        
    return jsonify({
        'entries': entries,
        'orphan_explanation': orphan_text,
        'orphan_box': orphan_box
    })

@app.route('/api/save_csv', methods=['POST'])
def save_csv():
    """Assembles all reviewed/cached page states, connects entries using stitching flags, and saves CSV."""
    data = request.json
    pdf_name = data.get('pdf')
    page_indices = data.get('page_indices', [])
    
    if not pdf_name:
        return jsonify({'error': 'Missing PDF name'}), 400
        
    pdf_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), pdf_name)
    if not os.path.exists(pdf_path):
        return jsonify({'error': 'PDF not found'}), 404
        
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    doc.close()
    
    # If page_indices is provided, compile ONLY those pages. Otherwise, default to all pages
    if not page_indices:
        page_indices = list(range(total_pages))
    
    compiled_pages = []
    for idx in page_indices:
        state_key = (pdf_name, idx)
        
        if state_key in PAGE_STATES:
            compiled_pages.append(PAGE_STATES[state_key])
        else:
            doc = fitz.open(pdf_path)
            page = doc[idx]
            cache_key = (pdf_name, idx)
            if cache_key in OCR_CACHE:
                lines = OCR_CACHE[cache_key]
            else:
                lines = run_apple_ocr(page, idx)
                OCR_CACHE[cache_key] = lines
            doc.close()
            
            entries, orphan_text, orphan_box = get_initial_boxes(lines)
            has_cat = len(entries) > 0
            
            compiled_pages.append({
                'page_idx': idx,
                'page_num': idx + 1,
                'stitch_poem': False,
                'stitch_explanation': not has_cat,
                'orphan_explanation': orphan_text,
                'orphan_box': orphan_box,
                'entries': entries
            })
            
    final_entries = []
    for page in compiled_pages:
        page_num = page['page_num']
        entries = page.get('entries', [])
        orphan = page.get('orphan_explanation', '').strip()
        
        # 1. Handle page-level stitching (stitching orphan text to the last entry's explanation)
        if page.get('stitch_explanation', False) or (not page.get('stitch_poem', False) and orphan):
            if final_entries:
                if orphan:
                    if final_entries[-1]['explanation']:
                        final_entries[-1]['explanation'] += " " + orphan
                    else:
                        final_entries[-1]['explanation'] = orphan
                if page_num not in final_entries[-1]['pages_sourced']:
                    final_entries[-1]['pages_sourced'].append(page_num)
            else:
                if orphan:
                    final_entries.append({
                        'poem': '',
                        'category': '',
                        'explanation': orphan,
                        'pages_sourced': [page_num]
                    })
        elif page.get('stitch_poem', False):
            if final_entries:
                poem_to_stitch = orphan if orphan else (entries[0]['poem_corrected'] if entries else '')
                if poem_to_stitch:
                    if final_entries[-1]['poem']:
                        final_entries[-1]['poem'] += "\n" + poem_to_stitch
                    else:
                        final_entries[-1]['poem'] = poem_to_stitch
                if page_num not in final_entries[-1]['pages_sourced']:
                    final_entries[-1]['pages_sourced'].append(page_num)
            else:
                poem_to_stitch = orphan if orphan else (entries[0]['poem_corrected'] if entries else '')
                final_entries.append({
                    'poem': poem_to_stitch,
                    'category': '',
                    'explanation': '',
                    'pages_sourced': [page_num]
                })
                
        # 2. Add new entries from this page
        start_entry_idx = 0
        if page.get('stitch_poem', False) and not orphan and entries:
            if final_entries:
                first_entry = entries[0]
                if first_entry.get('category'):
                    final_entries[-1]['category'] = first_entry['category']
                if first_entry.get('explanation_corrected'):
                    if final_entries[-1]['explanation']:
                        final_entries[-1]['explanation'] += " " + first_entry['explanation_corrected']
                    else:
                        final_entries[-1]['explanation'] = first_entry['explanation_corrected']
            start_entry_idx = 1
            
        for entry in entries[start_entry_idx:]:
            poem_val = entry.get('poem_corrected', '').strip()
            cat_val = entry.get('category', '').strip()
            exp_val = entry.get('explanation_corrected', '').strip()
            
            final_entries.append({
                'poem': poem_val,
                'category': cat_val,
                'explanation': exp_val,
                'pages_sourced': [page_num]
            })

    filename = "poem-extraction.csv"
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
    
    try:
        with open(csv_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['poem', 'category', 'explanation', 'pages'])
            writer.writeheader()
            for entry in final_entries:
                poem_val = entry['poem'].strip()
                cat_val = entry['category'].strip()
                exp_val = entry['explanation'].strip()
                pages_str = ",".join(str(p) for p in entry['pages_sourced'])
                
                if not poem_val and not cat_val and not exp_val:
                    continue
                    
                writer.writerow({
                    'poem': poem_val,
                    'category': cat_val,
                    'explanation': exp_val,
                    'pages': pages_str
                })
                
        # Save progress tracker to user-progress.json
        progress_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "user-progress.json")
        progress_data = {}
        if os.path.exists(progress_path):
            try:
                with open(progress_path, 'r', encoding='utf-8') as pf:
                    progress_data = py_json.load(pf)
            except Exception:
                pass
                
        last_page_idx = max(page_indices)
        progress_data[pdf_name] = {
            'last_processed_page': last_page_idx + 1,
            'page_indices': page_indices
        }
        
        with open(progress_path, 'w', encoding='utf-8') as pf:
            py_json.dump(progress_data, pf, ensure_ascii=False, indent=2)
            
        return jsonify({'success': True, 'filename': filename})
    except Exception as e:
        print(f"Error exporting CSV: {e}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
