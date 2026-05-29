import os
import csv
import sys
import re
import fitz  # PyMuPDF
import objc
from Foundation import NSURL

# 1. Load Vision framework via PyObjC
try:
    objc.loadBundle('Vision', bundle_path='/System/Library/Frameworks/Vision.framework', module_globals=globals())
except Exception as e:
    print(f"Error: Failed to load macOS Vision framework: {e}")
    sys.exit(1)

def is_category(text):
    """
    Checks if a line of text is a category anchor.
    Regex matches (ALL_CAPS) at the start of the line.
    """
    text_clean = text.strip()
    match = re.match(r'^\(([^)]+)\)', text_clean)
    if match:
        content = match.group(1).strip()
        if content.isupper() and 3 <= len(content) <= 25:
            return True
    return False

def join_explanation_lines(exp_lines):
    """
    Joins explanation text lines while preserving paragraph spacing.
    Uses vertical gap spacing to determine if a newline should be inserted.
    """
    if not exp_lines:
        return ""
    
    text_parts = [exp_lines[0]['text'].strip()]
    for i in range(1, len(exp_lines)):
        prev = exp_lines[i-1]
        curr = exp_lines[i]
        
        # Calculate normalized vertical gap between lines
        gap = curr['y'] - (prev['y'] + prev['h'])
        
        # If gap is relatively large, we insert a newline to start a new paragraph
        if gap >= 0.003:
            text_parts.append("\n" + curr['text'].strip())
        else:
            text_parts.append(curr['text'].strip())
            
    # Assemble parts
    result = ""
    for part in text_parts:
        if not result:
            result = part
        elif part.startswith('\n'):
            result += part
        else:
            result += " " + part
            
    return result

def run_extraction():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pdf_path = os.path.join(script_dir, "2-split.pdf")
    csv_path = os.path.join(script_dir, "2-extracted-poems.csv")
    
    if not os.path.exists(pdf_path):
        print(f"Error: Source PDF not found at {pdf_path}")
        return
        
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    print(f"Loaded output.pdf: {total_pages} pages to extract.")
    
    final_entries = []
    
    for p_idx in range(total_pages):
        page_num = p_idx + 1
        page = doc[p_idx]
        print(f"Processing Page {page_num}/{total_pages}...")
        
        # Render page to a temporary image file for Apple Vision OCR
        pix = page.get_pixmap(dpi=150)
        temp_img_path = os.path.join(script_dir, f"temp_p{page_num}.png")
        pix.save(temp_img_path)
        
        # Perform Apple Vision OCR
        url = NSURL.fileURLWithPath_(temp_img_path)
        handler = VNImageRequestHandler.alloc().initWithURL_options_(url, None)
        request = VNRecognizeTextRequest.alloc().init()
        request.setRecognitionLanguages_(["vi-VN", "en-US"])
        request.setUsesLanguageCorrection_(True)
        
        success = handler.performRequests_error_([request], None)
        if not success:
            print(f"  OCR failed for page {page_num}")
            if os.path.exists(temp_img_path):
                os.remove(temp_img_path)
            continue
            
        results = request.results()
        
        # Parse observations into a structured dictionary list
        lines = []
        for obs in results:
            box = obs.boundingBox()
            # Convert Vision bottom-left coords to standard top-left coords
            x = box.origin.x
            y = 1.0 - box.origin.y - box.size.height
            w = box.size.width
            h = box.size.height
            
            candidates = obs.topCandidates_(1)
            text = candidates[0].string() if candidates else ""
            
            # SPLITTING LOGIC FOR INLINE CATEGORIES
            text_clean = text.strip()
            match = re.match(r'^\(([^)]+)\)', text_clean)
            if match:
                content = match.group(1).strip()
                if content.isupper() and 3 <= len(content) <= 25:
                    category_text = f"({content})"
                    rest = text_clean[match.end():].strip()
                    # Strip leading punctuation (like . or :) and space
                    if rest.startswith('.') or rest.startswith(':'):
                        rest = rest[1:].strip()
                    
                    if rest:
                        # Append category part
                        lines.append({
                            'text': category_text,
                            'x': x,
                            'y': y,
                            'w': w,
                            'h': h * 0.4,
                            'raw_obs': obs
                        })
                        # Append the rest of the text as a separate line
                        lines.append({
                            'text': rest,
                            'x': x,
                            'y': y + h * 0.5,
                            'w': w,
                            'h': h * 0.5,
                            'raw_obs': obs
                        })
                        continue
            
            lines.append({
                'text': text,
                'x': x,
                'y': y,
                'w': w,
                'h': h,
                'raw_obs': obs
            })
            
        # Sort lines top-to-bottom, left-to-right
        lines.sort(key=lambda item: (item['y'], item['x']))
        
        # Find category anchor indices
        cat_indices = []
        for idx, line in enumerate(lines):
            if is_category(line['text']):
                cat_indices.append(idx)
                
        # Parse layout components based on category anchors
        orphan_lines = []
        parsed_entries = []
        poem_start_indices = {}
        
        for j, cat_idx in enumerate(cat_indices):
            curr = cat_idx - 1
            lower_limit = cat_indices[j-1] + 1 if j > 0 else 0
            
            poem_start = curr
            for i in range(curr, lower_limit - 1, -1):
                line_text = lines[i]['text'].strip()
                
                if i < curr:
                    # Heuristic Stop 1: Vertical Gap (indicates paragraph or entry boundary)
                    gap = lines[i+1]['y'] - (lines[i]['y'] + lines[i]['h'])
                    if gap >= 0.012:
                        break
                        
                    # Heuristic Stop 2: If the line starts with a lowercase letter,
                    # it is a prose explanation line (folk poetry lines always start with a capital letter).
                    first_alpha = None
                    for char in line_text:
                        if char.isalpha():
                            first_alpha = char
                            break
                    if first_alpha and first_alpha.islower():
                        break
                        
                poem_start = i
                
            poem_start_indices[cat_idx] = poem_start
            
        # Group lines
        if cat_indices:
            # Orphan lines at the top of the page (continuation of previous explanation)
            first_poem_start = poem_start_indices[cat_indices[0]]
            orphan_lines = lines[0:first_poem_start]
            
            # Form entry segments
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
                    
                parsed_entries.append({
                    'poem_lines': poem_lines,
                    'category_line': category_line,
                    'explanation_lines': explanation_lines
                })
        else:
            # No categories on page -> entire page continues the previous explanation
            orphan_lines = lines
            
        # ------------------ STITCHING LOGIC ------------------
        # Append orphan text to the last entry's explanation, if available
        if orphan_lines:
            orphan_text = join_explanation_lines(orphan_lines)
            if final_entries:
                if final_entries[-1]['explanation']:
                    final_entries[-1]['explanation'] += " " + orphan_text
                else:
                    final_entries[-1]['explanation'] = orphan_text
            else:
                # Page 1 starts with text without poem headers (unlikely, but fallback)
                final_entries.append({
                    'poem': "",
                    'category': "",
                    'explanation': orphan_text
                })
                
        # Append newly parsed entries
        for entry in parsed_entries:
            poem_text = "\n".join(line['text'].strip() for line in entry['poem_lines'])
            category_text = entry['category_line']['text'].strip()
            explanation_text = join_explanation_lines(entry['explanation_lines'])
            
            final_entries.append({
                'poem': poem_text,
                'category': category_text,
                'explanation': explanation_text
            })
            
        # Cleanup temporary page image
        if os.path.exists(temp_img_path):
            os.remove(temp_img_path)
            
    doc.close()
    
    # ------------------ CSV EXPORT ------------------
    print(f"Writing {len(final_entries)} entries to CSV: {csv_path}")
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['poem', 'category', 'explanation'])
        writer.writeheader()
        for entry in final_entries:
            # Skip empty rows (if any fallback)
            if not entry['poem'] and not entry['category'] and not entry['explanation']:
                continue
            writer.writerow(entry)
            
    print("Poem extraction successfully completed!")

if __name__ == "__main__":
    run_extraction()
