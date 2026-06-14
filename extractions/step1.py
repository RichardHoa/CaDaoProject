import os
import cv2
import fitz  # PyMuPDF
import numpy as np
import re
import tempfile
import objc
from Foundation import NSURL

# Load macOS Vision framework via PyObjC
try:
    objc.loadBundle('Vision', bundle_path='/System/Library/Frameworks/Vision.framework', module_globals=globals())
except Exception as e:
    print(f"Error: Failed to load macOS Vision framework: {e}")


def detect_column_separator(image, binary):
    """
    Detects the x-coordinate of the line separating the two columns in the image.
    Uses vertical projection profile analysis in the center region of the page.
    """
    height, width = image.shape[:2]
    
    # Calculate vertical projection (sum of pixels along columns)
    col_sums = np.sum(binary, axis=0) / (height * 255.0)
    
    # Define search region around the center (from 35% to 65% of the page width)
    start_x = int(width * 0.35)
    end_x = int(width * 0.65)
    
    center_col_sums = col_sums[start_x:end_x]
    
    # Smooth the profile to remove high-frequency noise
    kernel_size = int(width * 0.02)  # 2% of page width
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    smoothed = cv2.GaussianBlur(center_col_sums.reshape(-1, 1), (1, kernel_size), 0).flatten()
    
    # Find the x-coordinate with the minimum density of ink (the center of the gutter)
    min_idx_relative = np.argmin(smoothed)
    detected_x = start_x + min_idx_relative
    
    return detected_x

def detect_letter_box(image, binary):
    """
    Detects a large letter section box (like the box enclosing the letter 'M' on page 8).
    Returns (x, y, w, h) of the box if found, otherwise None.
    """
    height, width = image.shape[:2]
    
    # Find external contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    letter_box = None
    max_area = 0
    
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        
        # Criteria for the alphabet section header box:
        # 1. Must be in the upper half of the page
        # 2. Must be wide (at least 20% of page width)
        # 3. Must have a decent height (at least 40 pixels)
        # 4. Must not cover the entire page
        if y < height * 0.4 and w > width * 0.20 and h > 40 and h < height * 0.25:
            area = w * h
            if area > max_area:
                max_area = area
                letter_box = (x, y, w, h)
                
    return letter_box

def detect_header_page_number_and_icon(image, binary):
    """
    Detects the page number and the brand icon beside it in the top header.
    Returns (x, y, w, h) bounding box of the header if found, otherwise None.
    """
    height, width = image.shape[:2]
    
    # Search within the top 12% of the page height
    top_limit = int(height * 0.12)
    if top_limit < 100:
        top_limit = 100
        
    # Find external contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # First, find candidate contours for the brand icon (a square-like icon containing the letter 'M')
    icon_cnts = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # The icon is small, square-ish, and located in the top band
        if y + h < top_limit and 12 <= w <= 50 and 12 <= h <= 50:
            aspect_ratio = float(w) / h
            if 0.70 <= aspect_ratio <= 1.40:
                icon_cnts.append((x, y, w, h))
                
    if not icon_cnts:
        return None
        
    # Sort candidate icons by height (closer to the top of page)
    # and pick the best one
    icon_cnts.sort(key=lambda item: item[1])
    ix, iy, iw, ih = icon_cnts[0]
    
    # Start bounding box for the entire header with the icon itself
    hx0, hy0, hx1, hy1 = ix, iy, ix + iw, iy + ih
    
    # Now look for page number digits close to the icon
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # The page number digit(s) must:
        # 1. Be inside the top band
        # 2. Be close to the icon vertically (strict center alignment within 50% of icon height)
        # 3. Be close to the icon horizontally (within ~80 pixels)
        # 4. Have a height comparable to the icon (prevent merging body text or larger elements)
        # 5. Not be the icon itself
        if y + h < top_limit and not (x == ix and y == iy and w == iw and h == ih):
            vertical_close = abs((y + h/2.0) - (iy + ih/2.0)) < (ih * 0.5)
            height_comparable = h < (ih * 1.3)
            horizontal_dist = min(abs(x - (ix + iw)), abs((x + w) - ix))
            if vertical_close and height_comparable and horizontal_dist < 80:
                hx0 = min(hx0, x)
                hy0 = min(hy0, y)
                hx1 = max(hx1, x + w)
                hy1 = max(hy1, y + h)
                
    # Add a small padding of 4 pixels around the detected header bounding box
    padding = 4
    xmin = max(0, hx0 - padding)
    ymin = max(0, hy0 - padding)
    xmax = min(width, hx1 + padding)
    ymax = min(height, hy1 + padding)
    
    return (xmin, ymin, xmax - xmin, ymax - ymin)

def run_ocr_on_crop(crop):
    if 'VNImageRequestHandler' not in globals():
        return None
    temp_dir = os.path.dirname(os.path.abspath(__file__))
    for scale in [3.0, 2.0]:
        crop_resized = cv2.resize(crop, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        with tempfile.NamedTemporaryFile(suffix=".png", dir=temp_dir, delete=False) as temp_file:
            temp_img_path = temp_file.name
        try:
            cv2.imwrite(temp_img_path, crop_resized)
            url = NSURL.fileURLWithPath_(temp_img_path)
            handler = VNImageRequestHandler.alloc().initWithURL_options_(url, None)
            request = VNRecognizeTextRequest.alloc().init()
            request.setRecognitionLanguages_(["en-US"])
            request.setUsesLanguageCorrection_(False)
            success = handler.performRequests_error_([request], None)
            if success:
                results = request.results()
                for obs in results:
                    candidates = obs.topCandidates_(1)
                    if candidates:
                        text = candidates[0].string()
                        digits = re.findall(r'\d+', text)
                        if digits:
                            return int(digits[-1])
        except Exception as e:
            print(f"Error during OCR: {e}")
        finally:
            if os.path.exists(temp_img_path):
                try:
                    os.remove(temp_img_path)
                except Exception:
                    pass
    return None

def extract_page_number(image, header_box):
    """
    Extracts the printed page number. First tries OCR on the detected header box.
    If that fails or header_box is None, runs OCR on the top-left and top-right corners.
    """
    # 1. Try OCR on header_box if it exists
    if header_box:
        hx, hy, hw, hh = header_box
        crop = image[hy:hy+hh, hx:hx+hw]
        val = run_ocr_on_crop(crop)
        if val is not None:
            return val
            
    # 2. Try OCR on top-left and top-right corners of the page (checking both sides)
    height, width = image.shape[:2]
    top_limit = int(height * 0.12)
    if top_limit < 100:
        top_limit = 100
        
    left_crop = image[0:top_limit, 0:int(width * 0.25)]
    right_crop = image[0:top_limit, int(width * 0.75):width]
    
    for crop in [left_crop, right_crop]:
        val = run_ocr_on_crop(crop)
        if val is not None:
            return val
            
    return None

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pdf_path = os.path.join(script_dir, "2.pdf")
    output_pdf_path = os.path.join(script_dir, "2-split.pdf")
    output_dir = os.path.join(script_dir, "debug_output")
    os.makedirs(output_dir, exist_ok=True)
    
    doc = fitz.open(pdf_path)
    
    # Create the output PDF in memory
    out_doc = fitz.open()
    
    start_page = 5
    end_page = 862
    
    last_valid_page_number = None
    
    for idx in range(start_page, end_page):
        page_num = idx + 1
        page = doc[idx]
        
        # Render page to image at 150 DPI (zoom factor ~2.0) for layout analysis
        zoom = 2.0
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        
        # Convert pixmap to numpy array
        img_np = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
        
        # Convert RGB to BGR for OpenCV
        if pix.n == 4:
            image = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR)
        else:
            image = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            
        # Convert to grayscale and binary for layout analysis
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # 1. Detect column separator line (Red in debug image)
        x_sep = detect_column_separator(image, binary)
        
        # 2. Detect Alphabetical Letter Box (Green in debug image)
        letter_box = detect_letter_box(image, binary)
        
        # 3. Detect Page Number & Brand Icon (Blue in debug image)
        header_box = detect_header_page_number_and_icon(image, binary)
        
        # Extract printed page number from header (checking both header_box and corners)
        printed_page_num = extract_page_number(image, header_box)
        
        # Log the printed page number
        print(f"Page {page_num} (header number: {printed_page_num})")
            
        # --- PDF Modification (Redactions) ---
        # Add redactions directly on the page in-memory
        has_redactions = False
        
        # If page number/icon header is found, redact its bounding box (just the blue box)
        if header_box:
            hx, hy, hw, hh = header_box
            rect = fitz.Rect(hx / zoom, hy / zoom, (hx + hw) / zoom, (hy + hh) / zoom)
            page.add_redact_annot(rect, fill=(1, 1, 1))
            has_redactions = True
            
        if has_redactions:
            page.apply_redactions()
            
        # --- PDF Splitting & Vert/Horiz Cropping (CropBoxes) ---
        # Copy the modified/redacted page twice into the output document
        out_doc.insert_pdf(doc, from_page=idx, to_page=idx)
        out_doc.insert_pdf(doc, from_page=idx, to_page=idx)
        
        # Get the two newly inserted pages in the output document
        left_page = out_doc[-2]
        right_page = out_doc[-1]
        
        cb = page.cropbox
        x_sep_pt = x_sep / zoom
        
        # Determine top vertical crop coordinate (cut at the green line if letter box exists)
        if letter_box:
            bx, by, bw, bh = letter_box
            split_y_pt = (by + bh + 10) / zoom
            current_y0 = cb.y0 + split_y_pt
        else:
            current_y0 = cb.y0
            
        # Set CropBox for left page (left column)
        # Shifted by cb.x0 to account for original CropBox origin
        left_page.set_cropbox(fitz.Rect(cb.x0, current_y0, cb.x0 + x_sep_pt, cb.y1))
        
        # Set CropBox for right page (right column)
        # Shifted by cb.x0 to account for original CropBox origin
        right_page.set_cropbox(fitz.Rect(cb.x0 + x_sep_pt, current_y0, cb.x1, cb.y1))
        
        # --- Debug visual annotation (save image of original detected layout for reference) ---
        annotated_image = image.copy()
        cv2.line(annotated_image, (x_sep, 0), (x_sep, image.shape[0]), (0, 0, 255), 3)
        if letter_box:
            bx, by, bw, bh = letter_box
            split_y = by + bh + 10
            cv2.line(annotated_image, (0, split_y), (image.shape[1], split_y), (0, 255, 0), 3)
            cv2.rectangle(annotated_image, (bx, by), (bx + bw, by + bh), (0, 255, 0), 1)
        if header_box:
            hx, hy, hw, hh = header_box
            cv2.rectangle(annotated_image, (hx, hy), (hx + hw, hy + hh), (255, 0, 0), 2)
            
        out_img_path = os.path.join(output_dir, f"page_{page_num}_layout.png")
        cv2.imwrite(out_img_path, annotated_image)
        
        x_sep_pt = x_sep / zoom
        
    out_doc.save(output_pdf_path, garbage=4, deflate=True)
    out_doc.close()
    doc.close()

if __name__ == "__main__":
    main()
