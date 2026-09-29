#!/usr/bin/env python3
"""
Apple Vision OCR CLI Utility
---------------------------
A standalone script to perform high-accuracy OCR on images and PDF files using the
native macOS Vision framework via PyObjC.

Usage:
    python apple_ocr.py input_file.pdf --output output.txt
    python apple_ocr.py input_image.png --preprocess
"""

import os
import sys
import argparse
import tempfile
import json as py_json

# 1. Load macOS Vision framework dynamically via PyObjC
try:
    import objc
    from Foundation import NSURL
    # Dynamically load the macOS Vision framework bundle into the global namespace
    objc.loadBundle('Vision', bundle_path='/System/Library/Frameworks/Vision.framework', module_globals=globals())
except ImportError:
    print("Error: PyObjC is not installed. Please install it using:")
    print("  pip install pyobjc-core pyobjc-framework-Vision pyobjc-framework-Cocoa")
    sys.exit(1)
except Exception as e:
    print(f"Error: Failed to load macOS Vision framework: {e}")
    print("Note: This script requires macOS to run Apple Vision OCR.")
    sys.exit(1)

# Check if optional dependencies are available
try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False

try:
    import cv2
    import numpy as np
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False


def _apply_preprocessing_pipeline(gray_img):
    """
    Applies the OpenCV pre-processing pipeline from step2_improved.py to improve OCR.
    Steps:
      1. Bilateral filter     - smooth noise, keep character edges sharp
      2. Adaptive threshold  - handle uneven lighting
      3. Global deskew       - correct tilt using contour angles
      4. Diacritic thickening - thicken black strokes for better diacritics recognition
    """
    # Step 1: Bilateral filter
    denoised = cv2.bilateralFilter(gray_img, d=9, sigmaColor=75, sigmaSpace=75)

    # Step 2: Adaptive binarization
    binary = cv2.adaptiveThreshold(
        denoised, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=31, C=10
    )

    # Step 3: Global deskew via contour median angle
    inv = cv2.bitwise_not(binary)
    contours, _ = cv2.findContours(inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    angles = []
    for cnt in contours:
        if cv2.contourArea(cnt) < 100:  # skip tiny noise blobs
            continue
        rect = cv2.minAreaRect(cnt)
        angle = rect[-1]  # angle in [-90, 0)
        if angle < -45:
            angle = 90 + angle
        if abs(angle) <= 45:
            angles.append(angle)

    if angles:
        median_angle = float(np.median(angles))
        if abs(median_angle) > 0.5:  # only rotate if skew is meaningful
            h, w = binary.shape
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
            binary = cv2.warpAffine(
                binary, M, (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=255
            )

    # Step 4: Diacritic thickening (vertical erosion of white background → thickens black strokes)
    kernel = np.ones((2, 1), np.uint8)
    enhanced = cv2.erode(binary, kernel, iterations=1)

    return enhanced


def perform_vision_ocr(image_path, languages=None, use_lang_correction=True):
    """
    Performs Apple Vision OCR on a given image file path and returns text lines with coordinates.
    
    Vision coordinates use a normalized system where (0,0) is bottom-left, (1,1) is top-right.
    We convert the y-coordinate to top-left format (0,0 is top-left) for standard coordinate systems.
    """
    if languages is None:
        languages = ["vi-VN", "en-US"]

    # Convert the file path to NSURL as required by VNImageRequestHandler
    url = NSURL.fileURLWithPath_(image_path)
    
    # Initialize handler and request
    handler = VNImageRequestHandler.alloc().initWithURL_options_(url, None)
    request = VNRecognizeTextRequest.alloc().init()
    
    # Configure recognition options
    request.setRecognitionLanguages_(languages)
    request.setUsesLanguageCorrection_(use_lang_correction)
    
    # Execute the request
    success = handler.performRequests_error_([request], None)
    
    lines = []
    if success:
        results = request.results()
        for obs in results:
            box = obs.boundingBox()
            # Vision uses bottom-left origin; convert to top-left origin format
            x = box.origin.x
            y = 1.0 - box.origin.y - box.size.height
            w = box.size.width
            h = box.size.height
            
            candidates = obs.topCandidates_(1)
            text = candidates[0].string() if candidates else ""
            
            lines.append({
                'text': text,
                'bbox': {'x': x, 'y': y, 'w': w, 'h': h}
            })
            
    # Sort top-to-bottom, then left-to-right
    lines.sort(key=lambda l: (l['bbox']['y'], l['bbox']['x']))
    return lines


def process_image_file(image_path, preprocess=False, languages=None, use_lang_correction=True):
    """Processes a single image file, optionally applying preprocessing."""
    if preprocess:
        if not HAS_OPENCV:
            raise ImportError("OpenCV and NumPy are required for preprocessing. Please run: pip install opencv-python numpy")
        
        # Read image as grayscale
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not read image at: {image_path}")
            
        processed_img = _apply_preprocessing_pipeline(img)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            cv2.imwrite(tmp_path, processed_img)
            return perform_vision_ocr(tmp_path, languages, use_lang_correction)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    else:
        return perform_vision_ocr(image_path, languages, use_lang_correction)


def process_pdf_file(pdf_path, preprocess=False, languages=None, use_lang_correction=True, dpi=300):
    """Processes a PDF file, rendering pages to images using PyMuPDF and running OCR on each."""
    if not HAS_PYMUPDF:
        raise ImportError("PyMuPDF (fitz) is required to process PDF files. Please run: pip install PyMuPDF")

    doc = fitz.open(pdf_path)
    results = {}

    for page_idx in range(len(doc)):
        page = doc[page_idx]
        # Render page to high-res pixmap (DPI=300 for OCR accuracy)
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        
        # Save to temp PNG for processing
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name
            
        try:
            if preprocess:
                if not HAS_OPENCV:
                    raise ImportError("OpenCV and NumPy are required for preprocessing. Please run: pip install opencv-python numpy")
                # Read pixmap bytes into numpy, preprocess, and save
                nparr = np.frombuffer(pix.tobytes("png"), np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
                processed = _apply_preprocessing_pipeline(img)
                cv2.imwrite(tmp_path, processed)
            else:
                pix.save(tmp_path)
                
            page_lines = perform_vision_ocr(tmp_path, languages, use_lang_correction)
            results[page_idx + 1] = page_lines
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
                
    doc.close()
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Apple Vision OCR utility for PDF and image files.",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("input", help="Path to the input PDF or image file.")
    parser.add_argument("-o", "--output", help="Path to save the output text file (default: stdout).")
    parser.add_argument("-p", "--preprocess", action="store_true",
                        help="Enable OpenCV preprocessing (noise reduction, binarization, deskew, stroke thickening). Recommended for scans/photos.")
    parser.add_argument("-l", "--langs", nargs="+", default=["vi-VN", "en-US"],
                        help="List of language codes to recognize (default: vi-VN en-US).")
    parser.add_argument("--no-correction", action="store_false", dest="correction",
                        help="Disable language correction/spellchecking.")
    parser.add_argument("--json", action="store_true",
                        help="Output detailed OCR result (with bounding boxes) as JSON.")
    parser.add_argument("--dpi", type=int, default=300,
                        help="DPI resolution for rendering PDF pages (default: 300).")

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input path '{args.input}' does not exist.")
        sys.exit(1)

    # Detect file type based on extension
    _, ext = os.path.splitext(args.input.lower())
    
    is_pdf = (ext == '.pdf')
    
    print(f"Loading macOS Vision Framework OCR...", file=sys.stderr)
    print(f"Target languages: {', '.join(args.langs)}", file=sys.stderr)
    print(f"Preprocessing: {'Enabled' if args.preprocess else 'Disabled'}", file=sys.stderr)

    try:
        if is_pdf:
            print(f"Processing PDF file: {args.input} ...", file=sys.stderr)
            ocr_results = process_pdf_file(
                args.input, 
                preprocess=args.preprocess, 
                languages=args.langs, 
                use_lang_correction=args.correction,
                dpi=args.dpi
            )
        else:
            print(f"Processing Image file: {args.input} ...", file=sys.stderr)
            lines = process_image_file(
                args.input, 
                preprocess=args.preprocess, 
                languages=args.langs, 
                use_lang_correction=args.correction
            )
            ocr_results = {1: lines}
    except Exception as e:
        print(f"Error running OCR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Format the output
    output_str = ""
    if args.json:
        output_str = py_json.dumps(ocr_results, indent=2, ensure_ascii=False)
    else:
        # Text output format
        for page_num, lines in sorted(ocr_results.items()):
            if is_pdf:
                output_str += f"\n--- PAGE {page_num} ---\n"
            for item in lines:
                output_str += item['text'] + "\n"

    # Write to output file or print
    if args.output:
        try:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(output_str)
            print(f"Successfully wrote OCR output to {args.output}")
        except Exception as e:
            print(f"Error writing to output file: {e}")
            sys.exit(1)
    else:
        print(output_str)


if __name__ == '__main__':
    main()
