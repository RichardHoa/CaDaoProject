import os
import sys
import fitz  # PyMuPDF

def cut_pdf(pdf_path, pages_to_remove, output_path=None):
    if not os.path.exists(pdf_path):
        print(f"Error: File not found at {pdf_path}")
        return False
        
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    
    # Convert 1-based page numbers to 0-based indices
    indices_to_remove = set()
    for p in pages_to_remove:
        idx = p - 1
        if 0 <= idx < total_pages:
            indices_to_remove.add(idx)
        else:
            print(f"Warning: Page number {p} is out of bounds (1-{total_pages}). Skipping.")
            
    if not indices_to_remove:
        print("No valid pages to remove.")
        doc.close()
        return False
        
    # Keep pages that are NOT in indices_to_remove
    keep_indices = [i for i in range(total_pages) if i not in indices_to_remove]
    
    # Select pages to keep in memory
    doc.select(keep_indices)
    
    if output_path is None:
        base, ext = os.path.splitext(pdf_path)
        output_path = f"{base}_cut{ext}"
        
    print(f"Saving modified PDF ({len(doc)} pages remaining) to: {output_path}")
    doc.save(output_path, garbage=4, deflate=True)
    doc.close()
    print("Successfully completed!")
    return True

def parse_pages(pages_str):
    pages = []
    # Parse comma-separated page numbers and ranges (e.g., 51,52 or 51-53)
    parts = pages_str.split(',')
    for part in parts:
        part = part.strip()
        if '-' in part:
            try:
                start, end = map(int, part.split('-'))
                pages.extend(range(start, end + 1))
            except ValueError:
                print(f"Warning: Could not parse range '{part}'")
        else:
            try:
                pages.append(int(part))
            except ValueError:
                if part:
                    print(f"Warning: Could not parse page '{part}'")
    return sorted(list(set(pages)))

def main():
    if len(sys.argv) < 3:
        print("Usage: python cut.py <pdf_path> <page_numbers>")
        print("Example: python cut.py 1.pdf 51,52")
        print("\nEntering interactive mode:")
        pdf_path = input("Enter PDF path (e.g., 1.pdf): ").strip()
        if not pdf_path:
            print("No PDF path entered. Exiting.")
            return
        pages_str = input("Enter page numbers to cut out (comma-separated, e.g. 51,52): ").strip()
        if not pages_str:
            print("No page numbers entered. Exiting.")
            return
    else:
        pdf_path = sys.argv[1]
        pages_str = sys.argv[2]
        
    pages = parse_pages(pages_str)
    if not pages:
        print("No valid page numbers parsed.")
        return
        
    cut_pdf(pdf_path, pages)

if __name__ == "__main__":
    main()
