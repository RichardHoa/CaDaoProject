import json
import argparse
import os
import re

# Paths relative to the project root
JSON_PATH = '../data/learning_data.json'
TXT_PATH = 'learning_data_editor.txt'

def export_data():
    """Reads JSON and writes to a human-editable text file."""
    if not os.path.exists(JSON_PATH):
        print(f"Error: {JSON_PATH} not found.")
        return

    with open(JSON_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    with open(TXT_PATH, 'w', encoding='utf-8') as f:
        poem_id = 0
        for topic_obj in data:
            f.write("="*80 + "\n")
            f.write(f"TOPIC: {topic_obj.get('topic', 'Unknown Topic')}\n")
            f.write("="*80 + "\n\n")
            
            for poem in topic_obj['poems']:
                f.write("-" * 80 + "\n")
                f.write(f"POEM_ID: {poem_id}\n")
                f.write(f"POEM:\n{poem['poem_text']}\n\n")
                f.write(f"INTRODUCTION:\n{poem.get('introduction', '')}\n\n")
                f.write(f"INTERPRETATION:\n{poem.get('interpretation', '')}\n")
                f.write("-" * 80 + "\n\n")
                poem_id += 1
                
    print(f"Successfully exported {poem_id} poems to {TXT_PATH}")
    print("You can now edit the INTRODUCTION and INTERPRETATION fields in the text file.")
    print("Note: Do not modify the POEM_ID or the field markers (INTRODUCTION:, INTERPRETATION:).")

def import_data():
    """Reads the edited text file and updates the JSON file."""
    if not os.path.exists(TXT_PATH):
        print(f"Error: {TXT_PATH} not found. Run with --out first to generate it.")
        return

    if not os.path.exists(JSON_PATH):
        print(f"Error: {JSON_PATH} not found.")
        return

    with open(JSON_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    with open(TXT_PATH, 'r', encoding='utf-8') as f:
        content = f.read()

    # Split by the dashed separator to get individual poem blocks
    segments = re.split(r'-{40,}', content)
    
    updates = {}
    for segment in segments:
        if 'POEM_ID:' not in segment:
            continue
        
        try:
            # Extract POEM_ID
            id_match = re.search(r'POEM_ID:\s*(\d+)', segment)
            if not id_match:
                continue
            pid = int(id_match.group(1))
            
            # Split segment by field markers
            # This handles variable whitespace and multi-line content
            parts = re.split(r'\n(INTRODUCTION|INTERPRETATION):\n', segment)
            
            intro = None
            interp = None
            
            for i in range(1, len(parts), 2):
                marker = parts[i]
                text = parts[i+1].strip()
                if marker == 'INTRODUCTION':
                    intro = text
                elif marker == 'INTERPRETATION':
                    interp = text
            
            if intro is not None and interp is not None:
                updates[pid] = {
                    'introduction': intro,
                    'interpretation': interp
                }
        except Exception as e:
            print(f"Warning: Could not parse segment near POEM_ID {pid if 'pid' in locals() else 'unknown'}: {e}")

    # Update the JSON structure
    poem_id = 0
    updated_count = 0
    for topic_obj in data:
        for poem in topic_obj['poems']:
            if poem_id in updates:
                poem['introduction'] = updates[poem_id]['introduction']
                poem['interpretation'] = updates[poem_id]['interpretation']
                updated_count += 1
            poem_id += 1
    
    # Save back to JSON
    with open(JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"Successfully imported {updated_count} updates into {JSON_PATH}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Editor for Ca Dao learning data.")
    parser.add_argument('--in', dest='mode_in', action='store_true', help="Import shifts from TXT to JSON")
    parser.add_argument('--out', dest='mode_out', action='store_true', help="Export JSON to TXT (default behavior)")
    
    args = parser.parse_args()
    
    # Run in the requested mode
    if args.mode_in:
        import_data()
    else:
        export_data()
