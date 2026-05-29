#!/usr/bin/env python3
import os
import re
import csv
import sys
import time

def ends_with_sentence_ender(line):
    """
    Checks if a line ends with sentence-ending punctuation (., ?, !),
    optionally followed by closing quotes or brackets.
    """
    line = line.strip()
    if not line:
        return True
    return bool(re.search(r'[.?!]["\')\]\u201d\u2019]*\s*$', line))

def ends_with_any_punctuation(line):
    """
    Checks if a line ends with ANY punctuation (., ?, !, comma, ;, :, +),
    optionally followed by closing quotes or brackets.
    Used for poem lines: commas/semicolons mark legitimate verse line endings
    and should NOT trigger a line join.
    """
    line = line.strip()
    if not line:
        return True
    return bool(re.search(r'[.?!,;:+]["\')\]\u201d\u2019]*\s*$', line))

def starts_with_capitalized(line):
    """
    Checks if the first alphabetic character of a line is capitalized.
    """
    line = line.strip()
    if not line:
        return False
    for char in line:
        if char.isalpha():
            return char.isupper()
    return False

def clean_poem_string(poem):
    """
    Cleans the poem column by:
    1. Removing leading random symbols (only - is allowed)
    2. Stripping outer parentheses wrappers
    3. Discarding standalone tag notes (Bt, Di doan, Kieu, etc.)
    4. Joining fragmented lines: if a line has no trailing punctuation
       and the next line starts lowercase, they are merged with a space.
       Hyphen-prefixed lines (speaker turns) always start a new group.
    """
    if not poem:
        return ""

    p = poem.strip()

    # Handle parenthesized wrapper for the whole poem first
    changed = True
    while changed:
        changed = False
        if p.startswith("("):
            if p.endswith(")"):
                p = p[1:-1].strip()
                changed = True
            elif p.endswith(")."):
                p = p[1:-2].strip() + "."
                changed = True
            elif p.endswith(")?"):
                p = p[1:-2].strip() + "?"
                changed = True
            elif p.endswith(")!"):
                p = p[1:-2].strip() + "!"
                changed = True

    lines = p.split("\n")
    cleaned_lines = []

    # Tag list to discard if they are on a line by themselves
    DISCARD_TAGS = {"bt", "di doan", "dị đoan", "kieu", "kiều", "quan niem cu", "quan niệm cũ"}

    for line in lines:
        line_clean = line.strip()
        if not line_clean:
            continue

        # 1. Handle leading hyphen preservation while cleaning leading symbols
        starts_with_hyphen = False
        if line_clean.startswith("-"):
            starts_with_hyphen = True
            line_clean = line_clean[1:].strip()

        # 2. Remove any leading symbols (non-alphanumeric, non-whitespace, non-hyphen)
        line_clean = re.sub(r"^[^\w\s\-]+", "", line_clean).strip()

        # 3. Check for parenthesized wrapping of the individual line
        if line_clean.startswith("("):
            if line_clean.endswith(")"):
                line_clean = line_clean[1:-1].strip()
            elif line_clean.endswith(")."):
                line_clean = line_clean[1:-2].strip() + "."
            elif line_clean.endswith(")?"):
                line_clean = line_clean[1:-2].strip() + "?"
            elif line_clean.endswith(")!"):
                line_clean = line_clean[1:-2].strip() + "!"
            else:
                # Starts with "(" but does not end with closing parenthesis.
                match = re.match(r"^\(([^)]+)\)\.?\s*", line_clean)
                if match:
                    tag = match.group(1).strip().lower()
                    rest = line_clean[match.end():].strip()
                    if rest:
                        line_clean = rest
                    elif tag in DISCARD_TAGS:
                        continue

        # 4. Check if the line is a discard tag (ignoring trailing punctuation)
        clean_tag_check = re.sub(r"[^\w\s]", "", line_clean).strip().lower()
        if clean_tag_check in DISCARD_TAGS:
            continue

        # 5. Prepend hyphen back if needed
        if starts_with_hyphen:
            line_clean = "- " + line_clean

        if line_clean:
            cleaned_lines.append(line_clean)

    if not cleaned_lines:
        return ""

    # === Line continuation pass for poem ===
    # In poems, verse lines legitimately end with commas/semicolons.
    # We only join a line to the next if:
    #   1. The current line ends WITHOUT any punctuation at all, AND
    #   2. The next line starts with a lowercase letter.
    # Hyphen-prefixed lines (speaker turns) always start a new group.
    merged_lines = []
    current = cleaned_lines[0]

    for next_l in cleaned_lines[1:]:
        next_stripped = next_l.strip()
        # Never merge across a hyphen-speaker-turn boundary
        if next_stripped.startswith("-"):
            merged_lines.append(current)
            current = next_l
        elif (not ends_with_any_punctuation(current)) and (not starts_with_capitalized(next_stripped)):
            # Fragmented continuation — join with a space
            current = current + " " + next_stripped
        else:
            merged_lines.append(current)
            current = next_l

    merged_lines.append(current)
    return "\n".join(merged_lines)

def clean_explanation_string(explanation):
    """
    Cleans the explanation column by joining fragmented lines.
    Lines are connected if the current line does not end with a sentence ender,
    or if the next line does not start with a capitalized letter.
    """
    if not explanation:
        return ""

    lines = [line.strip() for line in explanation.split("\n") if line.strip()]
    if not lines:
        return ""

    merged_lines = []
    current_line = lines[0]

    for next_line in lines[1:]:
        if (not ends_with_sentence_ender(current_line)) or (not starts_with_capitalized(next_line)):
            # Connect the lines with a space
            current_line = current_line + " " + next_line
        else:
            merged_lines.append(current_line)
            current_line = next_line

    merged_lines.append(current_line)
    return "\n".join(merged_lines)

def main():
    input_path = "extractions/input.csv"
    temp_path = "extractions/input_temp.csv"

    if not os.path.exists(input_path):
        print(f"Error: Input file {input_path} not found.")
        sys.exit(1)

    print(f"Starting CSV filtering on {input_path}...")
    start_time = time.time()

    row_count = 0
    cleaned_poems = 0
    cleaned_exps = 0

    with open(input_path, "r", encoding="utf-8") as infile, \
         open(temp_path, "w", encoding="utf-8", newline="") as outfile:

        reader = csv.DictReader(infile)
        # Ensure we write exact same fields in the same order
        fieldnames = reader.fieldnames
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            original_poem = row.get("poem", "")
            original_exp = row.get("explanation", "")

            cleaned_poem = clean_poem_string(original_poem)
            cleaned_exp = clean_explanation_string(original_exp)

            if cleaned_poem != original_poem:
                cleaned_poems += 1
            if cleaned_exp != original_exp:
                cleaned_exps += 1

            row["poem"] = cleaned_poem
            row["explanation"] = cleaned_exp

            writer.writerow(row)
            row_count += 1

            if row_count % 5000 == 0:
                print(f"Processed {row_count} rows...")

    # Replace the original file with the cleaned one in-place
    os.replace(temp_path, input_path)

    elapsed = time.time() - start_time
    print(f"Finished processing in {elapsed:.2f} seconds.")
    print(f"Total rows processed: {row_count}")
    print(f"Poems updated: {cleaned_poems}")
    print(f"Explanations updated: {cleaned_exps}")

if __name__ == "__main__":
    main()
