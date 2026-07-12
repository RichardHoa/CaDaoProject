import argparse
import csv
import json


def json_to_csv(json_path, csv_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []
    for topic_data in data:
        topic = topic_data.get("topic", "")
        for poem in topic_data.get("poems", []):
            rows.append(
                {
                    "topic": topic,
                    "poem_text": poem.get("poem_text", ""),
                    "introduction": poem.get("introduction", ""),
                    "interpretation": poem.get("interpretation", ""),
                    "blanks": json.dumps(
                        poem.get("blanks", []), ensure_ascii=False
                    ),
                }
            )

    fieldnames = [
        "topic",
        "poem_text",
        "introduction",
        "interpretation",
        "blanks",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Successfully converted {json_path} -> {csv_path}")


def csv_to_json(csv_path, json_path):
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    topics_order = []
    topics_map = {}

    for row in rows:
        topic = row["topic"]
        if topic not in topics_map:
            topics_order.append(topic)
            topics_map[topic] = []

        poem = {
            "poem_text": row["poem_text"],
            "blanks": (
                json.loads(row["blanks"]) if row.get("blanks") else []
            ),
            "introduction": row["introduction"],
            "interpretation": row["interpretation"],
        }
        topics_map[topic].append(poem)

    output_data = []
    for topic in topics_order:
        output_data.append({"topic": topic, "poems": topics_map[topic]})

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    print(f"Successfully converted {csv_path} -> {json_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert between nested poem JSON and flat CSV formats."
    )
    parser.add_argument("input", help="Path to the input file")
    parser.add_argument("output", help="Path to the output file")
    parser.add_argument(
        "--reverse",
        action="store_true",
        help="Convert CSV back to JSON format",
    )

    args = parser.parse_args()

    if args.reverse:
        csv_to_json(args.input, args.output)
    else:
        json_to_csv(args.input, args.output)
