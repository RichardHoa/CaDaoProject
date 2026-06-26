import sqlite3

conn = sqlite3.connect("data/wiki.db")
cursor = conn.cursor()

# Get unique categories
cursor.execute("SELECT DISTINCT category FROM wiki")
categories = [row[0] for row in cursor.fetchall() if row[0]]

print(f"=== Found {len(categories)} unique categories in wiki.db ===")
print("All categories:")
for cat in sorted(categories):
    print(f" - {cat}")

conn.close()
