"""Direct SQLite cleanup: delete extras, keep only top 10."""
import sqlite3

conn = sqlite3.connect("data/aitrading.db")
conn.row_factory = sqlite3.Row

# Count current TOP strategies
rows = conn.execute("SELECT id, name FROM strategies WHERE name LIKE 'TOP%'").fetchall()
print(f"Total TOP strategies: {len(rows)}")

# Delete all that still have 'TOP1:' prefix (these are the un-renamed extras)
cur = conn.execute("DELETE FROM strategies WHERE name LIKE 'TOP1:%'")
print(f"Deleted {cur.rowcount} extra TOP1: strategies")
conn.commit()

# Verify remaining
rows = conn.execute("SELECT id, name, description FROM strategies ORDER BY id").fetchall()
print(f"\nRemaining strategies ({len(rows)}):")
for r in rows:
    row = dict(r)
    print(f"  id={row['id']:>3}  {row['name']}")
    print(f"         {row['description']}")

conn.close()
