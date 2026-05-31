"""Clean up strategies: rename top 10 with correct rank, delete extras."""
import requests

BASE = "http://localhost:8000"
resp = requests.get(f"{BASE}/api/strategies")
strategies = resp.json()

# Filter TOP strategies (sorted by id = insertion order = rank order by return)
tops = [s for s in strategies if s["name"].startswith("TOP1:")]
print(f"Found {len(tops)} TOP strategies")

# Keep first 10 (highest return), rename with correct rank
for i, s in enumerate(tops[:10]):
    new_name = s["name"].replace("TOP1:", f"TOP{i+1}:")
    sid = s["id"]
    payload = {"name": new_name}
    r = requests.put(f"{BASE}/api/strategies/{sid}", json=payload)
    desc = s.get("description", "")
    print(f"  Kept #{i+1}: id={sid} -> {new_name}  ({desc})")

# Delete the extras
deleted = 0
for s in tops[10:]:
    sid = s["id"]
    requests.delete(f"{BASE}/api/strategies/{sid}")
    deleted += 1

print(f"\nDeleted {deleted} extra strategies. Kept top 10.")
