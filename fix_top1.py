"""Re-create the missing TOP1 strategy."""
import sqlite3, json

conn = sqlite3.connect("data/aitrading.db")
from datetime import datetime
now = datetime.utcnow().isoformat()

params = json.dumps({
    "fast_window": 5,
    "slow_window": 20,
    "stop_loss_pct": 5.0,
    "take_profit_pct": 10.0
})
desc = "Return: +19.12%, MaxDD: 3.44%, Sharpe: 0.66, PF: 1.71, SL: wide"

conn.execute(
    "INSERT INTO strategies (name, type, params, description, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
    ("TOP1: SMA(5/20) (wide)", "sma_crossover", params, desc, now, now)
)
conn.commit()

# Verify
conn.row_factory = sqlite3.Row
rows = conn.execute("SELECT id, name, description FROM strategies WHERE name LIKE 'TOP%' ORDER BY name").fetchall()
print(f"Top strategies ({len(rows)}):")
for r in rows:
    row = dict(r)
    print(f"  id={row['id']:>3}  {row['name']:40s}  {row['description']}")
conn.close()
