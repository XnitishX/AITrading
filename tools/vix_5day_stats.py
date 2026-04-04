import sys
import pathlib
import json
import numpy as np
import pandas as pd

# Ensure repository root is on sys.path
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.loader import load_master


def compute(threshold=18.0, window=5):
    df = load_master()
    mask = df['vix_close'] > threshold
    rets = []
    indices = []
    for i in range(len(df) - (window - 1)):
        if mask.iloc[i:i+window].all():
            start = df['Close'].iloc[i]
            end = df['Close'].iloc[i+window-1]
            if pd.notna(start) and pd.notna(end) and start > 0:
                rets.append((end / start) - 1)
                indices.append(df['Date'].iloc[i].strftime('%Y-%m-%d'))
    if not rets:
        print(json.dumps({'error': 'no windows found'}))
        return
    arr = np.array(rets)
    out = {
        'threshold': threshold,
        'window_days': window,
        'count_windows': int(len(arr)),
        'mean_pct': float(arr.mean() * 100),
        'median_pct': float(np.median(arr) * 100),
        'mode_pct': [float(m * 100) for m in pd.Series(arr).mode().tolist()],
        'std_pct': float(arr.std(ddof=1) * 100),
    }
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    compute()
