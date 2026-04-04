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


def compute_mode_round(threshold=18.0, window=5, ndigits=2):
    df = load_master()
    mask = df['vix_close'] > threshold
    rets = []
    for i in range(len(df) - (window - 1)):
        if mask.iloc[i:i+window].all():
            start = df['Close'].iloc[i]
            end = df['Close'].iloc[i+window-1]
            if pd.notna(start) and pd.notna(end) and start > 0:
                rets.append((end / start) - 1)
    arr = np.array(rets)
    rounded = np.round(arr * 100, ndigits)
    # compute most frequent rounded value(s)
    vals, counts = np.unique(rounded, return_counts=True)
    maxc = counts.max()
    modes = vals[counts == maxc].tolist()
    out = {
        'count_windows': int(len(arr)),
        'mode_rounded_pct': [float(m) for m in modes],
        'mode_count': int(maxc),
    }
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    compute_mode_round()
