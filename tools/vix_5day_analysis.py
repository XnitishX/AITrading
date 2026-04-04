import json
import sys
import pathlib
import numpy as np
import pandas as pd

# Ensure repository root is on sys.path so `src` is importable
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.loader import load_master


def analyze(threshold=18.0, window=5):
    df = load_master()
    if 'vix_close' not in df.columns:
        print(json.dumps({'error': 'vix_close not available in master dataframe'}))
        return
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
        print(json.dumps({'error': f'No {window}-day windows found with vix>{threshold}'}))
        return
    arr = np.array(rets)
    mean = float(arr.mean() * 100)
    median = float(np.median(arr) * 100)
    std = float(arr.std(ddof=1) * 100)
    p5 = float(np.percentile(arr, 5) * 100)
    p95 = float(np.percentile(arr, 95) * 100)
    # mode (may be multiple) - use pandas
    modes = pd.Series(arr).mode().tolist()
    modes_pct = [float(m * 100) for m in modes]

    out = {
        'threshold': threshold,
        'window_days': window,
        'count_windows': len(arr),
        'mean_pct': mean,
        'median_pct': median,
        'mode_pct': modes_pct,
        'std_pct': std,
        'p5_pct': p5,
        'p95_pct': p95,
        'sample_start_dates': indices[:10],
    }
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    analyze(threshold=18.0, window=5)
