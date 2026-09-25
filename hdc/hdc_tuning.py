"""
HDC hyperparameter tuning for top-performing PepBERT and ESM2 models.

Flow:
  1. Extract embeddings for focus models (cached to disk after first run)
  2. Sweep HDC hyperparameters via grid search
  3. Save per-config results to CSV, print top configs
"""
import os
import sys
import itertools
import numpy as np
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)    # for helpers.py
sys.path.insert(0, HERE)    # for pepbert_hdc / esm2_hdc

from helpers import load_sequences, extract_embeddings, run_kfold_hdc

# ---------- Config ----------
DATA_DIR = os.path.join(ROOT, 'data')
CACHE_DIR = os.path.join(ROOT, 'embeddings')
RESULTS_DIR = os.path.join(HERE, 'tuning_results')
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# (label, family, repo_id, weights_file_or_None)
FOCUS_MODELS = [
    ('PepBERT-large-UniParc', 'pepbert',
     'dzjxzyd/PepBERT-large-UniParc', 'tmodel_17.pt'),
    ('esm2_t12_35M_UR50D', 'esm2',
     'facebook/esm2_t12_35M_UR50D', None),
]

HDC_GRID = {
    'dim': [10000, 20000, 40000],
    'lr': [0.01, 0.035, 0.1],
    'epochs': [20, 40, 80],
}

# ---------- Embedding cache ----------
def cache_path(label):
    return os.path.join(CACHE_DIR, f'{label}.npz')

def get_embeddings(label, family, repo_id, weights):
    path = cache_path(label)
    if os.path.exists(path):
        print(f"  Loading cached embeddings: {path}")
        d = np.load(path)
        return d['X'], d['y']

    print(f"  Extracting embeddings (slow, runs once)...")
    amp_seqs = load_sequences(os.path.join(DATA_DIR, 'Poreforming_AMPS.txt'))
    cpp_seqs = load_sequences(os.path.join(DATA_DIR, 'Poreforming_CPPs.txt'))

    if family == 'pepbert':
        import pepbert_hdc as m
        m.model, m.tokenizer, m.max_seq_len = m.load_model(repo_id, weights)
        get_emb = m.get_embedding
    else:
        import esm2_hdc as m
        m.tokenizer, m.model = m.load_model(repo_id)
        get_emb = m.get_embedding

    amp_emb = extract_embeddings(get_emb, amp_seqs, 'AMP')
    cpp_emb = extract_embeddings(get_emb, cpp_seqs, 'CPP')
    X = np.vstack([amp_emb, cpp_emb])
    y = np.array([0] * len(amp_seqs) + [1] * len(cpp_seqs))
    np.savez(path, X=X, y=y)
    print(f"  Cached to: {path}")
    return X, y

# ---------- Grid search ----------
def grid_search(X, y, label, grid):
    keys = list(grid.keys())
    combos = list(itertools.product(*[grid[k] for k in keys]))
    print(f"\nGrid search: {len(combos)} configs for {label}")
    results = []
    for i, combo in enumerate(combos, 1):
        cfg = dict(zip(keys, combo))
        print(f"  [{i}/{len(combos)}] {cfg}")
        accs, _, _ = run_kfold_hdc(X, y, label, n_folds=5, **cfg)
        results.append({**cfg,
                        'mean_acc': float(np.mean(accs)),
                        'std': float(np.std(accs))})
    return results

def save_results(results, label):
    import csv
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    path = os.path.join(RESULTS_DIR, f'{label}_{ts}.csv')
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)
    print(f"\nResults: {path}")
    top = sorted(results, key=lambda r: r['mean_acc'], reverse=True)[:5]
    print(f"Top 5 for {label}:")
    for r in top:
        print(f"  acc={r['mean_acc']:.4f} ± {r['std']:.4f}  {r}")

# ---------- Main ----------
def main():
    for label, family, repo_id, weights in FOCUS_MODELS:
        print(f"\n=== {label} ===")
        X, y = get_embeddings(label, family, repo_id, weights)
        results = grid_search(X, y, label, HDC_GRID)
        save_results(results, label)

if __name__ == '__main__':
    main()
