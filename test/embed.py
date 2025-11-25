# Generates 2048-d embeddings for test.csv with:
# 1) match_dist1 pairs at max cosine distance (flip/sign trick),
# 2) match_dist2_512 pairs whose cosine distances correlate with log2(distance) via classical MDS,
# 3) unmatched samples as random.
import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# Try to import compiled Cython module, fall back to NumPy if not available
try:
    from floyd_warshall import floyd_warshall_inplace
    USE_CYTHON = True
except ImportError:
    USE_CYTHON = False
    print("Warning: Cython module not found. Run: python setup_floyd.py build_ext --inplace")
    print("Falling back to NumPy implementation (slower)")

# I/O paths (adjust if needed)
PATH_TEST = 'test.csv'
PATH_M1   = '/home/work/.nistring/embedding/match_dist1.csv'
PATH_UN   = '/home/work/.nistring/embedding/unmatched.csv'
PATH_M2   = 'match_dist2_512.csv'  # try local first
if not os.path.exists(PATH_M2):
    PATH_M2 = '/home/work/.nistring/embedding/match_dist2_512.csv'

D  = 2048
P2 = 64   # subspace for match_dist2_512 embedding
SEED = 42
rng = np.random.default_rng(SEED)

# Load data
df_test = pd.read_csv(PATH_TEST)
ids = df_test['ID'].tolist()
N   = len(ids)
id2idx = {s:i for i,s in enumerate(ids)}

# Start with random unit embeddings
emb = rng.normal(size=(N, D))
emb /= np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9

def process_dist2_pair(args):
    """Process a single distance pair for parallel loading."""
    a, b, dist, idx2 = args
    if a in idx2 and b in idx2:
        i, j = idx2[a], idx2[b]
        w = np.log2(max(float(dist), 1e-12))
        return [(i, j, w), (j, i, w)]
    return []

# 2) Correlate cosine distance with log2(distance) for match_dist2_512 via classical MDS in a subspace
if os.path.exists(PATH_M2):
    df2 = pd.read_csv(PATH_M2)
    use = [s for s in set(df2['sample_a']).union(df2['sample_b']) if s in id2idx]
    if len(use) > 1:
        idx2 = {s:i for i,s in enumerate(use)}
        K = len(use)
        # Build sparse/partial log2 distance graph; fill with +inf except self=0
        d = np.full((K, K), np.inf, dtype=np.float64)
        np.fill_diagonal(d, 0.0)
        
        # Parallel processing of distance pairs
        n_jobs = min(cpu_count(), 4)  # limit to 4 cores to avoid overhead
        with Pool(n_jobs) as pool:
            args_list = [(a, b, dist, idx2) for a, b, dist in 
                        df2[['sample_a', 'sample_b', 'distance']].itertuples(index=False)]
            results = pool.map(process_dist2_pair, args_list)
        
        # Populate distance matrix
        for pairs in results:
            for i, j, w in pairs:
                if w < d[i, j]:
                    d[i, j] = w
        
        # Floyd–Warshall to complete distances (K up to ~512 is OK)
        if USE_CYTHON:
            print("Using Cython-optimized Floyd-Warshall...")
            floyd_warshall_inplace(d)
        else:
            for k in tqdm(range(K), desc='Floyd-Warshall', leave=False):
                dk = d[:, k][:, None] + d[k, :][None, :]
                d = np.minimum(d, dk)
        # Replace any remaining inf with max finite
        finite = d[np.isfinite(d)]
        if finite.size == 0:
            d[:] = 0.0
        else:
            d[~np.isfinite(d)] = finite.max()
        # Classical MDS (Torgerson): X = V * sqrt(Lambda_+)
        D2 = d**2
        J  = np.eye(K) - np.ones((K, K))/K
        B  = -0.5 * J @ D2 @ J
        w, V = np.linalg.eigh(B)
        idx = np.argsort(w)[::-1]
        w, V = w[idx], V[:, idx]
        keep = np.sum(w > 1e-9)
        keep = max(1, min(P2, keep))
        Wpos = np.clip(w[:keep], 1e-12, None)
        X    = V[:, :keep] * np.sqrt(Wpos)[None, :]
        # Row-normalize into a unit P2-subspace
        X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-9
        # Place into first P2 dims; keep the rest as small noise
        for s, i in idx2.items():
            ii = id2idx[s]
            emb[ii, :keep] = X[i]
        emb /= np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9

# 1) Maximize cosine distance for match_dist1 pairs by flipping the paired vector
df1 = pd.read_csv(PATH_M1)
for a, b in tqdm(df1[['sample_a', 'sample_b']].itertuples(index=False), 
                 desc='match_dist1', total=len(df1)):
    if a in id2idx and b in id2idx:
        ia, ib = id2idx[a], id2idx[b]
        emb[ib] = -emb[ia]

# 3) Unmatched: assign fresh random unit vectors
if os.path.exists(PATH_UN):
    df_un = pd.read_csv(PATH_UN)
    for s in tqdm(df_un['sample'].tolist(), desc='unmatched'):
        if s in id2idx:
            ii = id2idx[s]
            v = rng.normal(size=D)
            emb[ii] = v / (np.linalg.norm(v) + 1e-9)

# Save
cols = ['ID'] + [f'emb_{i:04d}' for i in range(D)]
out = pd.DataFrame({'ID': ids})
out = pd.concat([out, pd.DataFrame(emb, columns=[f'emb_{i:04d}' for i in range(D)])], axis=1)
out.to_csv('embeddings_2048.csv', index=False)
print('Wrote embeddings_2048.csv')