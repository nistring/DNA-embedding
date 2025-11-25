import pandas as pd
try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(x, total=None, desc=None):
        return x
from multiprocessing import Pool
from _hdist import hdist

def gc_content(seq):
    gc = sum(1 for c in seq if c in 'GCgc')
    return gc / len(seq) if len(seq) > 0 else 0.0

def nearest(args):
    i, seqs, samples = args
    si = seqs[i]
    best_j, best_d = None, float("inf")
    for j, sj in enumerate(seqs):
        if j == i:
            continue
        d = hdist(si, sj)
        if d < best_d:
            best_j, best_d = j, d
    return samples[i], samples[best_j], best_d

if __name__ == "__main__":
    df = pd.read_csv("test.csv", usecols=[0, 1], names=["ID", "seq"], header=0)
    samples, seqs = df["ID"].tolist(), df["seq"].tolist()
    gc_map = {s: gc_content(seq) for s, seq in zip(samples, seqs)}

    tasks = [(i, seqs, samples) for i in range(len(seqs))]
    with Pool(processes=48) as pool:
        it = pool.imap_unordered(nearest, tasks, chunksize=64)
        pairs = [p for p in tqdm(it, total=len(tasks), desc="Matching")]

    # dedupe symmetric pairs, keep smallest distance on conflicts
    uniq = {}
    for a, b, d in pairs:
        x, y = (a, b) if a <= b else (b, a)
        if (x, y) not in uniq or d < uniq[(x, y)]:
            uniq[(x, y)] = d

    dist1 = [(x, y, d, gc_map[x], gc_map[y]) for (x, y), d in uniq.items() if d == 1]
    dist2_512 = [(x, y, d, gc_map[x], gc_map[y]) for (x, y), d in uniq.items() if 2 <= d <= 512]
    
    matched = set()
    for (x, y), d in uniq.items():
        if d <= 512:
            matched.add(x)
            matched.add(y)
    unmatched = [(s, gc_map[s]) for s in samples if s not in matched]

    pd.DataFrame(dist1, columns=["sample_a", "sample_b", "distance", "gc_a", "gc_b"]).sort_values("sample_a").to_csv("match_dist1.csv", index=False)
    pd.DataFrame(dist2_512, columns=["sample_a", "sample_b", "distance", "gc_a", "gc_b"]).sort_values("sample_a").to_csv("match_dist2_512.csv", index=False)
    pd.DataFrame(unmatched, columns=["sample", "gc_content"]).to_csv("unmatched.csv", index=False)

    # Print GC statistics
    if dist1:
        gc_a = [x[3] for x in dist1]
        gc_b = [x[4] for x in dist1]
        print(f"match_dist1.csv - gc_a: mean={sum(gc_a)/len(gc_a):.4f}, std={pd.Series(gc_a).std():.4f}")
        print(f"match_dist1.csv - gc_b: mean={sum(gc_b)/len(gc_b):.4f}, std={pd.Series(gc_b).std():.4f}")
    
    if dist2_512:
        gc_a = [x[3] for x in dist2_512]
        gc_b = [x[4] for x in dist2_512]
        print(f"match_dist2_512.csv - gc_a: mean={sum(gc_a)/len(gc_a):.4f}, std={pd.Series(gc_a).std():.4f}")
        print(f"match_dist2_512.csv - gc_b: mean={sum(gc_b)/len(gc_b):.4f}, std={pd.Series(gc_b).std():.4f}")
    
    if unmatched:
        gc_vals = [x[1] for x in unmatched]
        print(f"unmatched.csv - gc_content: mean={sum(gc_vals)/len(gc_vals):.4f}, std={pd.Series(gc_vals).std():.4f}")