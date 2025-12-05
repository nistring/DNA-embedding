import pandas as pd
try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(x, total=None, desc=None):
        return x
from multiprocessing import Pool
from _hdist import hdist

if __name__ == "__main__":
    df = pd.read_csv("../data/test.csv", usecols=[0, 1], names=["ID", "seq"], header=0)
    samples, seqs = df["ID"].tolist(), df["seq"].tolist()

    # Compute distances for all unordered pairs (i < j) and keep those with d <= 512.
    # This ensures dist_le_512 contains every pair whose hdist <= 512, not only nearest neighbors.
    from itertools import combinations

    def pair_hd(args):
        i, j = args
        d = hdist(seqs[i], seqs[j])
        if d <= 512:
            return samples[i], samples[j], d
        return None

    idx_iter = combinations(range(len(seqs)), 2)
    total_pairs = len(seqs) * (len(seqs) - 1) // 2

    pairs = []
    with Pool(processes=48) as pool:
        it = pool.imap_unordered(pair_hd, idx_iter, chunksize=64)
        for res in tqdm(it, total=total_pairs, desc="Matching"):
            if res is not None:
                pairs.append(res)

    # dedupe symmetric pairs, keep smallest distance on conflicts (pairs already have d <= 512)
    clinvar = []
    mut = []
    for a, b, d in pairs:
        if d == 1:
            if df.loc[df["ID"] == a, "seq"].values[0][511] != df.loc[df["ID"] == b, "seq"].values[0][511]:
                clinvar.append((a, b, d))
            else:
                mut.append((a, b, d))
        elif d in (2, 4, 8, 16, 32, 64, 128, 256, 512):
            mut.append((a, b, d))

    pd.DataFrame(clinvar, columns=["sample_a", "sample_b", "distance"]).sort_values("sample_a").to_csv("match_clinvar.csv", index=False)
    pd.DataFrame(mut, columns=["sample_a", "sample_b", "distance"]).sort_values("sample_a").to_csv("match_mut.csv", index=False)