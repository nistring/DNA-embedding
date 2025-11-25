import pandas as pd
from multiprocessing import Pool

SEQ = {}

def hamming_pair(pair):
    a, b = pair
    sa, sb = SEQ.get(a), SEQ.get(b)
    if sa is None or sb is None:
        return (a, b, None)
    m = min(len(sa), len(sb))
    mism = sum(c1 != c2 for c1, c2 in zip(sa[:m], sb[:m])) + abs(len(sa) - len(sb))
    return (a, b, mism)

if __name__ == "__main__":
    seq_df = pd.read_csv("test.csv")
    name_col = "ID" if "ID" in seq_df.columns else seq_df.columns[0]
    seq_col = "seq" if "seq" in seq_df.columns else seq_df.columns[1]
    SEQ = dict(zip(seq_df[name_col], seq_df[seq_col]))

    pairs_df = pd.read_csv("match_pairs.csv", usecols=["sample_a", "sample_b"])
    pairs = list(zip(pairs_df["sample_a"], pairs_df["sample_b"]))

    with Pool(processes=48) as pool:
        results = pool.map(hamming_pair, pairs)

    pd.DataFrame(results, columns=["sample_a", "sample_b", "mismatches"]).to_csv(
        "pair_mismatches.csv", index=False
    )