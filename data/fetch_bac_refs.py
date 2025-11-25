import argparse
import csv
import math
from typing import Iterable, List, Tuple

import requests
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


def esearch_ids(query: str, retmax: int, email: str) -> List[str]:
    params = {
        "db": "nucleotide",
        "term": query,
        "retmode": "json",
        "retmax": str(retmax),
        "tool": "dnabert2-finetune",
        "email": email,
    }
    r = requests.get(f"{EUTILS_BASE}/esearch.fcgi", params=params, timeout=60)
    r.raise_for_status()
    js = r.json()
    return js.get("esearchresult", {}).get("idlist", [])


def chunked(xs: List[str], n: int) -> Iterable[List[str]]:
    for i in range(0, len(xs), n):
        yield xs[i:i + n]


def efetch_fasta(ids: List[str], email: str) -> str:
    params = {
        "db": "nucleotide",
        "id": ",".join(ids),
        "rettype": "fasta",
        "retmode": "text",
        "tool": "dnabert2-finetune",
        "email": email,
    }
    r = requests.get(f"{EUTILS_BASE}/efetch.fcgi", params=params, timeout=120)
    r.raise_for_status()
    return r.text


def parse_fasta(text: str) -> List[Tuple[str, str]]:
    records = []
    header = None
    seq_lines: List[str] = []
    for line in text.splitlines():
        if not line:
            continue
        if line.startswith(">"):
            if header is not None and seq_lines:
                records.append((header, "".join(seq_lines).upper()))
            header = line[1:].strip()
            seq_lines = []
        else:
            seq_lines.append(line.strip())
    if header is not None and seq_lines:
        records.append((header, "".join(seq_lines).upper()))
    return records


def clean_seq(s: str) -> str:
    # Map any non-ACGT to A
    allowed = set("ACGT")
    s = s.upper()
    return "".join(c if c in allowed else "A" for c in s)


def windows_from_sequence(seq: str, seq_len: int) -> List[str]:
    n = len(seq)
    if n < seq_len:
        return []
    # Extract all non-overlapping windows
    out = []
    for i in range(0, n - seq_len + 1, seq_len):
        out.append(seq[i:i + seq_len])
    return out


def main():
    p = argparse.ArgumentParser("Fetch BAC clone reference sequences and produce 1024bp windows CSV for dataset2")
    p.add_argument("--out_csv", type=str, required=True, help="Output CSV path with a 'seq' column")
    p.add_argument("--query", type=str, default='("Homo sapiens"[Organism] OR Human[Organism]) AND "BAC"[Title] AND "complete sequence"[Title]')
    p.add_argument("--extra_terms", type=str, nargs="*", default=["Homo sapiens BAC clone RP13-868N24 from 2, complete sequence"],
                   help="Additional phrases to OR into the query")
    p.add_argument("--email", type=str, default="user@example.com", help="Contact email for NCBI E-utilities")
    p.add_argument("--retmax", type=int, default=3000000, help="Max records to fetch from NCBI")
    p.add_argument("--seq_len", type=int, default=1024)
    p.add_argument("--workers", type=int, default=48, help="Parallel workers for NCBI efetch")
    args = p.parse_args()

    # Build final query
    q = args.query
    if args.extra_terms:
        ors = " OR ".join([f'"{t}"' for t in args.extra_terms])
        q = f"({q}) OR ({ors})"

    print(f"Searching NCBI nucleotide for: {q}")
    ids = esearch_ids(q, retmax=args.retmax, email=args.email)
    print(f"Found {len(ids)} IDs")
    if not ids:
        print("No results; exiting.")
        return

    rows = []
    batches = list(chunked(ids, 100))

    def process_batch(batch_ids: List[str]) -> List[dict]:
        try:
            text = efetch_fasta(batch_ids, email=args.email)
            recs = parse_fasta(text)
            out = []
            for header, seq in recs:
                seq = clean_seq(seq)
                slices = windows_from_sequence(seq, seq_len=args.seq_len)
                for s in slices:
                    out.append({"seq": s, "source": header})
            return out
        except Exception:
            return []

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(process_batch, b) for b in batches]
        for fut in tqdm(as_completed(futures), total=len(batches), desc="Downloading FASTA (parallel)"):
            rows.extend(fut.result())

    if not rows:
        print("No sequences of sufficient length; exiting.")
        return

    df = pd.DataFrame(rows)
    df.to_csv(args.out_csv, index=False)
    print(f"Wrote {len(df)} sequences to {args.out_csv}")


if __name__ == "__main__":
    main()
