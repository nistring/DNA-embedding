import argparse
import csv
from bisect import bisect_right
from typing import Dict, List, Tuple
import pysam

DNA = set("ACGT")

def load_gene_intervals(gtf_path: str) -> Dict[str, List[Tuple[int, int]]]:
    chrom2iv = {}
    with open(gtf_path, "r") as f:
        for line in f:
            if not line or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 5 or parts[2] != "gene":
                continue
            chrom = parts[0]
            try:
                start = int(parts[3])  # 1-based inclusive
                end = int(parts[4])    # 1-based inclusive
            except ValueError:
                continue
            if start > end:
                start, end = end, start
            chrom2iv.setdefault(chrom, []).append((start, end))
    # merge overlaps to speed membership test
    for chrom, ivs in list(chrom2iv.items()):
        ivs.sort()
        merged = []
        for s, e in ivs:
            if not merged or s > merged[-1][1] + 1:
                merged.append([s, e])
            else:
                merged[-1][1] = max(merged[-1][1], e)
        chrom2iv[chrom] = [(s, e) for s, e in merged]
    return chrom2iv

def pos_in_gene(chrom2iv: Dict[str, List[Tuple[int, int]]], chrom: str, pos1: int) -> bool:
    ivs = chrom2iv.get(chrom)
    if not ivs:
        return False
    starts = [s for s, _ in ivs]
    i = bisect_right(starts, pos1) - 1
    if i >= 0:
        s, e = ivs[i]
        return s <= pos1 <= e
    return False

def parse_clnsig(info_value) -> str:
    if info_value is None:
        return ""
    else:
        s = str(info_value)
    return s

def main():
    ap = argparse.ArgumentParser(description="Prepare ClinVar SNV dataset (compact) for contrastive training.")
    ap.add_argument("--vcf", default="clinvar.vcf", help="Path to clinvar.vcf")
    ap.add_argument("--fasta", default="hg38.fa", help="Path to hg38.fa (indexed)")
    ap.add_argument("--gtf", default="gencode.v48.chr_patch_hapl_scaff.basic.annotation.gtf", help="Path to gencode.v48.gtf")
    ap.add_argument("--out", default="clinvar_compact.csv", help="Output CSV path")
    ap.add_argument("--window", type=int, default=1024, help="Window length (default: 1024)")
    args = ap.parse_args()

    if args.window % 2 != 0:
        raise ValueError("Window length must be even; 1024 recommended.")
    half = args.window // 2
    # place SNV at 512th nucleotide (1-based) => 0-based index 511 in a 1024 window
    mut_idx = half - 1  # 511 for 1024

    gene_iv = load_gene_intervals(args.gtf)
    fa = pysam.FastaFile(args.fasta)
    vcf = pysam.VariantFile(args.vcf)

    out_f = open(args.out, "w", newline="")
    writer = csv.writer(out_f)
    writer.writerow(["ref_seq", "mut_idx", "alt", "label"])

    kept = 0
    seen = 0
    for rec in vcf:
        chrom = 'chr' + str(rec.chrom)
        pos1 = rec.pos  # 1-based
        ref = rec.ref.upper()
        alts = rec.alts or ()
        cl = parse_clnsig(rec.info.get("CLNSIG"))
        # if not pos_in_gene(gene_iv, chrom, pos1):
        #     continue
        for alt in alts:
            alt = (alt or "").upper()
            if alt == ref:
                print(f"Warning: alt == ref for {chrom}:{pos1} {ref}>{alt}")
                continue
            # Build window [pos1-511, pos1+511] inclusive
            start1 = pos1 - (half - 1)
            end1 = pos1 + half
            # Boundaries check
            if start1 < 1:
                print(f"Warning: start {start1} < 1 for {chrom}:{pos1}")
                continue
            # fetch uses 0-based, half-open [start0, end0)
            start0 = start1 - 1
            try:
                seq = fa.fetch(chrom, start0, end1)  # length should be window
            except:
                print(f"Warning: could not fetch {chrom}:{start1}-{end1}")
                continue
            if len(seq) != args.window:
                print(f"Warning: fetched sequence length {len(seq)} != window {args.window} for {chrom}:{start1}-{end1}")
                continue
            seq = seq.upper()
            # sanity: reference base matches fasta at center
            if seq[mut_idx] != ref:
                continue
            if "Pathogenic" in cl:
                label = 1
            elif "Likely_pathogenic" in cl:
                label = 1
            elif "Benign" in cl:
                label = 0
            else:
                # Remove: Uncertain, Likely Benign, Conflicting, Not Provided, Other
                continue
                
            writer.writerow([seq, mut_idx, alt, label])
            kept += 1
            seen += 1
        seen += 1

    out_f.close()
    fa.close()
    # print minimal stats
    print(f"Wrote {kept} SNVs to {args.out}")


if __name__ == "__main__":
    main()
