import argparse
import csv
import pysam

DNA = set("ACGT")


def main():
    ap = argparse.ArgumentParser(description="Prepare ClinVar SNV dataset (compact) for contrastive training.")
    ap.add_argument("--vcf", default="clinvar.vcf", help="Path to clinvar.vcf")
    ap.add_argument("--fasta", default="hg38.fa", help="Path to hg38.fa (indexed)")
    ap.add_argument("--out", default="clinvar_compact.csv", help="Output CSV path")
    ap.add_argument("--window", type=int, default=1024, help="Window length (default: 1024)")
    args = ap.parse_args()

    if args.window % 2 != 0:
        raise ValueError("Window length must be even; 1024 recommended.")
    half = args.window // 2
    # place SNV at 512th nucleotide (1-based) => 0-based index 511 in a 1024 window
    mut_idx = half - 1  # 511 for 1024

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
        if not rec.info.get("CLNSIG", None):
            continue
        cl = rec.info["CLNSIG"][0]
        if rec.info.get("CLNVC", None) != "single_nucleotide_variant":
            continue
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
            if "Pathogenic" in cl or "Likely_pathogenic" in cl:
                label = -1
            elif "Benign" in cl:
                label = 1
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
