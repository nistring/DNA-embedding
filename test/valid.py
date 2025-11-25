import pandas as pd
from collections import Counter
import math

# Load the data
test_df = pd.read_csv('test.csv')
unmatched_df = pd.read_csv('unmatched.csv')

# Create a dictionary for quick lookup
test_sequences = dict(zip(test_df['ID'], test_df['seq']))

def kmers(seq, k=3):
    """Calculate k-mer frequencies"""
    return Counter(seq[i:i+k] for i in range(len(seq)-k+1))

def orf_lengths(s):
    """Calculate ORF lengths in 3 frames"""
    stops = {"TAA", "TAG", "TGA"}
    lens = []
    for frame in range(3):
        l = 0
        for i in range(frame, len(s)-2, 3):
            codon = s[i:i+3]
            if codon in stops:
                lens.append(l)
                l = 0
            else:
                l += 1
        lens.append(l)
    return lens

def reverse_complement(seq):
    """Get reverse complement"""
    return seq[::-1].translate(str.maketrans("ACGT", "TGCA"))

def analyze_sequence(seq):
    """Analyze sequence for natural vs random characteristics"""
    seq = seq.upper()
    
    # Base composition
    counts = Counter(seq)
    gc = counts.get('G', 0) + counts.get('C', 0)
    gc_content = gc / len(seq) if len(seq) > 0 else 0
    
    # Dinucleotide and trinucleotide frequencies
    di = kmers(seq, 2)
    tri = kmers(seq, 3)
    
    # Calculate entropy (lower = more repetitive/suspicious)
    total_tri = sum(tri.values())
    tri_entropy = -sum((c/total_tri) * math.log2(c/total_tri) for c in tri.values() if c > 0)
    
    # ORF analysis
    orf_5p = orf_lengths(seq)
    orf_3p = orf_lengths(reverse_complement(seq))
    
    # Homopolymer runs (long runs suspicious)
    max_run = max([len(s) for s in ''.join(['1' if seq[i]==seq[i-1] else '0' for i in range(1, len(seq))]).split('0')] + [0])
    
    return {
        'length': len(seq),
        'gc_content': gc_content,
        'base_counts': counts,
        'max_orf_fwd': max(orf_5p) if orf_5p else 0,
        'max_orf_rev': max(orf_3p) if orf_3p else 0,
        'top_trimers': tri.most_common(5),
        'tri_entropy': tri_entropy,
        'max_homopolymer': max_run,
        'num_trimers': len(tri)
    }

print("Analyzing sequences for natural vs random characteristics:")
print("=" * 80)

for idx, row in unmatched_df.head(10).iterrows():
    sample_id = row['sample']
    if sample_id in test_sequences:
        seq = test_sequences[sample_id]
        analysis = analyze_sequence(seq)
        
        print(f"\n{sample_id}:")
        print(f"  Length: {analysis['length']}")
        print(f"  GC%: {analysis['gc_content']*100:.2f}")
        print(f"  Base counts: {dict(analysis['base_counts'])}")
        print(f"  Max ORF (fwd): {analysis['max_orf_fwd']}")
        print(f"  Max ORF (rev): {analysis['max_orf_rev']}")
        print(f"  Max homopolymer run: {analysis['max_homopolymer']}")
        print(f"  Trinucleotide diversity: {analysis['num_trimers']}")
        print(f"  Top 5 trinucleotides: {analysis['top_trimers']}")
        
        # Heuristics for detecting random sequences
        suspicious = []
        if analysis['max_orf_fwd'] < 10 and analysis['max_orf_rev'] < 10:
            suspicious.append("Very short ORFs (random-like)")
        if analysis['max_homopolymer'] > 20:
            suspicious.append("Unusual homopolymer runs")
        if abs(analysis['gc_content'] - 0.5) < 0.02:
            suspicious.append("GC% too close to 50% (random-like)")
        
        if suspicious:
            print(f"  ⚠️  Suspicious features: {', '.join(suspicious)}")
        else:
            print(f"  ✓ Appears natural")
    else:
        print(f"{sample_id}: NOT FOUND in test.csv")

print("\n" + "=" * 80)
print("SUMMARY:")
print(f"Total unmatched samples: {len(unmatched_df)}")
print(f"Samples found in test.csv: {sum(unmatched_df['sample'].isin(test_sequences))}")