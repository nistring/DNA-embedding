import pandas as pd
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

def find_matching_sequences(test_file, clinvar_file, match_file, enable_progress=True):
    """
    Creates matched_pairs_labeled.csv with sample_a, sample_b, and label columns from ClinVar matches.
    Also writes clinvar_compact_removed.csv with matching ref_seq rows removed.
    """
    # Read CSV files
    test_df = pd.read_csv(test_file)
    clinvar_df = pd.read_csv(clinvar_file)
    match_df = pd.read_csv(match_file)
    
    # Map (ref_seq, alt) to label in clinvar
    clinvar_to_label = {}
    iterator = tqdm(clinvar_df.iterrows(), total=len(clinvar_df), desc="Building ClinVar mapping") if enable_progress and tqdm else clinvar_df.iterrows()
    for _, row in iterator:
        ref_seq = row['ref_seq']
        alt = row['alt']
        if pd.notna(ref_seq) and pd.notna(alt):
            clinvar_to_label[(ref_seq, alt)] = row['label']
    
    # Map sample_id to (sequence, 512th position SNV)
    sample_to_seq_and_alt = {}
    iterator = tqdm(test_df.iterrows(), total=len(test_df), desc="Building test sample mapping") if enable_progress and tqdm else test_df.iterrows()
    for _, row in iterator:
        seq = row['seq']
        if pd.notna(seq) and len(seq) >= 512:
            alt_char = seq[511]  # 512th position (0-indexed)
            sample_to_seq_and_alt[row['ID']] = (seq, alt_char)
     
    # Create paired results from match_df and build removal set
    paired_results = []
    matched_seqs_and_alts = set()
    iterator = match_df.iterrows() if not enable_progress or not tqdm else tqdm(match_df.iterrows(), total=len(match_df), desc="Processing pairs")
    
    for _, row in iterator:
        sample_a, sample_b = row['sample_a'], row['sample_b']
        label = None
        seq_a, alt_a = sample_to_seq_and_alt[sample_a]
        seq_b, alt_b = sample_to_seq_and_alt[sample_b]

        label = clinvar_to_label.get((seq_b, alt_a))
        
        if label is None:
            label = clinvar_to_label.get((seq_a, alt_b))
        
        if label is not None:
            if sample_a > sample_b:
                sample_a, sample_b = sample_b, sample_a
            paired_results.append({'sample_a': sample_a, 'sample_b': sample_b, 'label': label})
            
            # Add to removal set
            matched_seqs_and_alts.add((seq_a, alt_b))
            matched_seqs_and_alts.add((seq_b, alt_a))
    
    paired_df = pd.DataFrame(paired_results)
    
    # Print summary
    print(f"Total test samples: {len(test_df)}")
    print(f"Paired samples with labels: {len(paired_df)}")
    
    if not paired_df.empty:
        label_counts = paired_df['label'].value_counts().sort_index()
        print(f"\nPaired samples by label:")
        print(f"  Pathogenic (label=-1): {label_counts.get(-1, 0)}")
        print(f"  Benign (label=1): {label_counts.get(1, 0)}")
    
    # Remove matched sequences from ClinVar
    clinvar_removed_df = clinvar_df[~clinvar_df.apply(lambda row: (row['ref_seq'], row['alt']) in matched_seqs_and_alts, axis=1)]
    clinvar_removed_df.to_csv('clinvar_compact_removed.csv', index=False)
    
    print(f"\nClinVar rows removed: {len(clinvar_df) - len(clinvar_removed_df)}. Saved to clinvar_compact_removed.csv")
    
    # Save paired results
    paired_df.to_csv('matched_pairs_labeled.csv', index=False)
    print("Paired results saved to matched_pairs_labeled.csv")
    
    return paired_df

if __name__ == "__main__":
    results = find_matching_sequences(
        'test.csv',
        'clinvar_compact.csv',
        '../test/results/match_clinvar.csv',
        enable_progress=True
    )
    
    if not results.empty:
        print(f"\nSample of paired results (first 10):")
        print(results.head(10))
    else:
        print("\nNo matching pairs found.")