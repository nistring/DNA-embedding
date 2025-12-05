import pandas as pd
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

def find_matching_sequences(test_file, bac_refs_file, clinvar_file, match_file, enable_progress=True):
    """
    Creates matched_pairs_labeled.csv with sample_a, sample_b, and label columns from ClinVar matches.
    Also writes clinvar_compact_removed.csv with matching ref_seq rows removed.
    """
    # Read CSV files
    test_df = pd.read_csv(test_file)
    clinvar_df = pd.read_csv(clinvar_file)
    match_df = pd.read_csv(match_file)
    
    # Map ref_seq to label in clinvar
    clinvar_seq_to_label = dict(zip(clinvar_df['ref_seq'].dropna(), clinvar_df.loc[clinvar_df['ref_seq'].notna(), 'label']))
    
    # Map sample_id to sequence
    sample_to_seq = dict(zip(test_df['ID'], test_df['seq']))
    matched_samples = {sid for sid, seq in sample_to_seq.items() if seq in clinvar_seq_to_label}
    
    # Create paired results from match_df
    paired_results = []
    iterator = match_df.iterrows() if not enable_progress or not tqdm else tqdm(match_df.iterrows(), total=len(match_df), desc="Processing pairs")
    
    for _, row in iterator:
        sample_a, sample_b = row['sample_a'], row['sample_b']
        label = clinvar_seq_to_label.get(sample_to_seq.get(sample_a)) or clinvar_seq_to_label.get(sample_to_seq.get(sample_b))
        
        if label is not None:
            if sample_a > sample_b:
                sample_a, sample_b = sample_b, sample_a
            paired_results.append({'sample_a': sample_a, 'sample_b': sample_b, 'label': label})
    
    paired_df = pd.DataFrame(paired_results)
    
    # Print summary
    print(f"Total test samples: {len(test_df)}")
    print(f"Samples matching ClinVar: {len(matched_samples)}")
    print(f"Paired samples with labels: {len(paired_df)}")
    
    if not paired_df.empty:
        label_counts = paired_df['label'].value_counts().sort_index()
        print(f"\nPaired samples by label:")
        print(f"  Pathogenic (label=-1): {label_counts.get(-1, 0)}")
        print(f"  Benign (label=1): {label_counts.get(1, 0)}")
    
    # Remove matched sequences from ClinVar
    matched_seqs = {sample_to_seq[s] for s in matched_samples if sample_to_seq[s] in clinvar_seq_to_label}
    clinvar_removed_df = clinvar_df[~clinvar_df['ref_seq'].isin(matched_seqs)]
    clinvar_removed_df.to_csv('clinvar_compact_removed.csv', index=False)
    
    print(f"\nClinVar rows removed: {len(clinvar_df) - len(clinvar_removed_df)}. Saved to clinvar_compact_removed.csv")
    
    # Save paired results
    paired_df.to_csv('matched_pairs_labeled.csv', index=False)
    print("Paired results saved to matched_pairs_labeled.csv")
    
    return paired_df

if __name__ == "__main__":
    results = find_matching_sequences(
        'test.csv',
        'bac_refs.csv',
        'clinvar_compact.csv',
        '../test/results/match_clinvar.csv',
        enable_progress=True
    )
    
    if not results.empty:
        print(f"\nSample of paired results (first 10):")
        print(results.head(10))
    else:
        print("\nNo matching pairs found.")