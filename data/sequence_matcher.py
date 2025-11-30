import pandas as pd
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

def find_matching_sequences(test_file, bac_refs_file, clinvar_file, enable_progress=True):
    """
    Scan test.csv and find matching sequences in bac_refs.csv or clinvar_compact.csv
    ClinVar sequences are taken from its 'ref_seq' column.
    Also writes clinvar_compact_removed.csv with matching ref_seq rows removed.
    enable_progress: if True and tqdm installed, show a progress bar while scanning.
    """
    # Read the CSV files
    test_df = pd.read_csv(test_file)
    bac_refs_df = pd.read_csv(bac_refs_file)
    clinvar_df = pd.read_csv(clinvar_file)
    
    # Get sequences from reference files as sets for faster lookup
    bac_refs_sequences = set(bac_refs_df['seq'].dropna())
    clinvar_sequences = set(clinvar_df['ref_seq'].dropna())  # changed from clinvar_df['seq']
    
    # Track variation types in clinvar if label column exists
    clinvar_label_counts = {}
    if 'label' in clinvar_df.columns:
        clinvar_label_counts = clinvar_df['label'].value_counts().to_dict()
    
    # Find matches
    results = []
    iterator = test_df.iterrows()
    if enable_progress and tqdm:
        iterator = tqdm(iterator, total=len(test_df), desc="Scanning sequences")
    for idx, row in iterator:
        test_seq = row['seq']
        
        if pd.isna(test_seq):
            continue
            
        match_info = {
            'test_index': idx,
            'sequence': test_seq,
            'found_in_bac_refs': test_seq in bac_refs_sequences,
            'found_in_clinvar': test_seq in clinvar_sequences
        }
        results.append(match_info)
    
    # Create results dataframe
    results_df = pd.DataFrame(results)
    
    # Print summary
    print(f"Total sequences scanned: {len(results_df)}")
    print(f"Matches in bac_refs.csv: {results_df['found_in_bac_refs'].sum()}")
    print(f"Matches in clinvar_compact.csv: {results_df['found_in_clinvar'].sum()}")
    print(f"Matches in both: {(results_df['found_in_bac_refs'] & results_df['found_in_clinvar']).sum()}")
    
    # Print ClinVar variation type counts
    if clinvar_label_counts:
        print(f"\nClinVar variation type counts:")
        print(f"  Pathogenic/Likely Pathogenic (label=1): {clinvar_label_counts.get(1, 0)}")
        print(f"  Benign (label=0): {clinvar_label_counts.get(0, 0)}")
                  
    # Prepare and write ClinVar with matched ref_seq removed
    matched_clinvar_sequences = {r['sequence'] for r in results if r['found_in_clinvar']}
    clinvar_removed_df = clinvar_df[~clinvar_df['ref_seq'].isin(matched_clinvar_sequences)]
    clinvar_removed_df.to_csv('clinvar_compact_removed.csv', index=False)
    rows_removed = len(clinvar_df) - len(clinvar_removed_df)
    distinct_clinvar_sequences_removed = clinvar_df[clinvar_df['ref_seq'].isin(matched_clinvar_sequences)]['ref_seq'].nunique()
    duplicate_removed_rows = rows_removed - distinct_clinvar_sequences_removed
    
    # Print ClinVar rows removed by category
    removed_by_category = clinvar_df[clinvar_df['ref_seq'].isin(matched_clinvar_sequences)]['label'].value_counts().sort_index()
    print(f"\nClinVar rows removed by category:")
    if not removed_by_category.empty:
        print(f"  Pathogenic/Likely Pathogenic (label=1): {removed_by_category.get(1, 0)}")
        print(f"  Benign (label=0): {removed_by_category.get(0, 0)}")
    
    print(f"ClinVar rows removed: {rows_removed}. Saved to clinvar_compact_removed.csv")
    print(f"Matched test sequences (rows in test with ClinVar hits): {results_df['found_in_clinvar'].sum()}")
    print(f"Unique matched test sequences set size: {len(matched_clinvar_sequences)}")
    print(f"Distinct ClinVar ref_seq values removed: {distinct_clinvar_sequences_removed}")
    print(f"Duplicate ClinVar rows among removed (inflates removal count): {duplicate_removed_rows}")
    
    # Save results
    results_df.to_csv('sequence_matches.csv', index=False)
    print("\nResults saved to sequence_matches.csv")
    
    return results_df

if __name__ == "__main__":
    # Run the matching
    results = find_matching_sequences(
        'test.csv',
        'bac_refs.csv',
        'clinvar_compact.csv',
        enable_progress=True
    )
    
    # Display matches
    matches = results[results['found_in_bac_refs'] | results['found_in_clinvar']]
    if not matches.empty:
        print("\nMatching sequences:")
        print(matches)
    else:
        print("\nNo matching sequences found.")