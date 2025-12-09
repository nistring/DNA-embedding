import pandas as pd

def find_matching_sequences(test_file, clinvar_file):
    """
    Removes sequences from ClinVar that appear in test.csv.
    Writes clinvar_compact_removed.csv with matching ref_seq rows removed.
    """
    # Read CSV files
    test_df = pd.read_csv(test_file)
    clinvar_df = pd.read_csv(clinvar_file)
    
    # Build set of test sequences (first 511 bases only, excluding 512th position SNV)
    test_seqs = set(seq[:511] + seq[512:] for seq in test_df['seq'].values)
    
    # Remove matched sequences from ClinVar
    clinvar_removed_df = clinvar_df[~clinvar_df.apply(lambda row: (row['ref_seq'][:511] + row['ref_seq'][512:]) in test_seqs, axis=1)]
    clinvar_removed_df.to_csv('clinvar_compact_removed.csv', index=False)
    
    print(f"\nClinVar rows removed: {len(clinvar_df) - len(clinvar_removed_df)}. Saved to clinvar_compact_removed.csv")

if __name__ == "__main__":
    find_matching_sequences(
        'test.csv',
        'clinvar_compact.csv',
    )