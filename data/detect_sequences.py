#!/usr/bin/env python3
"""
Detects if clinvar_compact_removed.csv contains any sequences from test.csv
"""

import pandas as pd
from pathlib import Path

def detect_sequences(clinvar_file, test_file):
    """
    Check if any sequences from test.csv appear in clinvar_compact_removed.csv
    
    Args:
        clinvar_file: Path to clinvar_compact_removed.csv
        test_file: Path to test.csv
    
    Returns:
        Dictionary with results
    """
    
    # Read CSV files
    print(f"Reading {clinvar_file}...")
    clinvar_df = pd.read_csv(clinvar_file)
    
    print(f"Reading {test_file}...")
    test_df = pd.read_csv(test_file)
    
    # Extract sequences
    clinvar_seqs = set(clinvar_df['ref_seq'].values)
    test_seqs = test_df['seq'].values
    
    print(f"\nTotal sequences in clinvar_compact_removed.csv: {len(clinvar_seqs)}")
    print(f"Total sequences in test.csv: {len(test_seqs)}")
    
    # Find matches
    matches = []
    for idx, test_seq in enumerate(test_seqs):
        test_id = test_df.iloc[idx]['ID']
        if test_seq in clinvar_seqs:
            matches.append({
                'test_id': test_id,
                'sequence': test_seq[:100] + '...' if len(test_seq) > 100 else test_seq
            })
            print(f"\n✓ Match found: {test_id}")
    
    # Results
    print(f"\n{'='*60}")
    print(f"RESULTS:")
    print(f"{'='*60}")
    print(f"Total matches found: {len(matches)}")
    
    if matches:
        print(f"\nSequences from test.csv found in clinvar_compact_removed.csv:")
        for match in matches:
            print(f"  - {match['test_id']}: {match['sequence']}")
    else:
        print("\nNo sequences from test.csv were found in clinvar_compact_removed.csv")
    
    return {
        'total_matches': len(matches),
        'matches': matches
    }

if __name__ == "__main__":
    # Update paths as needed
    data_dir = Path("/home/work/.nistring/embedding/data")
    clinvar_file = data_dir / "clinvar_compact_removed.csv"
    test_file = data_dir / "test.csv"
    
    if not clinvar_file.exists():
        print(f"Error: {clinvar_file} not found")
        exit(1)
    
    if not test_file.exists():
        print(f"Error: {test_file} not found")
        exit(1)
    
    results = detect_sequences(clinvar_file, test_file)
