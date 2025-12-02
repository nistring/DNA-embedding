import pandas as pd

# Read the CSV file
df = pd.read_csv('clinvar_compact_removed.csv')

# Group by 'ref_seq'
grouped = df.groupby('ref_seq')

# Lists to hold the new rows
paired_rows = []
unpaired_rows = []

for name, group in grouped:
    # Separate positives (label 1 or 2) and negatives (label -1)
    positives = group[group['label'] >= 1]
    negatives = group[group['label'] == -1]
    
    if len(positives) > 0 and len(negatives) > 0:
        # For each negative, pair with each positive
        for neg in negatives.itertuples():
            for pos in positives.itertuples():
                new_row = {
                    'ref_seq': name,
                    'mut_idx': neg.mut_idx,  # Assuming mut_idx is the same for the group
                    'alt_pos': pos.alt,
                    'alt_neg': neg.alt
                }
                paired_rows.append(new_row)
    else:
        # Unpaired: add all rows from this group
        unpaired_rows.extend(group[group['label'] <= 1].to_dict('records'))

# Create dataframes
paired_df = pd.DataFrame(paired_rows)
unpaired_df = pd.DataFrame(unpaired_rows)

# Save to CSVs
paired_df.to_csv('paired_sequences.csv', index=False)
unpaired_df.to_csv('unpaired_sequences.csv', index=False)

# Optional: print the dataframes
print("Paired sequences:")
print(paired_df)
print("\nUnpaired sequences:")
print(unpaired_df)