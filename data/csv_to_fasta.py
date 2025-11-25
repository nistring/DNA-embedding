import csv, math

with open('test.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    rows = list(reader)

total = len(rows)
if total == 0:
    raise SystemExit("No rows found in test.csv")

chunks = 1
chunk_size = math.ceil(total / chunks)

for i in range(chunks):
    start = i * chunk_size
    end = start + chunk_size
    chunk = rows[start:end]
    if not chunk:
        break
    fname = f"fasta/test_{i+1:02d}.fasta"
    with open(fname, 'w') as fastafile:
        for row in chunk:
            fastafile.write(f">{row['ID']}\n{row['seq']}\n")

# Example BLAST (adjust db as needed):
# for f in test_*.fasta; do blastn -query "$f" -db nt -out "results_${f%.fasta}.txt" -num_threads $(nproc); done