# DNA Embedding: ClinVar Contrastive Training

This repository prepares genomic datasets and trains a contrastive DNA embedding model geared toward variant pathogenicity and mutation robustness. It is tailored for the Dacon DNA sequence learning challenge (see competition description: https://dacon.io/competitions/official/236630/overview/description).

The pipeline:
- Build a compact ClinVar SNV dataset centered in 1024bp windows.
- Exclude any sequences that leak into the provided `test.csv` via exact window match at the SNV position.
- Optionally compute Hamming-distance pairs among test sequences to derive labels.
- Train a GPN-based model with contrastive objectives using distributed `torchrun`.

## Setup
- OS: Linux; Shell: `bash`.
- Model backend: Hugging Face Transformers + PyTorch.

### Create Conda Environment
```bash
conda create -n dna-embed python=3.12 -y
conda activate dna-embed
```

### Install Dependencies
Install packages used in this repo:
```bash
pip install -r requirements.txt
```

## Dataset Construction

### 1) Download References
Download hg38 reference FASTA (indexed later) and ClinVar VCF (GRCh38):
```bash
mkdir -p data
cd data
wget https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz
wget https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/clinvar.vcf.gz
gunzip -k hg38.fa.gz
gunzip -k clinvar.vcf.gz
```

Index the FASTA for fast region fetches:
```bash
samtools faidx hg38.fa
```

### 2) Generate Compact ClinVar CSV
`prepare_clinvar_dataset.py` extracts 1024bp windows centered on SNVs, placing the reference base at index 511 (0-based) and records the alternate base and label. Only Benign and Pathogenic (including Likely Pathogenic) entries are kept.

```bash
cd /home/work/.nistring/embedding/data
python prepare_clinvar_dataset.py --vcf clinvar.vcf --fasta hg38.fa --out clinvar_compact.csv --window 1024
```

Outputs:
- `clinvar_compact.csv` with columns: `ref_seq, mut_idx, alt, label`
	- `mut_idx` is typically `511` for window 1024.
	- `label` is `1` for Benign and `-1` for Pathogenic/Likely Pathogenic.

### 3) Prevent Test Leakage and Generate Test Labels
`sequence_matcher.py` performs two tasks:
- Removes ClinVar rows whose `(ref_seq, alt)` match pairs that would leak into `test.csv` windows.
- Produces labels for `test.csv` pairs using precomputed Hamming-distance neighbor pairs.

Run it after placing `test.csv` and `clinvar_compact.csv` in `data/` and precomputing matches into `test/results/match_clinvar.csv`:
```bash
cd /home/work/.nistring/embedding/data
python sequence_matcher.py
```

Outputs:
- `clinvar_compact_removed.csv`: filtered ClinVar without matched leakage entries.
- `matched_pairs_labeled.csv`: pairs from `test.csv` with labels inferred via ClinVar mapping on the SNV position.

## Hamming Distance Pair Computation

`test/match.py` computes Hamming distances among all unordered pairs in `data/test.csv` and emits two CSVs:
- `match_clinvar.csv`: pairs at distance 1 with a differing base at position 512 (1-based), used to infer ClinVar labels.
- `match_mut.csv`: generic mutation pairs.

It uses a compiled `_hdist` module for speed. To build and run on your machine:
```bash
cd /home/work/.nistring/embedding/test
python setup.py build_ext --inplace
```

### Run Pair Matching
```bash
cd /home/work/.nistring/embedding/test
python match.py
```

## Training

Training is orchestrated by `script/train.sh` using `torchrun` with multi-GPU DDP.

```bash
cd /home/work/.nistring/embedding
bash script/train.sh
```

Logs are written to `output/<RUN_NAME>/training.log`, and checkpoints under `output/<RUN_NAME>/joint`.

## Evaluation Metrics

Evaluation is computed in `eval_metrics.py` after embeddings are generated:

- **`cd` (mean cosine distance)**: Average of cosine distance $\mathrm{cd} = \frac{1-\cos}{2}$ over all ClinVar-matched benign and pathogenic pairs found in `data/matched_pairs_labeled.csv`. Lower values indicate closer ref/alt pairs overall.
- **`cdd` (pathogenic minus benign distance)**: Difference between mean pathogenic and mean benign distances. Higher `cdd` means pathogenic pairs are farther than benign pairs, which is desired.
- **`pcc` (Pearson correlation with mutations)**: Pearson correlation between cosine distance and mutation count (from `test/results/match_mut.csv`). Higher correlation indicates that embeddings reflect increasing dissimilarity with more mutations.

## Inference Using `script/generate.sh`

The `generate.sh` script provides a ready command to run embedding generation against a chosen checkpoint directory and input CSV.
```bash
cd /home/work/.nistring/embedding
bash script/generate.sh
```

## Design: Model, Loss, and Datasets

### How the model separates benign vs pathogenic SNVs
The base GPN encoder is wrapped with a lightweight projection head that explicitly focuses on the single-nucleotide variant (SNV) position while preserving global context. Together with the datasets and losses below, this yields embeddings where benign SNVs are close to their reference and pathogenic SNVs are farther away.

- Projection head (`model.py` → `ProjectionHead`):
	- Multi-Head Attention over pooled hidden states to capture global sequence context.
	- A local convolutional feature around the SNV index (default `snv_pos=511`) to emphasize the mutation effect on neighboring codons.
	- Explicit SNV token feature (the embedding at the SNV index) concatenated with global pooled features.
	- A final dense layer maps the concatenated features to the embedding space (default 2048-D).

- Training objective (`model.py` → `ContrastiveTrainer`), connected to the datasets in the next subsection:
	- For ClinVar ref/alt pairs (batch_type == 0; see `ClinVarRefAltDataset`), optimize cosine similarity such that:
		- Benign pairs (label = +1) are encouraged to be close to their reference (high cosine similarity; margin threshold applies).
		- Pathogenic pairs (label = -1) are pushed away from the reference (low cosine similarity).
	- For mutation severity batches (batch_type == 1; see `ContrastiveMutateDataset`), regress the cosine distance `cd = (1 - cos) / 2` to the normalized mutation count `k/512`.

These SNV-focused features (local convolution + exact SNV token) strengthen sensitivity at the mutation locus while pooled attention preserves global sequence context.

### Datasets (`dataset.py`)
- `ClinVarRefAltDataset`: yields (ref, alt) pairs for SNVs centered at index 511 with labels `±1`. Drives contrastive discrimination between benign and pathogenic changes.
- `ContrastiveMutateDataset`: samples random 1024bp windows from `hg38.fa` and applies `k` random point mutations (`1 ≤ k ≤ 512`). Target is `k/512` (normalized), enabling regression/ranking on mutation severity that correlates with `pcc`.
- `BalancedAlternatingDataset`: interleaves batches from multiple datasets in a round-robin fashion for balanced multi-task training; reshuffles each epoch.
- `ContrastiveDataCollator`: keeps pair structure by stacking two inputs per sample and flattens into a batch suitable for contrastive objectives.

### Loss
- Combine a cosine-margin contrastive loss (controlled by `--cos_loss_margin`, e.g., `0.9`) for ref/alt embeddings and a regression loss on mutation level (`k/512`) from `ContrastiveMutateDataset`.
- Gradient clipping and cosine LR scheduling are configured in `train.sh`.