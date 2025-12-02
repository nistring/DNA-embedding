"""Dataset classes for DNABERT-2 fine-tuning."""
import csv
import logging
import random
from typing import Dict, List
import torch
from torch.utils.data import Dataset
import transformers

logger = logging.getLogger(__name__)


class ClinVarRefAltDataset(Dataset):
    """ClinVar dataset for cd_loss - returns (ref, alt) pairs, balanced 1:1 pos/neg."""
    def __init__(self, path: str, tokenizer: transformers.PreTrainedTokenizer, sep: str = ",", label: int = 1):
        super().__init__()
        with open(path, "r") as f:
            samples = [{"ref": row["ref_seq"], "mut_idx": int(row["mut_idx"]), 
                       "alt": row["alt"].upper(), "label": int(row["label"])}
                      for row in csv.DictReader(f, delimiter=sep)]
        self.sample = [s for s in samples if s["label"] == label]
        self.tokenizer = tokenizer
        logger.info(f"Loaded {len(self.sample)} samples with label {label} ClinVar ref-alt pairs")

    def __len__(self):
        return len(self.sample)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        s = self.sample[idx]
        ref_seq, mi, alt, label = s["ref"], s["mut_idx"], s["alt"], s["label"]
        snv_seq = ref_seq[:mi] + alt + ref_seq[mi+1:]

        output = self.tokenizer([ref_seq, snv_seq], padding="max_length", 
                               max_length=self.tokenizer.model_max_length, 
                               truncation=True, return_tensors="pt")

        # We don't return attention_mask to save memory; model can run without it.
        return {"input_ids": output["input_ids"], \
            "labels": torch.tensor(label, dtype=torch.float32), "batch_type": torch.tensor(0, dtype=torch.long)}


class ClinVarTripletDataset(Dataset):
    """ClinVar dataset for triplet loss - returns (ref, pos, neg) triplets."""
    def __init__(self, path: str, tokenizer: transformers.PreTrainedTokenizer, sep: str = ","):
        super().__init__()
        with open(path, "r") as f:
            self.samples = [{"ref_seq": row["ref_seq"], "mut_idx": int(row["mut_idx"]), 
                           "alt_pos": row["alt_pos"].upper(), "alt_neg": row["alt_neg"].upper()}
                          for row in csv.DictReader(f, delimiter=sep)]
        self.tokenizer = tokenizer
        logger.info(f"Loaded {len(self.samples)} ClinVar triplets for triplet loss")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        s = self.samples[idx]
        ref_seq, mi, alt_pos, alt_neg = s["ref_seq"], s["mut_idx"], s["alt_pos"], s["alt_neg"]
        pos_seq = ref_seq[:mi] + alt_pos + ref_seq[mi+1:]
        neg_seq = ref_seq[:mi] + alt_neg + ref_seq[mi+1:]

        output = self.tokenizer([ref_seq, pos_seq, neg_seq], padding="max_length", 
                               max_length=self.tokenizer.model_max_length, 
                               truncation=True, return_tensors="pt")

        # Omit attention_mask to reduce batch memory footprint; keep labels and batch_type
        return {"input_ids": output["input_ids"], \
            "labels": torch.tensor([0], dtype=torch.float32), "batch_type": torch.tensor(1, dtype=torch.long)}


class ContrastiveMutateDataset(Dataset):
    """
    Contrastive mutation dataset for regression.
    Returns (ref, mutated) pairs with log2(mutation count) as target.
    The loss should maximize cosine distance proportional to log2(mutation count).
    """
    DNA = ["A", "C", "G", "T"]

    def __init__(self, path: str, tokenizer: transformers.PreTrainedTokenizer, seq_col: str = "seq", sep: str = ",", mut_levels=(1, 2, 8, 64, 128, 256, 512), use_reverse_complement: bool = False):
        super().__init__()
        with open(path, "r") as f:
            data = list(csv.reader(f, delimiter=sep))
        seq_idx = data[0].index(seq_col) if data else 0
        self.seqs = [row[seq_idx] for row in data[1:]]
        
        # Double dataset with reverse complement versions
        if use_reverse_complement:
            original_seqs = self.seqs.copy()
            for seq in original_seqs:
                self.seqs.append(self._reverse_complement(seq))
        
        self.mut_levels, self.tokenizer = tuple(mut_levels), tokenizer
        logger.info(f"Loaded {len(self.seqs)} mutation samples (reverse_complement={use_reverse_complement})")

    def __len__(self):
        return len(self.seqs)

    @staticmethod
    def _mutate(seq: str, k: int) -> str:
        seq_list = list(seq)
        for i in random.sample(range(len(seq)), min(k, len(seq))):
            seq_list[i] = random.choice([b for b in ContrastiveMutateDataset.DNA if b != seq_list[i]])
        return "".join(seq_list)

    @staticmethod
    def _reverse_complement(seq: str) -> str:
        """Compute reverse complement of DNA sequence."""
        complement = {"A": "T", "T": "A", "G": "C", "C": "G"}
        return "".join(complement.get(base, base) for base in reversed(seq))

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        ref, k = self.seqs[idx], random.choice(self.mut_levels)
        mutated = self._mutate(ref, k)
        
        output = self.tokenizer([ref, mutated], padding="max_length",
                               max_length=self.tokenizer.model_max_length, 
                               truncation=True, return_tensors="pt")
        # Only return input_ids, labels and batch_type to minimize memory usage
        return {"input_ids": output["input_ids"], \
            "labels": torch.log2(torch.tensor(k, dtype=torch.float32)), \
            "batch_type": torch.tensor(2, dtype=torch.long)}


class BalancedAlternatingDataset(Dataset):
    """Interleaves items from three datasets in round-robin fashion for balanced training.
    
    Cycles through datasets sequentially by batch: datasets repeat.
    Each dataset is shuffled independently for proper randomization.
    """
    def __init__(self, datasets: List[Dataset], batch_size: int = 32, seed: int = 42, shuffle: bool = True):
        self.datasets = datasets
        self.batch_size, self.seed, self.shuffle = batch_size, seed, shuffle
        
        rng = random.Random(seed)
        self.indices = [list(range(len(ds))) for ds in self.datasets]
        for idx_list in self.indices:
            rng.shuffle(idx_list)
        
        num_batches = max((len(ds) + batch_size - 1) // batch_size for ds in self.datasets) * len(self.datasets)
        self.num_batches, self.length = num_batches, num_batches * batch_size

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        batch_num, idx_in_batch = divmod(idx, self.batch_size)
        dataset_idx = batch_num % len(self.datasets)
        sample_idx = (batch_num // len(self.datasets)) * self.batch_size + idx_in_batch
        return self.datasets[dataset_idx][self.indices[dataset_idx][sample_idx % len(self.datasets[dataset_idx])]]
    
    def set_epoch(self, epoch: int):
        """Reshuffle data at each epoch for proper randomization in distributed training."""
        if not self.shuffle:
            return
        rng = random.Random(self.seed + epoch)
        self.indices = [list(range(len(ds))) for ds in self.datasets]
        for idx_list in self.indices:
            rng.shuffle(idx_list)


class ContrastiveDataCollator:
    """
    Custom data collator for contrastive learning that preserves pair structure.
    
    Converts (B, 2, seq_len) inputs to (B*2, seq_len) for model forward pass,
    while preserving metadata needed to reshape back in the loss function.
    """
    
    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Collate batch of samples, flattening the pair dimension."""
        collated = {}
        # Stack and flatten input_ids (B, 2, seq_len) -> (B*2, seq_len)
        stacked = torch.stack([sample['input_ids'] for sample in batch])
        collated['input_ids'] = stacked.view(-1, stacked.shape[-1])

        # Stack labels as-is (some labels are vectors, some scalars)
        collated['labels'] = torch.stack([sample['labels'] for sample in batch])

        # Pass batch_type as an integer (single scalar per batch)
        # It is sufficient to inspect the first sample because batches are homogeneous in type.
        bt_sample = batch[0]['batch_type']
        collated['batch_type'] = int(bt_sample.item())
        return collated
