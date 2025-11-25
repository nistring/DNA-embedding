"""Dataset classes for DNABERT-2 fine-tuning."""
import csv
import logging
import random
from typing import Dict, List
import torch
from torch.utils.data import Dataset, Sampler
import transformers

logger = logging.getLogger(__name__)


class ContrastiveClinVarDataset(Dataset):
    """Contrastive learning dataset for ClinVar - returns (ref, snv) pairs.
    
    Labels: 0 = benign, 1 = pathogenic
    SupConLoss will maximize distance between benign and pathogenic classes,
    and also push ref and snv apart (they get same label but are different sequences).
    """
    def __init__(self, path: str, tokenizer: transformers.PreTrainedTokenizer, sep: str = ","):
        super().__init__()
        with open(path, "r") as f:
            self.samples = [{"ref": row["ref_seq"], "mut_idx": int(row["mut_idx"]), 
                           "alt": row["alt"].upper(), "label": int(row["label"])}
                          for row in csv.DictReader(f, delimiter=sep)]
        self.tokenizer = tokenizer
        self.num_labels = 2
        logger.info(f"Loaded {len(self.samples)} ClinVar samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        s = self.samples[idx]
        ref_seq, mi, alt, label = s["ref"], s["mut_idx"], s["alt"], s["label"]
        snv_seq = (ref_seq[:mi] + alt + ref_seq[mi+1:]) if (0 <= mi < len(ref_seq) and ref_seq[mi] != alt) else ref_seq

        output = self.tokenizer([ref_seq, snv_seq], padding="max_length", 
                               max_length=self.tokenizer.model_max_length, 
                               truncation=True, return_tensors="pt")

        return {"input_ids": output["input_ids"], "attention_mask": output["attention_mask"],
                "labels": torch.tensor(label, dtype=torch.long),
                "is_clinvar": torch.tensor(True, dtype=torch.bool)}


class ContrastiveMutateDataset(Dataset):
    """
    Contrastive mutation dataset for regression.
    Returns (ref, mutated) pairs with log2(mutation count) as target.
    The loss should maximize cosine distance proportional to log2(mutation count).
    """
    DNA = ["A", "C", "G", "T"]

    def __init__(self, path: str, tokenizer: transformers.PreTrainedTokenizer, seq_col: str = "seq", sep: str = ",", mut_levels=(2, 8, 64, 128, 256, 512)):
        super().__init__()
        with open(path, "r") as f:
            data = list(csv.reader(f, delimiter=sep))
        seq_idx = data[0].index(seq_col) if data else 0
        self.seqs = [row[seq_idx] for row in data[1:]]
        self.mut_levels, self.tokenizer = tuple(mut_levels), tokenizer
        logger.info(f"Loaded {len(self.seqs)} mutation samples")

    def __len__(self):
        return len(self.seqs)

    @staticmethod
    def _mutate(seq: str, k: int) -> str:
        seq_list = list(seq)
        for i in random.sample(range(len(seq)), min(k, len(seq))):
            seq_list[i] = random.choice([b for b in ContrastiveMutateDataset.DNA if b != seq_list[i]])
        return "".join(seq_list)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        ref, k = self.seqs[idx], random.choice(self.mut_levels)
        output = self.tokenizer([ref, self._mutate(ref, k)], padding="max_length",
                               max_length=self.tokenizer.model_max_length, 
                               truncation=True, return_tensors="pt")
        return {"input_ids": output["input_ids"], "attention_mask": output["attention_mask"],
                "labels": torch.log2(torch.tensor(k, dtype=torch.float32)),
                "is_clinvar": torch.tensor(False, dtype=torch.bool)}


class BatchGroupedSampler(Sampler):
    """Sampler that groups indices into batches while maintaining order for alternating datasets.
    
    Ensures all samples in a minibatch come from the same dataset source while
    preserving the strict alternating pattern between datasets.
    """
    def __init__(self, dataset_len: int, batch_size: int, shuffle: bool = True, seed: int = 42):
        self.dataset_len = dataset_len
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.num_batches = (dataset_len + batch_size - 1) // batch_size

    def __iter__(self):
        batch_indices = list(range(self.num_batches))
        if self.shuffle:
            random.Random(self.seed).shuffle(batch_indices)
        for batch_idx in batch_indices:
            yield from range(batch_idx * self.batch_size, 
                           min((batch_idx + 1) * self.batch_size, self.dataset_len))

    def __len__(self):
        return self.dataset_len

    def set_epoch(self, epoch: int):
        """Set epoch for proper shuffling in distributed training."""
        self.seed = self.seed + epoch


class BalancedAlternatingDataset(Dataset):
    """Interleaves items from two datasets in 1:1 ratio for balanced training.
    
    Alternates batches: batch 0 from dataset_a, batch 1 from dataset_b, etc.
    Ensures each minibatch is homogeneous (all from one dataset) while keeping
    training balanced across both datasets. Data within each dataset is shuffled.
    
    Use with BatchGroupedSampler for shuffled but homogeneous batches.
    If len(dataset_b) < len(dataset_a), dataset_b is cycled.
    """
    def __init__(self, dataset_a: Dataset, dataset_b: Dataset, batch_size: int = 32, seed: int = 42):
        self.dataset_a, self.dataset_b = dataset_a, dataset_b
        self.batch_size, self.seed = batch_size, seed
        
        rng = random.Random(seed)
        self.indices_a, self.indices_b = list(range(len(dataset_a))), list(range(len(dataset_b)))
        rng.shuffle(self.indices_a)
        rng.shuffle(self.indices_b)
        
        num_batches = max((len(dataset_a) + batch_size - 1) // batch_size,
                         (len(dataset_b) + batch_size - 1) // batch_size) * 2
        self.num_batches, self.length = num_batches, num_batches * batch_size
        self.column_names = getattr(dataset_a, "column_names", ["input_ids", "attention_mask", "labels"])

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        batch_num, idx_in_batch = divmod(idx, self.batch_size)
        sample_idx = (batch_num // 2) * self.batch_size + idx_in_batch
        if batch_num % 2 == 0:
            return self.dataset_a[self.indices_a[sample_idx % len(self.dataset_a)]]
        return self.dataset_b[self.indices_b[sample_idx % len(self.dataset_b)]]
    
    def set_epoch(self, epoch: int):
        """Reshuffle data at each epoch for proper randomization in distributed training."""
        rng = random.Random(self.seed + epoch)
        self.indices_a, self.indices_b = list(range(len(self.dataset_a))), list(range(len(self.dataset_b)))
        rng.shuffle(self.indices_a), rng.shuffle(self.indices_b)


class ContrastiveDataCollator:
    """
    Custom data collator for contrastive learning that preserves pair structure.
    
    Converts (B, 2, seq_len) inputs to (B*2, seq_len) for model forward pass,
    while preserving metadata needed to reshape back in the loss function.
    """
    def __init__(self, pad_token_id=0):
        self.pad_token_id = pad_token_id
    
    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Collate batch of samples, flattening the pair dimension."""
        collated = {'_batch_size': len(batch)}
        for key in batch[0].keys():
            if key in ['input_ids', 'attention_mask']:
                stacked = torch.stack([sample[key] for sample in batch])  # (B, 2, seq_len)
                collated[key] = stacked.view(-1, stacked.shape[-1])  # (B*2, seq_len)
            elif key in ['labels', 'is_clinvar']:
                collated[key] = torch.stack([sample[key] for sample in batch])
            else:
                try:
                    collated[key] = torch.stack([sample[key] for sample in batch])
                except (ValueError, RuntimeError):
                    collated[key] = [sample[key] for sample in batch]
        return collated
