import csv
import logging
import random
from typing import Dict, List
import torch
from torch.utils.data import Dataset
import transformers
from Bio import SeqIO

logger = logging.getLogger(__name__)


class ClinVarRefAltDataset(Dataset):
    """ClinVar dataset for cd_loss - returns (ref, alt) pairs."""
    def __init__(self, path: str, tokenizer: transformers.PreTrainedTokenizer, sep: str = ",", label: int = 1):
        super().__init__()
        with open(path, "r") as f:
            samples = [{"ref": row["ref_seq"], "mut_idx": int(row["mut_idx"]), 
                       "alt": row["alt"].upper(), "label": int(row["label"])}
                      for row in csv.DictReader(f, delimiter=sep) if int(row["label"]) == label]
        self.sample = samples
        self.tokenizer = tokenizer
        logger.info(f"Loaded {len(self.sample)} ClinVar samples with label {label}")

    def __len__(self):
        return len(self.sample)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        s = self.sample[idx]
        ref_seq = s["ref"]
        snv_seq = ref_seq[:s["mut_idx"]] + s["alt"] + ref_seq[s["mut_idx"]+1:]

        output = self.tokenizer([ref_seq, snv_seq], padding="max_length", 
                               max_length=self.tokenizer.model_max_length, 
                               truncation=True, return_tensors="pt")

        return {"input_ids": output["input_ids"],
                "labels": torch.tensor(s["label"], dtype=torch.float32),
                "batch_type": torch.tensor(0, dtype=torch.long)}


class ContrastiveMutateDataset(Dataset):
    """Contrastive mutation dataset with mutation count as target."""
    DNA = ["A", "C", "G", "T"]

    def __init__(self, path: str, tokenizer: transformers.PreTrainedTokenizer, seq_length: int = 1024,
                 seed: int = 42, num_samples: int = None):
        super().__init__()
        self.fasta_path = path
        self.seq_length = seq_length
        self.tokenizer = tokenizer
        self.rng = random.Random(seed)
        self.num_samples = num_samples
        self.sequences: List[tuple[str, str]] = []
        self.seq_info = []
        for record in SeqIO.parse(path, "fasta"):
            seq_str = str(record.seq).upper()
            self.sequences.append((record.id, seq_str))
            self.seq_info.append((record.id, len(seq_str)))
        logger.info(f"Loaded FASTA with {len(self.seq_info)} sequences")

    def __len__(self):
        return self.num_samples if self.num_samples is not None else sum(max(0, length - self.seq_length + 1) for _, length in self.seq_info)

    @staticmethod
    def _mutate(seq: str, k: int) -> str:
        seq_list = list(seq)
        for i in random.sample(range(len(seq)), min(k, len(seq))):
            seq_list[i] = random.choice([b for b in ContrastiveMutateDataset.DNA if b != seq_list[i]])
        return "".join(seq_list)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        # Randomly sample a sequence and position
        seq_id, seq = self.rng.choice(self.sequences)
        start = self.rng.randint(0, max(0, len(seq) - self.seq_length))
        ref = seq[start:start + self.seq_length]
        
        k = self.rng.randint(1, 512)
        mutated = self._mutate(ref, k)
        
        output = self.tokenizer([ref, mutated], padding="max_length",
                               max_length=self.tokenizer.model_max_length, 
                               truncation=True, return_tensors="pt")
        
        return {"input_ids": output["input_ids"],
                "labels": torch.tensor(k, dtype=torch.float32) / 512.0,
                "batch_type": torch.tensor(1, dtype=torch.long)}


class BalancedAlternatingDataset(Dataset):
    """Interleaves items from multiple datasets in round-robin fashion by batch."""
    def __init__(self, datasets: List[Dataset], batch_size: int = 32, seed: int = 42, shuffle: bool = True):
        self.datasets = datasets
        self.batch_size = batch_size
        self.seed = seed
        self.shuffle = shuffle
        
        rng = random.Random(seed)
        self.indices = [list(range(len(ds))) for ds in datasets]
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
        """Reshuffle data at each epoch for distributed training."""
        if not self.shuffle:
            return
        rng = random.Random(self.seed + epoch)
        self.indices = [list(range(len(ds))) for ds in self.datasets]
        for idx_list in self.indices:
            rng.shuffle(idx_list)


class ContrastiveDataCollator:
    """Data collator that preserves pair structure for contrastive learning."""
    
    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        stacked = torch.stack([sample['input_ids'] for sample in batch])
        
        return {
            'input_ids': stacked.view(-1, stacked.shape[-1]),
            'labels': torch.stack([sample['labels'] for sample in batch]),
            'batch_type': int(batch[0]['batch_type'].item())
        }
