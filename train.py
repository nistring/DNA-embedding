import logging
from dataclasses import dataclass, field
from typing import Optional

import torch
from sklearn.model_selection import train_test_split

import transformers
from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM
from peft import get_peft_model, LoraConfig, TaskType

# Local imports
from my_datasets import (
    ClinVarRefAltDataset,
    ClinVarTripletDataset,
    ContrastiveMutateDataset,
    BalancedAlternatingDataset,
    ContrastiveDataCollator,
)
from model import WithProjection, ContrastiveTrainer

from copy import deepcopy

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="zhihan1996/DNABERT-2-117M")
    tokenizer_name: Optional[str] = field(default=None)
    model_type: str = field(default="DNABERT-2", metadata={"help": "Type of model to use (e.g., 'DNABERT-2', 'GPN', 'nucleotide-transformer-v2')"})
    use_lora: bool = field(default=False, metadata={"help": "whether to use LoRA"})
    lora_r: int = field(default=8, metadata={"help": "hidden dimension for LoRA"})
    lora_alpha: int = field(default=16, metadata={"help": "alpha for LoRA"})
    lora_dropout: float = field(default=0.05, metadata={"help": "dropout rate for LoRA"})
    lora_target_modules: str = field(default="ffn.0", metadata={"help": "where to perform LoRA"})
    use_reverse_complement: bool = field(default=False, metadata={"help": "Use reverse complement augmentation in training dataset"})
    trust_remote_code: bool = field(default=True, metadata={"help": "Trust remote code for custom models"})
    projection_output_dim: int = field(default=2048, metadata={"help": "Output dimension of projection head"})

@dataclass
class DataArguments:
    clinvar_csv: str = field(metadata={"help": "Path to ClinVar CSV with ref_seq,snv_seq,label"})
    triplet_csv: str = field(metadata={"help": "Path to ClinVar triplet CSV with ref_seq,alt_pos,alt_neg"})
    refs_csv: str = field(metadata={"help": "Path to mutation CSV with reference sequences"})
    clinvar_sep: str = field(default=",")
    refs_sep: str = field(default=",")

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    run_name: str = field(default="dnabert2_finetune")
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(default=1024, metadata={"help": "Maximum sequence length"})
    load_best_model_at_end: bool = field(default=True, metadata={"help": "Load best model at training end"})
    metric_for_best_model: str = field(default="eval_loss", metadata={"help": "Metric to use for best model selection"})
    greater_is_better: bool = field(default=False, metadata={"help": "Whether higher metric is better"})
    cos_loss_margin: float = field(default=-0.8, metadata={"help": "Margin parameter for training CosineEmbeddingLoss"})

def main():
    """Main training function for contrastive learning."""
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARNING, 
    )
    logger.info(f"Training/evaluation parameters {training_args}")
    logger.info(f"Model type: {model_args.model_type}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        trust_remote_code=model_args.trust_remote_code,
    )
    
    logger.info(f"Loading datasets from {data_args.clinvar_csv}, {data_args.triplet_csv} and {data_args.refs_csv}")
    # Create three separate datasets
    refpos_dataset = ClinVarRefAltDataset(data_args.clinvar_csv, tokenizer, data_args.clinvar_sep, 1)
    refneg_dataset = ClinVarRefAltDataset(data_args.clinvar_csv, tokenizer, data_args.clinvar_sep, -1)
    # pathogenic_dataset = ClinVarTripletDataset(data_args.triplet_csv, tokenizer, data_args.clinvar_sep)
    mutate_dataset = ContrastiveMutateDataset(data_args.refs_csv, tokenizer, "seq", data_args.refs_sep, use_reverse_complement=model_args.use_reverse_complement)
    
    # Split each dataset into train/eval
    train_idx, eval_idx = train_test_split(list(range(len(refpos_dataset))), test_size=0.05, random_state=42)
    refpos_train, refpos_eval = torch.utils.data.Subset(refpos_dataset, train_idx), torch.utils.data.Subset(refpos_dataset, eval_idx)
    
    train_idx, eval_idx = train_test_split(list(range(len(refneg_dataset))), test_size=0.05, random_state=42)
    refneg_train, refneg_eval = torch.utils.data.Subset(refneg_dataset, train_idx), torch.utils.data.Subset(refneg_dataset, eval_idx)
    
    # refalt_train = BalancedAlternatingDataset([refpos_train, refneg_train], training_args.per_device_train_batch_size, 42)
    # refalt_eval = BalancedAlternatingDataset([refpos_eval, refneg_eval], training_args.per_device_eval_batch_size, 42, shuffle=False)
    
    # train_idx, eval_idx = train_test_split(list(range(len(pathogenic_dataset))), test_size=0.05, random_state=42)
    # pathogenic_train, pathogenic_eval = torch.utils.data.Subset(pathogenic_dataset, train_idx), torch.utils.data.Subset(pathogenic_dataset, eval_idx)
    
    train_idx, eval_idx = train_test_split(list(range(len(mutate_dataset))), test_size=0.05, random_state=42)
    mutate_train, mutate_eval = torch.utils.data.Subset(mutate_dataset, train_idx), torch.utils.data.Subset(mutate_dataset, eval_idx)
    
    # Combine three datasets with alternating batches
    train_dataset = BalancedAlternatingDataset([refpos_train, refneg_train, mutate_train], training_args.per_device_train_batch_size, 42)
    eval_dataset = BalancedAlternatingDataset([refpos_eval, refneg_eval, mutate_eval], training_args.per_device_eval_batch_size, 42, shuffle=False)

    # Load pretrained model and wrap with projection head
    base_model = AutoModel.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
    )

    # Apply LoRA if enabled
    if model_args.use_lora:
        peft_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
            target_modules=model_args.lora_target_modules.split(","),
            bias="none",
        )
        base_model = get_peft_model(base_model, peft_config)
        logger.info("LoRA enabled")

    # Create model with projection head
    model = WithProjection(
        base_model, 
        input_dim=None,  # Auto-detect from config
        output_dim=model_args.projection_output_dim,
        model_type=model_args.model_type
    )

    logger.info(f"Model hidden dimension: {base_model.config.hidden_size if hasattr(base_model.config, 'hidden_size') else 'unknown'}")
    
    trainer = ContrastiveTrainer(
        model=model, args=training_args, train_dataset=train_dataset, eval_dataset=eval_dataset,
        data_collator=ContrastiveDataCollator())

    logger.info("Starting training")
    trainer.train()

if __name__ == "__main__":
    import gpn.model
    import os
    os.environ["WANDB_DISABLED"] = "true"
    main()
