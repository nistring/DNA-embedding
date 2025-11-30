"""DNABERT-2 fine-tuning with contrastive learning."""
import os
import json
import logging
from dataclasses import dataclass, field
from typing import Optional

import torch
from sklearn.model_selection import train_test_split

import transformers
from transformers import AutoTokenizer, AutoModel, PreTrainedModel
from peft import get_peft_model

# Local imports
from my_datasets import (
    ContrastiveClinVarDataset,
    ContrastiveMutateDataset,
    BalancedAlternatingDataset,
    BatchGroupedSampler,
    ContrastiveDataCollator,
)
from utils import safe_save_model_for_hf_trainer


class ProjectionHead(torch.nn.Module):
    """Projection head combining SNV-specific token with multi-pooling sequence context."""
    def __init__(self, input_dim=512, output_dim=2048):
        super().__init__()
        self.dense = torch.nn.Linear(input_dim * 4, output_dim)
    
    def forward(self, x):
        # x: (B*2, seq_len, D)
        snv_feat = x[:, 511]  # SNV-specific: (B*2, D)
        mean_feat = x.mean(dim=1)  # Mean pooling: (B*2, D)
        max_feat = x.max(dim=1)[0]  # Max pooling: (B*2, D)
        min_feat = x.min(dim=1)[0]  # Min pooling: (B*2, D)
        
        # Concatenate: (B*2, 4*D) = (B*2, 2048)
        combined = torch.cat([snv_feat, mean_feat, max_feat, min_feat], dim=-1)
        return self.dense(combined)


class DNABertWithProjection(torch.nn.Module):
    """Wrapper model that adds projection head to pretrained DNABERT-2."""
    
    def __init__(self, base_model, input_dim=512, output_dim=2048):
        super().__init__()
        self.base_model = base_model
        self.projection_head = ProjectionHead(input_dim, output_dim)
        # Copy config from base model for compatibility
        self.config = base_model.config
    
    def forward(self, *args, **kwargs):
        """Forward pass through base model and projection head."""
        outputs = self.base_model(*args, **kwargs)
        # Extract last_hidden_state and apply projection
        if hasattr(outputs, 'last_hidden_state'):
            hidden_state = outputs.last_hidden_state
        else:
            hidden_state = outputs[0]
        projected = self.projection_head(hidden_state)
        return (projected,)

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
    model_type: str = field(default="DNABERT-2", metadata={"help": "Type of model to use"})
    soft_masked_loss_weight_train: float = field(default=0.0, metadata={"help": "Weight for soft-masked loss during training"})
    soft_masked_loss_weight_evaluation: float = field(default=0.0, metadata={"help": "Weight for soft-masked loss during evaluation"})
    use_lora: bool = field(default=False, metadata={"help": "whether to use LoRA"})
    lora_r: int = field(default=8, metadata={"help": "hidden dimension for LoRA"})
    lora_alpha: int = field(default=32, metadata={"help": "alpha for LoRA"})
    lora_dropout: float = field(default=0.05, metadata={"help": "dropout rate for LoRA"})
    lora_target_modules: str = field(default="query,value,key,dense", metadata={"help": "where to perform LoRA"})


@dataclass
class DataArguments:
    clinvar_csv: str = field(metadata={"help": "Path to ClinVar CSV with ref_seq,snv_seq,label"})
    refs_csv: str = field(metadata={"help": "Path to mutation CSV with reference sequences"})
    clinvar_sep: str = field(default=",")
    refs_sep: str = field(default=",")

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    run_name: str = field(default="dnabert2_finetune")
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(default=1024, metadata={"help": "Maximum sequence length"})
    find_unused_parameters: bool = field(default=False)
    load_best_model_at_end: bool = field(default=True, metadata={"help": "Load best model at training end"})
    metric_for_best_model: str = field(default="eval_loss", metadata={"help": "Metric to use for best model selection"})
    greater_is_better: bool = field(default=False, metadata={"help": "Whether higher metric is better"})
    lr_scheduler_type: str = field(default="exponential", metadata={"help": "Learning rate scheduler type"})
    torch_compile: bool = field(default=False, metadata={"help": "Disable torch.compile to avoid dynamo tracing issues"})
# -----------------------------
# Contrastive Trainer
# -----------------------------
class ContrastiveTrainer(transformers.Trainer):
    """Custom trainer for supervised contrastive learning and mutation regression."""
    def __init__(self, mutation_loss_weight=1.0, clinvar_loss_weight=1.0, train_sampler=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mutation_loss_weight = mutation_loss_weight
        self.clinvar_loss_weight = clinvar_loss_weight
        self.custom_train_sampler = train_sampler

    def _get_train_sampler(self, *args, **kwargs):
        """Return the custom batch-grouped sampler if available, otherwise use default."""
        if self.custom_train_sampler is not None:
            return self.custom_train_sampler
        # Fall back to parent's implementation for default sampling
        return super()._get_train_sampler(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Compute custom contrastive loss for both training and evaluation."""
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        
        # Check if we should log this step
        should_log = (self.state.global_step % self.args.logging_steps == 0) if self.state.global_step > 0 else False
        
        loss = contrastive_loss_func(
            outputs, labels, 
            self.mutation_loss_weight, self.clinvar_loss_weight,
            should_log=should_log)

        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """
        Perform a prediction step using custom contrastive loss for both training and evaluation.
        Overrides parent's prediction_step to use our custom loss function.
        """
        inputs = self._prepare_inputs(inputs)
        labels = inputs.pop("labels")
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        loss = contrastive_loss_func(
            outputs, labels,
            self.mutation_loss_weight, self.clinvar_loss_weight,
            should_log=False
        )
        
        if prediction_loss_only:
            return (loss, None, None)
        
        logits = outputs.get("last_hidden_state") if isinstance(outputs, dict) else (
            outputs[0] if isinstance(outputs, (tuple, list)) else outputs
        )
        
        return (loss * 2, logits, labels)


def supervised_contrastive_loss(projections, targets, temperature=0.07):
    """
    Supervised Contrastive Loss based on:
    https://github.com/GuillaumeErhard/Supervised_contrastive_loss_pytorch/blob/main/loss/spc.py
    
    Implementation of the loss described in the paper "Supervised Contrastive Learning":
    https://arxiv.org/abs/2004.11362
    
    :param projections: torch.Tensor, shape [batch_size, projection_dim]
    :param targets: torch.Tensor, shape [batch_size]
    :param temperature: float, temperature scaling factor
    :return: torch.Tensor, scalar loss value
    """
    device = projections.device
    
    dot_product_tempered = torch.mm(projections, projections.T) / temperature
    # Minus max for numerical stability with exponential. Same done in cross entropy. Epsilon added to avoid log(0)
    exp_dot_tempered = (
        torch.exp(dot_product_tempered - torch.max(dot_product_tempered, dim=1, keepdim=True)[0]) + 1e-5
    )
    
    mask_similar_class = (targets.unsqueeze(1).repeat(1, targets.shape[0]) == targets).to(device)
    mask_anchor_out = (1 - torch.eye(exp_dot_tempered.shape[0])).to(device)
    mask_combined = mask_similar_class * mask_anchor_out
    cardinality_per_samples = torch.sum(mask_combined, dim=1)
    
    log_prob = -torch.log(exp_dot_tempered / (torch.sum(exp_dot_tempered * mask_anchor_out, dim=1, keepdim=True)))
    supervised_contrastive_loss_per_sample = torch.sum(log_prob * mask_combined, dim=1) / cardinality_per_samples
    loss = torch.mean(supervised_contrastive_loss_per_sample)
    
    return loss


def contrastive_loss_func(outputs, labels, mutation_loss_weight=1.0, clinvar_loss_weight=1.0, alpha=0.5, num_items_in_batch=None, should_log=False, projection_head=None):
    """Custom loss function for supervised contrastive learning and mutation regression."""
    
    # Embeddings are already projected by the model forward pass: (B*2, D')
    embeddings = torch.nn.functional.normalize(outputs[0].view(labels.shape[0], 2, -1), dim=-1)
    
    if labels.max() <= 1:  # ClinVar: binary classification
        # cd_loss: push ref and alt apart
        cd_loss = (1 + (embeddings[:, 0] * embeddings[:, 1]).sum(-1).mean()) / 2  # Scale to [0,1]
        
        # cdd_loss: supervised contrastive loss on alt embeddings based on pathogenicity
        cdd_loss = supervised_contrastive_loss(embeddings[:, 1], labels)
        
        if should_log:
            logger.info(f"cd_loss: {cd_loss.item():.4f}, cdd_loss: {cdd_loss.item():.4f}")
        return clinvar_loss_weight * (cd_loss + cdd_loss)
    else:  # Mutation: regression
        cos_sim = (1 - (embeddings[:, 0] * embeddings[:, 1]).sum(-1)) / 2  # Scale to [0,1]
        labels = labels / 10
        pcc_loss = (mutation_loss_weight - alpha) * torch.nn.functional.mse_loss(cos_sim, labels) + \
            alpha * (1 - torch.nan_to_num(torch.corrcoef(torch.stack([cos_sim, labels]))[0,1])) / 2
        if should_log:
            logger.info(f"pcc_loss: {pcc_loss.item():.4f}")
        return pcc_loss

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

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
        trust_remote_code=True,
    )

    if not (data_args.clinvar_csv and getattr(data_args, "refs_csv", None)):
        raise ValueError("Both --clinvar_csv and --refs_csv must be provided for joint training.")
    
    logger.info(f"Loading datasets from {data_args.clinvar_csv} and {data_args.refs_csv}")
    clinvar_dataset = ContrastiveClinVarDataset(data_args.clinvar_csv, tokenizer, data_args.clinvar_sep)
    mutate_dataset = ContrastiveMutateDataset(data_args.refs_csv, tokenizer, "seq", data_args.refs_sep)
    
    train_idx, eval_idx = train_test_split(list(range(len(clinvar_dataset))), test_size=0.2, random_state=42)
    clinvar_train, clinvar_eval = torch.utils.data.Subset(clinvar_dataset, train_idx), torch.utils.data.Subset(clinvar_dataset, eval_idx)
    
    eval_dataset = BalancedAlternatingDataset(clinvar_eval, mutate_dataset, training_args.per_device_eval_batch_size, 42 )
    train_dataset = BalancedAlternatingDataset(clinvar_train, mutate_dataset, training_args.per_device_train_batch_size, 42)

    # Load pretrained model and wrap with projection head
    base_model = AutoModel.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
    )
    model = DNABertWithProjection(base_model, input_dim=512, output_dim=2048)
    
    trainer = ContrastiveTrainer(
        mutation_loss_weight=1.0, clinvar_loss_weight=1.0,
        model=model, args=training_args, train_dataset=train_dataset, eval_dataset=eval_dataset,
        train_sampler=BatchGroupedSampler(len(train_dataset), training_args.per_device_train_batch_size, False, 42),
        data_collator=ContrastiveDataCollator())

    logger.info("Starting training")
    trainer.train()

if __name__ == "__main__":
    import gpn.model
    main()
