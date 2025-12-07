import logging
from dataclasses import dataclass, field
from typing import Optional
import os
import csv
import shutil
from tqdm import tqdm

import torch
import transformers
from transformers import AutoTokenizer, AutoModel, TrainerCallback

from dataset import (
    ClinVarRefAltDataset,
    ContrastiveMutateDataset,
    BalancedAlternatingDataset,
    ContrastiveDataCollator,
)
from model import WithProjection, ContrastiveTrainer
from eval_metrics import compute_test_metrics

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    model_name_or_path: str = field(default=None)
    tokenizer_name: Optional[str] = field(default=None)
    model_type: str = field(default=None)
    trust_remote_code: bool = field(default=True)
    projection_output_dim: int = field(default=2048)

@dataclass
class DataArguments:
    clinvar_csv: str
    refs_fasta: str
    test_csv: str = field(default="")
    clinvar_sep: str = field(default=",")

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    run_name: str = field(default="dnabert2_finetune")
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(default=1024)
    load_best_model_at_end: bool = field(default=False)
    cos_loss_margin: float = field(default=-0.8)


class EvaluationCallback(TrainerCallback):
    def __init__(self, tokenizer, test_csv, training_args):
        self.tokenizer = tokenizer
        self.test_csv = test_csv
        self.test_dir = os.path.join(os.path.dirname(test_csv), "..", "test", "results")
        self.eval_batch_size = getattr(training_args, "per_device_eval_batch_size", 32)
    
    def on_epoch_end(self, args, state, control, **kwargs):
        if model := kwargs.get("model"):
            self._eval_and_log(model, f"epoch {state.epoch}")
    
    def _eval_and_log(self, model, step_info):
        logger.info(f"Running evaluation at {step_info}...")
        device = next(model.parameters()).device
        
        with open(self.test_csv, "r") as f:
            items = list(csv.DictReader(f))
            test_ids = [row["ID"] for row in items]
            test_seqs = [row["seq"] for row in items]
        
        embeddings = {}
        model.eval()
        with torch.no_grad():
            for i in tqdm(range(0, len(test_seqs), self.eval_batch_size), desc="Eval", leave=False):
                batch_ids = test_ids[i:i + self.eval_batch_size]
                batch_seqs = test_seqs[i:i + self.eval_batch_size]
                toks = self.tokenizer(batch_seqs, padding="max_length", max_length=self.tokenizer.model_max_length,
                                      truncation=True, return_tensors="pt")
                outputs = model(
                    input_ids=toks["input_ids"].to(device),
                    attention_mask=toks["attention_mask"].to(device, dtype=torch.bool),
                )
                projected = outputs[0] if isinstance(outputs, tuple) else outputs
                for j, sid in enumerate(batch_ids):
                    embeddings[sid] = projected[j].cpu().numpy()
        
        metrics = compute_test_metrics(embeddings, self.test_dir)
        for key, value in metrics.items():
            logger.info(f"eval_{key}: {value:.4f}")

def main():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    logger.info(f"Training parameters: {training_args}")
    logger.info(f"Model type: {model_args.model_type}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name or model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        trust_remote_code=model_args.trust_remote_code,
    )
    
    logger.info(f"Loading datasets from {data_args.clinvar_csv}, {data_args.refs_fasta} and {data_args.test_csv}")
    
    refpos_dataset = ClinVarRefAltDataset(data_args.clinvar_csv, tokenizer, data_args.clinvar_sep, 1)
    refneg_dataset = ClinVarRefAltDataset(data_args.clinvar_csv, tokenizer, data_args.clinvar_sep, -1)
    mutate_dataset = ContrastiveMutateDataset(data_args.refs_fasta, tokenizer, num_samples=max(len(refpos_dataset), len(refneg_dataset)))
    
    train_dataset = BalancedAlternatingDataset([refpos_dataset, refneg_dataset, mutate_dataset], training_args.per_device_train_batch_size, 42)

    base_model = AutoModel.from_pretrained(model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code)
    model = WithProjection(base_model, input_dim=None, output_dim=model_args.projection_output_dim, model_type=model_args.model_type)

    logger.info(f"Model hidden dimension: {getattr(base_model.config, 'hidden_size', 'unknown')}")
    
    # Create output directory and copy training script
    os.makedirs(training_args.output_dir, exist_ok=True)
    train_script_path = os.path.join(os.path.dirname(__file__), "script", "train.sh")
    if os.path.exists(train_script_path):
        dest_path = os.path.join(training_args.output_dir, "train.sh")
        shutil.copy2(train_script_path, dest_path)
        logger.info(f"Copied training script to {dest_path}")
    else:
        logger.warning(f"Training script not found at {train_script_path}")
    
    callbacks = [EvaluationCallback(tokenizer, data_args.test_csv, training_args)]
    
    trainer = ContrastiveTrainer(model=model, args=training_args, train_dataset=train_dataset, 
                                eval_dataset=None, data_collator=ContrastiveDataCollator(), callbacks=callbacks)

    logger.info("Starting training")
    trainer.train()

if __name__ == "__main__":
    import gpn.model
    os.environ["WANDB_DISABLED"] = "true"
    main()
