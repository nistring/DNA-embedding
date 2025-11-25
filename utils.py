"""Utility functions for training and evaluation."""
import torch
import transformers


def l2_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-9) -> torch.Tensor:
    """Normalize tensor along dimension."""
    return x / (x.norm(dim=dim, keepdim=True) + eps)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    if trainer.args.should_save:
        trainer._save(output_dir, state_dict={k: v.cpu() for k, v in trainer.model.state_dict().items()})
