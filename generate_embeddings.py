"""Generate embeddings using fine-tuned DNABERT-2 encoder."""
import os
import argparse
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from safetensors.torch import load_file
from tqdm import tqdm
import gpn.model

from model import WrapperModel
from eval_metrics import compute_test_metrics

MODEL_NAME = "songlab/gpn-brassicales"
TOKENIZER_NAME = "gonzalobenegas/tokenizer-dna-mlm"

def load_model(checkpoint_dir: str, device: torch.device, use_vanilla_gpn: bool = False):
    """Load model from checkpoint.
    
    Args:
        checkpoint_dir: Path to checkpoint directory
        device: torch device
        use_vanilla_gpn: If True, use vanilla GPN with mean pooling (512D output).
                        If False, use WrapperModel with projection (2048D output).
    """
    checkpoint_dir = os.path.abspath(checkpoint_dir)
    print(f"Loading model from {checkpoint_dir}")
    
    base_model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True, model_max_length=1024,
        padding_side="right",)
    
    if use_vanilla_gpn:
        model = base_model
    else:
        model = WrapperModel(base_model, input_dim=512, output_dim=2048)
    
        state_dict = load_file(os.path.join(checkpoint_dir, "model.safetensors"))
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    
        if missing_keys:
            print(f"Warning: Missing keys: {missing_keys[:5]}...")
        if unexpected_keys:
            print(f"Warning: Unexpected keys: {unexpected_keys[:5]}...")
    
        overlap = set(state_dict.keys()) & set(dict(model.named_parameters()).keys())
        print(f"Loaded {len(overlap)} weight tensors")
    
    return model.to(device).eval()


def main():
    parser = argparse.ArgumentParser(description="Generate embeddings")
    parser.add_argument("--input_csv", type=str, default="data/test.csv")
    parser.add_argument("--id_col", type=str, default="ID")
    parser.add_argument("--seq_col", type=str, default="seq")
    parser.add_argument("--output_csv", type=str)
    parser.add_argument("--checkpoint_dir", type=str)
    parser.add_argument("--batch_size", type=int, default=160)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--use_vanilla_gpn", action="store_true", help="Use vanilla GPN model with mean pooling instead of wrapper with projection")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint_dir, device, use_vanilla_gpn=args.use_vanilla_gpn)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, trust_remote_code=True)

    df = pd.read_csv(args.input_csv)
    ids = df[args.id_col].astype(str).tolist()
    seqs = df[args.seq_col].astype(str).tolist()

    output_csv = args.output_csv or os.path.join(args.checkpoint_dir, "embeddings.csv")

    embs = []
    with torch.no_grad():
        for i in tqdm(range(0, len(seqs), args.batch_size), desc="Embedding", leave=False):
            toks = tokenizer(seqs[i:i+args.batch_size], padding=True, truncation=True,
                           max_length=args.max_length, return_tensors="pt").to(device)
            outputs = model(input_ids=toks["input_ids"], attention_mask=toks["attention_mask"].to(torch.bool))
            
            if args.use_vanilla_gpn:
                # Apply mean pooling over sequence length for vanilla GPN
                projected = (outputs[0] * toks["attention_mask"].unsqueeze(-1)).sum(dim=1) / toks["attention_mask"].sum(dim=1, keepdim=True)
            else:
                projected = outputs[0] if isinstance(outputs, tuple) else outputs
            
            embs.append(projected.cpu())
    
    embs = torch.cat(embs, dim=0)
    out = pd.DataFrame({"ID": ids, **{f"emb_{i:04d}": embs[:, i].numpy() for i in range(embs.shape[1])}})
    out.to_csv(output_csv, index=False)
    print(f"Wrote {len(ids)} embeddings ({embs.shape[1]}D) to {output_csv}")
    
    embeddings_dict = {ids[i]: embs[i].numpy() for i in range(len(ids))}
    metrics, worst_benign_pairs, worst_pathogenic_pairs = compute_test_metrics(embeddings_dict, "test/results")
    
    print("\nEvaluation Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    
    print("\nTop 10 Worst-Performing Benign Pairs (Highest Distance):")
    print(f"{'Rank':<6} {'Sample A':<20} {'Sample B':<20} {'Distance':<12}")
    print("-" * 58)
    for i, pair in enumerate(worst_benign_pairs, 1):
        print(f"{i:<6} {pair['sample_a']:<20} {pair['sample_b']:<20} {pair['distance']:<12.4f}")
    
    print("\nTop 10 Worst-Performing Pathogenic Pairs (Lowest Distance):")
    print(f"{'Rank':<6} {'Sample A':<20} {'Sample B':<20} {'Distance':<12}")
    print("-" * 58)
    for i, pair in enumerate(worst_pathogenic_pairs, 1):
        print(f"{i:<6} {pair['sample_a']:<20} {pair['sample_b']:<20} {pair['distance']:<12.4f}")


if __name__ == "__main__":
    main()
