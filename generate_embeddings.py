"""Generate embeddings using fine-tuned DNABERT-2 encoder."""
import os
import argparse
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from safetensors.torch import load_file
import gpn.model

MODEL_NAME="songlab/gpn-brassicales"
TOKENIZER_NAME="gonzalobenegas/tokenizer-dna-mlm"

def load_model(checkpoint_dir: str, device: torch.device):
    """Load fine-tuned DNABERT2 from checkpoint directory."""
    checkpoint_dir = os.path.abspath(checkpoint_dir)
    print(f"Loading fine-tuned model from {checkpoint_dir}")
    # Load config from base model and weights from checkpoint
    model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True)
    # Load safetensors weights from checkpoint
    state_dict = load_file(os.path.join(checkpoint_dir, "model.safetensors"))
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    
    # Validate that weights were loaded
    if missing_keys:
        print(f"Warning: Missing keys in checkpoint: {missing_keys[:5]}...")  # Show first 5
    if unexpected_keys:
        print(f"Warning: Unexpected keys in checkpoint: {unexpected_keys[:5]}...")
    
    loaded_keys = set(state_dict.keys())
    model_keys = set(dict(model.named_parameters()).keys())
    overlap = loaded_keys & model_keys
    print(f"Successfully loaded {len(overlap)} weight tensors from checkpoint")
    
    return model.to(device).eval()


def main():
    parser = argparse.ArgumentParser(description="Generate embeddings using DNABERT-2")
    parser.add_argument("--input_csv", type=str, default="data/test.csv")
    parser.add_argument("--id_col", type=str, default="ID")
    parser.add_argument("--seq_col", type=str, default="seq")
    parser.add_argument("--output_csv", type=str, default="output/embeddings.csv")
    parser.add_argument("--checkpoint_dir", type=str, default="/home/work/.nistring/embedding/output/dnabert2_finetune/joint/checkpoint-813")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_length", type=int, default=1024)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint_dir, device)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, trust_remote_code=True)

    df = pd.read_csv(args.input_csv)
    ids, seqs = df[args.id_col].astype(str).tolist(), df[args.seq_col].astype(str).tolist()

    embs = []
    with torch.no_grad():
        for i in range(0, len(seqs), args.batch_size):
            toks = tokenizer(seqs[i:i+args.batch_size], padding=True, truncation=True,
                           max_length=args.max_length, return_tensors="pt").to(device)
            outputs = model(input_ids=toks["input_ids"], 
                           attention_mask=toks["attention_mask"].to(torch.bool))
            # Handle both tuple and object outputs
            last_hidden_state = outputs[0] if isinstance(outputs, tuple) else outputs.last_hidden_state
            batch_embs = last_hidden_state[:, 511]#.mean(dim=1)
            embs.append(batch_embs.cpu())
    
    embs = torch.cat(embs, dim=0)
    out = pd.DataFrame({"ID": ids, **{f"emb_{i:04d}": embs[:, i].numpy() for i in range(embs.shape[1])}})
    out.to_csv(args.output_csv, index=False)
    print(f"Wrote {len(ids)} embeddings ({embs.shape[1]}D) to {args.output_csv}")


if __name__ == "__main__":
    main()
