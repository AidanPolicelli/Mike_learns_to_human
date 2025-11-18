import argparse
import os
import random
import json
import time
import numpy as np
import torch
import torch.optim as optim
from tokenizer import CharTokenizer
from model import ModelConfig, TinyTransformer

# ------------------------------------------
# Utility: Load text safely
# ------------------------------------------
def load_text(path: str):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

# ------------------------------------------
# Create batches for training
# ------------------------------------------
def batchify(data_ids, batch_size, seq_len, device):
    n = len(data_ids) - seq_len - 1
    while True:
        idxs = np.random.randint(0, n, size=batch_size)
        x = np.stack([data_ids[i:i+seq_len] for i in idxs])
        y = np.stack([data_ids[i+1:i+seq_len+1] for i in idxs])
        yield (torch.tensor(x, dtype=torch.long, device=device),
               torch.tensor(y, dtype=torch.long, device=device))

# ------------------------------------------
# Build model configuration for scaling
# ------------------------------------------
def get_model_config(scale, vocab_size, seq_len):
    if scale == "tiny":
        return ModelConfig(
            vocab_size=vocab_size,
            d_model=256,
            n_heads=4,
            n_layers=4,
            d_ff=1024,
            max_seq_len=seq_len,
            dropout=0.0,
        )
    elif scale == "medium":
        return ModelConfig(
            vocab_size=vocab_size,
            d_model=384,
            n_heads=6,
            n_layers=6,
            d_ff=1536,
            max_seq_len=seq_len,
            dropout=0.0,
        )
    elif scale == "large":
        return ModelConfig(
            vocab_size=vocab_size,
            d_model=512,
            n_heads=8,
            n_layers=8,
            d_ff=2048,
            max_seq_len=seq_len,
            dropout=0.0,
        )
    elif scale == "xl":
        return ModelConfig(
            vocab_size=vocab_size,
            d_model=768,
            n_heads=12,
            n_layers=12,
            d_ff=3072,
            max_seq_len=seq_len,
            dropout=0.0,
        )
    else:
        raise ValueError("Unknown scale: choose tiny, medium, large, or xl")

# ------------------------------------------
# Main training function
# ------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to training corpus")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--seq_len", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--scale", type=str, default="medium",
                    help="Model size: tiny / medium / large / xl")
    ap.add_argument("--save_dir", default="checkpoints")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    # --------------------------------------
    # Set seeds
    # --------------------------------------
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # --------------------------------------
    # Prepare folders and device
    # --------------------------------------
    os.makedirs(args.save_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\nUsing device: {device}\n")

    corpus_text = load_text(args.data)
    vocab_path = os.path.join(args.save_dir, "vocab.json")
    model_path = os.path.join(args.save_dir, "mike.pt")

    # --------------------------------------
    # Load or build tokenizer
    # --------------------------------------
    if os.path.exists(vocab_path):
        tok = CharTokenizer.load(vocab_path)
        print(f"Loaded existing vocab: {vocab_path}")
    else:
        print("Building new vocab...")
        tok = CharTokenizer.build_from_text(corpus_text)
        tok.save(vocab_path)
        print(f"Saved new vocab to: {vocab_path}")

    ids = tok.encode(corpus_text)

    # --------------------------------------
    # Load or initialize model depending on scale + checkpoint
    # --------------------------------------
    model = None
    cfg = None

    if os.path.exists(model_path):
        print(f"Found checkpoint: {model_path}")
        ckpt = torch.load(model_path, map_location=device)

        # read checkpoint config
        old_cfg = ModelConfig(**ckpt["cfg"])

        # build desired config
        new_cfg = get_model_config(args.scale, len(tok.stoi), args.seq_len)

        # If same model size, we can load weights safely
        same_shape = (
            old_cfg.d_model == new_cfg.d_model and
            old_cfg.n_heads == new_cfg.n_heads and
            old_cfg.n_layers == new_cfg.n_layers and
            old_cfg.d_ff == new_cfg.d_ff
        )

        if same_shape:
            print("Same model size → continuing training from checkpoint")
            cfg = new_cfg
            model = TinyTransformer(cfg).to(device)
            model.load_state_dict(ckpt["model"])
        else:
            print("Model size changed → training from scratch")
            cfg = new_cfg
            model = TinyTransformer(cfg).to(device)

    else:
        print("No checkpoint found → starting fresh model")
        cfg = get_model_config(args.scale, len(tok.stoi), args.seq_len)
        model = TinyTransformer(cfg).to(device)

    # --------------------------------------
    # Training setup
    # --------------------------------------
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    gen = batchify(ids, args.batch_size, args.seq_len, device)

    steps_per_epoch = max(1, len(ids) // (args.batch_size * args.seq_len))
    print(f"\nSteps per epoch: {steps_per_epoch}\n")

    # --------------------------------------
    # TRAINING LOOP
    # --------------------------------------
    model.train()
    for epoch in range(args.epochs):
        losses = []
        start = time.time()

        for _ in range(steps_per_epoch):
            x, y = next(gen)
            _, loss = model(x, y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())

        avg_loss = sum(losses) / len(losses)
        duration = time.time() - start
        print(f"Epoch {epoch+1}/{args.epochs} | loss={avg_loss:.4f} | time={duration:.2f}s")

    # --------------------------------------
    # SAVE UPDATED CHECKPOINT
    # --------------------------------------
    torch.save({"model": model.state_dict(), "cfg": cfg.__dict__}, model_path)
    print(f"\nSaved updated model to {model_path}\n")


if __name__ == "__main__":
    main()
