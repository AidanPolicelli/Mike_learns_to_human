import argparse, os, random, json
import torch
import torch.optim as optim
from tokenizer import CharTokenizer
from model import ModelConfig, TinyTransformer
import numpy as np

def load_text(path: str):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def batchify(data_ids, batch_size, seq_len, device):
    n = len(data_ids) - seq_len - 1
    while True:
        idxs = np.random.randint(0, n, size=batch_size)
        x = np.stack([data_ids[i:i+seq_len] for i in idxs])
        y = np.stack([data_ids[i+1:i+seq_len+1] for i in idxs])
        yield torch.tensor(x, dtype=torch.long, device=device), torch.tensor(y, dtype=torch.long, device=device)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--seq_len", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--save_dir", default="checkpoints")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    text = load_text(args.data)
    vocab_path = os.path.join(args.save_dir, "vocab.json")
    model_path = os.path.join(args.save_dir, "mike.pt")

    # ------------------------------
    # LOAD EXISTING TOKENIZER
    # ------------------------------
    if os.path.exists(vocab_path):
        tok = CharTokenizer.load(vocab_path)
        print(f"Loaded existing vocab from {vocab_path}")
    else:
        tok = CharTokenizer.build_from_text(text)
        tok.save(vocab_path)
        print(f"Built new vocab and saved to {vocab_path}")

    ids = tok.encode(text)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ------------------------------
    # LOAD EXISTING MODEL CHECKPOINT
    # ------------------------------
    if os.path.exists(model_path):
        ckpt = torch.load(model_path, map_location=device)
        cfg_dict = ckpt["cfg"]
        cfg = ModelConfig(**cfg_dict)
        model = TinyTransformer(cfg).to(device)
        model.load_state_dict(ckpt["model"])
        print(f"Loaded existing model from {model_path}")
    else:
        # Create new model
        cfg = ModelConfig(
            vocab_size=len(tok.stoi),
            d_model=512,
            n_heads=8,
            n_layers=8,
            d_ff=2048,
            max_seq_len=args.seq_len,
            dropout=0.0,
        )
        model = TinyTransformer(cfg).to(device)
        print("Initialized new model (no checkpoint found)")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    gen = batchify(ids, args.batch_size, args.seq_len, device)
    steps_per_epoch = max(1, len(ids) // (args.batch_size * args.seq_len))

    # ------------------------------
    # TRAINING LOOP
    # ------------------------------
    model.train()
    for epoch in range(args.epochs):
        losses = []
        for _ in range(steps_per_epoch):
            x, y = next(gen)
            _, loss = model(x, y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())
        print(f"epoch {epoch+1}/{args.epochs} - loss {sum(losses)/len(losses):.4f}")

    # ------------------------------
    # SAVE UPDATED CHECKPOINT
    # ------------------------------
    torch.save({"model": model.state_dict(), "cfg": cfg.__dict__}, model_path)
    print(f"Saved updated model to {model_path}")

if __name__ == "__main__":
    main()
