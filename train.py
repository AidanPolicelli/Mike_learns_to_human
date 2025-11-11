import argparse, os, random, json
import torch
import torch.optim as optim
from tokenizer import CharTokenizer
from model import ModelConfig, TinyTransformer
import numpy as np

# Keep it isolated
import offline_mode  # noqa: F401

def load_text(path: str):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def batchify(data_ids, batch_size, seq_len, device):
    # Produce infinite stream of (x,y) batches
    n = len(data_ids) - seq_len - 1
    while True:
        idxs = np.random.randint(0, n, size=batch_size)
        x = np.stack([data_ids[i:i+seq_len] for i in idxs])
        y = np.stack([data_ids[i+1:i+seq_len+1] for i in idxs])
        yield torch.tensor(x, dtype=torch.long, device=device), torch.tensor(y, dtype=torch.long, device=device)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="path to training text (.txt)")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--seq_len", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--n_heads", type=int, default=4)
    ap.add_argument("--n_layers", type=int, default=4)
    ap.add_argument("--d_ff", type=int, default=1024)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--save_dir", default="checkpoints")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    os.makedirs(args.save_dir, exist_ok=True)

    text = load_text(args.data)
    tok = CharTokenizer.build_from_text(text)
    vocab_path = os.path.join(args.save_dir, "vocab.json")
    tok.save(vocab_path)

    ids = tok.encode(text)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = ModelConfig(
        vocab_size=len(tok.stoi),
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        max_seq_len=args.seq_len,
        dropout=args.dropout
    )
    model = TinyTransformer(cfg).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    gen = batchify(ids, args.batch_size, args.seq_len, device)

    steps_per_epoch = max(1, len(ids) // (args.batch_size * args.seq_len))
    model.train()
    for epoch in range(args.epochs):
        losses = []
        for step in range(steps_per_epoch):
            x, y = next(gen)
            _, loss = model(x, y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())
        avg = sum(losses)/len(losses)
        print(f"epoch {epoch+1}/{args.epochs} - loss {avg:.4f}")

    model_path = os.path.join(args.save_dir, "mike.pt")
    torch.save({"model": model.state_dict(), "cfg": cfg.__dict__}, model_path)
    print(f"Saved model to {model_path}")
    print(f"Saved vocab to {vocab_path}")

if __name__ == "__main__":
    main()
