import argparse, json, torch, os
from model import TinyTransformer, ModelConfig
from tokenizer import CharTokenizer

# Keep it isolated
import offline_mode  # noqa: F401

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seq_len", type=int, default=256)
    args = ap.parse_args()

    device = "cpu"
    ckpt = torch.load(args.model, map_location=device)
    cfg = ModelConfig(**ckpt["cfg"])
    model = TinyTransformer(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    dummy = torch.randint(low=0, high=cfg.vocab_size, size=(1, min(cfg.max_seq_len, args.seq_len)), dtype=torch.long)
    torch.onnx.export(
        model,
        (dummy, None),
        args.out,
        input_names=["tokens", "targets"],
        output_names=["logits", "loss"],
        dynamic_axes={"tokens": {1: "seq"}, "logits": {1: "seq"}},
        opset_version=17
    )
    print(f"Exported ONNX to {args.out}")

if __name__ == "__main__":
    main()
