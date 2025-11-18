import argparse, os, json, torch
from tokenizer import CharTokenizer
from model import TinyTransformer, ModelConfig

def load(model_path, vocab_path, device):
    ckpt = torch.load(model_path, map_location=device)
    cfg = ModelConfig(**ckpt["cfg"])
    model = TinyTransformer(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    tok = CharTokenizer.load(vocab_path)
    return model, tok

# ultra-simple generator for small models
def generate_text(model, tok, prompt, device, max_new_tokens=200, temperature=0.3, top_k=50):
    model.eval()
    ids = tok.encode(prompt)
    if not ids: 
        return ""
    x = torch.tensor([ids[-256:]], dtype=torch.long, device=device)  # limit context
    with torch.no_grad():
        y = model.generate(
            x, 
            max_new_tokens=max_new_tokens, 
            temperature=temperature, 
            top_k=top_k
        )
    return tok.decode(y[0].tolist())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--vocab", required=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok = load(args.model, args.vocab, device)

    print("Offline REPL. Type 'exit' to quit.")
    while True:
        user = input("You: ")
        if user.strip().lower() in {"exit", "quit"}:
            break
        
        # tiny char-model friendly prompt
        prompt = f"User: {user}\nAssistant:"
        out = generate_text(model, tok, prompt, device, max_new_tokens= 100)

        # just take everything after Assistant:
        resp = out.split("Assistant:")[-1].split("\n")[0].strip()
        print("MIKE:", resp)


if __name__ == "__main__":
    main()
