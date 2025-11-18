import argparse, os, json, torch
from tokenizer import CharTokenizer
from model import TinyTransformer, ModelConfig

# Keep it isolated
import offline_mode  # noqa: F401

SYSTEM = (
"System: You are MIKE-like: logical, friendly, and curious. "
"Prefer step-by-step explanations when solving problems. "
"Occasionally ask for one good and one bad joke and reflect on what makes them so. "
"Stay offline; do not claim to browse."
)

def load(model_path, vocab_path, device):
    ckpt = torch.load(model_path, map_location=device)
    cfg = ModelConfig(**ckpt["cfg"])
    model = TinyTransformer(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    tok = CharTokenizer.load(vocab_path)
    return model, tok

def generate_text(model, tok, prompt, device, max_new_tokens=200, temperature=0.7, top_k=200):
    model.eval()
    ids = tok.encode(prompt)
    if not ids: return ""
    x = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.no_grad():
        y = model.generate(x, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k)
    return tok.decode(y[0].tolist())

def assemble_prompt(history, humor_memory):
    jokes = humor_memory.get("examples", [])
    fewshot = ""
    if jokes:
        # Use a compact few-shot style
        for j in jokes[-3:]:
            fewshot += f"\n[GoodJoke]{j['good']}\n[WhyGood]{j['why_good']}\n[BadJoke]{j['bad']}\n[WhyBad]{j['why_bad']}"
    convo = "\n".join(history[-8:])
    return SYSTEM + "\n" + fewshot + "\n" + convo + "\nAssistant:"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--humor_memory", default="checkpoints/humor_memory.json")
    ap.add_argument("--temperature", type=float, default=0.8)
    args = ap.parse_args()

    os.makedirs("checkpoints", exist_ok=True)
    if os.path.exists(args.humor_memory):
        humor_memory = json.load(open(args.humor_memory, "r", encoding="utf-8"))
    else:
        humor_memory = {"examples": []}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok = load(args.model, args.vocab, device)
    history = ["User: Hello!", "Assistant: Hi! I’m your offline MIKE-like helper. What shall we reason about today?"]

    print("Offline REPL. Type 'exit' to quit.")
    while True:
        user = input("You: ")
        if user.strip().lower() in {"exit", "quit"}:
            break
        history.append(f"User: {user}")
        prompt = assemble_prompt(history, humor_memory)
        out = generate_text(model, tok, prompt, device)
        # Extract only the new assistant portion after the last "Assistant:"
        resp = out.split("Assistant:")[-1]
        print("MIKE:", resp)
        history.append(f"Assistant: {resp}")

if __name__ == "__main__":
    main()
