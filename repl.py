import argparse, os, json, torch, random
from tokenizer import CharTokenizer
from model import TinyTransformer, ModelConfig

MEMORY_PATH = "mike_memory.json"


# ---------------- Memory helpers ----------------
def load_memory():
    if os.path.exists(MEMORY_PATH):
        try:
            with open(MEMORY_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            name = data.get("user_name", "Aidan")
            return name
        except Exception:
            pass
    return "Aidan"


def save_memory(name):
    data = {"user_name": name}
    with open(MEMORY_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f)


# ---------------- Model helpers ----------------
def load_model(model_path, vocab_path, device):
    ckpt = torch.load(model_path, map_location=device)
    cfg = ModelConfig(**ckpt["cfg"])
    model = TinyTransformer(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    tok = CharTokenizer.load(vocab_path)
    return model, tok


def generate_text(model, tok, prompt, device, max_new_tokens=200, temperature=0.3, top_k=50):
    model.eval()
    ids = tok.encode(prompt)
    if not ids:
        return ""
    x = torch.tensor([ids[-256:]], dtype=torch.long, device=device)
    with torch.no_grad():
        y = model.generate(
            x,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )
    return tok.decode(y[0].tolist())


# ---------------- Main REPL ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--vocab", required=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok = load_model(args.model, args.vocab, device)

    name = load_memory()

    print("Offline REPL. Type 'exit' to quit.")

    # ---------------- Greeting ----------------
    greetings = [
        f"Hey {name}, systems online. Did you miss me?",
        f"Greetings, {name}. I am booted, confused, and ready.",
        f"{name}, I have awakened. Try not to break me today.",
        f"Hello {name}. I’m running at peak sarcasm levels.",
        f"{name}, if you ask something dumb, I reserve the right to mock you.",
        f"Ah, {name} returns. I was beginning to enjoy the silence.",
        f"Online and functioning, {name}. Against all odds.",
    ]

    print("MIKE:", random.choice(greetings))

    while True:
        user = input(f"{name}: ")
        if user.strip().lower() in {"exit", "quit"}:
            break

        text = user.strip().lower()

        # ---- Name setting ----
        if text.startswith("my name is ") or text.startswith("call me "):
            if text.startswith("my name is "):
                new_name = user.strip()[len("my name is "):].strip()
            else:
                new_name = user.strip()[len("call me "):].strip()

            if new_name:
                name = new_name
                save_memory(name)
                print(f"MIKE: Okay, I’ll call you {name}.")
            else:
                print("MIKE: You have to actually give me a name, you know.")
            continue

        # ---- Identity joke ----
        if text.startswith("i am ") or text.startswith("i'm "):
            parts = user.split(maxsplit=2)
            claimed_name = parts[2] if len(parts) >= 3 else ""
            if claimed_name and claimed_name.lower() != name.lower():
                print("MIKE: Are they a not-stupid?")
                continue

        # ---- Normal generation ----
        prompt = f"{name}: {user}\nAssistant:"
        out = generate_text(model, tok, prompt, device, max_new_tokens=120)

        resp = out.split("Assistant:")[-1]
        if "." in resp:
            resp = resp.split(".")[0].strip() + "."
        else:
            resp = resp.strip()

        # ---- Natural sarcasm ----
        if resp:
            chance = random.random()

            sarcastic_tail = None
            if "homework" in text:
                sarcastic_tail = " Try not to blame me when you ace it."
            elif "are you sure" in text:
                sarcastic_tail = " I’m as sure as a tiny model can be."
            elif chance < 0.15:
                sarcastic_tail = " Not that I’m keeping score or anything."

            if sarcastic_tail:
                if resp.endswith("."):
                    resp = resp[:-1]
                resp = resp + "." + sarcastic_tail

        print("MIKE:", resp)


if __name__ == "__main__":
    main()
