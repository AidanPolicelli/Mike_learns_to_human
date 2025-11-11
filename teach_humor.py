import argparse, os, json

# Keep it isolated
import offline_mode  # noqa: F401

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--memory", default="checkpoints/humor_memory.json")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.memory) or ".", exist_ok=True)
    if os.path.exists(args.memory):
        mem = json.load(open(args.memory, "r", encoding="utf-8"))
    else:
        mem = {"examples": []}

    print("Teach humor: provide one GOOD joke and one BAD joke. Type 'exit' to stop.")
    while True:
        good = input("Good joke: ")
        if good.strip().lower() in {"exit", "quit"}: break
        why_good = input("Why is it good? ")
        bad = input("Bad joke: ")
        if bad.strip().lower() in {"exit", "quit"}: break
        why_bad = input("Why is it bad? ")

        mem["examples"].append({
            "good": good.strip(),
            "why_good": why_good.strip(),
            "bad": bad.strip(),
            "why_bad": why_bad.strip()
        })
        json.dump(mem, open(args.memory, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print(f"Saved example. Total examples: {len(mem['examples'])}")

if __name__ == "__main__":
    main()
