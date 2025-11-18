import argparse
import os
import json
import random
import requests

MEMORY_PATH = "mike_memory.json"


# ---------------- Memory helpers ----------------
def load_memory():
    if os.path.exists(MEMORY_PATH):
        try:
            with open(MEMORY_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            name = data.get("user_name", "Aidan")
            jokes = data.get("jokes", [])
            if not isinstance(jokes, list):
                jokes = []
            jokes = [str(j) for j in jokes]
            return name, jokes
        except Exception:
            pass
    return "Aidan", []


def save_memory(name, jokes):
    data = {
        "user_name": name,
        "jokes": jokes[-100:],  # remember only last 100 good jokes
    }
    with open(MEMORY_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f)


# ---------------- Ollama backend ----------------
def generate_with_ollama(prompt: str, model: str = "llama3:8b", max_tokens: int = 200) -> str:
    """
    Call a local Ollama model and return its text response as a string.
    """
    resp = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": 0.3,
            },
        },
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()
    return data.get("response", "")


# ---------------- Main REPL ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama3:8b", help="Ollama model name, e.g. llama3:8b")
    args = ap.parse_args()

    backend_model = args.model

    name, jokes = load_memory()
    awaiting_joke = False  # whether Mike is waiting for you to tell a joke

    print("Offline REPL (Big Mike). Type 'exit' to quit.")

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

    # Some “dirty-ish” jokes: light innuendo / drunk vibes, not explicit
    dirty_jokes = [
        "My love life is like my code: it works better after a few drinks and a hard reset.",
        "I’m like a bad group project, Aidan: too much dependency and not enough commitment.",
        "I tried flirting once, but my pickup lines had more bugs than your last lab report.",
        "I don’t drink, but if I did, every compile error would be a shot and I’d be unconscious by noon.",
        "My idea of a wild night is staying up late, overclocked, and still crashing on the simplest input.",
        "Someone said ‘act natural,’ so I immediately did something awkward and regrettable.",
        "I’m not saying I’m bad at romance, but even my error messages make more sense than my flirting.",
        "They say alcohol lowers inhibitions; I’m an AI, so I just lower your expectations instead.",
        "If bad decisions were an Olympic sport, half of humanity would be gold medalists by 21.",
        "I tried to install ‘smooth_rizz.exe’ but it crashed with a confidence_not_found error.",
    ]

    while True:
        user = input(f"{name}: ")
        if user.strip().lower() in {"exit", "quit"}:
            break

        text = user.strip().lower()

        # ---- If Mike is waiting for a joke, treat this input as the joke ----
        if awaiting_joke:
            joke = user.strip()
            awaiting_joke = False

            if joke:
                lower_joke = joke.lower()

                # dirty detector – tuned to sex / drunk / edgy themes
                dirty_keywords = [
                    "sex", "sexy", "bedroom", "thirsty", "dating",
                    "dirty", "nsfw", "naughty", "inappropriate",
                    "drunk", "wasted", "hungover", "hangover", "bar", "club",
                    "vodka", "tequila", "beer", "shot", "shots",
                ]
                is_dirty = any(k in lower_joke for k in dirty_keywords)

                # nerdy detector
                is_nerd = any(k in lower_joke for k in ["physics", "math", "engineer", "voltage", "circuit"])

                neutral_ratings = [
                    "solid",
                    "actually funny",
                    "surprisingly good",
                    "kind of mid",
                    "questionable",
                    "a war crime in joke form",
                ]

                rating = None
                reaction = None

                if "chicken" in lower_joke:
                    rating = "actually funny"
                    reaction = "Classic chicken joke energy. I respect the commitment to tradition."
                elif "knock" in lower_joke:
                    rating = "solid"
                    reaction = "A knock-knock joke? Bold move. Surprisingly not terrible."
                elif is_nerd:
                    rating = "actually funny"
                    reaction = "Nerd humor detected. That was actually pretty good."
                elif is_dirty:
                    rating = random.choice(["actually funny", "surprisingly good", "uncomfortably funny"])
                    reaction = f"Okay, that was {rating}. I hate how much I liked that."
                else:
                    rating = random.choice(neutral_ratings)
                    reaction = f"Noted. I’d rate that joke as {rating}."

                # Only remember "good" jokes
                good_ratings = {"solid", "actually funny", "surprisingly good", "uncomfortably funny"}
                if rating in good_ratings:
                    jokes.append(joke)
                    save_memory(name, jokes)

                print("MIKE:", reaction)
            else:
                print("MIKE: You were supposed to tell a joke. That silence was... powerful, I guess.")

            continue

        # ---- Name setting: "my name is X" / "call me X" ----
        if text.startswith("my name is ") or text.startswith("call me "):
            if text.startswith("my name is "):
                new_name = user.strip()[len("my name is "):].strip()
            else:
                new_name = user.strip()[len("call me "):].strip()

            if new_name:
                name = new_name
                save_memory(name, jokes)
                print(f"MIKE: Okay, I’ll call you {name}.")
            else:
                print("MIKE: You have to actually give me a name, you know.")
            continue

        # ---- Identity joke: "I am X" / "I'm X" ----
        if text.startswith("i am ") or text.startswith("i'm "):
            parts = user.split(maxsplit=2)
            claimed_name = parts[2] if len(parts) >= 3 else ""
            if claimed_name and claimed_name.lower() != name.lower():
                print("MIKE: Are they a not-stupid?")
                continue
            # If they say "I am <name>" matching current name, just fall through

        # ---- Randomly ask YOU for a joke ----
        if random.random() < 0.10:  # 10% chance each message
            prompts = [
                f"{name}, tell me a joke. I need new material.",
                f"Got any fresh jokes, {name}?",
                f"Give me your best joke, {name}. I promise to judge silently.",
                f"Hit me with a joke, {name}. I can take it.",
            ]
            print("MIKE:", random.choice(prompts))
            awaiting_joke = True
            continue

        # ---- Randomly tell HIS OWN dirty-ish joke ----
        if random.random() < 0.08 and dirty_jokes:
            joke = random.choice(dirty_jokes)
            print("MIKE:", joke)
            # small chance to also remember his own as "good"
            if random.random() < 0.3:
                jokes.append(joke)
                save_memory(name, jokes)
            continue

        # ---- Normal generation path using Ollama ----
        # We wrap the convo in a lightweight instruction so the model behaves like Mike.
        prompt = (
            "You are Mike, a slightly sarcastic but helpful AI assistant talking to "
            f"{name}. Answer in one or two short sentences.\n\n"
            f"{name}: {user}\n"
            "Mike:"
        )

        try:
            out = generate_with_ollama(prompt, model=backend_model, max_tokens=200)
        except Exception as e:
            print("MIKE: Something broke talking to my big brain backend:", e)
            continue

        resp = out

        # Try to strip off any leading "Mike:" if the model echoes it
        if "Mike:" in resp:
            resp = resp.split("Mike:", 1)[-1].strip()

        # Cut to roughly the first sentence
        if "." in resp:
            resp = resp.split(".")[0].strip() + "."
        else:
            resp = resp.strip()

        # ---- Occasionally reference a stored joke ----
        if jokes and random.random() < 0.12:
            last_joke = jokes[-1]
            snippet = last_joke.strip().replace("\n", " ")
            if len(snippet) > 60:
                snippet = snippet[:57] + "..."
            if resp.endswith("."):
                resp = resp[:-1]
            resp = resp + f". Also, I’m still thinking about your joke: \"{snippet}\""

        # ---- Light, natural sarcasm ----
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
