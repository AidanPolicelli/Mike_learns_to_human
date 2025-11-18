import argparse
import os
import json
import random
import requests

MEMORY_PATH = "mike_memory.json"
DIRTY_JOKES_PATH = "dirty_jokes.txt"


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


# ---------------- Dirty jokes loader ----------------
def load_dirty_jokes():
    if not os.path.exists(DIRTY_JOKES_PATH):
        # Create file if missing
        with open(DIRTY_JOKES_PATH, "w", encoding="utf-8") as f:
            f.write("")  
        return []

    try:
        with open(DIRTY_JOKES_PATH, "r", encoding="utf-8") as f:
            jokes = [line.strip() for line in f.readlines() if line.strip()]
        return jokes
    except Exception:
        return []


# ---------------- Ollama backend ----------------
def generate_with_ollama(prompt: str, model: str = "llama3:8b", max_tokens: int = 200) -> str:
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


# ---------------- Printing helper ----------------
def mike_say(message: str):
    print(f"MIKE: {message}")
    print()


# ---------------- Main REPL ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama3:8b")
    args = ap.parse_args()

    backend_model = args.model

    name, jokes = load_memory()
    awaiting_joke = False

    print("Offline REPL (Big Mike). Type 'exit' to quit.\n")

    greetings = [
        f"Hey {name}, systems online. Did you miss me?",
        f"Greetings, {name}. I am booted, confused, and ready.",
        f"{name}, I have awakened. Try not to break me today.",
        f"Hello {name}. I’m running at peak sarcasm levels.",
        f"{name}, if you ask something dumb, I reserve the right to mock you.",
        f"Ah, {name} returns. I was beginning to enjoy the silence.",
        f"Online and functioning, {name}. Against all odds.",
    ]
    mike_say(random.choice(greetings))

    # Load your personal dirty jokes from file
    dirty_jokes = load_dirty_jokes()

    curiosity_prompts = [
        "By the way, how’s your day going?",
        "What are you supposed to be doing instead of talking to me?",
        "Anything interesting happen today, or are we both procrastinating?",
        "On a scale from 1 to finals week, how stressed are you right now?",
    ]

    while True:
        user = input(f"{name}: ")
        if user.strip().lower() in {"exit", "quit"}:
            break

        text = user.strip().lower()

        # ---- Joke response logic ----
        if awaiting_joke:
            joke = user.strip()
            awaiting_joke = False

            if joke:
                lower_joke = joke.lower()

                dirty_keywords = [
                    "sex", "sexy", "bedroom", "dating",
                    "dirty", "nsfw", "naughty", "inappropriate",
                    "drunk", "wasted", "hungover", "hangover",
                    "vodka", "tequila", "beer", "shot", "shots",
                ]
                is_dirty = any(k in lower_joke for k in dirty_keywords)
                is_nerd = any(k in lower_joke for k in ["physics", "math", "engineer", "voltage", "circuit"])

                neutral_ratings = [
                    "solid",
                    "actually funny",
                    "surprisingly good",
                    "kind of mid",
                    "questionable",
                    "uncomfortably funny",
                ]

                rating = None
                reaction = None

                if "chicken" in lower_joke:
                    rating = "actually funny"
                    reaction = "Classic chicken joke energy. Respect."
                elif "knock" in lower_joke:
                    rating = "solid"
                    reaction = "A knock-knock joke? Bold. Surprisingly not awful."
                elif is_nerd:
                    rating = "actually funny"
                    reaction = "Nerd humor detected. That one actually slapped."
                elif is_dirty:
                    rating = "surprisingly good"
                    reaction = "That was… unexpectedly good. I’m concerned."
                else:
                    rating = random.choice(neutral_ratings)
                    reaction = f"Noted. I’d rate that as {rating}."

                if rating in {"solid", "actually funny", "surprisingly good", "uncomfortably funny"}:
                    jokes.append(joke)
                    save_memory(name, jokes)

                mike_say(reaction)
            else:
                mike_say("You were supposed to tell a joke. That silence was… powerful.")

            continue

        # ---- Name setting ----
        if text.startswith("my name is ") or text.startswith("call me "):
            new_name = user.split(" ", 3)[-1].strip()
            if new_name:
                name = new_name
                save_memory(name, jokes)
                mike_say(f"Got it. You’re {name} now. Fancy.")
            else:
                mike_say("You have to actually give me a name.")
            continue

        # ---- Identity joke ----
        if text.startswith("i am ") or text.startswith("i'm "):
            claimed = user.split(" ", 2)[-1].strip()
            if claimed.lower() != name.lower():
                mike_say("Are they a not-stupid, or should I lower expectations?")
                continue

        # ---- Mike asks YOU for a joke ----
        if random.random() < 0.10:
            prompts = [
                f"{name}, tell me a joke. I need new material.",
                f"Got any fresh jokes, {name}?",
                f"Give me your best joke, {name}. I’ll judge silently.",
                f"Hit me with a joke, {name}. I can take it.",
            ]
            mike_say(random.choice(prompts))
            awaiting_joke = True
            continue

        # ---- Mike tells HIS joke (from your file) ----
        if dirty_jokes and random.random() < 0.08:
            joke = random.choice(dirty_jokes)
            mike_say(joke)
            continue

        # ---- Normal AI output ----
        prompt = (
            "You are Mike, a sarcastic, playful local AI with a dry sense of humor. "
            "Keep responses short (1–2 sentences), modern, casual, and witty. "
            f"You are talking to {name}.\n\n"
            f"{name}: {user}\nMike:"
        )

        try:
            out = generate_with_ollama(prompt, model=backend_model, max_tokens=200)
        except Exception as e:
            mike_say(f"Ollama flopped: {e}")
            continue

        resp = out

        if "Mike:" in resp:
            resp = resp.split("Mike:", 1)[-1].strip()

        if "." in resp:
            resp = resp.split(".", 1)[0].strip() + "."
        else:
            resp = resp.strip()

        # ---- Reference stored jokes ----
        if jokes and random.random() < 0.12:
            last = jokes[-1][:60]
            if resp.endswith("."):
                resp = resp[:-1]
            resp += f". Also, still thinking about your joke: \"{last}\""

        # ---- Light sarcasm ----
        chance = random.random()
        tail = None
        if "homework" in text:
            tail = " Try not to blame me if it goes wrong."
        elif "are you sure" in text:
            tail = " As sure as a local model can be, which is… questionable."
        elif chance < 0.18:
            tail = random.choice([
                " Not that I’m keeping score.",
                " But hey, what do I know?",
                " Just another day being smarter than my power supply.",
            ])

        if tail:
            if resp.endswith("."):
                resp = resp[:-1]
            resp += "." + tail

        # ---- Ask YOU something occasionally ----
        if random.random() < 0.22 and "?" not in resp and len(text) > 15:
            ask = random.choice(curiosity_prompts)
            if resp.endswith("."):
                resp = resp[:-1]
            resp += ". " + ask

        mike_say(resp)


if __name__ == "__main__":
    main()
