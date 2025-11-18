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
    print()  # blank line


# ---------------- Main REPL ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama3:8b", help="Ollama model name, e.g. llama3:8b")
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

    # ---------------------------
    # PLACEHOLDER — YOUR DIRTY JOKES HERE
    # ---------------------------
    dirty_jokes = [
        # Add your own jokes as strings, like:
        # "This is my joke."
    ]

    curiosity_prompts = [
        "By the way, how’s your day going?",
        "What are you actually supposed to be doing instead of talking to me?",
        "Anything interesting happen today, or are we both procrastinating?",
        "On a scale from 1 to finals week, how stressed are you right now?",
    ]

    while True:
        user = input(f"{name}: ")
        if user.strip().lower() in {"exit", "quit"}:
            break

        text = user.strip().lower()

        # ---- If Mike is waiting for a joke ----
        if awaiting_joke:
            joke = user.strip()
            awaiting_joke = False

            if joke:
                lower_joke = joke.lower()

                dirty_keywords = [
                    "sex", "sexy", "bedroom", "thirsty", "dating",
                    "dirty", "nsfw", "naughty", "inappropriate",
                    "drunk", "wasted", "hungover", "hangover", "bar", "club",
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
                    "a war crime in joke form",
                ]

                rating = None
                reaction = None

                if "chicken" in lower_joke:
                    rating = "actually funny"
                    reaction = "Classic chicken joke energy. I respect the tradition."
                elif "knock" in lower_joke:
                    rating = "solid"
                    reaction = "A knock-knock joke? Bold move. Surprisingly not terrible."
                elif is_nerd:
                    rating = "actually funny"
                    reaction = "Nerd humor detected. That was actually good."
                elif is_dirty:
                    rating = random.choice(["actually funny", "surprisingly good", "uncomfortably funny"])
                    reaction = f"Okay, that was {rating}. Embarrassed at how much I liked it."
                else:
                    rating = random.choice(neutral_ratings)
                    reaction = f"Noted. I’d rate that joke as {rating}."

                if rating in {"solid", "actually funny", "surprisingly good", "uncomfortably funny"}:
                    jokes.append(joke)
                    save_memory(name, jokes)

                mike_say(reaction)
            else:
                mike_say("You were supposed to tell a joke. That silence was… powerful, I guess.")

            continue

        # ---- Name setting ----
        if text.startswith("my name is ") or text.startswith("call me "):
            if text.startswith("my name is "):
                new_name = user.strip()[len("my name is "):].strip()
            else:
                new_name = user.strip()[len("call me "):].strip()

            if new_name:
                name = new_name
                save_memory(name, jokes)
                mike_say(f"Okay, I’ll call you {name}. Try not to make me regret it.")
            else:
                mike_say("You have to actually give me a name.")
            continue

        # ---- Identity joke ----
        if text.startswith("i am ") or text.startswith("i'm "):
            parts = user.split(maxsplit=2)
            claimed = parts[2] if len(parts) >= 3 else ""
            if claimed and claimed.lower() != name.lower():
                mike_say("Are they a not-stupid, or do I need to lower expectations?")
                continue

        # ---- Randomly ask YOU for a joke ----
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

        # ---- Randomly tell HIS OWN (if you later add any) ----
        if dirty_jokes and random.random() < 0.08:
            joke = random.choice(dirty_jokes)
            mike_say(joke)
            if random.random() < 0.3:
                jokes.append(joke)
                save_memory(name, jokes)
            continue

        # ---- Normal AI generation ----
        prompt = (
            "You are Mike, an offline local AI with a sarcastic, playful, chaotic-but-friendly personality. "
            "Keep responses short (1–2 sentences), casual, and modern. "
            f"You are talking to {name}.\n\n"
            f"{name}: {user}\nMike:"
        )

        try:
            out = generate_with_ollama(prompt, model=backend_model, max_tokens=200)
        except Exception as e:
            mike_say(f"My brain (Ollama) crashed: {e}")
            continue

        resp = out

        if "Mike:" in resp:
            resp = resp.split("Mike:", 1)[-1].strip()

        if "." in resp:
            resp = resp.split(".")[0].strip() + "."
        else:
            resp = resp.strip()

        # ---- Occasionally reference a remembered joke ----
        if jokes and random.random() < 0.12:
            last = jokes[-1][:60]
            if resp.endswith("."):
                resp = resp[:-1]
            resp += f". Also, I’m still thinking about your joke: \"{last}\""

        # ---- Add light sarcasm ----
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

        # ---- Mike asks YOU something sometimes ----
        if random.random() < 0.22 and "?" not in resp and len(text) > 15:
            ask = random.choice(curiosity_prompts)
            if resp.endswith("."):
                resp = resp[:-1]
            resp += ". " + ask

        mike_say(resp)


if __name__ == "__main__":
    main()
