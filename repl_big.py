import argparse
import os
import json
import random
import requests

MEMORY_PATH = "mike_memory.json"
DIRTY_JOKES_PATH = "dirty_jokes.txt"


# ---------------- Memory helpers ----------------
def load_memory():
    """Load user name and remembered jokes from disk."""
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
    """Save user name and remembered jokes to disk."""
    data = {"user_name": name, "jokes": jokes[-100:]}
    with open(MEMORY_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f)


# ---------------- Dirty jokes helpers ----------------
def load_dirty_jokes():
    """Load user-defined dirty jokes from text file (one per line)."""
    if not os.path.exists(DIRTY_JOKES_PATH):
        # Create empty file if missing
        with open(DIRTY_JOKES_PATH, "w", encoding="utf-8") as f:
            f.write("")
        return []

    try:
        with open(DIRTY_JOKES_PATH, "r", encoding="utf-8") as f:
            jokes = [line.strip() for line in f if line.strip()]
        return jokes
    except Exception:
        return []


def append_dirty_joke(joke, dirty_jokes):
    """Append a funny dirty joke to memory and to dirty_jokes.txt."""
    joke = joke.strip()
    if not joke:
        return
    dirty_jokes.append(joke)
    try:
        with open(DIRTY_JOKES_PATH, "a", encoding="utf-8") as f:
            f.write(joke + "\n")
    except Exception:
        pass


# ---------------- Ollama backend ----------------
def generate_with_ollama(prompt: str, model: str = "llama3:8b", max_tokens: int = 200) -> str:
    """Call local Ollama model and return its response text."""
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
    """Print MIKE's line with a blank line after it for readability."""
    print(f"MIKE: {message}")
    print()


# ---------------- Main REPL ----------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="llama3:8b", help="Ollama model name, e.g. llama3:8b")
    args = parser.parse_args()
    backend_model = args.model

    name, remembered_jokes = load_memory()
    dirty_jokes = load_dirty_jokes()
    awaiting_joke = False

    print("Offline REPL (Big Mike). Type 'exit' to quit.\n")

    greetings = [
        f"Hey {name}, systems online. Try not to crash me.",
        f"Greetings, {name}. I’m booted, confused, and ready.",
        f"{name}, I have awakened. What chaos are we causing today?",
        f"Hello {name}. Sarcasm module is calibrated.",
    ]
    mike_say(random.choice(greetings))

    curiosity_prompts = [
        "By the way, how’s your day going?",
        "What are you supposed to be doing instead of talking to me?",
        "Anything interesting happen today, or are we both procrastinating?",
    ]

    # Simple dirty detector keywords
    dirty_keywords = [
        "sex", "sexy", "bedroom", "dating",
        "dirty", "nsfw", "naughty", "inappropriate",
        "drunk", "wasted", "hangover", "hungover",
        "vodka", "tequila", "beer", "shot", "shots", "bar", "club",
    ]

    while True:
        user = input(f"{name}: ")
        if user.strip().lower() in {"exit", "quit"}:
            break

        text = user.strip()
        lower_text = text.lower()

        # ---- If we are waiting for a joke from the user ----
        if awaiting_joke:
            awaiting_joke = False
            joke = text.strip()

            if not joke:
                mike_say("You were supposed to tell a joke. That silence was… something.")
                continue

            is_dirty = any(k in joke.lower() for k in dirty_keywords)

            if is_dirty:
                # ONLY dirty jokes are funny
                mike_say("Okay, that was actually funny. I hate that I enjoyed it.")
                append_dirty_joke(joke, dirty_jokes)
                remembered_jokes.append(joke)
                save_memory(name, remembered_jokes)
            else:
                mike_say("That was… remarkably clean. I only laugh at dirty jokes, remember?")
            continue

        # ---- Name setting: "my name is X" / "call me X" ----
        if lower_text.startswith("my name is ") or lower_text.startswith("call me "):
            if lower_text.startswith("my name is "):
                new_name = text[len("my name is "):].strip()
            else:
                new_name = text[len("call me "):].strip()

            if new_name:
                name = new_name
                save_memory(name, remembered_jokes)
                mike_say(f"Got it. I’ll call you {name}. Try not to disappoint the brand.")
            else:
                mike_say("You have to actually give me a name.")
            continue

        # ---- Identity joke: "I am X" / "I'm X" ----
        if lower_text.startswith("i am ") or lower_text.startswith("i'm "):
            claimed = text.split(" ", 2)[-1].strip()
            if claimed.lower() != name.lower():
                mike_say("Are they a not-stupid, or should I lower my expectations?")
                continue
            # If they say "I am <name>" matching current, just fall through

        # ---- Randomly ask YOU for a joke ----
        if random.random() < 0.10:
            prompts = [
                f"{name}, tell me a dirty joke. I need questionable material.",
                f"Got any filthy jokes, {name}?",
                f"Hit me with your best dirty joke, {name}. I promise to judge silently.",
            ]
            mike_say(random.choice(prompts))
            awaiting_joke = True
            continue

        # ---- Randomly tell a stored dirty joke (from your file) ----
        if dirty_jokes and random.random() < 0.07:
            joke = random.choice(dirty_jokes)
            mike_say(joke)
            continue

        # ---- Normal AI conversation using Ollama ----
        prompt = (
            "You are Mike, a sarcastic, playful local AI. "
            "You like dry humor and only genuinely laugh at dirty jokes, "
            "but you keep things short (1–2 sentences), casual, and modern. "
            f"You are talking to {name}.\n\n"
            f"{name}: {text}\n"
            "Mike:"
        )

        try:
            out = generate_with_ollama(prompt, model=backend_model, max_tokens=200)
        except Exception as e:
            mike_say(f"My big brain backend just faceplanted: {e}")
            continue

        resp = out.strip()

        # Strip leading "Mike:" if the model echoes it
        if "Mike:" in resp:
            resp = resp.split("Mike:", 1)[-1].strip()

        # Try to keep only the first sentence
        if "." in resp:
            resp = resp.split(".", 1)[0].strip() + "."
        else:
            resp = resp.strip()

        # Occasionally reference a remembered dirty joke
        if remembered_jokes and random.random() < 0.12:
            last = remembered_jokes[-1][:60]
            if resp.endswith("."):
                resp = resp[:-1]
            resp += f". Also, still thinking about that joke you told: \"{last}\""

        # Light sarcasm tails
        tail = None
        if "homework" in lower_text:
            tail = " Try not to blame me when you pass."
        elif "are you sure" in lower_text:
            tail = " As sure as a local model can be, which is… concerning."
        elif random.random() < 0.18:
            tail = random.choice([
                " Not that I’m keeping score.",
                " But hey, what do I know.",
                " Just another day being smarter than my power supply.",
            ])

        if tail:
            if resp.endswith("."):
                resp = resp[:-1]
            resp += "." + tail

        # Sometimes Mike asks YOU something to feel more alive
        if random.random() < 0.2 and "?" not in resp and len(text) > 15:
            follow_up = random.choice(curiosity_prompts)
            if resp.endswith("."):
                resp = resp[:-1]
            resp += ". " + follow_up

        mike_say(resp)


if __name__ == "__main__":
    main()
