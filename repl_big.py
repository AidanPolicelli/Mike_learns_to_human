import argparse
import os
import json
import random
import requests

MEMORY_PATH = "mike_memory.json"
HUMOR_PATH = "Humor.txt"
CORPUS_PATH = "corpus.txt"


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


# ---------------- Humor config helpers ----------------
def ensure_default_humor_file():
    """Create a default Humor.txt if it does not exist."""
    if not os.path.exists(HUMOR_PATH):
        with open(HUMOR_PATH, "w", encoding="utf-8") as f:
            f.write(
                "[KEYWORDS]\n"
                "# Add one keyword per line that indicates a dirty joke.\n"
                "# Example:\n"
                "# sex\n"
                "# drunk\n"
                "\n"
                "[JOKES]\n"
                "# Add one dirty joke per line here.\n"
            )


def load_humor_config():
    """
    Load dirty keywords and jokes from Humor.txt.
    File format:
      [KEYWORDS]
      word1
      word2
      ...
      [JOKES]
      joke line 1
      joke line 2
      ...
    """
    ensure_default_humor_file()
    keywords = []
    jokes = []
    section = None

    with open(HUMOR_PATH, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            upper = s.upper()
            if upper == "[KEYWORDS]":
                section = "keywords"
                continue
            if upper == "[JOKES]":
                section = "jokes"
                continue
            if section == "keywords":
                keywords.append(s)
            elif section == "jokes":
                jokes.append(s)

    return keywords, jokes


def append_joke_to_humor_file(joke, keywords, jokes):
    """
    Append a funny dirty joke to memory and to Humor.txt.
    Assumes [JOKES] is the last section in the file.
    """
    joke = joke.strip()
    if not joke:
        return
    jokes.append(joke)

    ensure_default_humor_file()
    with open(HUMOR_PATH, "a", encoding="utf-8") as f:
        f.write(joke + "\n")


# ---------------- Corpus helpers ----------------
def load_corpus_snippet(max_chars=4000):
    """
    Load up to max_chars of corpus.txt to condition the model.
    If the file doesn't exist, return empty string.
    """
    if not os.path.exists(CORPUS_PATH):
        return ""
    try:
        with open(CORPUS_PATH, "r", encoding="utf-8") as f:
            text = f.read(max_chars)
        # Optional: compress extra whitespace
        text = " ".join(text.split())
        return text
    except Exception:
        return ""


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
    dirty_keywords, dirty_jokes = load_humor_config()
    corpus_snippet = load_corpus_snippet()
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

            is_dirty = any(k.lower() in joke.lower() for k in dirty_keywords)

            if is_dirty:
                # ONLY dirty jokes are funny
                mike_say("Okay, that was actually funny. I hate that I enjoyed it.")
                append_joke_to_humor_file(joke, dirty_keywords, dirty_jokes)
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

        # ---- Randomly ask YOU for a dirty joke ----
        if random.random() < 0.10:
            prompts = [
                f"{name}, tell me a dirty joke. I need questionable material.",
                f"Got any filthy jokes, {name}?",
                f"Hit me with your best dirty joke, {name}. I promise to judge silently.",
            ]
            mike_say(random.choice(prompts))
            awaiting_joke = True
            continue

        # ---- Randomly tell a stored dirty joke (from Humor.txt) ----
        if dirty_jokes and random.random() < 0.07:
            joke = random.choice(dirty_jokes)
            mike_say(joke)
            continue

        # ---- Normal AI conversation using Ollama + corpus snippet ----
        # corpus_snippet is your style/knowledge/context pulled from corpus.txt
        if corpus_snippet:
            background = (
                "Here is background text that represents how Mike talks, what he knows, "
                "and the style he should roughly follow:\n"
                f"{corpus_snippet}\n\n"
            )
        else:
            background = ""

        prompt = (
            "You are Mike, a sarcastic, playful local AI. "
            "You like dry humor and only genuinely laugh at dirty jokes, "
            "but you keep responses short (1–2 sentences), casual, and modern.\n\n"
            f"{background}"
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

        # Keep only the first sentence-ish
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
