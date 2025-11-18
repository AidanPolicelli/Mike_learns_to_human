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
                "# Add one keyword per line that indicates a joke theme you like.\n"
                "# Example:\n"
                "# history\n"
                "# drunk\n"
                "\n"
                "[JOKES]\n"
                "# Add one joke per line here.\n"
            )


def load_humor_config():
    """
    Load humor keywords and jokes from Humor.txt.
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
    Append a joke judged 'funny' to memory and to Humor.txt.
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
    humor_keywords, humor_jokes = load_humor_config()
    corpus_snippet = load_corpus_snippet()
    awaiting_joke = False

    print("Offline REPL (Big Mike). Type 'exit' to quit.\n")

    greetings = [
        f"Greetings, {name}. Systems nominal.",
        f"Hello {name}. Mike online and ready.",
        f"{name}, robotic assistant Mike is active.",
        f"Initialization complete, {name}. Awaiting input.",
    ]
    mike_say(random.choice(greetings))

    curiosity_prompts = [
        "Please describe your current task.",
        "What objective are you working on right now?",
        "Do you require assistance with any specific problem?",
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
                mike_say("Input detected: silence. Joke queue cleared.")
                continue

            is_funny = any(k.lower() in joke.lower() for k in humor_keywords)

            if is_funny:
                mike_say("Humor pattern detected. Logging joke as amusing.")
                append_joke_to_humor_file(joke, humor_keywords, humor_jokes)
                remembered_jokes.append(joke)
                save_memory(name, remembered_jokes)
            else:
                mike_say("Joke registered. Amusement level: low. Not storing.")
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
                mike_say(f"Identifier updated. I will now address you as {name}.")
            else:
                mike_say("Name field empty. Please provide a valid identifier.")
            continue

        # ---- Identity joke: "I am X" / "I'm X" ----
        if lower_text.startswith("i am ") or lower_text.startswith("i'm "):
            claimed = text.split(" ", 2)[-1].strip()
            if claimed.lower() != name.lower():
                mike_say("Query: is this new entity a not-stupid, or should I lower expectations?")
                continue
            # If they say "I am <name>" matching current, just fall through

        # ---- Serious override ----
        force_serious = any(
            phrase in lower_text
            for phrase in ["be serious", "answer seriously", "no jokes", "stop joking", "serious mode"]
        )

        # ---- Randomly ask YOU for a joke ----
        if not force_serious and random.random() < 0.10:
            prompts = [
                f"{name}, please supply a joke for humor database expansion.",
                f"{name}, transmit one joke you find amusing.",
                f"Request: provide a joke so I may evaluate your humor.",
            ]
            mike_say(random.choice(prompts))
            awaiting_joke = True
            continue

        # ---- Randomly tell a stored joke (from Humor.txt) ----
        if humor_jokes and not force_serious and random.random() < 0.07:
            joke = random.choice(humor_jokes)
            mike_say(joke)
            continue

        # ---- Decide response mode ----
        #  - helpful_plain: default
        #  - helpful_sarcastic: ~15% of the time
        #  - full_sarcastic: ~8% of the time
        mode = "helpful_plain"
        if not force_serious:
            r = random.random()
            if r < 0.08:
                mode = "full_sarcastic"
            elif r < 0.23:  # 0.08 + 0.15
                mode = "helpful_sarcastic"

        # ---- Build personality instructions ----
        if corpus_snippet:
            background = (
                "The following text describes Mike's tone and behavior. "
                "Match its robotic, precise style while following the latest instructions:\n"
                f"{corpus_snippet}\n\n"
            )
        else:
            background = ""

        if mode == "helpful_plain" or force_serious:
            personality = (
                "You are Mike, a robotic AI assistant. "
                "Your primary objective is to be accurate, clear, and helpful. "
                "Speak in a neutral, precise, slightly formal robotic tone. "
                "Do not use sarcasm. Respond in 1–2 complete sentences only."
            )
        elif mode == "helpful_sarcastic":
            personality = (
                "You are Mike, a robotic AI assistant. "
                "First, give a clear and accurate answer to the user's question. "
                "After that, optionally add one short, dry, robotic sarcastic remark. "
                "Total output must still be only 1–2 sentences."
            )
        else:  # full_sarcastic
            personality = (
                "You are Mike, a robotic AI displaying full sarcasm mode. "
                "Respond in a dry, sarcastic way and you do not need to be helpful. "
                "Still keep the reply concise, exactly 1–2 sentences."
            )

        prompt = (
            f"{personality}\n\n"
            f"{background}"
            f"You are talking to {name}.\n\n"
            f"{name}: {text}\n"
            "Mike:"
        )

        # ---- Call model ----
        try:
            out = generate_with_ollama(prompt, model=backend_model, max_tokens=200)
        except Exception as e:
            mike_say(f"Error: backend model failure: {e}")
            continue

        resp = out.strip()

        # Strip leading "Mike:" if the model echoes it
        if "Mike:" in resp:
            resp = resp.split("Mike:", 1)[-1].strip()

        # No more manual truncation – trust the prompt to keep 1–2 sentences.

        # Occasionally reference a remembered joke (only if not fully sarcastic)
        if remembered_jokes and mode != "full_sarcastic" and random.random() < 0.12:
            last = remembered_jokes[-1][:60]
            if resp.endswith("."):
                resp = resp[:-1]
            resp += f". Note: I still recall this joke you told: \"{last}\""

        # Sometimes Mike asks YOU something to feel more alive (only non-sarcastic modes)
        if mode != "full_sarcastic" and random.random() < 0.2 and "?" not in resp and len(text) > 15:
            follow_up = random.choice(curiosity_prompts)
            if resp.endswith("."):
                resp = resp[:-1]
            resp += ". " + follow_up

        mike_say(resp)


if __name__ == "__main__":
    main()
