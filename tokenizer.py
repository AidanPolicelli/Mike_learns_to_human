import json
from typing import List, Dict

class CharTokenizer:
    """
    Simple character-level tokenizer. Robust offline.
    """
    def __init__(self, vocab: Dict[str, int] | None = None):
        self.stoi = vocab or {}
        self.itos = {i: s for s, i in self.stoi.items()}

    @staticmethod
    def build_from_text(text: str):
        chars = sorted(list(set(text)))
        stoi = {ch: i for i, ch in enumerate(chars)}
        return CharTokenizer(stoi)

    def encode(self, s: str) -> List[int]:
        return [self.stoi[c] for c in s if c in self.stoi]

    def decode(self, ids: List[int]) -> str:
        return ''.join(self.itos[i] for i in ids if i in self.itos)

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.stoi, f, ensure_ascii=False, indent=2)

    @staticmethod
    def load(path: str):
        with open(path, "r", encoding="utf-8") as f:
            vocab = json.load(f)
        return CharTokenizer(vocab)
