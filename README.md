# mike_offline_llm

An *isolated*, locally-trainable, **no‑internet** tiny language model inspired by the playful spirit of “MIKE.”  
It’s focused on **logical, step‑by‑step responses** and a **childlike curiosity**.  
You can train it on your own text, keep it offline, and export to **ONNX** for compiler toolchains.

## Features
- **No internet by default** — `offline_mode.py` blocks sockets and HTTP.
- **Character-level tokenizer** (simple & robust offline).
- **Tiny decoder-only Transformer** in PyTorch.
- **Training** on your local `.txt` corpus.
- **Interactive REPL** (`repl.py`) with a MIKE-like persona focused on logic and curiosity.
- **Humor learning loop** (`teach_humor.py`) where you provide *good* and *bad* jokes; stored as few-shot hints.
- **ONNX export** (`export_onnx.py`) for isolated/embedded compiler flows.
- MIT license. Minimal deps: `torch`, `numpy`, `pyyaml`

> ⚠️ This is an educational starter. It’s small and simple by design, not SOTA. Grow it as you like.

## Quick start (no internet)
1. Create and activate a Python 3.10+ venv.
2. Install deps from a local wheelhouse if you can, or `pip install torch numpy pyyaml` (once, while you still have access).
3. Put your training text files under `data/` (e.g., `data/corpus.txt`).  
4. Train:
   ```bash
   python train.py --data data/corpus.txt --epochs 2 --batch_size 64 --seq_len 256
   ```
5. Chat offline (REPL):
   ```bash
   python repl.py --model checkpoints/mike.pt --vocab checkpoints/vocab.json
   ```
6. Teach humor (optional, improves few-shot prompts):
   ```bash
   python teach_humor.py --memory checkpoints/humor_memory.json
   ```
7. Export to ONNX for isolated compilers:
   ```bash
   python export_onnx.py --model checkpoints/mike.pt --vocab checkpoints/vocab.json --out mike.onnx
   ```

## Philosophy
- **Logic-first**: Prompts nudge the model to show steps and stay grounded.
- **Curiosity**: Initial turns invite you to share jokes and feedback.
- **Isolation**: `offline_mode.py` disables networking so evaluation is hermetic.

## Repo layout
```
mike_offline_llm/
├─ offline_mode.py
├─ tokenizer.py
├─ model.py
├─ train.py
├─ repl.py
├─ teach_humor.py
├─ export_onnx.py
├─ persona.md
├─ config.yaml
├─ data/
│  └─ corpus.txt            # (placeholder)
├─ checkpoints/             # saved model/vocab/memory
└─ LICENSE
```

## GitHub
You can initialize a local git repo and push to GitHub:
```bash
git init
git add .
git commit -m "Initial offline LLM (MIKE-like)"
git branch -M main
git remote add origin <your-github-repo-url>
git push -u origin main
```

Have fun, and be kind to your baby MIKE. :)
