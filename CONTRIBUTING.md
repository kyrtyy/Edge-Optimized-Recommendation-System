# Contributing

Contributions are welcome — bug fixes, improved documentation, or new experiments.

## Setup

```bash
git clone https://github.com/<your-username>/otto-recommender.git
cd otto-recommender
python -m venv .venv && source .venv/bin/activate
pip install torch faiss-cpu onnx onnxruntime numpy pandas tqdm pytest ruff
```

## Before opening a PR

1. **Lint** — `ruff check .`
2. **Tests** — `pytest tests/ -v`
3. Keep commits focused: one logical change per commit
4. Update the README if you change public interfaces

## Reporting issues

Open a GitHub Issue with: Python version, OS, error traceback, and the minimal code to reproduce.
