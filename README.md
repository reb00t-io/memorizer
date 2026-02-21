# Memorizer

A lightweight Python library that provides a structured, long‑term memory architecture for LLMs. It combines in‑context learning with periodic LoRA fine‑tuning, keeping facts, preferences, and behaviours in persistent memory while leaving the base model unchanged.

## Quick Start
```bash
# Clone and install
git clone git@github.com:reb00t-io/memorizer.git
cd memorizer
python3 -m venv .venv && source .venv/bin/activate
pip install .

# Run interactive chat
python -m src.chat.chat
```

## Core Concepts
- **Fixed context layout**: system, long‑term, short‑term, recall (optional), working.
- **Context class** (`src/model/context.py`): manages memory sections, appends messages, compresses short‑term + working into long‑term.
- **Compression** uses a summarisation LLM to create concise long‑term updates.

## Documentation
- Detailed project description: `CLAUDE.md`
- Tests: `pytest src`
