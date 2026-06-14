# Memorizer

A lightweight Python library that provides a structured, long‑term memory architecture for LLMs. It combines in‑context learning with periodic LoRA fine‑tuning, keeping facts, preferences, and behaviours in persistent memory while leaving the base model unchanged.

## Install

```bash
# As a library dependency (public repo)
pip install "git+https://github.com/reb00t-io/memorizer.git"

# Pin to a released tag
pip install "memorizer @ git+https://github.com/reb00t-io/memorizer.git@v1.0.0"
```

Only `requests` is pulled in for library use. The interactive chat CLI needs the
optional `chat` extra (`prompt-toolkit`):

```bash
pip install "memorizer[chat] @ git+https://github.com/reb00t-io/memorizer.git"
```

## Use as a library

The `Model` class can be driven directly from another project — the endpoint,
model id, system prompt and completion-token budget are all supplied at startup:

```python
from memorizer import Model

model = Model.create(
    model_id="gpt-oss-120b",
    base_url="http://host:8080/v1",
    api_key="...",                       # defaults to "dummy" for local servers
    system_prompt="You are <MODEL_ID>.", # <MODEL_ID> is substituted at startup
    max_completion_tokens=1500,          # required — no default
)

model.context.append("user", "Hello!")
text, tool_calls = model.stream_and_process()
print(text)
```

`Model.create()` builds the backing `Context`; pass an existing `Context` to
`Model(...)` directly if you manage it yourself. Memory is persisted under
`~/.memorizer/` by default (`data_dir=` / `persist=False` to change).

## Run the chat CLI

```bash
git clone git@github.com:reb00t-io/memorizer.git
cd memorizer
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[chat]"

memorizer-chat            # or: python -m memorizer.chat.chat
```

## Core Concepts
- **Fixed context layout**: system, long‑term, short‑term, recall (optional), working.
- **Context class** (`memorizer/model/context.py`): manages memory sections, appends messages, compresses short‑term + working into long‑term.
- **Compression** uses a summarisation LLM to create concise long‑term updates.

## Documentation
- Detailed project description: `CLAUDE.md`
- Tests: `pytest tests`
