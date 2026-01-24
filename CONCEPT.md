# In‑Context Learning + LoRA: A Practical Memory Architecture

## Core Idea

Use a **model‑managed context** with explicit memory sections, lightweight metadata, and background compression to keep long‑running conversations compact while preserving fidelity at the tail.

- **Context is structured** into fixed sections (system, long‑term, short‑term, recall, working) so the model can reliably “know where to look.”
- **Messages carry timestamps**, and the latest message is prefixed with the current time to anchor recency.
- **Assistant outputs are compressed asynchronously** into a stored `compressed_content` field.
- **Only the tail stays uncompressed** (configurable), while older working messages can be served from their compressed form.

This keeps **behavior in weights** while **facts and evolving state** live in memory — with a stable protocol for retrieval and summarization.

---

## What Improves Over Time

- Better routing of attention to relevant memory
- More efficient use of compressed information
- Learned behaviors, preferences, and response patterns
- Reduced reliance on prompt hacks or brittle instructions

The system does not accumulate facts in weights — it improves at *using* external memory.

---

## Memory Compression Loop (Nightly)

- Daily interactions are logged.
- Long‑term memory is recompressed overnight using the current model.
- Compression is **knowledge‑aware**: only genuinely new or still‑relevant information is retained.
- LoRA training replays interactions **with the compressed memory present**, reinforcing correct memory usage.

Raw data should be retained separately to avoid irreversible loss or feedback loops.

---

## Context Layout (Fixed Order)

All inference uses a stable context layout:

1. **System message**
2. **Long‑term memory**
   - Nightly updated
   - Highly compressed
   - Stable schema
3. **Short‑term memory**
   - Updated during the day
   - Low or no compression
4. **Optional: Recall memory**
   - Retrieved from offline storage
   - Updated every ~50 tokens
   - Inserted only when needed
5. **Working memory**
   - Current thoughts and conversation
   - New tokens are appended here

The fixed structure is critical: LoRA learns *where to look* and *how to interpret* each block.

## Structured Thinking, Workspace, and Mode Control

### Motivation

While ICL + LoRA enables memory and behavioral adaptation, vanilla inference still suffers from two core limitations:
- The model jumps directly into solution mode instead of first understanding intent and ambiguity.
- Thinking is one-shot and front‑loaded, rather than evolving over time as new information appears.

To address this, we introduce an explicit **workspace** and a lightweight **mode control loop** that separates thinking from speaking and allows reasoning, retrieval, and planning to happen intermittently during generation.

---

### Workspace (Evolving State)

The workspace acts as an externalized working memory that is continuously updated.

Minimal stable schema:

WORKSPACE:
- User intent (hypothesis)
- Why the user might be asking
- Confidence in understanding (low / medium / high)
- Current theory of the problem
- Plan
- Open questions / uncertainties
- Next step

The workspace is **not** user-facing. It represents the model’s current beliefs and plan and is revised rather than regenerated.

---

### Modes of Operation

Inference is decomposed into explicit modes:

- INTERPRET: Understand intent, ambiguity, and success criteria
- QUESTION: Ask for clarification when confidence is low
- PLAN: Decide on an approach before execution
- EXECUTE: Produce user-facing output
- REFLECT: Re-check assumptions, detect contradictions, revise plan
- UPDATE: Update the workspace after new information or generated content

At any point, the correct action is to **switch modes**, not necessarily to continue generating output.

---

### Intermittent Retrieval and Thinking

Retrieval and reasoning do not happen only upfront.

Instead:
- Generation is interrupted every N tokens or semantic boundary (e.g., paragraph, decision point).
- The model updates the workspace.
- Retrieval is triggered conditionally:
  - From compressed long‑term memory for stable facts and preferences.
  - From uncompressed memory or external data for detailed or high‑fidelity information.
- The updated workspace and retrieved context are re‑inserted before continuing generation.

This creates an evolving thought process rather than a single-pass answer.

---

### Control Loop (Hackable First, Learned Later)

Initial implementation uses orchestration around an existing model:

1. Generate until a checkpoint.
2. Pause generation.
3. Run a workspace update step.
4. Decide next mode (question / plan / execute / reflect).
5. Perform retrieval if required.
6. Resume generation with updated context.

No architectural changes are required.

Later, the same behavior is reinforced via LoRA fine‑tuning by training on trajectories that include:
- Explicit workspace updates
- Mode switching decisions
- Clarification before answering
- Self-questioning and plan revision

---

### Training Objective (LoRA Phase)

Fine‑tuning does not teach facts.
It teaches **control over cognition**:
- When to think vs speak
- When to ask vs answer
- How to revise beliefs over time
- How to use memory blocks correctly

Training examples are multi‑step trajectories:
user input → workspace update → mode decision → retrieval → revised workspace → output

This aligns training with inference-time control flow.

---

### Resulting System Properties

The integrated system:
- Uses compressed long‑term memory efficiently
- Retrieves high‑resolution data only when needed
- Thinks incrementally rather than upfront
- Revises its understanding and plan over time
- Avoids premature commitment to answers

Attention remains the interface:
- Memory stores information
- Weights store behavior
- The workspace and mode controller determine *when* and *how* attention is applied

## Bottom Line

ICL + LoRA does not give transformers intrinsic memory —
it gives the **system** memory, learning, and adaptation.

This creates a clean separation of concerns:
- Memory stores facts
- Weights store behavior
- Attention is the interface between them
