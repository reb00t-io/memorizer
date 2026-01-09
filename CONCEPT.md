# In‑Context Learning + LoRA: A Practical Memory Architecture

## Core Idea

Combine **in‑context learning (ICL)** with **periodic LoRA fine‑tuning** to give transformers effective long‑term memory and the ability to learn new behavior over time — without changing the base architecture.

- **ICL** provides fast, flexible adaptation using explicit memory in the context.
- **LoRA** provides slow, persistent adaptation by learning *how to use* that memory.
- The model is trained *with memory present*, so it learns a stable “memory protocol” rather than memorizing facts.

This separates **knowledge (stored in memory)** from **behavior (stored in weights)**.

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

---

## Bottom Line

ICL + LoRA does not give transformers intrinsic memory —
it gives the **system** memory, learning, and adaptation.

This creates a clean separation of concerns:
- Memory stores facts
- Weights store behavior
- Attention is the interface between them
