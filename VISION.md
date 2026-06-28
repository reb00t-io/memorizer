# Vision: Memorizer as Scaled In-Context Learning

## The Goal in One Sentence

Memorizer becomes a **shared, self-maintaining knowledge substrate** that lets a
fleet of LLM agents learn continuously from their interactions — compressing what
they learn, forgetting what goes stale, retrieving the rest on demand, and sharing
an organization-wide memory that rides in the prompt prefix nearly for free.

It stays a drop-in **chat-completions** layer: the host application talks to an
ordinary completions endpoint, and everything — long-term memory, retrieved
context, org knowledge — is injected behind that interface.

This document describes the *final* state. It builds on the compression loop and
fixed context layout in [CONCEPT.md](CONCEPT.md); read that first for the
ICL + LoRA control loop. Here we describe what changes when memory is scaled out
across time and across agents.

---

## The Shape of the System

```
                          ┌─────────────────────────────┐
   Host app  ───chat───▶  │          Memorizer          │  ───chat───▶  LLM
   (its own system        │  injects memory + recall   │     (Kimi / local /
    prompt merged in)     └──────────────┬──────────────┘      any OpenAI API)
                                         │
                    ┌────────────────────┼───────────────────────┐
                    ▼                    ▼                       ▼
            ┌───────────────┐    ┌────────────────┐     ┌─────────────────┐
            │  Org memory   │    │  Agent memory  │     │   Qdrant store  │
            │ (read-mostly, │    │ (per-agent LTM,│     │ semantic + BM25 │
            │  prompt-cached│    │  goals, facts) │     │  + rerank, with │
            │  prefix)      │    │                │     │   provenance)   │
            └───────────────┘    └────────────────┘     └─────────────────┘
```

Two stores, one retrieval engine:

- **Agent memory** — what a single agent has learned and consolidated (the
  existing episodic / factual / goal / workspace layers).
- **Org memory** — knowledge promoted to the whole organization: shared facts,
  policies, glossaries, hard-won lessons. Read-mostly, curated, stable.
- **Qdrant** indexes both as a retrieval corpus and answers recall queries.

---

## Core Capabilities

### 1. Compress with provenance, store for retrieval

Every learned unit — a compressed message, an episodic summary, a factual bullet —
carries a **stable ID and a link back to its source**. Compression never destroys
the trail: a factual bullet points to the episodic summary it came from, which
points to the raw messages archived behind it.

This provenance graph is the keystone. It is what makes the other three
capabilities (recall-with-citation, unlearning, hierarchical compression) the same
mechanism rather than three ad-hoc ones. Each unit is embedded and indexed in
Qdrant at write time.

### 2. Hierarchical compression with pointers back

Knowledge flows up a ladder, getting terser at each rung, but every rung keeps a
pointer down:

```
raw messages ──▶ compressed message ──▶ episodic summary ──▶ factual bullet
   (archived,        (working mem)         (long-term)         (long-term)
    retrievable)
```

The top rungs live in the prompt for free recall; the lower rungs live in Qdrant
and are pulled in only when a query needs the detail. Compression always runs
against the same stable knowledge prefix, both to keep the prefix cache warm and
to ensure the agent's own knowledge informs the summary.

### 3. Retrieval: Qdrant hybrid search + rerank

Recall is a **hybrid query** over the compressed history and its archived sources:

- **Semantic search** (dense vectors) for meaning-level matches.
- **BM25** (sparse) for exact terms, names, identifiers, error codes.
- **Fusion + rerank** to produce a single ranked, deduplicated set.
- **Trust and recency as ranking signals** — facts carry source attribution and a
  "do I question this?" flag (already produced today); stale or superseded units
  are demoted or filtered.

Retrieval is exposed two ways: as **conditional injection** into the `recall`
section during the control loop (every ~N tokens, per CONCEPT.md), and as an
explicit **`recall` tool** the model can call to fetch history mid-response. The
tool plumbing already exists in `Model.stream`; the `recall` memory section is
already a placeholder waiting to be filled.

### 4. Unlearn outdated knowledge

Overwriting is not forgetting. A fact that has been superseded must stop being
retrieved, not just disappear from one summary. Unlearning is first-class:

- Units carry **validity metadata** (`valid_from`, `valid_to`, `supersedes`).
- Superseding a fact **tombstones** the old unit and its derived children via the
  provenance graph.
- Retrieval filters tombstoned units; rerank demotes the merely-stale.

This keeps the corpus honest as the world changes, instead of letting dead vectors
resurface forever.

---

## Organization-Scale Memory

### Shared knowledge, promoted deliberately

An agent's consolidated facts can be **promoted** to org memory when they're
general and durable: org policies, domain glossaries, recurring lessons. Promotion
is a curated step (human- or policy-gated), not automatic leakage — the org store
is the trusted, slow-moving tier.

Once in org memory, knowledge is **shared across every agent** in the
organization. One agent's lesson becomes every agent's baseline.

### Org knowledge first — cached across the org

The context is layered **most-shared and most-stable first**, so the longest
possible prefix is byte-identical across the whole fleet:

```
1. Org memory        ← identical across all agents, changes rarely  ┐
2. Memorizer system  ← stable per deployment                        │ cacheable
3. App / user system ← the host app's own system prompt, merged in  ┘ prefix
4. Agent long-term memory (facts, goals)
5. Short-term + recall (retrieved, conditional)
6. Workspace + working memory   ← timestamps and "now" live only here
```

Because the org prefix is stable and identical, it stays warm in the inference
server's **prefix cache** — so for most requests the org knowledge is, in effect,
**carried for free**. The host application's own system prompt is *merged in below*
the cached prefix, so adopting Memorizer doesn't bust the shared cache.

**The discipline that makes this work:** nothing volatile may appear above the
working layer. Timestamps, "current time," and per-turn state render only in the
tail. The prefix is a contract: stable bytes in, free cache out.

> Caching is a measured outcome, not a guarantee. Cross-agent reuse depends on the
> inference server's cache scope and warmth. We treat the hit rate as something to
> instrument, not assume.

---

## Learning, Measured

A memory system is only worth its complexity if it demonstrably makes agents
better over time. The final system ships with an **eval harness that runs with
learning enabled**:

- Run a benchmark, let the agent consolidate what it learned, run it again.
- Always against a **no-memory control**, so improvement is attributable to memory.
- Separate **memorization** (it retrieved a near-identical past answer) from
  **generalization** (it got better at a class of problems) — and report both.
- Reset / seed memory cleanly between conditions so runs are comparable.

The metric that matters is the slope: does accuracy, or token cost, or
time-to-answer improve across repeated runs *because* of accumulated memory?

---

## Design Invariants

These hold across the whole system and should not be traded away:

1. **Chat completions is the only interface.** Everything is injected; the host
   app never learns the memory machinery exists.
2. **Fixed context layout.** A stable schema is what LoRA learns to read and what
   the prefix cache rewards. Order does not drift.
3. **Compression preserves provenance.** No learned unit is a dead end; every one
   links to its source.
4. **The prefix is a stability contract.** Volatile content lives only in the tail.
5. **Forgetting is explicit.** Stale knowledge is invalidated and stops being
   retrieved, not silently overwritten.
6. **Background, never on the request path.** Compression, consolidation, indexing,
   and workspace updates run async so they never add user-facing latency.

---

## How Today's Code Maps to the Vision

Much of the skeleton already exists:

| Vision capability | Today |
|---|---|
| Hierarchical compression | `working → episodic → factual` cascade in `model.py` |
| Source attribution & trust | factual prompt records source + "do I question this?" |
| Retrieval corpus | `add_uncompressed()` archives raw messages behind each summary |
| Soft unlearning | factual-update prompt can override / remove facts |
| Recall slot + tool plumbing | empty `recall` memory section + `tools` path in `Model.stream` |
| Cache-friendly fixed order | the fixed-layout `Context` |
| Chat-completions, injected | the core design |

The new build-out, in dependency order:

1. **Stable IDs + provenance graph** over memory units — the keystone.
2. **Pluggable store** behind `Memory` persistence, backed by **Qdrant** for
   shared, concurrent, indexed access (today's per-file JSON can't be shared-written).
3. **Hybrid retrieval** (semantic + BM25 + rerank) and the **recall tool**.
4. **True unlearning** (validity metadata, tombstones, supersedes-edges).
5. **Org memory tier** + promotion flow + the cached-prefix layering.
6. **Eval-with-learning harness** with a no-memory control.

---

## Bottom Line

Memorizer's end state is not "a model with memory." It is a **system** that learns:
it compresses experience into a provenance-linked knowledge graph, retrieves the
right slice on demand with Qdrant, forgets what's no longer true, and shares an
organization's accumulated knowledge through a prompt prefix that costs almost
nothing to carry — all behind an ordinary chat-completions call, and all measured
by whether agents actually get better over time.
