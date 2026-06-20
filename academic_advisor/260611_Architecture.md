# Academic Advisor — Architecture & Code Guide

A multi-agent Retrieval-Augmented Generation (RAG) system that answers student
questions about an academic program. Each of four PDFs becomes its own
specialist agent; a router reads each question and forwards it to the single
most relevant agent, which answers grounded **only** in its document.

---

## 1. High-level architecture

```
                              ┌──────────────────────┐
   question ──────────────────►        ROUTER         │  Qwen reads the question +
                              │  (picks 1 of 4 agents)│  the 4 agent descriptions and
                              └───────────┬───────────┘  replies with one agent name.
                                          │              Falls back to embedding
                                          │              similarity if the reply is
                                          │              ambiguous.
        ┌──────────────┬──────────────────┼──────────────────┬──────────────────┐
        ▼              ▼                  ▼                  ▼
   curriculum      calendar       internship_faq     internship_process     ← 4 agents
        │              │                  │                  │
   FAISS index    FAISS index        FAISS index        FAISS index          ← per-PDF
        └──────────────┴──────────────────┴──────────────────┘
                                   │
                shared embedding model (MiniLM)  +  shared LLM (Qwen2.5-3B-Instruct)
```

**Core idea:** four knowledge sources, four retrievers, but **one** embedding
model and **one** LLM loaded into memory and reused everywhere. Loading the LLM
four times would waste several gigabytes of GPU memory for no benefit.

---

## 2. Request lifecycle

What happens when you run `answer("When does the resit period start?")`:

1. **`init()`** (once) — builds or loads all four agents. Each agent loads its
   PDF, splits it into chunks, embeds them, and stores them in a FAISS index.
   Indexes are cached to `indexes/`, so this only does real work the first time.
2. **`route(question)`** — the LLM is shown the question plus the four agent
   descriptions and asked to name the best source. Here it returns `calendar`.
3. **`agent.ask(question)`** — the chosen agent embeds the question, retrieves
   the `TOP_K` most similar chunks from *its* FAISS index, and builds a prompt
   containing only those chunks as context.
4. **`llm.chat(system, user)`** — Qwen generates an answer constrained to the
   retrieved context, and is instructed to say it cannot find the answer rather
   than invent one.
5. The result (answer + source page snippets) is returned and printed.

---

## 3. File-by-file

| File | Responsibility |
|------|----------------|
| `config.py` | Single source of truth: model paths, chunk/retrieval settings, and the four `AGENTS` (name → PDF → router description). Change behavior here, not in the logic files. |
| `llm.py` | Loads the LLM **once** (lazy, on first call) and exposes `chat(system, user)`. Applies the model's chat template and runs greedy decoding for deterministic output. |
| `agent.py` | The `Agent` class — one per PDF. Builds/loads its cached FAISS index and implements `ask()` (retrieve → prompt → answer). Also holds the one shared embedding model via `get_embeddings()`. |
| `router.py` | `route()` — LLM-based agent selection with an embedding-similarity fallback for robustness. |
| `main.py` | Entry point. Builds the agents, exposes `answer()`, and provides a CLI (one-shot, `--agent` override, or interactive loop). |
| `requirements.txt` | Dependencies. |
| `Data/` | The four source PDFs. |
| `indexes/` | Auto-generated FAISS caches (safe to delete to force a rebuild). |

---

## 4. How the code works — pointers

### `config.py` — the control panel
Everything tunable lives here. The `AGENTS` list is the heart of it: each entry
maps an agent `name` to its `pdf` filename and a `description`. **The
`description` is what the router reads**, so its wording directly affects routing
quality — this matters most for the two internship PDFs, which overlap.

### `llm.py` — load once, reuse everywhere
`_model` and `_tokenizer` are module-level globals, populated on the first
`chat()` call (lazy loading). Because Python caches imported modules, every
agent and the router share that single loaded model. `device_map="auto"` places
it on the GPU (needs the `accelerate` package). `chat()` formats messages with
`apply_chat_template` — this is why an **Instruct** model matters; the template
encodes the system/user roles the model was trained on.

### `agent.py` — one PDF, one index
- `build_or_load()`: if a cached index exists in `indexes/<name>/`, it loads it
  instantly; otherwise it reads the PDF (`PyPDFLoader`), splits it with
  `RecursiveCharacterTextSplitter` (overlapping chunks so context isn't cut
  mid-sentence), embeds the chunks, and saves the FAISS index for next time.
- `ask()`: `retriever.invoke(question)` returns the `TOP_K` nearest chunks. They
  are concatenated (with page numbers) into the prompt context. The system
  prompt forces grounded, no-hallucination answers. Returns a dict with the
  answer plus source snippets so you can verify grounding.

### `router.py` — pick the right specialist
`route()` asks the LLM to name one source. The reply is matched against the
valid agent names. If it is ambiguous or unrecognized, `_embedding_route()`
takes over: it embeds the question and every agent description and returns the
highest cosine-similarity match. This two-tier design keeps routing working even
when the LLM phrases its answer oddly.

### `main.py` — wiring it together
`AGENT_MAP` is built once by `init()` and reused. `answer()` is the public API:
pass just a question to auto-route, or pass `agent_name=` to bypass the router
and call a specific agent. The `__main__` block gives you a CLI and an
interactive loop. The module is also importable, so `from main import answer`
works inside a notebook.

---

## 5. Extending the system

**Add a fifth knowledge source:** drop the PDF in `Data/`, then add one entry to
`AGENTS` in `config.py`:

```python
{
    "name": "grading",
    "pdf": "Grading_Policy.pdf",
    "description": "Grading rules: how grades are calculated, rounding, "
                   "pass marks, and resit grade caps.",
},
```

That's it — no logic changes. A new agent and index are created automatically on
the next run. (Delete `indexes/` if you also changed chunk settings.)

---

## 6. Running it

```bash
cd vault/academic_advisor
pip install -r requirements.txt

python main.py "When does the resit period start?"          # auto-routed
python main.py --agent curriculum "List the year-1 modules" # force an agent
python main.py                                              # interactive loop
```

First run downloads the embedding model (~90 MB), loads the LLM, and builds the
four indexes. Every run after that loads the cached indexes and answers
immediately.

---

## 7. Tuning knobs (all in `config.py`)

| Setting | Effect | When to change |
|---------|--------|----------------|
| `AGENTS[*].description` | Router accuracy | If questions go to the wrong agent (esp. the two internship sources). |
| `TOP_K` | How many chunks feed the answer | Raise for more context, lower if answers get noisy/slow. |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | Retrieval granularity | Smaller chunks = more precise retrieval; larger = more context per chunk. Delete `indexes/` after changing. |
| `MAX_NEW_TOKENS` | Answer length cap | Raise for longer answers. |
| `LLM_MODEL` | Which model runs | Swap between the local Coder models and the downloaded Instruct model. |

---

## 8. Design decisions & trade-offs

Why the system is built the way it is — the reasoning behind each choice and what
was given up.

### LLM router (with embedding fallback) vs. pure embedding routing
- **Chosen:** ask the LLM to name the agent; fall back to embedding similarity.
- **Why:** the LLM understands *intent*, not just surface word overlap. "Can I do
  my internship abroad?" routes to the internship agents even if it shares few
  exact words with their descriptions. Pure embedding routing matches on semantic
  proximity and struggles when two sources overlap (the FAQ vs. process PDFs).
- **Trade-off:** an LLM call per question adds latency. The embedding fallback
  costs almost nothing and guarantees a valid choice when the LLM's reply is
  malformed — so we get the LLM's judgment without the brittleness.

### One agent per PDF vs. one combined index
- **Chosen:** a separate FAISS index per PDF, selected by the router.
- **Why:** it gives a clean, explainable "which document answered this" trace
  (good for a report and for debugging), and keeps each source's retrieval from
  being drowned out by a larger document. It also matches the assignment's
  "four agents" framing.
- **Trade-off:** a question that genuinely spans two PDFs only gets one of them.
  A single merged index would retrieve across all sources at once. We prioritized
  clarity and grounding over cross-document synthesis.

### Local open models vs. a hosted API
- **Chosen:** Qwen run locally on the GPU, plus a local MiniLM embedder.
- **Why:** no API key, no per-call cost, full privacy (the PDFs never leave the
  machine), and it works on a lab GPU. Reproducible for grading.
- **Trade-off:** a multi-gigabyte model download and GPU memory pressure. A larger
  hosted model would give better answers with zero local resources, at the cost
  of keys, money, and sending data off-machine.

### FAISS vs. a managed vector database
- **Chosen:** FAISS, stored on disk and loaded in-process.
- **Why:** for a handful of PDFs (a few hundred chunks) it is instant, has zero
  infrastructure, and serializes to a folder we can cache and ship.
- **Trade-off:** no network service, no concurrent multi-user access, no
  metadata-filtered queries out of the box. None of which this project needs.

### Instruct model vs. base / Coder model
- **Chosen:** an instruction-tuned chat model (`Qwen2.5-3B-Instruct`).
- **Why:** routing and grounded answering both rely on following instructions and
  the chat template. Instruct models do this reliably; base models and
  code-specialized models produce terser, less natural advisor answers.
- **Trade-off:** the local Coder models were already on disk (no download), but
  their specialization made them a poorer fit, so we accepted the download.

### Cached indexes vs. rebuild every run
- **Chosen:** build once, save to `indexes/`, load thereafter.
- **Why:** embedding a PDF on every startup is wasted work; caching makes restarts
  near-instant.
- **Trade-off:** the cache can go stale — editing a PDF or chunk setting requires
  deleting `indexes/` to force a rebuild (noted in Gotchas).

---

## 9. Gotchas

- **Flat imports.** `main.py` uses `import agent`, `import llm`, etc., so run it
  from *inside* `academic_advisor/`.
- **Stale indexes.** Changing a PDF or the chunk settings does **not** rebuild
  the index automatically — delete `indexes/` to force a rebuild.
- **Instruct vs base/Coder models.** The chat template and answer quality assume
  an instruction-tuned model. A base (non-instruct) model will route and answer
  poorly.
- **`allow_dangerous_deserialization=True`** is set when loading FAISS. This is
  safe here because the indexes are generated locally by this same code; never
  load a FAISS index from an untrusted source with this flag.
