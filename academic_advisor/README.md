# Academic Advisor — Multi-Agent RAG System

A multi-agent Retrieval-Augmented Generation (RAG) system that answers student questions about the ADSAI academic program. Each of four source PDFs is handled by its own specialist agent; a router reads each question and forwards it to the single most relevant agent, which answers grounded only in its document.

---

## Important: Run on Data-lab Servers

> **This project must be run on the Data-lab servers.**

The system loads `Qwen2.5-3B-Instruct` locally from the shared `models/` directory. This requires a GPU with sufficient VRAM and access to the pre-downloaded model files that live on the lab file system. It will not run on a personal laptop.

---

## Architecture

```
                          ┌──────────────────────┐
   question ──────────────► ROUTER (Qwen LLM)     │  picks 1 of 4 agents by
                          │  + embedding fallback  │  reading agent descriptions.
                          └──────────┬─────────────┘
                                     │
        ┌────────────┬───────────────┼───────────────────┐
        ▼            ▼               ▼                   ▼
  curriculum     calendar    internship_faq    internship_process
        │            │               │                   │
   FAISS index  FAISS index    FAISS index          FAISS index
                          │
           shared: MiniLM embedder + Qwen2.5-3B-Instruct
```

- **4 agents** — one per PDF, each with its own FAISS vector index.
- **1 LLM** — loaded once into GPU memory and shared by all agents and the router.
- **1 embedding model** — `all-MiniLM-L6-v2`, shared across all agents.
- **FAISS indexes** are cached in `indexes/` after the first run for fast restarts.

---

## Project Structure

```
academic_advisor/
├── main.py          # Entry point — CLI and interactive loop
├── agent.py         # Agent class: builds/loads FAISS index, runs ask()
├── router.py        # Routes each question to the right agent
├── llm.py           # Loads Qwen once and exposes chat()
├── config.py        # All settings: model paths, chunk size, agent list
├── requirements.txt # Python dependencies
├── Data/            # The four source PDFs
│   ├── ADSAI_Curriculum_2025.pdf
│   ├── Academic_Calendar_New_Grid_EN.pdf
│   ├── Internship FAQs.pdf
│   └── Internship-Process-V1.pdf
└── indexes/         # Auto-generated FAISS caches (safe to delete to rebuild)
```

---

## Agents

| Agent name | Source PDF | Covers |
|---|---|---|
| `curriculum` | `ADSAI_Curriculum_2025.pdf` | Courses, modules, ECTS credits, program structure |
| `calendar` | `Academic_Calendar_New_Grid_EN.pdf` | Semester dates, exam/resit periods, holidays |
| `internship_faq` | `Internship FAQs.pdf` | Eligibility, duration, payment, credits |
| `internship_process` | `Internship-Process-V1.pdf` | Forms, approval steps, supervisor roles, deadlines |

---

## Setup

Run all commands from inside the `academic_advisor/` folder on the Data-lab server:

```bash
cd academic_advisor
pip install -r requirements.txt
```

Dependencies: `langchain`, `langchain-community`, `langchain-huggingface`, `faiss-cpu`, `sentence-transformers`, `transformers>=4.43`, `torch`, `accelerate`, `pypdf`.

The first run will:
1. Download the `all-MiniLM-L6-v2` embedding model (~90 MB).
2. Load `Qwen2.5-3B-Instruct` from the local `models/` directory.
3. Build and cache all four FAISS indexes in `indexes/`.

Every subsequent run loads the cached indexes and answers immediately.

---

## Usage

```bash
# Ask a single question (auto-routed to the right agent)
python main.py "When does the resit period start?"

# Force a specific agent and skip the router
python main.py --agent curriculum "List the year-1 modules"
python main.py --agent calendar "What are the exam week dates?"

# Start an interactive chat loop
python main.py
```

To import `answer()` directly in a notebook:

```python
import sys
sys.path.insert(0, "path/to/academic_advisor")
from main import answer

result = answer("Can I do my internship abroad?")
```

---

## Configuration

All tunable settings live in `config.py` — change behavior there, not in the logic files.

| Setting | Default | Effect |
|---|---|---|
| `LLM_MODEL` | `Qwen2.5-3B-Instruct` | Swap to a bigger/smaller local model |
| `TOP_K` | `4` | Number of chunks fed to the LLM; raise for more context |
| `CHUNK_SIZE` | `800` | Words per chunk; smaller = more precise retrieval |
| `CHUNK_OVERLAP` | `120` | Overlap between chunks to avoid cutting context mid-sentence |
| `MAX_NEW_TOKENS` | `512` | Maximum answer length |
| `AGENTS[*].description` | — | What the router reads; tune if questions go to the wrong agent |

> After changing `CHUNK_SIZE`, `CHUNK_OVERLAP`, or a PDF, delete `indexes/` to force a rebuild.

---

## Adding a New Knowledge Source

Drop the PDF in `Data/`, then add one entry to `AGENTS` in `config.py`:

```python
{
    "name": "grading",
    "pdf": "Grading_Policy.pdf",
    "description": "Grading rules: how grades are calculated, rounding, pass marks, and resit grade caps.",
},
```

No logic changes needed — a new agent and index are created automatically on the next run.

---

## Gotchas

- **Run from inside `academic_advisor/`** — flat imports (`import agent`, `import llm`) require the working directory to be the project folder.
- **Stale indexes** — changing a PDF or chunk settings does not auto-rebuild. Delete `indexes/` manually.
- **Instruct model required** — the chat template and routing assume an instruction-tuned model. A base or code-only model will route and answer poorly.
- **Data-lab only** — the local `models/` path and GPU requirement mean this cannot run on a personal machine without significant changes to `config.py`.
