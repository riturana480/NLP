"""Central configuration for the Academic Advisor multi-agent system.

Edit model names, chunking, and the agent/PDF mapping here.
"""

from pathlib import Path

BASE_DIR = Path(__file__).parent
VAULT_DIR = BASE_DIR / "Data"        # where the 4 PDFs live
INDEX_DIR = BASE_DIR / "indexes"     # cached FAISS indexes (auto-generated)

# --- Models (run locally on GPU, no API key needed) ---
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Local models live in vault/models, i.e. the parent of this project folder.
# Anchoring to BASE_DIR.parent means it resolves correctly regardless of the
# directory you launch `python main.py` from.
# Pick ONE based on your GPU VRAM (see `nvidia-smi`):
#   - big GPU (>=24 GB): the 32B
#   - small GPU / safe default: the 1.5B
MODELS_DIR = BASE_DIR.parent / "models"

LLM_MODEL = str(MODELS_DIR / "Qwen" / "Qwen2.5-3B-Instruct")  # downloading now
# LLM_MODEL = str(MODELS_DIR / "Qwen" / "Qwen2.5-Coder-32B")   # local, big, code-tuned
# LLM_MODEL = str(BASE_DIR.parent / "old_models" / "Qwen2.5-Coder-1.5B")  # local, tiny

# --- Retrieval / generation settings ---
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120
TOP_K = 4
MAX_NEW_TOKENS = 512

# --- The four agents. `description` is what the router reads to pick an agent,
#     so keep them distinct (note the two internship sources). ---
AGENTS = [
    {
        "name": "curriculum",
        "pdf": "ADSAI_Curriculum_2025.pdf",
        "description": (
            "Course curriculum and program structure: the list of courses and modules, "
            "their content and learning outcomes, credits (ECTS), and how the ADSAI "
            "study program is organized across the years."
        ),
    },
    {
        "name": "calendar",
        "pdf": "Academic_Calendar_New_Grid_EN.pdf",
        "description": (
            "Academic calendar: important dates and deadlines, semester start and end "
            "dates, holidays, exam and resit periods, and the weekly schedule grid."
        ),
    },
    {
        "name": "internship_faq",
        "pdf": "Internship FAQs.pdf",
        "description": (
            "Internship frequently asked questions: short answers to common student "
            "questions about internships, such as eligibility, duration, payment, "
            "finding a company, and how internships count for credits."
        ),
    },
    {
        "name": "internship_process",
        "pdf": "Internship-Process-V1.pdf",
        "description": (
            "Internship process and official procedures: the step-by-step workflow to "
            "arrange an internship, required forms and documents, approval and "
            "registration steps, supervisor roles, and deadlines within the process."
        ),
    },
]
