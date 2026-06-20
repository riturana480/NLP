"""LLM-based router: picks which agent should handle a question.

Sends the question + the four agent descriptions to Qwen and asks for a single
agent name. If the model returns anything unexpected, falls back to embedding
similarity against the agent descriptions (robust for the overlapping internship
sources).
"""

import numpy as np

import llm
from config import AGENTS

VALID_NAMES = [a["name"] for a in AGENTS]

ROUTER_SYSTEM = (
    "You are a routing system for an academic advisor. Given a student's question, "
    "choose the single most relevant knowledge source from the list. "
    "Reply with ONLY the source name exactly as written, and nothing else."
)


def _menu():
    return "\n".join(f"- {a['name']}: {a['description']}" for a in AGENTS)


def route(question, verbose=False):
    """Return the name of the agent best suited to answer `question`."""
    user = f"Sources:\n{_menu()}\n\nQuestion: {question}\n\nSource name:"
    raw = llm.chat(ROUTER_SYSTEM, user, max_new_tokens=16).strip().lower()

    # Accept the answer if it clearly names one of the valid agents.
    matches = [name for name in VALID_NAMES if name in raw]
    if len(matches) == 1:
        if verbose:
            print(f"[router] LLM -> {matches[0]}")
        return matches[0]

    # Ambiguous or unrecognized -> fall back to embedding similarity.
    choice = _embedding_route(question)
    if verbose:
        print(f"[router] fallback (embedding) -> {choice}  (LLM said: {raw!r})")
    return choice


def _embedding_route(question):
    """Cosine similarity between the question and each agent's description."""
    from agent import get_embeddings  # local import avoids a circular import

    emb = get_embeddings()
    q = np.array(emb.embed_query(question))
    q_norm = q / (np.linalg.norm(q) + 1e-9)

    best_name, best_score = VALID_NAMES[0], -1.0
    for a in AGENTS:
        d = np.array(emb.embed_query(a["description"]))
        d_norm = d / (np.linalg.norm(d) + 1e-9)
        score = float(q_norm @ d_norm)
        if score > best_score:
            best_name, best_score = a["name"], score
    return best_name
