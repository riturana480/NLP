"""Academic Advisor — multi-agent RAG entry point.

Builds all four agents once, then routes each question to the right one.

Usage:
    python main.py                                  # interactive chat loop
    python main.py "When does the exam period start?"   # one-shot question
    python main.py --agent calendar "..."           # force a specific agent
"""

import argparse

from agent import Agent
from router import route
from config import AGENTS

# Built once, reused for every question.
AGENT_MAP = {}


def init():
    """Build/load all four agents (and their cached indexes)."""
    if not AGENT_MAP:
        for cfg in AGENTS:
            AGENT_MAP[cfg["name"]] = Agent(**cfg).build_or_load()
    return AGENT_MAP


def answer(question, agent_name=None, verbose=True):
    """Route a question to the best agent and return its grounded answer.

    Pass `agent_name` to bypass the router and call a specific agent directly.
    """
    agents = init()
    if agent_name is None:
        agent_name = route(question, verbose=verbose)

    result = agents[agent_name].ask(question)

    if verbose:
        print(f"\n[answered by: {result['agent']}]\n")
        print(result["answer"])
        print("\n--- Sources ---")
        for i, s in enumerate(result["sources"], 1):
            print(f"[{i}] page {s['page']}: {s['snippet']}...")
    return result


def interactive():
    init()
    print("\nAcademic Advisor ready. Ask a question (type 'quit' to exit).\n")
    while True:
        question = input("You: ").strip()
        if question.lower() in {"quit", "exit", "q", ""}:
            print("Goodbye!")
            break
        answer(question)
        print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Academic Advisor multi-agent RAG")
    parser.add_argument(
        "question", nargs="*", help="Question to ask (omit to start interactive mode)"
    )
    parser.add_argument(
        "--agent",
        choices=[a["name"] for a in AGENTS],
        help="Force a specific agent and skip the router",
    )
    args = parser.parse_args()

    if args.question:
        answer(" ".join(args.question), agent_name=args.agent)
    else:
        interactive()
