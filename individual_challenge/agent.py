import json, ollama, chromadb
from chromadb.utils import embedding_functions
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from config import MODEL,EMBEDDING,collection_name


def get_collection(company):
    client = chromadb.PersistentClient(path="./db")
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING)
    return client.get_or_create_collection(collection_name(company), embedding_function=ef)


def search_tool(col, query, k=6):
    res = col.query(query_texts=[query], n_results=k)
    return list(zip(res["documents"][0], res["metadatas"][0]))


def _ask_json(prompt):
    out = ollama.chat(model=MODEL, format="json", messages=[{"role": "user", "content": prompt}])
    return json.loads(out["message"]["content"])


def analyze(company, max_steps=6):
    col = get_collection(company)
    gathered = {}          
    trace = []             
    queries_done = []
    next_query = f"{company} strategy opportunities risks"

    for step in range(1, max_steps + 1):
        for doc, m in search_tool(col, next_query):
            gathered[m["url"] or m["title"]] = (doc, m)
        queries_done.append(next_query)
        trace.append({"step": step, "action": "search",
                      "query": next_query, "evidence": len(gathered)})

        preview = "\n".join(f"- {doc}" for doc, _ in list(gathered.values())[:18])
        decision = _ask_json(f"""You are researching {company} to brief its CEO.
Queries already run: {queries_done}
Evidence gathered so far ({len(gathered)} items):
{preview}

Decide the next action. Return JSON:
{{"action": "search" or "finish",
  "query": "<a NEW query covering a gap such as competitors, regulation, or public sentiment — only if action is search>",
  "reason": "<one short sentence>"}}
Choose "finish" only once opportunities, risks, competitors and trends are all covered.""")
        trace.append({"step": step, "action": "decide",
                      "decision": decision.get("action"), "reason": decision.get("reason", "")})

        if decision.get("action") != "search" or not decision.get("query"):
            break
        next_query = decision["query"]

    items = list(gathered.values())
    evidence = "\n".join(f"[{i}] ({m['source']}) {doc}" for i, (doc, m) in enumerate(items))
    result = _ask_json(f"""You are the AI strategy advisor to the CEO of {company}.
Use ONLY the numbered evidence below. Cite evidence by its [index] number.

EVIDENCE:
{evidence}

Return a JSON object with exactly these keys (3-4 items per list):
"opportunities": list of {{"title", "impact":"High|Medium|Low", "evidence":[indices], "confidence":0-1}}
"risks": list of {{"title", "category", "severity":"High|Medium|Low", "evidence":[indices], "confidence":0-1}}
"trends": list of short strings
"recommendations": list of {{"recommendation", "priority":"High|Medium|Low", "evidence":[indices], "expected_impact", "risk_level":"High|Medium|Low"}}
"briefing": {{"what_happened", "why_it_matters", "what_to_do_next"}}
Return only valid JSON, no extra text.""")

    result["_evidence"] = [{"i": i, "source": m["source"], "title": m["title"], "url": m["url"]}
                           for i, (doc, m) in enumerate(items)]
    result["_trace"] = trace
    return result


def sentiment(col):
    sia = SentimentIntensityAnalyzer()
    return [{"source": m["source"],
             "title": m["title"],
             "score": sia.polarity_scores(m["title"])["compound"]}
            for m in col.get()["metadatas"]]