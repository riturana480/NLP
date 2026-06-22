import feedparser, chromadb, hashlib
from chromadb.utils import embedding_functions
from config import collection_name, EMBEDDING

def fetch_rss(url, source, limit):
    feed = feedparser.parse(url)
    return [{
        "source": source,
        "title":  e.get("title", "").strip(),
        "text":   e.get("summary", "").strip(),
        "url":    e.get("link", ""),
        "date":   e.get("published", ""),
    } for e in feed.entries[:limit]]

def collect(company, ticker):
    docs = []
    for q in [company, f"{company} AI", f"{company} competitors", f"{company} earnings"]:
        q = q.replace(" ", "+")
        docs += fetch_rss(f"https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en",
                          "google_news", 25)
    docs += fetch_rss(f"https://www.reddit.com/search.rss?q={company}&sort=new&limit=50",
                      "reddit", 50)
    if ticker:
        docs += fetch_rss(f"https://finance.yahoo.com/rss/headline?s={ticker}",
                          "yahoo_finance", 30)
    return docs

def dedupe(docs):
    seen, out = set(), []
    for d in docs:
        key = d["url"] or d["title"]
        if d["title"] and key not in seen:
            seen.add(key)
            out.append(d)
    return out

def _id(d):
    return hashlib.md5((d["url"] or d["title"]).encode()).hexdigest()

def index(company, docs):
    client = chromadb.PersistentClient(path="./db")
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING)
    col = client.get_or_create_collection(collection_name(company), embedding_function=ef)
    col.upsert(
        ids=[_id(d) for d in docs],
        documents=[f"{d['title']}. {d['text']}" for d in docs],
        metadatas=docs,
    )
    return col

def build(company, ticker):
    docs = dedupe(collect(company, ticker))
    index(company, docs)
    return docs

if __name__ == "__main__":  #CLI test
    from config import COMPANIES
    c = "NVIDIA"
    docs = build(c, COMPANIES[c])
    print(f"Indexed {len(docs)} docs for {c} from {len(set(d['source'] for d in docs))} sources.")