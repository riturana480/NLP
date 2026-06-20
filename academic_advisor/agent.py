"""A single RAG agent backed by one PDF.

Each Agent owns its own FAISS index (its PDF's chunks) but shares the global
embedding model and the global LLM. Indexes are cached to disk so restarts are
instant.
"""

import torch
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

import llm
from config import (
    VAULT_DIR,
    INDEX_DIR,
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    TOP_K,
)

_embeddings = None


def get_embeddings():
    """One shared embedding model for all agents and the router fallback."""
    global _embeddings
    if _embeddings is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": device},
        )
    return _embeddings


SYSTEM_PROMPT = (
    "You are an academic advisor assistant. Answer the student's question using ONLY "
    "the provided context, which comes from the '{name}' document. "
    "If the answer is not contained in the context, say you could not find it in that "
    "document and suggest the student check another source. Be concise and accurate, "
    "and do not invent information."
)


class Agent:
    def __init__(self, name, pdf, description):
        self.name = name
        self.pdf = pdf
        self.description = description
        self.vector_store = None
        self.retriever = None

    @property
    def index_path(self):
        return INDEX_DIR / self.name

    def build_or_load(self):
        """Load the cached FAISS index, or build it from the PDF on first run."""
        embeddings = get_embeddings()
        if self.index_path.exists():
            self.vector_store = FAISS.load_local(
                str(self.index_path),
                embeddings,
                allow_dangerous_deserialization=True,  # our own local index, safe
            )
        else:
            print(f"[{self.name}] building index from {self.pdf} ...")
            pages = PyPDFLoader(str(VAULT_DIR / self.pdf)).load()
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
                separators=["\n\n", "\n", ". ", " ", ""],
            )
            chunks = splitter.split_documents(pages)
            self.vector_store = FAISS.from_documents(chunks, embeddings)
            self.index_path.parent.mkdir(parents=True, exist_ok=True)
            self.vector_store.save_local(str(self.index_path))
            print(f"[{self.name}] indexed {len(chunks)} chunks.")

        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": TOP_K})
        return self

    def ask(self, question):
        """Retrieve relevant chunks and answer the question, grounded in this PDF."""
        docs = self.retriever.invoke(question)
        context = "\n\n".join(
            f"[page {d.metadata.get('page', '?')}] {d.page_content}" for d in docs
        )
        system = SYSTEM_PROMPT.format(name=self.name)
        user = f"Context:\n{context}\n\nQuestion: {question}"
        answer = llm.chat(system, user)

        sources = [
            {
                "page": d.metadata.get("page", "?"),
                "snippet": d.page_content[:160].strip().replace("\n", " "),
            }
            for d in docs
        ]
        return {"agent": self.name, "answer": answer, "sources": sources}
