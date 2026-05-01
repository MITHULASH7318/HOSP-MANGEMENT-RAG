"""
src/rag_engine.py — RAG backend for MedBot
"""

import os
import requests
from pathlib import Path
from typing import List, Optional, Dict, Any
from dotenv import load_dotenv

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

# ✅ Use langchain-huggingface (fixes BertModel error with new transformers)
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    # Fallback for older installs
    from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
OPENROUTER_API_KEY  = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

MODEL_NAME      = "mistralai/mixtral-8x7b-instruct"
AVAILABLE_MODELS = [
    "mistralai/mixtral-8x7b-instruct",
    "mistralai/mistral-7b-instruct",
    "openai/gpt-3.5-turbo",
    "openai/gpt-4o-mini",
    "anthropic/claude-3-haiku",
]

FAISS_INDEX_DIR = "./data/faiss_index"
DOCUMENTS_DIR   = "./data/documents"


# ── Embeddings ────────────────────────────────────────────────────────────────
def get_embedding_function() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )


# ── Document stats ─────────────────────────────────────────────────────────────
def get_document_stats(documents_dir: str = DOCUMENTS_DIR) -> Dict[str, Any]:
    """Return total file count and filenames. Accepts optional directory path."""
    path = Path(documents_dir)
    if not path.exists():
        return {"total_files": 0, "filenames": []}
    files = list(path.glob("**/*.txt")) + list(path.glob("**/*.pdf"))
    return {
        "total_files": len(files),
        "filenames":   [f.name for f in sorted(files)],
    }


# ── Load raw documents ────────────────────────────────────────────────────────
def load_documents(documents_dir: str = DOCUMENTS_DIR) -> List[Document]:
    docs = []
    path = Path(documents_dir)
    if not path.exists():
        return docs

    for file in path.glob("**/*.txt"):
        try:
            loader = TextLoader(str(file), encoding="utf-8")
            docs.extend(loader.load())
        except Exception:
            try:
                loader = TextLoader(str(file), encoding="latin-1")
                docs.extend(loader.load())
            except Exception:
                pass

    for file in path.glob("**/*.pdf"):
        try:
            loader = PyPDFLoader(str(file))
            docs.extend(loader.load())
        except Exception:
            pass

    return docs


# ── Chunk documents ───────────────────────────────────────────────────────────
def chunk_documents(documents: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    return splitter.split_documents(documents)


# ── Build FAISS vectorstore ───────────────────────────────────────────────────
def build_vectorstore(chunks: List[Document]):
    embeddings = get_embedding_function()
    vs = FAISS.from_documents(chunks, embeddings)
    Path(FAISS_INDEX_DIR).mkdir(parents=True, exist_ok=True)
    vs.save_local(FAISS_INDEX_DIR)
    return vs


# ── Load existing FAISS index ─────────────────────────────────────────────────
def load_vectorstore():
    path = Path(FAISS_INDEX_DIR)
    if not path.exists():
        return None
    try:
        embeddings = get_embedding_function()
        return FAISS.load_local(
            FAISS_INDEX_DIR,
            embeddings,
            allow_dangerous_deserialization=True,
        )
    except Exception:
        return None


# ── Initialize RAG (build or load) ───────────────────────────────────────────
def initialize_rag(force_rebuild: bool = False):
    if not force_rebuild:
        vs = load_vectorstore()
        if vs is not None:
            return vs

    docs = load_documents()
    if not docs:
        raise ValueError(
            f"No documents found in '{DOCUMENTS_DIR}'. "
            "Add .txt or .pdf files and rebuild."
        )
    chunks = chunk_documents(docs)
    return build_vectorstore(chunks)


# ── Retrieve context chunks ───────────────────────────────────────────────────
def retrieve_context(vs, query: str, k: int = 5) -> List[Document]:
    return vs.similarity_search(query, k=k)


# ── Call OpenRouter LLM ───────────────────────────────────────────────────────
def call_openrouter(messages: list, model: str = MODEL_NAME) -> str:
    if not OPENROUTER_API_KEY:
        raise ValueError(
            "OPENROUTER_API_KEY is not set. "
            "Add it to .env (local) or Streamlit Secrets (deploy)."
        )

    resp = requests.post(
        f"{OPENROUTER_BASE_URL}/chat/completions",
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "HTTP-Referer":  "https://medbot-rag.streamlit.app",
            "X-Title":       "MedBot Hospital RAG",
        },
        json={"model": model, "messages": messages},
        timeout=60,
    )

    if resp.status_code != 200:
        raise ValueError(
            f"OpenRouter API error {resp.status_code}: {resp.text[:300]}"
        )

    data = resp.json()
    return data["choices"][0]["message"]["content"]


# ── Main QA function ──────────────────────────────────────────────────────────
def answer_question(
    vectorstore,
    question: str,
    chat_history: Optional[list] = None,
    model: str = MODEL_NAME,
    k: int = 5,
) -> Dict[str, Any]:
    """
    Retrieve relevant chunks, call LLM, return answer + source filenames.
    """
    docs    = retrieve_context(vectorstore, question, k=k)
    context = "\n\n---\n\n".join(d.page_content for d in docs)
    sources = list({
        Path(d.metadata.get("source", "")).name
        for d in docs
        if d.metadata.get("source")
    })

    system_prompt = (
        "You are MedBot, a professional hospital knowledge assistant. "
        "Answer only based on the provided hospital documents. "
        "If the answer is not in the documents, clearly say so. "
        "Be concise, accurate, and professional."
    )

    messages = [{"role": "system", "content": system_prompt}]

    if chat_history:
        messages.extend(chat_history[-6:])  # Keep last 3 turns

    messages.append({
        "role":    "user",
        "content": f"Context from hospital documents:\n{context}\n\nQuestion: {question}",
    })

    answer = call_openrouter(messages, model=model)
    return {"answer": answer, "sources": sources}