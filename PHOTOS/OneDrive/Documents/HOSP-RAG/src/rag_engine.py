"""
rag_engine.py
RAG pipeline using FAISS (no C++ build tools needed on Windows)
"""

import os
import pickle
import requests
from pathlib import Path
from typing import List, Optional
from dotenv import load_dotenv

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

load_dotenv()

# ─── Configuration ────────────────────────────────────────────────────────────
OPENROUTER_API_KEY  = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
MODEL_NAME          = os.getenv("MODEL_NAME", "mistralai/mixtral-8x7b-instruct")
EMBEDDING_MODEL     = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
FAISS_INDEX_DIR     = os.getenv("FAISS_INDEX_DIR", "./data/faiss_index")
DOCUMENTS_DIR       = os.getenv("DOCUMENTS_DIR", "./data/documents")

AVAILABLE_MODELS = [
    "mistralai/mixtral-8x7b-instruct",
    "mistralai/mistral-7b-instruct",
    "meta-llama/llama-3-70b-instruct",
    "meta-llama/llama-3-8b-instruct",
    "openai/gpt-3.5-turbo",
    "openai/gpt-4o-mini",
    "anthropic/claude-3-haiku",
    "google/gemma-2-9b-it",
]


# ─── Embeddings ───────────────────────────────────────────────────────────────
def get_embedding_function() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


# ─── Document Loading ─────────────────────────────────────────────────────────
def load_documents(docs_dir: str = DOCUMENTS_DIR) -> List[Document]:
    docs = []
    docs_path = Path(docs_dir)

    if not docs_path.exists():
        raise FileNotFoundError(f"Documents directory not found: {docs_dir}")

    for txt_file in docs_path.glob("**/*.txt"):
        try:
            loader = TextLoader(str(txt_file), encoding="utf-8")
            for doc in loader.load():
                doc.metadata["filename"] = txt_file.name
                docs.append(doc)
        except Exception as e:
            print(f"⚠️  Skipping {txt_file.name}: {e}")

    for pdf_file in docs_path.glob("**/*.pdf"):
        try:
            loader = PyPDFLoader(str(pdf_file))
            for doc in loader.load():
                doc.metadata["filename"] = pdf_file.name
                docs.append(doc)
        except Exception as e:
            print(f"⚠️  Skipping {pdf_file.name}: {e}")

    print(f"✅ Loaded {len(docs)} document section(s)")
    return docs


# ─── Chunking ─────────────────────────────────────────────────────────────────
def chunk_documents(documents: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f"✅ Created {len(chunks)} chunks")
    return chunks


# ─── FAISS Vector Store ───────────────────────────────────────────────────────
def build_vectorstore(chunks: List[Document], index_dir: str = FAISS_INDEX_DIR) -> FAISS:
    embeddings = get_embedding_function()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    Path(index_dir).mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(index_dir)
    print(f"✅ FAISS index saved to: {index_dir}")
    return vectorstore


def load_vectorstore(index_dir: str = FAISS_INDEX_DIR) -> Optional[FAISS]:
    index_path = Path(index_dir)
    if not index_path.exists() or not any(index_path.iterdir()):
        return None
    try:
        embeddings = get_embedding_function()
        vs = FAISS.load_local(index_dir, embeddings, allow_dangerous_deserialization=True)
        print(f"✅ FAISS index loaded from: {index_dir}")
        return vs
    except Exception as e:
        print(f"❌ Could not load FAISS index: {e}")
        return None


def initialize_rag(force_rebuild: bool = False) -> FAISS:
    if not force_rebuild:
        vs = load_vectorstore()
        if vs is not None:
            return vs

    print("🔨 Building knowledge base...")
    documents = load_documents()
    if not documents:
        raise ValueError("No documents found. Add .txt or .pdf files to data/documents/")
    chunks = chunk_documents(documents)
    return build_vectorstore(chunks)


# ─── Retrieval ────────────────────────────────────────────────────────────────
def retrieve_context(vectorstore: FAISS, query: str, k: int = 5) -> List[Document]:
    return vectorstore.similarity_search(query, k=k)


def format_context(docs: List[Document]) -> str:
    parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("filename", "Unknown")
        parts.append(f"[Source {i}: {source}]\n{doc.page_content.strip()}")
    return "\n\n---\n\n".join(parts)


# ─── OpenRouter LLM ───────────────────────────────────────────────────────────
def call_openrouter(messages: list, model: str = MODEL_NAME,
                    temperature: float = 0.2, max_tokens: int = 1024) -> str:
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY not set. Add it to your .env file.")

    response = requests.post(
        f"{OPENROUTER_BASE_URL}/chat/completions",
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://hospital-rag.internal",
            "X-Title": "Hospital Knowledge Assistant",
        },
        json={
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
        timeout=60,
    )

    if response.status_code != 200:
        raise Exception(f"OpenRouter error {response.status_code}: {response.text}")

    return response.json()["choices"][0]["message"]["content"]


# ─── System Prompt ────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are MedBot, an expert hospital knowledge assistant for internal staff use.
Help nurses and administrative staff find accurate information from hospital SOPs, policies, and manuals.

RULES:
- Answer ONLY from the provided document context
- Use bullet points or numbered steps for procedures
- Cite the source document name in your answer
- If not found in context, say: "This is not in the current hospital documents. Please consult your supervisor."
- Never give personal medical advice or diagnoses
- Keep responses clear and professional"""


# ─── Main Q&A Function ────────────────────────────────────────────────────────
def answer_question(vectorstore: FAISS, question: str,
                    chat_history: List[dict] = None,
                    model: str = MODEL_NAME, k: int = 5) -> dict:
    context_docs = retrieve_context(vectorstore, question, k=k)
    context_str  = format_context(context_docs)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if chat_history:
        messages.extend(chat_history[-8:])

    messages.append({"role": "user", "content":
        f"HOSPITAL DOCUMENT CONTEXT:\n{context_str}\n\nSTAFF QUESTION: {question}\n\n"
        "Answer clearly based only on the documents above."
    })

    answer  = call_openrouter(messages, model=model)
    sources = list({doc.metadata.get("filename", "Unknown") for doc in context_docs})

    return {"answer": answer, "sources": sources, "context_docs": context_docs}


# ─── Document Stats ───────────────────────────────────────────────────────────
def get_document_stats(docs_dir: str = DOCUMENTS_DIR) -> dict:
    docs_path = Path(docs_dir)
    txt_files = list(docs_path.glob("**/*.txt")) if docs_path.exists() else []
    pdf_files = list(docs_path.glob("**/*.pdf")) if docs_path.exists() else []
    all_files = txt_files + pdf_files
    return {
        "total_files": len(all_files),
        "txt_files":   len(txt_files),
        "pdf_files":   len(pdf_files),
        "filenames":   [f.name for f in all_files],
    }