import os, requests
from pathlib import Path
from typing import List, Optional
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

# Always resolve from this file's location — works on any OS, any CWD
_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_ROOT / ".env")

OPENROUTER_API_KEY  = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
MODEL_NAME          = os.getenv("MODEL_NAME", "openai/gpt-4o-mini")
EMBEDDING_MODEL     = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
DOCUMENTS_DIR       = str(_ROOT / "data" / "documents")
FAISS_INDEX_DIR     = str(_ROOT / "data" / "faiss_index")

AVAILABLE_MODELS = [
    "openai/gpt-4o-mini",
    "openai/gpt-3.5-turbo",
    "anthropic/claude-3-haiku",
    "mistralai/mistral-small",
    "mistralai/mixtral-8x7b-instruct",
    "meta-llama/llama-3-8b-instruct",
]

def get_embedding_function():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

def load_documents(docs_dir: str = None) -> List[Document]:
    p = Path(docs_dir) if docs_dir else Path(DOCUMENTS_DIR)
    if not p.exists():
        raise FileNotFoundError(f"Documents folder not found: {p}")
    docs = []
    for f in sorted(list(p.glob("**/*.txt")) + list(p.glob("**/*.pdf"))):
        try:
            loader = TextLoader(str(f), encoding="utf-8") if f.suffix == ".txt" else PyPDFLoader(str(f))
            for doc in loader.load():
                doc.metadata["filename"] = f.name
                docs.append(doc)
        except Exception as e:
            print(f"Skipping {f.name}: {e}")
    print(f"Loaded {len(docs)} sections from {p}")
    return docs

def chunk_documents(documents: List[Document]) -> List[Document]:
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=800, chunk_overlap=150,
        separators=["\n\n", "\n", ".", " ", ""],
    ).split_documents(documents)
    print(f"Created {len(chunks)} chunks")
    return chunks

def build_vectorstore(chunks: List[Document]) -> FAISS:
    idx = Path(FAISS_INDEX_DIR)
    idx.mkdir(parents=True, exist_ok=True)
    vs = FAISS.from_documents(chunks, get_embedding_function())
    vs.save_local(str(idx))
    print(f"FAISS index saved to {idx}")
    return vs

def load_vectorstore() -> Optional[FAISS]:
    idx = Path(FAISS_INDEX_DIR)
    if not idx.exists() or not any(idx.iterdir()):
        return None
    try:
        return FAISS.load_local(str(idx), get_embedding_function(),
                                allow_dangerous_deserialization=True)
    except Exception as e:
        print(f"Load failed: {e}")
        return None

def initialize_rag(force_rebuild: bool = False) -> FAISS:
    if not force_rebuild:
        vs = load_vectorstore()
        if vs:
            return vs
    docs   = load_documents()
    if not docs:
        raise ValueError("No documents found in data/documents/")
    chunks = chunk_documents(docs)
    return build_vectorstore(chunks)

def retrieve_context(vectorstore: FAISS, query: str, k: int = 5) -> List[Document]:
    return vectorstore.similarity_search(query, k=k)

def format_context(docs: List[Document]) -> str:
    return "\n\n---\n\n".join(
        f"[Source {i}: {d.metadata.get('filename','?')}]\n{d.page_content.strip()}"
        for i, d in enumerate(docs, 1)
    )

def call_openrouter(messages: list, model: str = None,
                    temperature: float = 0.1, max_tokens: int = 800) -> str:
    key = OPENROUTER_API_KEY or os.getenv("OPENROUTER_API_KEY", "")
    if not key:
        raise ValueError("OPENROUTER_API_KEY not set in .env file")
    if model is None:
        model = MODEL_NAME
    r = requests.post(
        f"{OPENROUTER_BASE_URL}/chat/completions",
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://hospital-rag.internal",
            "X-Title": "MedBot",
        },
        json={"model": model, "messages": messages,
              "temperature": temperature, "max_tokens": max_tokens},
        timeout=30,
    )
    if r.status_code != 200:
        raise Exception(f"OpenRouter {r.status_code}: {r.json().get('error',{}).get('message', r.text)[:200]}")
    return r.json()["choices"][0]["message"]["content"]

SYSTEM_PROMPT = """You are MedBot, a hospital knowledge assistant for internal staff.
You have access to hospital SOPs, policies, clinical manuals, infection control guidelines,
HR policies, medication protocols, nursing handoff procedures, and emergency codes.

RULES:
- Answer from the provided document context
- If the question is broad (like "hospital policy"), summarize ALL key topics available across the documents
- Use numbered steps for procedures, bullet points for lists
- Cite source document names
- Be helpful — if context has related info, use it even if question is vague
- Only say "not found" if truly nothing relevant exists in any document"""

def answer_question(vectorstore: FAISS, question: str,
                    chat_history: List[dict] = None,
                    model: str = None, k: int = 6) -> dict:
    if model is None:
        model = MODEL_NAME
    docs    = retrieve_context(vectorstore, question, k=k)
    context = format_context(docs)
    msgs    = [{"role": "system", "content": SYSTEM_PROMPT}]
    if chat_history:
        msgs.extend(chat_history[-6:])
    msgs.append({"role": "user", "content":
        f"HOSPITAL DOCUMENT CONTEXT:\n{context}\n\nSTAFF QUESTION: {question}"})
    answer  = call_openrouter(msgs, model=model)
    sources = list({d.metadata.get("filename", "?") for d in docs})
    return {"answer": answer, "sources": sources, "context_docs": docs}

def get_document_stats(docs_dir: str = None) -> dict:
    p   = Path(docs_dir) if docs_dir else Path(DOCUMENTS_DIR)
    txt = sorted(p.glob("**/*.txt")) if p.exists() else []
    pdf = sorted(p.glob("**/*.pdf")) if p.exists() else []
    return {
        "total_files": len(txt) + len(pdf),
        "txt_files":   len(txt),
        "pdf_files":   len(pdf),
        "filenames":   [f.name for f in txt + pdf],
    }