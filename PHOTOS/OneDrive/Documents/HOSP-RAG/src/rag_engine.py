import os
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

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

MODEL_NAME = "mistralai/mixtral-8x7b-instruct"
FAISS_INDEX_DIR = "./data/faiss_index"
DOCUMENTS_DIR = "./data/documents"

# ✅ FIXED EMBEDDING (NO META ERROR)
def get_embedding_function():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )

# ─── DOCUMENT LOAD ───
def load_documents():
    docs = []
    path = Path(DOCUMENTS_DIR)

    for file in path.glob("**/*.txt"):
        loader = TextLoader(str(file))
        docs.extend(loader.load())

    for file in path.glob("**/*.pdf"):
        loader = PyPDFLoader(str(file))
        docs.extend(loader.load())

    return docs

# ─── CHUNK ───
def chunk_documents(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    return splitter.split_documents(documents)

# ─── BUILD ───
def build_vectorstore(chunks):
    embeddings = get_embedding_function()
    vs = FAISS.from_documents(chunks, embeddings)
    Path(FAISS_INDEX_DIR).mkdir(parents=True, exist_ok=True)
    vs.save_local(FAISS_INDEX_DIR)
    return vs

# ─── LOAD ───
def load_vectorstore():
    path = Path(FAISS_INDEX_DIR)
    if not path.exists():
        return None

    try:
        embeddings = get_embedding_function()
        return FAISS.load_local(
            FAISS_INDEX_DIR,
            embeddings,
            allow_dangerous_deserialization=True
        )
    except:
        return None

# ─── INIT ───
def initialize_rag(force_rebuild=False):
    if not force_rebuild:
        vs = load_vectorstore()
        if vs:
            return vs

    docs = load_documents()
    chunks = chunk_documents(docs)
    return build_vectorstore(chunks)

# ─── RETRIEVE ───
def retrieve_context(vs, query):
    return vs.similarity_search(query, k=5)

# ─── LLM CALL ───
def call_openrouter(messages):
    if not OPENROUTER_API_KEY:
        raise ValueError("API KEY missing")

    res = requests.post(
        f"{OPENROUTER_BASE_URL}/chat/completions",
        headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}"},
        json={"model": MODEL_NAME, "messages": messages},
    )

    return res.json()["choices"][0]["message"]["content"]

# ─── QA ───
def answer_question(vectorstore, question):
    docs = retrieve_context(vectorstore, question)
    context = "\n".join([d.page_content for d in docs])

    messages = [{
        "role": "user",
        "content": f"Context:\n{context}\n\nQuestion: {question}"
    }]

    answer = call_openrouter(messages)
    return {"answer": answer}