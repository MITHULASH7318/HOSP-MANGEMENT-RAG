# --- KEEP ALL YOUR IMPORTS SAME ---
import streamlit as st
import os
import re
from datetime import datetime

from src.rag_engine import (
    initialize_rag,
    answer_question,
    get_document_stats,
    AVAILABLE_MODELS,
    MODEL_NAME,
)

# ✅ API KEY FIX (WORKS IN BOTH LOCAL + DEPLOY)
def get_api_key():
    return os.getenv("OPENROUTER_API_KEY") or st.secrets.get("OPENROUTER_API_KEY", "")

# ✅ CACHE VECTORSTORE (PREVENT FREEZE)
@st.cache_resource
def load_vectorstore():
    return initialize_rag(force_rebuild=False)

# ─── Session state ───
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "kb_ready" not in st.session_state:
    st.session_state.kb_ready = False

# ─── SIDEBAR ───
with st.sidebar:
    st.title("🏥 MedBot")

    if st.button("Build Knowledge Base"):
        with st.spinner("Building KB..."):
            try:
                vs = initialize_rag(force_rebuild=True)
                st.session_state.vectorstore = vs
                st.session_state.kb_ready = True
                st.success("KB Ready ✅")
            except Exception as e:
                st.error(e)

# ❌ REMOVED AUTO LOAD LOOP (VERY IMPORTANT)

# ─── LOAD KB ONCE (SAFE) ───
if not st.session_state.kb_ready:
    try:
        vs = load_vectorstore()
        if vs:
            st.session_state.vectorstore = vs
            st.session_state.kb_ready = True
    except:
        pass

# ─── CHAT ───
st.title("🏥 Hospital Knowledge Assistant")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

prompt = st.chat_input("Ask something...", disabled=not st.session_state.kb_ready)

if prompt:
    api_key = get_api_key()

    if not api_key:
        st.error("API key missing in Streamlit Secrets")
        st.stop()

    import src.rag_engine as _re
    _re.OPENROUTER_API_KEY = api_key

    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = answer_question(
                    vectorstore=st.session_state.vectorstore,
                    question=prompt,
                )
                st.write(result["answer"])

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["answer"]
                })

            except Exception as e:
                st.error(e)