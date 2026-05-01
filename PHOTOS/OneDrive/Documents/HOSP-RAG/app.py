"""
app.py — Hospital Knowledge Assistant (MedBot v2.0)
Theme: Dark Charcoal + Copper — Elegant Professional Medical RAG
Run: streamlit run app.py
"""

import streamlit as st
import os
import re
import sys
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

try:
    from dotenv import load_dotenv
    load_dotenv(BASE_DIR / ".env")
except ImportError:
    pass

from src.rag_engine import (
    initialize_rag,
    answer_question,
    get_document_stats,
    AVAILABLE_MODELS,
    MODEL_NAME,
)
import src.rag_engine as _rag_engine


def get_api_key() -> str:
    key = os.getenv("OPENROUTER_API_KEY", "")
    if not key:
        try:
            key = st.secrets.get("OPENROUTER_API_KEY", "")
        except Exception:
            pass
    return key


@st.cache_resource(show_spinner=False)
def load_cached_vectorstore():
    return initialize_rag(force_rebuild=False)


st.set_page_config(
    page_title="MedBot — Hospital Knowledge Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

:root {
    --bg-base:       #0e0e10;
    --bg-surface:    #141416;
    --bg-card:       #1a1a1e;
    --bg-elevated:   #202026;
    --copper-600:    #8B4513;
    --copper-500:    #A0522D;
    --copper-400:    #B8703A;
    --copper-300:    #CD8B52;
    --copper-200:    #DFB08A;
    --copper-100:    #F0D4B8;
    --silver-700:    #3a3a42;
    --silver-600:    #52525e;
    --silver-500:    #6e6e7e;
    --silver-400:    #8e8e9e;
    --silver-300:    #aeaebe;
    --silver-200:    #cecede;
    --silver-100:    #e8e8f0;
    --text-primary:  #f0f0f4;
    --text-secondary:#9090a0;
    --text-muted:    #505060;
    --border:        rgba(160,82,45,0.15);
    --border-bright: rgba(184,112,58,0.30);
    --border-silver: rgba(142,142,158,0.18);
}

*, body { font-family: 'DM Sans', sans-serif; }
html, body, .stApp { background: var(--bg-base) !important; color: var(--text-primary); }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #090909 0%, #0e0e10 50%, #111114 100%) !important;
    border-right: 1px solid rgba(184,112,58,0.2) !important;
}
section[data-testid="stSidebar"] > div { padding-top: 0 !important; }
section[data-testid="stSidebar"] * { color: var(--text-primary) !important; }
section[data-testid="stSidebar"] label {
    color: var(--copper-300) !important;
    font-size: 0.67rem !important; font-weight: 600 !important;
    letter-spacing: 0.12em !important; text-transform: uppercase !important;
}

.sb-logo {
    background: linear-gradient(145deg, #111112, #1a1a1e);
    border-bottom: 1px solid var(--border-bright);
    padding: 28px 20px 22px; margin: -1rem -1rem 0; text-align: center;
}
.sb-logo-emblem {
    width: 60px; height: 60px;
    background: linear-gradient(145deg, #1e1e22, #282830);
    border: 1px solid var(--border-bright); border-radius: 16px;
    display: flex; align-items: center; justify-content: center;
    margin: 0 auto 12px; font-size: 1.7rem;
}
.sb-logo-name { font-size: 1rem; font-weight: 700; color: var(--silver-100) !important; letter-spacing: 0.06em; }
.sb-logo-tag  { font-size: 0.6rem; color: var(--text-muted) !important; letter-spacing: 0.18em; text-transform: uppercase; margin-top: 4px; }

.sb-status {
    display: flex; align-items: center; gap: 8px;
    background: rgba(184,112,58,0.07); border: 1px solid var(--border-bright);
    border-radius: 8px; padding: 8px 12px; margin: 14px 0;
    font-size: 0.74rem; font-weight: 600;
}
.sb-dot     { width: 7px; height: 7px; border-radius: 50%; background: var(--copper-300); animation: pulse 2.5s infinite; flex-shrink: 0; }
.sb-dot-off { width: 7px; height: 7px; border-radius: 50%; background: var(--silver-500); flex-shrink: 0; }
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.3} }

.metric-card {
    background: var(--bg-card); border: 1px solid var(--border-silver);
    border-radius: 10px; padding: 14px 10px; text-align: center;
}
.metric-val { font-size: 1.5rem; font-weight: 700; color: var(--copper-300); font-family: 'DM Mono', monospace; }
.metric-lbl { font-size: 0.62rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.1em; margin-top: 3px; }

.sb-section {
    font-size: 0.62rem; font-weight: 600; color: var(--silver-400) !important;
    letter-spacing: 0.14em; text-transform: uppercase;
    padding: 16px 0 7px; border-top: 1px solid var(--border-silver); margin-top: 4px;
}

.doc-pill {
    display: flex; align-items: center; gap: 6px;
    background: rgba(255,255,255,0.03); border: 1px solid var(--border-silver);
    border-radius: 6px; padding: 5px 10px; margin: 3px 0;
    font-size: 0.69rem; color: var(--silver-300) !important; font-family: 'DM Mono', monospace;
    overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}

/* ── Suggestion buttons (main area) ── */
.stButton > button {
    background: var(--bg-card) !important;
    color: var(--silver-200) !important;
    border: 1px solid var(--border-silver) !important;
    border-radius: 10px !important;
    font-weight: 400 !important; font-size: 0.82rem !important;
    font-family: 'DM Sans', sans-serif !important;
    transition: all 0.18s ease !important;
    text-align: left !important;
    padding: 0.65rem 1rem !important;
    height: auto !important; min-height: 52px !important;
    line-height: 1.45 !important; white-space: normal !important;
    box-shadow: none !important;
}
.stButton > button:hover {
    background: var(--bg-elevated) !important;
    color: var(--copper-200) !important;
    border-color: var(--border-bright) !important;
    transform: translateY(-1px) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* Sidebar button overrides */
div[data-testid="stSidebar"] .stButton > button {
    background: var(--bg-elevated) !important;
    color: var(--copper-200) !important;
    border-color: var(--border-bright) !important;
    font-weight: 500 !important; text-align: center !important;
    min-height: 40px !important;
}
div[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(184,112,58,0.14) !important;
    color: var(--copper-100) !important;
}

/* ── Sidebar selectbox / slider ── */
div[data-testid="stSidebar"] .stSelectbox > div > div {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-bright) !important;
    border-radius: 8px !important; color: var(--text-primary) !important;
}
.stSlider [data-testid="stThumb"] { background: var(--copper-400) !important; }
.stSlider [data-testid="stSliderTrack"] > div { background: var(--copper-600) !important; }

/* ── Main area ── */
.main .block-container { background: transparent; padding: 1.5rem 2rem 2rem; max-width: 1040px; }

/* ── Header ── */
.medbot-header {
    position: relative; overflow: hidden;
    background: linear-gradient(145deg, #141416, #1a1a1e, #141416);
    border: 1px solid var(--border-bright); border-radius: 18px;
    padding: 26px 32px; margin-bottom: 16px;
}
.medbot-header::before {
    content: ''; position: absolute; top: -60px; right: -60px;
    width: 220px; height: 220px;
    background: radial-gradient(circle, rgba(184,112,58,0.08) 0%, transparent 70%);
    border-radius: 50%;
}
.header-inner { position: relative; z-index: 1; display: flex; align-items: center; gap: 20px; }
.header-emblem {
    width: 58px; height: 58px;
    background: linear-gradient(145deg, #1e1e22, #282830);
    border: 1px solid var(--border-bright); border-radius: 14px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.75rem; flex-shrink: 0;
}
.header-title { font-size: 1.6rem; font-weight: 700; color: var(--text-primary); letter-spacing: -0.02em; line-height: 1.1; }
.header-title span { color: var(--copper-300); }
.header-sub { font-size: 0.78rem; color: var(--text-secondary); margin-top: 5px; }
.header-right { margin-left: auto; text-align: right; flex-shrink: 0; }
.live-badge {
    display: inline-flex; align-items: center; gap: 6px;
    background: rgba(184,112,58,0.1); border: 1px solid var(--border-bright);
    color: var(--copper-200); padding: 5px 13px; border-radius: 20px;
    font-size: 0.68rem; font-weight: 600; letter-spacing: 0.1em; text-transform: uppercase;
}
.live-badge .dot { width: 6px; height: 6px; border-radius: 50%; background: var(--copper-300); animation: pulse 2.5s infinite; }
.header-time { font-size: 0.67rem; color: var(--text-muted); margin-top: 6px; font-family: 'DM Mono', monospace; }

/* ── Dept strip ── */
.dept-strip { display: flex; gap: 7px; flex-wrap: wrap; margin-bottom: 18px; }
.dept-badge {
    background: var(--bg-card); border: 1px solid var(--border-silver);
    color: var(--silver-300); padding: 4px 12px; border-radius: 20px;
    font-size: 0.68rem; font-weight: 500;
}

/* ── Welcome box ── */
.sug-intro {
    font-size: 0.67rem; font-weight: 600; color: var(--copper-300);
    letter-spacing: 0.16em; text-transform: uppercase; margin-bottom: 12px;
}
.welcome-box {
    background: var(--bg-card); border: 1px solid var(--border-bright);
    border-radius: 14px; padding: 22px 26px; margin-bottom: 20px;
}
.welcome-title { font-size: 0.96rem; font-weight: 600; color: var(--copper-200); margin-bottom: 8px; }
.welcome-text  { font-size: 0.82rem; color: var(--text-secondary); line-height: 1.7; }
.welcome-notice {
    margin-top: 14px; padding: 9px 13px;
    background: rgba(255,255,255,0.03); border-left: 2px solid var(--silver-600);
    font-size: 0.73rem; color: var(--silver-400); border-radius: 0 6px 6px 0;
}

/* ── Chat messages ── */
.msg-user {
    background: var(--bg-card); border: 1px solid var(--border-silver);
    border-radius: 16px 16px 4px 16px; padding: 15px 20px;
    margin: 10px 0 10px 60px; position: relative;
}
.msg-user::before { content: '👤'; position: absolute; left: -32px; top: 12px; font-size: 1rem; }
.msg-label-u { font-size: 0.59rem; font-weight: 600; color: var(--silver-400); letter-spacing: 0.14em; text-transform: uppercase; margin-bottom: 7px; }
.msg-text { color: var(--text-primary); font-size: 0.9rem; line-height: 1.65; }

.msg-bot {
    background: linear-gradient(145deg, #141416, #1a1a1e);
    border: 1px solid var(--border-bright); border-radius: 4px 16px 16px 16px;
    padding: 18px 22px; margin: 10px 60px 10px 0; position: relative;
}
.msg-bot::before { content: '🏥'; position: absolute; right: -34px; top: 12px; font-size: 1rem; }
.msg-label-b {
    font-size: 0.59rem; font-weight: 600; color: var(--copper-300);
    letter-spacing: 0.14em; text-transform: uppercase; margin-bottom: 10px;
    display: flex; align-items: center; gap: 6px;
}
.bot-dot { width: 6px; height: 6px; border-radius: 50%; background: var(--copper-300); animation: pulse 2.5s infinite; }
.msg-bot-text { color: var(--text-primary); font-size: 0.9rem; line-height: 1.72; }
.msg-bot-text strong { color: var(--copper-200); }
.msg-bot-text code {
    background: rgba(184,112,58,0.1); padding: 1px 6px; border-radius: 4px;
    font-family: 'DM Mono', monospace; font-size: 0.83em; color: var(--copper-200);
}

.sources-row { display: flex; flex-wrap: wrap; gap: 5px; margin-top: 12px; padding-top: 10px; border-top: 1px solid var(--border-silver); }
.src-chip {
    background: rgba(255,255,255,0.04); border: 1px solid var(--border-silver);
    color: var(--silver-300); padding: 3px 9px; border-radius: 8px;
    font-size: 0.66rem; font-weight: 500; font-family: 'DM Mono', monospace;
}

/* ── Chat input ── */
[data-testid="stChatInput"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-silver) !important;
    border-radius: 12px !important;
}
[data-testid="stChatInput"] textarea {
    color: var(--text-primary) !important;
    font-family: 'DM Sans', sans-serif !important; font-size: 0.88rem !important;
    background: transparent !important;
}
[data-testid="stChatInput"]:focus-within {
    border-color: var(--copper-400) !important;
    box-shadow: 0 0 0 3px rgba(184,112,58,0.1) !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: var(--bg-base); }
::-webkit-scrollbar-thumb { background: var(--silver-700); border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: var(--silver-600); }

#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }
details { background: var(--bg-card) !important; border: 1px solid var(--border-silver) !important; border-radius: 8px !important; }
summary { color: var(--copper-200) !important; font-size: 0.76rem !important; font-weight: 600 !important; }
hr { border-color: var(--border-silver) !important; margin: 10px 0 !important; }
</style>
""", unsafe_allow_html=True)


# ─── Session state ────────────────────────────────────────────────────────────
if "messages"    not in st.session_state: st.session_state.messages    = []
if "vectorstore" not in st.session_state: st.session_state.vectorstore = None
if "kb_ready"    not in st.session_state: st.session_state.kb_ready    = False
if "sel_model"   not in st.session_state: st.session_state.sel_model   = MODEL_NAME
if "retrieval_k" not in st.session_state: st.session_state.retrieval_k = 5

# ── Auto-load vectorstore once ────────────────────────────────────────────────
if not st.session_state.kb_ready:
    try:
        vs = load_cached_vectorstore()
        if vs is not None:
            st.session_state.vectorstore = vs
            st.session_state.kb_ready    = True
    except Exception:
        pass


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sb-logo">
        <div class="sb-logo-emblem">🏥</div>
        <div class="sb-logo-name">MedBot RAG</div>
        <div class="sb-logo-tag">Hospital Knowledge Assistant</div>
    </div>""", unsafe_allow_html=True)
    st.markdown("")

    ok = st.session_state.kb_ready
    dot_html = '<div class="sb-dot"></div>' if ok else '<div class="sb-dot-off"></div>'
    st.markdown(f"""
    <div class="sb-status">
        {dot_html}
        {"Knowledge Base Online" if ok else "Knowledge Base Offline"}
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="sb-section">Knowledge Base</div>', unsafe_allow_html=True)
    doc_stats = get_document_stats(str(BASE_DIR / "data" / "documents"))
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f'<div class="metric-card"><div class="metric-val">{doc_stats["total_files"]}</div><div class="metric-lbl">Docs</div></div>', unsafe_allow_html=True)
    with c2:
        col = "#CD8B52" if ok else "#6e6e7e"
        st.markdown(f'<div class="metric-card"><div class="metric-val" style="font-size:1.1rem;color:{col}">{"✓" if ok else "✗"}</div><div class="metric-lbl">{"Ready" if ok else "Build"}</div></div>', unsafe_allow_html=True)

    st.markdown("")
    if st.button("🔨  Build / Rebuild Knowledge Base", use_container_width=True):
        with st.spinner("Indexing documents…"):
            try:
                load_cached_vectorstore.clear()
                vs = initialize_rag(force_rebuild=True)
                st.session_state.vectorstore = vs
                st.session_state.kb_ready    = True
                st.success("✅ Knowledge base ready!")
                st.rerun()
            except Exception as e:
                st.error(f"Build failed: {e}")

    if doc_stats["filenames"]:
        st.markdown('<div class="sb-section">Indexed Documents</div>', unsafe_allow_html=True)
        for fname in doc_stats["filenames"]:
            st.markdown(f'<div class="doc-pill">▪ {fname}</div>', unsafe_allow_html=True)

    st.markdown('<div class="sb-section">Language Model</div>', unsafe_allow_html=True)
    st.session_state.sel_model = st.selectbox("Model", AVAILABLE_MODELS, index=0, label_visibility="collapsed")

    st.markdown('<div class="sb-section">Retrieval Settings</div>', unsafe_allow_html=True)
    st.session_state.retrieval_k = st.slider("Context chunks (k)", 3, 10, 5)

    st.markdown("")
    if st.button("🗑️  Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown("""
    <div style="text-align:center;margin-top:18px;padding-top:10px;border-top:1px solid rgba(142,142,158,0.1);">
        <div style="font-size:0.57rem;color:#303038;letter-spacing:0.1em;">MEDBOT v2.0 · INTERNAL USE ONLY</div>
    </div>""", unsafe_allow_html=True)


# ─── Header ───────────────────────────────────────────────────────────────────
now = datetime.now().strftime("%A, %d %B %Y · %H:%M")
ok  = st.session_state.kb_ready
st.markdown(f"""
<div class="medbot-header">
  <div class="header-inner">
    <div class="header-emblem">🏥</div>
    <div>
      <div class="header-title">Hospital <span>Knowledge</span> Assistant</div>
      <div class="header-sub">Internal RAG &nbsp;·&nbsp; SOPs &nbsp;·&nbsp; Policies &nbsp;·&nbsp; Clinical Manuals &nbsp;·&nbsp; Nursing Protocols</div>
    </div>
    <div class="header-right">
      <div class="live-badge"><div class="dot"></div>{"SYSTEM ONLINE" if ok else "OFFLINE"}</div>
      <div class="header-time">{now}</div>
    </div>
  </div>
</div>""", unsafe_allow_html=True)

st.markdown("""
<div class="dept-strip">
  <span class="dept-badge">Emergency</span>
  <span class="dept-badge">Pharmacy</span>
  <span class="dept-badge">ICU / CCU</span>
  <span class="dept-badge">Radiology</span>
  <span class="dept-badge">Infection Control</span>
  <span class="dept-badge">Nursing</span>
  <span class="dept-badge">Compliance</span>
  <span class="dept-badge">Administration</span>
</div>""", unsafe_allow_html=True)


# ─── Answer runner — stores result into session_state, never reruns itself ────
def run_answer(question: str):
    """Call LLM and append answer to messages. No st.rerun() inside."""
    api_key = get_api_key()
    if not api_key:
        st.session_state.messages.append({
            "role": "assistant",
            "content": "❌ API key missing. Add OPENROUTER_API_KEY to Streamlit Secrets or your .env file.",
            "sources": [],
        })
        return

    # ✅ Always inject key before calling
    _rag_engine.OPENROUTER_API_KEY = api_key

    if not st.session_state.kb_ready or st.session_state.vectorstore is None:
        st.session_state.messages.append({
            "role": "assistant",
            "content": "⚠️ Knowledge base not ready. Click **Build Knowledge Base** in the sidebar.",
            "sources": [],
        })
        return

    try:
        result = answer_question(
            vectorstore=st.session_state.vectorstore,
            question=question,
            model=st.session_state.sel_model,
            k=st.session_state.retrieval_k,
        )
        st.session_state.messages.append({
            "role":    "assistant",
            "content": result["answer"],
            "sources": result.get("sources", []),
        })
    except Exception as e:
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"❌ Error: {e}",
            "sources": [],
        })


# ─── Render a single message ──────────────────────────────────────────────────
def render_msg(msg: dict):
    if msg["role"] == "user":
        st.markdown(f"""
        <div class="msg-user">
          <div class="msg-label-u">Staff Question</div>
          <div class="msg-text">{msg['content']}</div>
        </div>""", unsafe_allow_html=True)
    elif msg["role"] == "assistant":
        html = msg["content"].replace("\n", "<br>")
        html = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', html)
        html = re.sub(r'`([^`]+)`', r'<code>\1</code>', html)
        srcs = "".join(f'<span class="src-chip">▪ {s}</span>' for s in msg.get("sources", []))
        src_block = f'<div class="sources-row">{srcs}</div>' if srcs else ""
        st.markdown(f"""
        <div class="msg-bot">
          <div class="msg-label-b"><div class="bot-dot"></div>MedBot Response</div>
          <div class="msg-bot-text">{html}</div>
          {src_block}
        </div>""", unsafe_allow_html=True)


# ─── FAQ suggestion questions ─────────────────────────────────────────────────
SUGGESTIONS = [
    "What are the 5 rights of medication administration?",
    "How do I handle a Code Blue emergency?",
    "What PPE is needed for contact precautions?",
    "Explain the SBAR communication framework",
    "What are the steps for patient admission?",
    "How do I report a medication error?",
    "When should I activate the Rapid Response Team?",
    "What is the HIPAA policy for patient PHI?",
    "What is the RACE protocol for Code Red fire?",
    "Which medications require double nurse verification?",
]

# ─── Main content area ────────────────────────────────────────────────────────
if not st.session_state.messages:
    # ── Welcome screen ──
    st.markdown("""
    <div class="welcome-box">
      <div class="welcome-title">Welcome, Healthcare Professional</div>
      <div class="welcome-text">
        Ask me anything about hospital policies, clinical SOPs, medication protocols,
        infection control, emergency codes, nursing handoffs, and administrative guidelines.
        I search the official hospital documents and return accurate, sourced answers instantly.
      </div>
      <div class="welcome-notice">
        ⚠ For clinical emergencies always follow direct physician orders.
        MedBot provides policy guidance only — not medical advice.
      </div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="sug-intro">Frequently Asked Questions</div>', unsafe_allow_html=True)

    cols = st.columns(2)
    clicked_q = None
    for i, sug in enumerate(SUGGESTIONS):
        with cols[i % 2]:
            if st.button(sug, key=f"sug_{i}", use_container_width=True):
                clicked_q = sug   # ✅ capture which button was clicked

    # ✅ THE FIX: process AFTER all buttons are rendered, in same Streamlit run
    # This avoids the rerun-before-answer problem entirely
    if clicked_q is not None:
        st.session_state.messages.append({"role": "user", "content": clicked_q})
        with st.spinner("Searching hospital documents…"):
            run_answer(clicked_q)
        st.rerun()  # safe here — messages already have both user + bot

else:
    # ── Chat history ──
    for msg in st.session_state.messages:
        render_msg(msg)


# ─── Chat input — always enabled ─────────────────────────────────────────────
prompt = st.chat_input("Ask about any hospital policy, SOP, or clinical procedure…")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    render_msg({"role": "user", "content": prompt})
    with st.spinner("Searching hospital documents…"):
        run_answer(prompt)
    # Render the bot reply immediately (last message)
    if st.session_state.messages[-1]["role"] == "assistant":
        render_msg(st.session_state.messages[-1])