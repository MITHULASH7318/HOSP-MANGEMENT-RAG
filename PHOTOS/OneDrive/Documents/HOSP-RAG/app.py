"""
app.py — Hospital Knowledge Assistant (MedBot v2.0)
Theme: Deep Charcoal + Silver + Copper — Elegant Professional Medical RAG
Run: streamlit run app.py
API Key loaded silently from .env — NO visible key field
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

from src.rag_engine import (
    initialize_rag,
    answer_question,
    get_document_stats,
    AVAILABLE_MODELS,
    MODEL_NAME,
    DOCUMENTS_DIR,
    FAISS_INDEX_DIR,
)

from dotenv import load_dotenv
load_dotenv(BASE_DIR / ".env")

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MedBot — Hospital Knowledge Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS — Deep Charcoal + Silver + Copper Theme ──────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,wght@0,300;0,400;0,500;0,600;0,700;1,400&family=DM+Mono:wght@400;500&display=swap');

:root {
    --bg-base:        #0e0e10;
    --bg-surface:     #141416;
    --bg-card:        #1a1a1e;
    --bg-elevated:    #202026;
    --copper-600:     #8B4513;
    --copper-500:     #A0522D;
    --copper-400:     #B8703A;
    --copper-300:     #CD8B52;
    --copper-200:     #DFB08A;
    --copper-100:     #F0D4B8;
    --silver-700:     #3a3a42;
    --silver-600:     #52525e;
    --silver-500:     #6e6e7e;
    --silver-400:     #8e8e9e;
    --silver-300:     #aeaebe;
    --silver-200:     #cecede;
    --silver-100:     #e8e8f0;
    --text-primary:   #f0f0f4;
    --text-secondary: #9090a0;
    --text-muted:     #505060;
    --border:         rgba(160,82,45,0.15);
    --border-bright:  rgba(184,112,58,0.28);
    --border-silver:  rgba(142,142,158,0.18);
}

*, body { font-family: 'DM Sans', sans-serif; }
html, body, .stApp { background: var(--bg-base); color: var(--text-primary); }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #090909 0%, #0e0e10 50%, #111114 100%) !important;
    border-right: 1px solid rgba(184,112,58,0.2);
}
section[data-testid="stSidebar"] > div { padding-top: 0 !important; }
section[data-testid="stSidebar"] * { color: var(--text-primary) !important; }
section[data-testid="stSidebar"] label {
    color: var(--copper-300) !important;
    font-size: 0.67rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
}

/* ── Sidebar logo ── */
.sb-logo {
    background: linear-gradient(145deg, #111112, #1a1a1e);
    border-bottom: 1px solid var(--border-bright);
    padding: 28px 20px 22px;
    margin: -1rem -1rem 0;
    text-align: center;
}
.sb-logo-emblem {
    width: 60px; height: 60px;
    background: linear-gradient(145deg, #1e1e22, #282830);
    border: 1px solid var(--border-bright);
    border-radius: 16px;
    display: flex; align-items: center; justify-content: center;
    margin: 0 auto 12px;
    font-size: 1.7rem;
}
.sb-logo-name {
    font-size: 1rem; font-weight: 700; color: var(--silver-100) !important;
    letter-spacing: 0.06em;
}
.sb-logo-tag {
    font-size: 0.6rem; color: var(--text-muted) !important;
    letter-spacing: 0.18em; text-transform: uppercase; margin-top: 4px;
}

/* ── Status badge ── */
.sb-status {
    display: flex; align-items: center; gap: 8px;
    background: rgba(184,112,58,0.07);
    border: 1px solid var(--border-bright);
    border-radius: 8px; padding: 8px 12px; margin: 14px 0;
    font-size: 0.74rem; font-weight: 600;
}
.sb-status-dot {
    width: 7px; height: 7px; border-radius: 50%;
    background: var(--copper-300);
    animation: pulse 2.5s infinite; flex-shrink: 0;
}
.sb-status-dot.offline { background: var(--silver-500); animation: none; }
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.3} }

/* ── Metric cards ── */
.metric-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin: 10px 0; }
.metric-card {
    background: var(--bg-card);
    border: 1px solid var(--border-silver);
    border-radius: 10px; padding: 14px 10px; text-align: center;
}
.metric-val { font-size: 1.5rem; font-weight: 700; color: var(--copper-300); font-family: 'DM Mono', monospace; }
.metric-lbl { font-size: 0.62rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.1em; margin-top: 3px; }

/* ── Sidebar section labels ── */
.sb-section {
    font-size: 0.62rem; font-weight: 600; color: var(--silver-400) !important;
    letter-spacing: 0.14em; text-transform: uppercase;
    padding: 16px 0 7px; border-top: 1px solid var(--border-silver);
    margin-top: 4px;
}

/* ── Doc pill ── */
.doc-pill {
    display: flex; align-items: center; gap: 6px;
    background: rgba(255,255,255,0.03);
    border: 1px solid var(--border-silver);
    border-radius: 6px; padding: 5px 10px; margin: 3px 0;
    font-size: 0.69rem; color: var(--silver-300) !important;
    font-family: 'DM Mono', monospace;
}

/* ── Main container ── */
.main .block-container {
    background: transparent;
    padding: 1.5rem 2rem 2rem;
    max-width: 1040px;
}

/* ── Header ── */
.medbot-header {
    position: relative; overflow: hidden;
    background: linear-gradient(145deg, #141416, #1a1a1e, #141416);
    border: 1px solid var(--border-bright);
    border-radius: 18px;
    padding: 26px 32px;
    margin-bottom: 16px;
}
.medbot-header::before {
    content: '';
    position: absolute; top: -60px; right: -60px;
    width: 220px; height: 220px;
    background: radial-gradient(circle, rgba(184,112,58,0.08) 0%, transparent 70%);
    border-radius: 50%;
}
.header-inner { position: relative; z-index: 1; display: flex; align-items: center; gap: 20px; }
.header-emblem {
    width: 58px; height: 58px;
    background: linear-gradient(145deg, #1e1e22, #282830);
    border: 1px solid var(--border-bright);
    border-radius: 14px; display: flex; align-items: center; justify-content: center;
    font-size: 1.75rem; flex-shrink: 0;
}
.header-title {
    font-size: 1.6rem; font-weight: 700; color: var(--text-primary);
    letter-spacing: -0.02em; line-height: 1.1;
}
.header-title span { color: var(--copper-300); }
.header-sub {
    font-size: 0.78rem; color: var(--text-secondary);
    margin-top: 5px; font-weight: 400; letter-spacing: 0.01em;
}
.header-right { margin-left: auto; text-align: right; flex-shrink: 0; }
.live-badge {
    display: inline-flex; align-items: center; gap: 6px;
    background: rgba(184,112,58,0.1);
    border: 1px solid var(--border-bright);
    color: var(--copper-200); padding: 5px 13px;
    border-radius: 20px; font-size: 0.68rem; font-weight: 600;
    letter-spacing: 0.1em; text-transform: uppercase;
}
.live-badge .dot { width: 6px; height: 6px; border-radius: 50%; background: var(--copper-300); animation: pulse 2.5s infinite; }
.header-time { font-size: 0.67rem; color: var(--text-muted); margin-top: 6px; font-family: 'DM Mono', monospace; }

/* ── Dept strip ── */
.dept-strip { display: flex; gap: 7px; flex-wrap: wrap; margin-bottom: 18px; }
.dept-badge {
    display: inline-flex; align-items: center; gap: 5px;
    background: var(--bg-card);
    border: 1px solid var(--border-silver);
    color: var(--silver-300); padding: 4px 12px;
    border-radius: 20px; font-size: 0.68rem; font-weight: 500;
    letter-spacing: 0.02em;
}

/* ── Welcome box ── */
.sug-intro {
    font-size: 0.67rem; font-weight: 600; color: var(--copper-300);
    letter-spacing: 0.16em; text-transform: uppercase; margin-bottom: 12px;
}
.welcome-box {
    background: var(--bg-card);
    border: 1px solid var(--border-bright);
    border-radius: 14px; padding: 22px 26px; margin-bottom: 20px;
}
.welcome-title { font-size: 0.96rem; font-weight: 600; color: var(--copper-200); margin-bottom: 8px; }
.welcome-text { font-size: 0.82rem; color: var(--text-secondary); line-height: 1.7; }
.welcome-notice {
    margin-top: 14px; padding: 9px 13px;
    background: rgba(255,255,255,0.03);
    border-left: 2px solid var(--silver-600);
    font-size: 0.73rem; color: var(--silver-400);
    border-radius: 0 6px 6px 0;
}

/* ── Chat messages ── */
.msg-user {
    background: var(--bg-card);
    border: 1px solid var(--border-silver);
    border-radius: 16px 16px 4px 16px;
    padding: 15px 20px; margin: 10px 0 10px 70px;
    position: relative;
}
.msg-user::before {
    content: '👤';
    position: absolute; left: -34px; top: 12px; font-size: 1rem;
}
.msg-label-user {
    font-size: 0.59rem; font-weight: 600; color: var(--silver-400);
    letter-spacing: 0.14em; text-transform: uppercase; margin-bottom: 7px;
}
.msg-text { color: var(--text-primary); font-size: 0.9rem; line-height: 1.65; }

.msg-bot {
    background: linear-gradient(145deg, #141416, #1a1a1e);
    border: 1px solid var(--border-bright);
    border-radius: 4px 16px 16px 16px;
    padding: 18px 22px; margin: 10px 70px 10px 0;
    position: relative;
}
.msg-bot::before {
    content: '🏥';
    position: absolute; right: -36px; top: 12px; font-size: 1rem;
}
.msg-label-bot {
    font-size: 0.59rem; font-weight: 600; color: var(--copper-300);
    letter-spacing: 0.14em; text-transform: uppercase; margin-bottom: 10px;
    display: flex; align-items: center; gap: 6px;
}
.bot-pulse { width: 6px; height: 6px; border-radius: 50%; background: var(--copper-300); animation: pulse 2.5s infinite; }

.msg-bot-text { color: var(--text-primary); font-size: 0.9rem; line-height: 1.72; }
.msg-bot-text strong { color: var(--copper-200); }
.msg-bot-text code {
    background: rgba(184,112,58,0.1);
    padding: 1px 6px; border-radius: 4px;
    font-family: 'DM Mono', monospace; font-size: 0.83em; color: var(--copper-200);
}

.sources-row { display: flex; flex-wrap: wrap; gap: 5px; margin-top: 12px; padding-top: 10px; border-top: 1px solid var(--border-silver); }
.src-chip {
    display: inline-flex; align-items: center; gap: 4px;
    background: rgba(255,255,255,0.04);
    border: 1px solid var(--border-silver);
    color: var(--silver-300); padding: 3px 9px; border-radius: 8px;
    font-size: 0.66rem; font-weight: 500;
    font-family: 'DM Mono', monospace;
}

/* ── Suggestion buttons ── */
.stButton button {
    background: var(--bg-card) !important;
    color: var(--silver-200) !important;
    border: 1px solid var(--border-silver) !important;
    border-radius: 10px !important;
    font-weight: 400 !important;
    font-size: 0.82rem !important;
    font-family: 'DM Sans', sans-serif !important;
    letter-spacing: 0.01em !important;
    transition: all 0.18s ease !important;
    text-align: left !important;
    padding: 0.6rem 1rem !important;
    height: auto !important;
    min-height: 50px !important;
    line-height: 1.4 !important;
    white-space: normal !important;
}
.stButton button:hover {
    background: var(--bg-elevated) !important;
    color: var(--copper-200) !important;
    border-color: var(--border-bright) !important;
    transform: translateY(-1px) !important;
}
.stButton button:active {
    transform: translateY(0) !important;
}

div[data-testid="stSidebar"] .stButton button {
    background: var(--bg-elevated) !important;
    color: var(--copper-200) !important;
    border-color: var(--border-bright) !important;
    font-weight: 500 !important;
    text-align: center !important;
    min-height: 40px !important;
}
div[data-testid="stSidebar"] .stButton button:hover {
    background: rgba(184,112,58,0.12) !important;
    color: var(--copper-100) !important;
}

/* ── Chat input ── */
[data-testid="stChatInput"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-silver) !important;
    border-radius: 12px !important;
}
[data-testid="stChatInput"] textarea {
    color: var(--text-primary) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.88rem !important;
    background: transparent !important;
}
[data-testid="stChatInput"]:focus-within {
    border-color: var(--copper-400) !important;
    box-shadow: 0 0 0 3px rgba(184,112,58,0.1) !important;
}

/* ── Selectbox ── */
.stSelectbox > div > div {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-silver) !important;
    border-radius: 8px !important;
    color: var(--text-primary) !important;
}

/* ── Slider ── */
.stSlider [data-testid="stThumb"] { background: var(--copper-400) !important; }
.stSlider [data-testid="stSliderTrack"] > div { background: var(--copper-600) !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: var(--bg-base); }
::-webkit-scrollbar-thumb { background: var(--silver-700); border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: var(--silver-600); }

/* ── Hide streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

/* ── Expander ── */
details { background: var(--bg-card) !important; border: 1px solid var(--border-silver) !important; border-radius: 8px !important; }
summary { color: var(--copper-200) !important; font-size: 0.76rem !important; font-weight: 600 !important; }

hr { border-color: var(--border-silver) !important; margin: 10px 0 !important; }
</style>
""", unsafe_allow_html=True)


# ─── Session state ────────────────────────────────────────────────────────────
if "messages"       not in st.session_state: st.session_state.messages       = []
if "vectorstore"    not in st.session_state: st.session_state.vectorstore    = None
if "kb_ready"       not in st.session_state: st.session_state.kb_ready       = False
if "selected_model" not in st.session_state: st.session_state.selected_model = MODEL_NAME
if "pending_prompt" not in st.session_state: st.session_state.pending_prompt = None


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sb-logo">
        <div class="sb-logo-emblem">🏥</div>
        <div class="sb-logo-name">MedBot RAG</div>
        <div class="sb-logo-tag">Hospital Knowledge Assistant</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("")

    ok = st.session_state.kb_ready
    dot_cls = "sb-status-dot" if ok else "sb-status-dot offline"
    status_text = "Knowledge Base Online" if ok else "Knowledge Base Offline"
    st.markdown(f"""
    <div class="sb-status">
        <div class="{dot_cls}"></div>
        {status_text}
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="sb-section">Knowledge Base</div>', unsafe_allow_html=True)

    doc_stats = get_document_stats(str(BASE_DIR / "data" / "documents"))
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-val">{doc_stats['total_files']}</div>
            <div class="metric-lbl">Documents</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        v_color = "#CD8B52" if ok else "#6e6e7e"
        st.markdown(f"""<div class="metric-card">
            <div class="metric-val" style="color:{v_color};font-size:1.1rem;">{"✓" if ok else "✗"}</div>
            <div class="metric-lbl">{"Ready" if ok else "Build"}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("")
    if st.button("Build / Rebuild Knowledge Base", use_container_width=True):
        with st.spinner("Indexing documents…"):
            try:
                vs = initialize_rag(force_rebuild=True)
                st.session_state.vectorstore = vs
                st.session_state.kb_ready    = True
                st.success("Knowledge base is ready.")
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")

    if not st.session_state.kb_ready:
        try:
            vs = initialize_rag(force_rebuild=False)
            if vs:
                st.session_state.vectorstore = vs
                st.session_state.kb_ready    = True
        except Exception:
            pass

    if doc_stats["filenames"]:
        st.markdown('<div class="sb-section">Indexed Documents</div>', unsafe_allow_html=True)
        for fname in doc_stats["filenames"]:
            icon = "▪" if fname.endswith(".txt") else "▸"
            st.markdown(f'<div class="doc-pill">{icon} {fname}</div>', unsafe_allow_html=True)

    st.markdown('<div class="sb-section">Language Model</div>', unsafe_allow_html=True)
    selected_model = st.selectbox("Model", AVAILABLE_MODELS, index=0, label_visibility="collapsed")
    st.session_state.selected_model = selected_model

    st.markdown('<div class="sb-section">Retrieval Settings</div>', unsafe_allow_html=True)
    retrieval_k = st.slider("Context chunks (k)", 3, 10, 5)
    temperature  = st.slider("Response style", 0.0, 1.0, 0.2, 0.1)

    st.markdown("")
    if st.button("Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.pending_prompt = None
        st.rerun()

    st.markdown("""
    <div style="text-align:center;margin-top:20px;padding-top:12px;border-top:1px solid rgba(142,142,158,0.1);">
        <div style="font-size:0.58rem;color:#303038;letter-spacing:0.1em;">MEDBOT v2.0 · INTERNAL USE ONLY</div>
    </div>""", unsafe_allow_html=True)


# ─── Header ───────────────────────────────────────────────────────────────────
now = datetime.now().strftime("%A, %d %B %Y · %H:%M")
ok  = st.session_state.kb_ready
badge_html = '<span class="dot"></span> SYSTEM ONLINE' if ok else 'OFFLINE'
st.markdown(f"""
<div class="medbot-header">
    <div class="header-inner">
        <div class="header-emblem">🏥</div>
        <div>
            <div class="header-title">Hospital <span>Knowledge</span> Assistant</div>
            <div class="header-sub">Internal RAG System &nbsp;·&nbsp; SOPs &nbsp;·&nbsp; Policies &nbsp;·&nbsp; Clinical Manuals &nbsp;·&nbsp; Nursing Protocols</div>
        </div>
        <div class="header-right">
            <div class="live-badge">{badge_html}</div>
            <div class="header-time">{now}</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

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
</div>
""", unsafe_allow_html=True)


# ─── Answer runner ────────────────────────────────────────────────────────────
def run_answer(prompt_text: str):
    api_key = os.getenv("OPENROUTER_API_KEY", "")
    if not api_key:
        st.error("OPENROUTER_API_KEY not found in .env file.")
        st.info("Create a `.env` file:\n```\nOPENROUTER_API_KEY=sk-or-v1-your-key\n```")
        return
    if not st.session_state.kb_ready or st.session_state.vectorstore is None:
        st.error("Please build the knowledge base first.")
        return

    import src.rag_engine as _re
    _re.OPENROUTER_API_KEY = api_key

    with st.spinner("Searching hospital documents…"):
        try:
            history = [
                {"role": m["role"], "content": m.get("content", "")}
                for m in st.session_state.messages[:-1]
                if m["role"] in ("user", "assistant")
            ]
            result = answer_question(
                vectorstore=st.session_state.vectorstore,
                question=prompt_text,
                chat_history=history,
                model=st.session_state.selected_model,
                k=retrieval_k,
            )
            bot_msg = {
                "role":    "assistant",
                "content": result["answer"],
                "sources": result["sources"],
            }
            st.session_state.messages.append(bot_msg)
        except Exception as e:
            st.error(f"Error: {e}")


# ─── Render message ───────────────────────────────────────────────────────────
def render_msg(msg: dict):
    if msg["role"] == "user":
        st.markdown(f"""
        <div class="msg-user">
            <div class="msg-label-user">Staff Question</div>
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
            <div class="msg-label-bot"><div class="bot-pulse"></div> MedBot Response</div>
            <div class="msg-bot-text">{html}</div>
            {src_block}
        </div>""", unsafe_allow_html=True)


# ─── Process pending suggestion click ─────────────────────────────────────────
if st.session_state.pending_prompt is not None:
    pending = st.session_state.pending_prompt
    st.session_state.pending_prompt = None
    for msg in st.session_state.messages:
        render_msg(msg)
    run_answer(pending)
    st.rerun()

# ─── Welcome / suggestion screen ──────────────────────────────────────────────
elif not st.session_state.messages:
    st.markdown("""
    <div class="welcome-box">
        <div class="welcome-title">Welcome, Healthcare Professional</div>
        <div class="welcome-text">
            Ask me anything about hospital policies, clinical SOPs, medication protocols,
            infection control, emergency codes, nursing handoffs, and administrative guidelines.
            I search official hospital documents and return accurate, sourced answers instantly.
        </div>
        <div class="welcome-notice">
            For clinical emergencies, always follow direct physician orders.
            MedBot provides policy guidance only — not medical advice.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sug-intro">Frequently Asked Questions</div>', unsafe_allow_html=True)

    suggestions = [
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

    cols = st.columns(2)
    for i, sug in enumerate(suggestions):
        with cols[i % 2]:
            if st.button(sug, key=f"sug_{i}", use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": sug})
                st.session_state.pending_prompt = sug
                st.rerun()

# ─── Normal chat history view ─────────────────────────────────────────────────
else:
    for msg in st.session_state.messages:
        render_msg(msg)


# ─── Chat input ───────────────────────────────────────────────────────────────
prompt = st.chat_input(
    "Ask about any hospital policy, SOP, or clinical procedure…",
    disabled=not st.session_state.kb_ready,
)

if prompt:
    api_key = os.getenv("OPENROUTER_API_KEY", "")
    if not api_key:
        st.error("OPENROUTER_API_KEY not found in .env file.")
        st.info("Create a `.env` file:\n```\nOPENROUTER_API_KEY=sk-or-v1-your-key-here\n```")
        st.stop()

    if not st.session_state.kb_ready or st.session_state.vectorstore is None:
        st.error("Please build the knowledge base first.")
        st.stop()

    import src.rag_engine as _re
    _re.OPENROUTER_API_KEY = api_key

    st.session_state.messages.append({"role": "user", "content": prompt})
    render_msg({"role": "user", "content": prompt})

    with st.spinner("Searching hospital documents…"):
        try:
            history = [
                {"role": m["role"], "content": m.get("content", "")}
                for m in st.session_state.messages[:-1]
                if m["role"] in ("user", "assistant")
            ]
            result = answer_question(
                vectorstore=st.session_state.vectorstore,
                question=prompt,
                chat_history=history,
                model=st.session_state.selected_model,
                k=retrieval_k,
            )
            bot_msg = {
                "role":    "assistant",
                "content": result["answer"],
                "sources": result["sources"],
            }
            st.session_state.messages.append(bot_msg)
            render_msg(bot_msg)

        except Exception as e:
            err = str(e)
            st.error(f"Error: {err}")
            if "401" in err or "api key" in err.lower():
                st.info("Your OpenRouter API key may be invalid. Check openrouter.ai")