import streamlit as st
import os, re, sys
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from dotenv import load_dotenv
load_dotenv(BASE_DIR / ".env")

from src.rag_engine import (
    initialize_rag, answer_question, get_document_stats,
    AVAILABLE_MODELS, MODEL_NAME,
)

st.set_page_config(page_title="MedBot — Hospital Knowledge Assistant",
                   page_icon="🏥", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;500;600;700;800&family=IBM+Plex+Mono:wght@400;500&display=swap');
:root {
    --bg:#071a1a; --surf:#0d2626; --card:#0f2d2d; --elev:#133333;
    --t700:#0f5e5e; --t600:#0d7a7a; --t500:#0d9488; --t400:#2dd4bf; --t300:#5eead4; --t200:#99f6e4;
    --e500:#10b981; --e400:#34d399; --e300:#6ee7b7;
    --amber:#fbbf24; --rose:#fb7185;
    --tx:#e2fafa; --txs:#94b8b8; --txm:#4d7a7a;
    --bdr:rgba(13,148,136,.2); --bdrb:rgba(45,212,191,.35); --bdre:rgba(52,211,153,.3);
    --glow:0 0 24px rgba(13,148,136,.35);
}
*,body{font-family:'Sora',sans-serif;}
html,body,.stApp{background:var(--bg);color:var(--tx);}

/* Sidebar */
section[data-testid="stSidebar"]{background:linear-gradient(170deg,#030f0f,#071a1a,#0a2020)!important;border-right:1px solid var(--bdrb);}
section[data-testid="stSidebar"] *{color:var(--tx)!important;}
section[data-testid="stSidebar"] label{color:var(--t400)!important;font-size:.68rem!important;font-weight:700!important;letter-spacing:.12em!important;text-transform:uppercase!important;}

.sb-logo{background:linear-gradient(135deg,#0a2020,#0d2e2e);border-bottom:1px solid var(--bdrb);padding:26px 20px 20px;margin:-1rem -1rem 0;text-align:center;}
.sb-logo-icon{font-size:2.5rem;display:block;filter:drop-shadow(0 0 14px rgba(45,212,191,.65));animation:float 3s ease-in-out infinite;}
@keyframes float{0%,100%{transform:translateY(0)}50%{transform:translateY(-5px)}}
.sb-logo-name{font-size:1.05rem;font-weight:800;color:var(--t300)!important;letter-spacing:.05em;margin-top:8px;}
.sb-logo-tag{font-size:.6rem;color:var(--txm)!important;letter-spacing:.16em;text-transform:uppercase;margin-top:3px;}

.sb-on{display:flex;align-items:center;gap:8px;background:rgba(16,185,129,.08);border:1px solid var(--bdre);border-radius:8px;padding:8px 12px;margin:12px 0;font-size:.75rem;font-weight:600;color:var(--e400)!important;}
.sb-off{display:flex;align-items:center;gap:8px;background:rgba(249,115,22,.08);border:1px solid rgba(249,115,22,.3);border-radius:8px;padding:8px 12px;margin:12px 0;font-size:.75rem;font-weight:600;color:#fb923c!important;}
.sb-dot-on{width:8px;height:8px;border-radius:50%;background:var(--e400);box-shadow:0 0 8px var(--e400);animation:pu 2s infinite;flex-shrink:0;}
.sb-dot-off{width:8px;height:8px;border-radius:50%;background:#f97316;box-shadow:0 0 8px #f97316;flex-shrink:0;}
@keyframes pu{0%,100%{opacity:1}50%{opacity:.35}}

.sb-section{font-size:.62rem;font-weight:700;color:var(--t500)!important;letter-spacing:.15em;text-transform:uppercase;padding:14px 0 6px;border-top:1px solid var(--bdr);margin-top:4px;}
.mc{background:var(--card);border:1px solid var(--bdr);border-radius:10px;padding:13px 8px;text-align:center;}
.mv{font-size:1.45rem;font-weight:800;color:var(--t400);font-family:'IBM Plex Mono',monospace;}
.ml{font-size:.6rem;color:var(--txm);text-transform:uppercase;letter-spacing:.1em;margin-top:3px;}
.doc-pill{display:flex;align-items:center;gap:6px;background:rgba(13,148,136,.07);border:1px solid var(--bdr);border-radius:6px;padding:5px 10px;margin:3px 0;font-size:.68rem;color:var(--t300)!important;font-family:'IBM Plex Mono',monospace;}

/* Main */
.main .block-container{background:transparent;padding:1.5rem 2rem 2rem;max-width:1050px;}

/* Header */
.hdr{position:relative;overflow:hidden;background:linear-gradient(135deg,#0a2020,#0d2e2e,#0f3535);border:1px solid var(--bdrb);border-radius:20px;padding:28px 34px;margin-bottom:18px;}
.hdr::before{content:'';position:absolute;top:-50px;right:-50px;width:240px;height:240px;background:radial-gradient(circle,rgba(13,148,136,.14),transparent 70%);border-radius:50%;}
.hdr::after{content:'';position:absolute;bottom:-40px;left:30%;width:180px;height:180px;background:radial-gradient(circle,rgba(16,185,129,.07),transparent 70%);border-radius:50%;}
.hi{position:relative;z-index:1;display:flex;align-items:center;gap:22px;}
.hem{width:62px;height:62px;background:linear-gradient(135deg,var(--t700),var(--t500));border-radius:16px;display:flex;align-items:center;justify-content:center;font-size:2rem;flex-shrink:0;box-shadow:var(--glow);}
.ht{font-size:1.7rem;font-weight:800;color:var(--tx);letter-spacing:-.03em;line-height:1.1;}
.ht span{color:var(--t400);}
.hs{font-size:.78rem;color:var(--txs);margin-top:4px;letter-spacing:.01em;}
.hr{margin-left:auto;text-align:right;flex-shrink:0;}
.bon{display:inline-flex;align-items:center;gap:6px;background:rgba(16,185,129,.12);border:1px solid var(--bdre);color:var(--e400);padding:5px 14px;border-radius:20px;font-size:.68rem;font-weight:700;letter-spacing:.1em;text-transform:uppercase;}
.boff{display:inline-flex;align-items:center;gap:6px;background:rgba(249,115,22,.1);border:1px solid rgba(249,115,22,.3);color:#fb923c;padding:5px 14px;border-radius:20px;font-size:.68rem;font-weight:700;letter-spacing:.1em;text-transform:uppercase;}
.bdot{width:7px;height:7px;border-radius:50%;animation:pu 2s infinite;}
.bdon{background:var(--e400);box-shadow:0 0 6px var(--e400);}
.bdoff{background:#f97316;}
.htime{font-size:.67rem;color:var(--txm);margin-top:6px;font-family:'IBM Plex Mono',monospace;}

.dept-strip{display:flex;gap:7px;flex-wrap:wrap;margin-bottom:18px;}
.dept-badge{background:rgba(13,148,136,.08);border:1px solid var(--bdr);color:var(--txs);padding:4px 13px;border-radius:20px;font-size:.68rem;font-weight:500;}

/* Welcome */
.welcome-box{background:linear-gradient(135deg,rgba(13,148,136,.07),rgba(16,185,129,.04));border:1px solid var(--bdrb);border-radius:16px;padding:22px 26px;margin-bottom:20px;}
.welcome-title{font-size:.97rem;font-weight:700;color:var(--t300);margin-bottom:7px;}
.welcome-text{font-size:.82rem;color:var(--txs);line-height:1.7;}
.welcome-notice{margin-top:12px;padding:9px 13px;background:rgba(251,191,36,.07);border-left:3px solid var(--amber);border-radius:0 6px 6px 0;font-size:.73rem;color:#fcd34d;}
.sug-intro{font-size:.67rem;font-weight:700;color:var(--t500);letter-spacing:.16em;text-transform:uppercase;margin-bottom:12px;}

/* Messages */
.msg-user{background:linear-gradient(135deg,#0e3030,#113838);border:1px solid rgba(45,212,191,.28);border-radius:18px 18px 5px 18px;padding:16px 20px;margin:10px 0 10px 60px;}
.msg-user-lbl{font-size:.58rem;font-weight:700;color:var(--t400);letter-spacing:.15em;text-transform:uppercase;margin-bottom:7px;}
.msg-text{color:var(--tx);font-size:.91rem;line-height:1.68;}

.msg-bot{background:linear-gradient(135deg,#081e1e,#0d2828);border:1px solid var(--bdrb);border-radius:5px 18px 18px 18px;padding:18px 22px;margin:10px 60px 10px 0;}
.msg-bot-lbl{font-size:.58rem;font-weight:700;color:var(--e400);letter-spacing:.15em;text-transform:uppercase;margin-bottom:10px;display:flex;align-items:center;gap:6px;}
.bp{width:6px;height:6px;border-radius:50%;background:var(--e400);animation:pu 2s infinite;}
.msg-bot-text{color:var(--tx);font-size:.91rem;line-height:1.72;}
.msg-bot-text strong{color:var(--t300);}
.sources-row{display:flex;flex-wrap:wrap;gap:5px;margin-top:12px;padding-top:10px;border-top:1px solid var(--bdr);}
.src-chip{background:rgba(13,148,136,.1);border:1px solid var(--bdr);color:var(--t400);padding:3px 9px;border-radius:8px;font-size:.65rem;font-weight:600;font-family:'IBM Plex Mono',monospace;}

/* Suggestion buttons */
.stButton button{background:rgba(13,148,136,.07)!important;color:var(--txs)!important;border:1px solid var(--bdr)!important;border-radius:10px!important;font-size:.82rem!important;font-family:'Sora',sans-serif!important;transition:all .18s!important;text-align:left!important;padding:.6rem 1rem!important;height:auto!important;min-height:52px!important;line-height:1.4!important;white-space:normal!important;font-weight:400!important;}
.stButton button:hover{background:rgba(13,148,136,.15)!important;color:var(--t300)!important;border-color:var(--bdrb)!important;transform:translateY(-1px)!important;box-shadow:0 4px 16px rgba(13,148,136,.15)!important;}
div[data-testid="stSidebar"] .stButton button{background:rgba(13,148,136,.1)!important;color:var(--t300)!important;border-color:var(--bdrb)!important;font-weight:600!important;text-align:center!important;min-height:40px!important;}
div[data-testid="stSidebar"] .stButton button:hover{background:rgba(13,148,136,.22)!important;color:var(--t200)!important;}

[data-testid="stChatInput"]{background:var(--surf)!important;border:1px solid var(--bdrb)!important;border-radius:14px!important;}
[data-testid="stChatInput"] textarea{color:var(--tx)!important;font-family:'Sora',sans-serif!important;font-size:.9rem!important;background:transparent!important;}
[data-testid="stChatInput"]:focus-within{border-color:var(--t500)!important;box-shadow:0 0 0 3px rgba(13,148,136,.15)!important;}
.stSelectbox>div>div{background:var(--card)!important;border:1px solid var(--bdrb)!important;border-radius:8px!important;color:var(--tx)!important;}
::-webkit-scrollbar{width:5px}::-webkit-scrollbar-track{background:var(--bg)}::-webkit-scrollbar-thumb{background:var(--t700);border-radius:3px}
#MainMenu,footer,header{visibility:hidden}.stDeployButton{display:none}
</style>
""", unsafe_allow_html=True)

# session state
if "messages"    not in st.session_state: st.session_state.messages    = []
if "vectorstore" not in st.session_state: st.session_state.vectorstore = None
if "kb_ready"    not in st.session_state: st.session_state.kb_ready    = False
if "sel_model"   not in st.session_state: st.session_state.sel_model   = MODEL_NAME
if "pending"     not in st.session_state: st.session_state.pending     = None

# load KB on startup
if st.session_state.vectorstore is None:
    try:
        vs = initialize_rag(force_rebuild=False)
        if vs:
            st.session_state.vectorstore = vs
            st.session_state.kb_ready    = True
    except Exception:
        pass

ok = st.session_state.kb_ready

# ── SIDEBAR ────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sb-logo"><span class="sb-logo-icon">🏥</span><div class="sb-logo-name">MedBot RAG</div><div class="sb-logo-tag">Hospital Knowledge Assistant</div></div>', unsafe_allow_html=True)
    st.markdown("")
    if ok:
        st.markdown('<div class="sb-on"><div class="sb-dot-on"></div>Knowledge Base Online</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="sb-off"><div class="sb-dot-off"></div>Knowledge Base Offline</div>', unsafe_allow_html=True)

    st.markdown('<div class="sb-section">📚 Knowledge Base</div>', unsafe_allow_html=True)
    ds = get_document_stats()
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f'<div class="mc"><div class="mv">{ds["total_files"]}</div><div class="ml">Docs</div></div>', unsafe_allow_html=True)
    with c2:
        vc = "#34d399" if ok else "#f97316"
        st.markdown(f'<div class="mc"><div class="mv" style="color:{vc};font-size:1.1rem;">{"✓" if ok else "✗"}</div><div class="ml">{"Ready" if ok else "Build"}</div></div>', unsafe_allow_html=True)

    st.markdown("")
    if st.button("🔨 Build / Rebuild Knowledge Base", use_container_width=True):
        with st.spinner("Indexing documents…"):
            try:
                vs = initialize_rag(force_rebuild=True)
                st.session_state.vectorstore = vs
                st.session_state.kb_ready    = True
                st.success("✅ Knowledge base ready!")
                st.rerun()
            except Exception as e:
                st.error(f"❌ {e}")

    if ds["filenames"]:
        st.markdown('<div class="sb-section">📄 Indexed Documents</div>', unsafe_allow_html=True)
        for f in ds["filenames"]:
            st.markdown(f'<div class="doc-pill">📋 {f}</div>', unsafe_allow_html=True)

    st.markdown('<div class="sb-section">🤖 Language Model</div>', unsafe_allow_html=True)
    st.session_state.sel_model = st.selectbox("Model", AVAILABLE_MODELS, index=0, label_visibility="collapsed")

    st.markdown('<div class="sb-section">⚙️ Retrieval Settings</div>', unsafe_allow_html=True)
    retrieval_k = st.slider("Context chunks (k)", 3, 10, 6)

    st.markdown("")
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.pending  = None
        st.rerun()

    st.markdown('<div style="text-align:center;margin-top:18px;padding-top:12px;border-top:1px solid rgba(13,148,136,.12);font-size:.58rem;color:#1d3d3d;letter-spacing:.1em;">MEDBOT v3.0 · INTERNAL USE ONLY</div>', unsafe_allow_html=True)

# ── HEADER ─────────────────────────────────────────────────
now = datetime.now().strftime("%A, %d %B %Y · %H:%M")
if ok:
    badge = '<div class="bon"><div class="bdot bdon"></div>SYSTEM ONLINE</div>'
else:
    badge = '<div class="boff"><div class="bdot bdoff"></div>OFFLINE</div>'

st.markdown(f'<div class="hdr"><div class="hi"><div class="hem">🏥</div><div><div class="ht">Hospital <span>Knowledge</span> Assistant</div><div class="hs">Internal RAG System · SOPs · Policies · Clinical Manuals · Nursing Protocols</div></div><div class="hr">{badge}<div class="htime">{now}</div></div></div></div>', unsafe_allow_html=True)
st.markdown('<div class="dept-strip"><span class="dept-badge">Emergency</span><span class="dept-badge">Pharmacy</span><span class="dept-badge">ICU / CCU</span><span class="dept-badge">Radiology</span><span class="dept-badge">Infection Control</span><span class="dept-badge">Nursing</span><span class="dept-badge">Compliance</span><span class="dept-badge">Administration</span></div>', unsafe_allow_html=True)

# ── RENDER MESSAGE ──────────────────────────────────────────
def render_msg(msg):
    if msg["role"] == "user":
        st.markdown(f'<div class="msg-user"><div class="msg-user-lbl">👤 Staff Question</div><div class="msg-text">{msg["content"]}</div></div>', unsafe_allow_html=True)
    elif msg["role"] == "assistant":
        html = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', msg["content"].replace("\n","<br>"))
        srcs = "".join(f'<span class="src-chip">📄 {s}</span>' for s in msg.get("sources",[]))
        src_block = f'<div class="sources-row">{srcs}</div>' if srcs else ""
        st.markdown(f'<div class="msg-bot"><div class="msg-bot-lbl"><div class="bp"></div>MedBot · AI Response</div><div class="msg-bot-text">{html}</div>{src_block}</div>', unsafe_allow_html=True)

# ── GET ANSWER FUNCTION ─────────────────────────────────────
def get_answer(question):
    api_key = os.getenv("OPENROUTER_API_KEY","")
    if not api_key:
        return {"answer":"❌ OPENROUTER_API_KEY not found. Create a .env file with your key.", "sources":[]}
    if not st.session_state.kb_ready or st.session_state.vectorstore is None:
        return {"answer":"❌ Knowledge base not ready. Click Build in the sidebar.", "sources":[]}
    try:
        import src.rag_engine as _re
        _re.OPENROUTER_API_KEY = api_key
        return answer_question(
            vectorstore=st.session_state.vectorstore,
            question=question,
            model=st.session_state.sel_model,
            k=retrieval_k,
        )
    except Exception as e:
        return {"answer":f"❌ Error: {e}", "sources":[]}

# ── PROCESS PENDING ─────────────────────────────────────────
if st.session_state.pending:
    question = st.session_state.pending
    st.session_state.pending = None
    # render all existing messages
    for msg in st.session_state.messages:
        render_msg(msg)
    # get and save answer
    with st.spinner("🔍 Searching hospital documents…"):
        result = get_answer(question)
    st.session_state.messages.append({
        "role":"assistant",
        "content":result["answer"],
        "sources":result["sources"]
    })
    st.rerun()

elif not st.session_state.messages:
    # ── WELCOME ─────────────────────────────────────────────
    st.markdown('<div class="welcome-box"><div class="welcome-title">👋 Welcome, Healthcare Professional</div><div class="welcome-text">Ask me anything about hospital policies, clinical SOPs, medication protocols, infection control, emergency codes, nursing handoffs, and administrative guidelines. I search your official hospital documents and return accurate sourced answers.</div><div class="welcome-notice">⚠️ For clinical emergencies, always follow direct physician orders. MedBot provides policy guidance only.</div></div>', unsafe_allow_html=True)
    st.markdown('<div class="sug-intro">💡 Frequently Asked Questions</div>', unsafe_allow_html=True)

    SUGS = [
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
    for i, sug in enumerate(SUGS):
        with cols[i % 2]:
            if st.button(sug, key=f"s{i}", use_container_width=True):
                st.session_state.messages.append({"role":"user","content":sug})
                st.session_state.pending = sug
                st.rerun()
else:
    # ── CHAT HISTORY ─────────────────────────────────────────
    for msg in st.session_state.messages:
        render_msg(msg)

# ── CHAT INPUT ──────────────────────────────────────────────
prompt = st.chat_input("Ask about any hospital policy, SOP, or clinical procedure…")
if prompt:
    st.session_state.messages.append({"role":"user","content":prompt})
    st.session_state.pending = prompt
    st.rerun()