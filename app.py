import streamlit as st
import os
import pickle
import numpy as np
# import faiss
import torch
import tempfile
import time

from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# ─────────────────────────────────────────────
#  PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Document PDF Intelligence Bolte",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  GLOBAL CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700;900&family=DM+Sans:wght@300;400;500&display=swap');

/* ── root tokens ── */
:root {
  --ink:    #0d0d0d;
  --paper:  #f5f0e8;
  --cream:  #ede8dc;
  --amber:  #c8923a;
  --amber2: #e8b05a;
  --rust:   #8b3a2a;
  --sage:   #4a5e4a;
  --shadow: rgba(13,13,13,.12);
}

/* ── global reset ── */
html, body, [class*="css"] {
  font-family: 'DM Sans', sans-serif;
  background: var(--paper) !important;
  color: var(--ink) !important;
}

/* ── hide default Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 2rem 4rem !important; max-width: 1100px; }

/* ── sidebar ── */
[data-testid="stSidebar"] {
  background: var(--ink) !important;
  border-right: 3px solid var(--amber);
}
[data-testid="stSidebar"] * { color: var(--paper) !important; }
[data-testid="stSidebar"] .stFileUploader label { color: var(--amber2) !important; font-weight: 500; }
[data-testid="stSidebar"] .stFileUploader section {
  background: #1a1a1a !important;
  border: 2px dashed var(--amber) !important;
  border-radius: 8px;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
  font-family: 'Playfair Display', serif !important;
  color: var(--amber2) !important;
}

/* ── hero title ── */
.hero {
  text-align: center;
  padding: 2.5rem 1rem 1.5rem;
  border-bottom: 2px solid var(--amber);
  margin-bottom: 2rem;
}
.hero h1 {
  font-family: 'Playfair Display', serif;
  font-size: 3.6rem;
  font-weight: 900;
  letter-spacing: -.02em;
  line-height: 1;
  color: var(--ink);
  margin: 0;
}
.hero h1 span { color: var(--amber); }
.hero p {
  font-size: .95rem;
  color: #555;
  margin-top: .6rem;
  letter-spacing: .08em;
  text-transform: uppercase;
}

/* ── chat bubbles ── */
.chat-row { display: flex; gap: 12px; margin-bottom: 1.4rem; align-items: flex-start; }
.chat-row.user  { flex-direction: row-reverse; }

.bubble {
  padding: .85rem 1.15rem;
  border-radius: 18px;
  max-width: 78%;
  line-height: 1.65;
  font-size: .93rem;
  position: relative;
  box-shadow: 0 2px 8px var(--shadow);
}
.bubble.user {
  background: var(--ink);
  color: var(--paper);
  border-bottom-right-radius: 4px;
}
.bubble.bot {
  background: white;
  color: var(--ink);
  border: 1.5px solid var(--cream);
  border-bottom-left-radius: 4px;
}
.bubble.bot::before {
  content: "🔮";
  position: absolute;
  top: -10px; left: -10px;
  font-size: 1.1rem;
}

.avatar {
  width: 36px; height: 36px;
  border-radius: 50%;
  display: flex; align-items: center; justify-content: center;
  font-size: 1rem;
  flex-shrink: 0;
}
.avatar.user { background: var(--rust); color: white; }
.avatar.bot  { background: var(--amber); color: white; }

/* ── score chips ── */
.score-row { display: flex; gap: 8px; flex-wrap: wrap; margin-top: .5rem; }
.score-chip {
  font-size: .7rem;
  background: var(--cream);
  border: 1px solid var(--amber);
  border-radius: 20px;
  padding: 2px 10px;
  color: var(--rust);
  font-weight: 500;
}

/* ── input bar ── */
.stTextInput > div > div > input {
  border: 2px solid var(--amber) !important;
  border-radius: 30px !important;
  padding: .65rem 1.2rem !important;
  font-size: .95rem !important;
  background: white !important;
  color: var(--ink) !important;
  box-shadow: 0 2px 10px var(--shadow) !important;
}
.stTextInput > div > div > input:focus {
  border-color: var(--rust) !important;
  box-shadow: 0 0 0 3px rgba(200,146,58,.18) !important;
}

/* ── buttons ── */
.stButton > button {
  border-radius: 30px !important;
  background: var(--amber) !important;
  color: var(--ink) !important;
  border: none !important;
  font-weight: 600 !important;
  padding: .5rem 1.6rem !important;
  font-size: .9rem !important;
  transition: background .2s, transform .1s !important;
}
.stButton > button:hover {
  background: var(--rust) !important;
  color: white !important;
  transform: translateY(-1px) !important;
}

/* ── status boxes ── */
.status-box {
  padding: .9rem 1.2rem;
  border-radius: 10px;
  margin: .8rem 0;
  font-size: .88rem;
  font-weight: 500;
  border-left: 4px solid;
}
.status-success { background:#edf7ed; border-color: var(--sage); color: var(--sage); }
.status-warning { background:#fff8ec; border-color: var(--amber); color: #7a5000; }
.status-error   { background:#fdf0ee; border-color: var(--rust);  color: var(--rust); }
.status-info    { background:#f0f4ff; border-color: #4a6ab0;      color: #2a4080; }

/* ── chunk expander ── */
.stExpander { border: 1.5px solid var(--cream) !important; border-radius: 10px !important; }
.stExpander summary { font-size: .82rem; color: #888; }

/* ── separator ── */
hr { border-color: var(--cream) !important; }

/* ── spinner text ── */
.stSpinner > div { border-top-color: var(--amber) !important; }

/* ── progress bar ── */
.stProgress > div > div { background: var(--amber) !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  IMPORTS FROM USER'S OWN MODULES
#  (wrapped so Streamlit doesn't crash if they
#   aren't present in the same directory)
# ─────────────────────────────────────────────
try:
    from Day01_pdf_utlis import extract_text, save_file
    from Day02_chunking  import load_file, split_text, save_chunks_pickle
    from Day3_embeddings import save_embeddings
    from Day4_Vector_Store import load_chunk, load_embeddings, build_faiss, save_index, load_index
    MODULES_AVAILABLE = True
except ImportError:
    MODULES_AVAILABLE = False

# ─────────────────────────────────────────────
#  FALLBACK IMPLEMENTATIONS
#  (used when user modules aren't importable;
#   keeps the app self-contained for demo)
# ─────────────────────────────────────────────
import pdfplumber

def _extract_text(pdf_path: str) -> str:
    text = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                text.append(t)
    return "\n\n".join(text)

def _split_text(text: str, chunk_size=500, overlap=50):
    words  = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks

def _create_embeddings(chunks, emb_model):
    return emb_model.encode(chunks, show_progress_bar=False)

def _build_faiss_index(embeddings):
    dim   = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings.astype("float32"))
    return index


# ─────────────────────────────────────────────
#  SESSION STATE INIT
# ─────────────────────────────────────────────
def _init_state():
    defaults = {
        "messages":      [],
        "index":         None,
        "chunks":        None,
        "llm":           None,
        "emb_model":     None,
        "pdf_name":      None,
        "pipeline_done": False,
        "llm_loaded":    False,
        "tmp_dir":       None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()

# ─────────────────────────────────────────────
#  CACHED RESOURCE LOADERS
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource(show_spinner=False)
def get_llm():
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer  = AutoTokenizer.from_pretrained(model_name)
    llm = pipeline(
        "text-generation",
        model=model_name,
        tokenizer=tokenizer,
        torch_dtype=torch.float32,
        device=-1,
    )
    return llm

# ─────────────────────────────────────────────
#  CORE FUNCTIONS
# ─────────────────────────────────────────────
def build_index_from_pdf(pdf_path: str):
    emb_model = get_embedding_model()
    if MODULES_AVAILABLE:
        text   = extract_text(pdf_path)
        chunks = split_text(text)
    else:
        text   = _extract_text(pdf_path)
        chunks = _split_text(text)

    embeddings = _create_embeddings(chunks, emb_model)
    index      = _build_faiss_index(np.array(embeddings))
    return index, chunks


def semantic_search(question: str, index, chunks, k=3):
    emb_model = get_embedding_model()
    q_vec     = emb_model.encode([question]).astype("float32")
    dists, idxs = index.search(q_vec, k=k)
    results  = [chunks[i] for i in idxs[0]]
    scores   = dists[0].tolist()
    return results, scores


def get_answer(llm, question: str, relevant_chunks: list) -> str:
    context = "\n\n".join(relevant_chunks)
    prompt  = f"""<|system|>
You are a strict PDF assistant. You ONLY answer from the context below.
RULES:
1. Use ONLY the context provided. No outside knowledge.
2. If the answer cannot be found in context, respond exactly: "This information is not available in the PDF."
3. Do NOT guess. Do NOT use training knowledge. Do NOT make up facts.
</s>
<|user|>
CONTEXT FROM PDF:
{context}

QUESTION: {question}

Answer ONLY from the context. If not found, say "This information is not available in the PDF."
</s>
<|assistant|>
"""
    output = llm(
        prompt,
        temperature=0.3,
        repetition_penalty=1.15,
        max_new_tokens=350,
        do_sample=True,
    )
    raw    = output[0]["generated_text"]
    answer = raw.split("<|assistant|>")[-1].strip()
    return answer

# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔮 DocOracle")
    st.markdown("---")
    st.markdown("### Upload Your PDF")

    uploaded = st.file_uploader(
        "Drop a PDF to begin",
        type=["pdf"],
        label_visibility="collapsed",
    )

    if uploaded:
        new_pdf = uploaded.name != st.session_state.get("pdf_name")

        if new_pdf:
            # reset state for new document
            st.session_state.messages      = []
            st.session_state.index         = None
            st.session_state.chunks        = None
            st.session_state.pipeline_done = False
            st.session_state.pdf_name      = uploaded.name

            # save to temp file
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
            tmp.write(uploaded.read())
            tmp.flush()
            st.session_state.tmp_dir = tmp.name

        if not st.session_state.pipeline_done:
            with st.spinner("📖 Indexing document…"):
                try:
                    idx, cks = build_index_from_pdf(st.session_state.tmp_dir)
                    st.session_state.index         = idx
                    st.session_state.chunks        = cks
                    st.session_state.pipeline_done = True
                    st.markdown(
                        f'<div class="status-success">✅ Indexed <b>{uploaded.name}</b><br>'
                        f'<span style="font-weight:300">{len(cks)} chunks created</span></div>',
                        unsafe_allow_html=True,
                    )
                except Exception as e:
                    st.markdown(
                        f'<div class="status-error">❌ Indexing failed:<br>{e}</div>',
                        unsafe_allow_html=True,
                    )
        else:
            st.markdown(
                f'<div class="status-success">✅ <b>{uploaded.name}</b> ready<br>'
                f'<span style="font-weight:300">{len(st.session_state.chunks)} chunks</span></div>',
                unsafe_allow_html=True,
            )

    st.markdown("---")
    st.markdown("### Load Language Model")

    if not st.session_state.llm_loaded:
        if st.button("⚡ Load TinyLlama"):
            with st.spinner("Loading model (first time may take a minute)…"):
                try:
                    st.session_state.llm        = get_llm()
                    st.session_state.llm_loaded = True
                    st.markdown(
                        '<div class="status-success">✅ TinyLlama loaded & ready</div>',
                        unsafe_allow_html=True,
                    )
                except Exception as e:
                    st.markdown(
                        f'<div class="status-error">❌ Model load failed:<br>{e}</div>',
                        unsafe_allow_html=True,
                    )
    else:
        st.markdown(
            '<div class="status-success">✅ TinyLlama is loaded</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ── config knobs ──
    st.markdown("### Settings")
    threshold   = st.slider("Relevance threshold", 0.5, 3.0, 1.5, 0.1,
                            help="Higher = stricter. Queries with distance above this return 'not found'.")
    top_k       = st.slider("Top-k chunks", 1, 6, 3,
                            help="Number of context chunks sent to the model.")
    show_chunks = st.checkbox("Show retrieved chunks", value=False)

    st.markdown("---")
    if st.button("🗑️ Clear chat"):
        st.session_state.messages = []
        st.rerun()

    st.markdown(
        "<p style='font-size:.72rem;color:#888;margin-top:1rem;'>"
        "DocOracle answers <em>only</em> from your uploaded PDF. "
        "Powered by TinyLlama + FAISS.</p>",
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────
#  MAIN PANEL
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>Doc<span>Oracle</span></h1>
  <p>PDF-Grounded Intelligence · No Hallucinations · Strictly Source-Based</p>
</div>
""", unsafe_allow_html=True)

# ── readiness banner ──
col1, col2 = st.columns(2)
with col1:
    if st.session_state.pipeline_done:
        st.markdown('<div class="status-success">📄 Document indexed</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-warning">📄 No document loaded — upload a PDF in the sidebar</div>', unsafe_allow_html=True)
with col2:
    if st.session_state.llm_loaded:
        st.markdown('<div class="status-success">🤖 TinyLlama ready</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-warning">🤖 Model not loaded — click "Load TinyLlama" in the sidebar</div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── chat history ──
chat_container = st.container()

with chat_container:
    if not st.session_state.messages:
        st.markdown("""
        <div style="text-align:center;padding:3rem 1rem;color:#aaa;">
          <div style="font-size:3rem">🔮</div>
          <div style="font-family:'Playfair Display',serif;font-size:1.3rem;color:#999;margin-top:.5rem;">
            Your conversation will appear here
          </div>
          <div style="font-size:.85rem;margin-top:.4rem;">
            Upload a PDF → Load the model → Ask anything
          </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        for msg in st.session_state.messages:
            role   = msg["role"]
            bubble = "user" if role == "user" else "bot"
            avatar = "👤" if role == "user" else "🔮"
            av_cls = "user" if role == "user" else "bot"

            chunks_html = ""
            if show_chunks and role == "assistant" and msg.get("chunks"):
                chips = "".join(
                    f'<span class="score-chip">chunk {i+1} · dist {s:.3f}</span>'
                    for i, s in enumerate(msg.get("scores", []))
                )
                chunks_html = f'<div class="score-row">{chips}</div>'

            st.markdown(f"""
            <div class="chat-row {role}">
              <div class="avatar {av_cls}">{avatar}</div>
              <div class="bubble {bubble}">
                {msg["content"]}
                {chunks_html}
              </div>
            </div>
            """, unsafe_allow_html=True)

            if show_chunks and role == "assistant" and msg.get("chunks"):
                with st.expander("📄 Retrieved context chunks"):
                    for i, (c, s) in enumerate(zip(msg["chunks"], msg.get("scores", []))):
                        st.markdown(f"**Chunk {i+1}** — distance `{s:.4f}`")
                        st.caption(c[:600] + ("…" if len(c) > 600 else ""))
                        st.markdown("---")

# ── input row ──
st.markdown("<br>", unsafe_allow_html=True)
input_col, btn_col = st.columns([5, 1])

with input_col:
    question = st.text_input(
        "question",
        placeholder="Ask anything about your PDF…",
        label_visibility="collapsed",
        key="question_input",
    )
with btn_col:
    send = st.button("Ask →", use_container_width=True)

# ─────────────────────────────────────────────
#  QUERY HANDLER
# ─────────────────────────────────────────────
if send and question.strip():
    # guards
    if not st.session_state.pipeline_done:
        st.warning("⚠️ Please upload and index a PDF first.")
        st.stop()
    if not st.session_state.llm_loaded:
        st.warning("⚠️ Please load the TinyLlama model first.")
        st.stop()

    # append user message
    st.session_state.messages.append({"role": "user", "content": question})

    with st.spinner("🔍 Searching document…"):
        rel_chunks, scores = semantic_search(
            question,
            st.session_state.index,
            st.session_state.chunks,
            k=top_k,
        )

    if scores[0] > threshold:
        answer = "⚠️ This information is not available in the PDF."
        rel_chunks, scores = [], []
    else:
        with st.spinner("🧠 Generating answer…"):
            answer = get_answer(st.session_state.llm, question, rel_chunks)

    st.session_state.messages.append({
        "role":    "assistant",
        "content": answer,
        "chunks":  rel_chunks,
        "scores":  scores,
    })

    st.rerun()

# ─────────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────────
st.markdown("""
<hr style="margin-top:3rem">
<p style="text-align:center;font-size:.75rem;color:#bbb;letter-spacing:.06em;">
  DOCORACLE · STRICTLY PDF-GROUNDED · TINYLLAMA 1.1B · FAISS VECTOR SEARCH
</p>
""", unsafe_allow_html=True)
