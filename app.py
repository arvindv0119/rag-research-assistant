import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import streamlit as st
from rag_pipeline import RAGPipeline

st.set_page_config(page_title="DocQuery", page_icon="📄", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Syne:wght@400;600;700&display=swap');

* { font-family: 'Syne', sans-serif; }
code, pre { font-family: 'DM Mono', monospace !important; }

.stApp { background: #0f1117; }
section[data-testid="stSidebar"] { background: #1a1a1a; }

.main .block-container { padding-top: 2rem; max-width: 1100px; }

h1 { 
    font-size: 2.4rem !important; 
    font-weight: 700 !important;
    color: #ffffff !important;
    letter-spacing: -1px;
    border-bottom: 3px solid #ffffff;
    padding-bottom: 8px;
    margin-bottom: 4px !important;
}

h3 { 
    font-size: 0.75rem !important; 
    font-weight: 600 !important;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #aaaaaa !important;
    margin-top: 1.5rem !important;
}

.stTextArea textarea {
    background: #1e2029 !important;
    border: 1.5px solid #333 !important;
    border-radius: 4px !important;
    font-family: 'DM Mono', monospace !important;
    font-size: color: #ffffff !important;
    color: #ffffff !important;
}
.stTextArea textarea:focus { border-color: #1a1a1a !important; box-shadow: none !important; }

.stTextInput input {
    background: #1e2029 !important;
    border: 1.5px solid #333 !important;
    border-radius: 4px !important;
    font-size: 0.95rem !important;
    color: #ffffff !important;
    padding: 12px !important;
}
.stTextInput input:focus { border-color: #1a1a1a !important; box-shadow: none !important; }

.stButton button {
    background: #1a1a1a !important;
    color: #f5f0e8 !important;
    border: none !important;
    border-radius: 4px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    letter-spacing: 1px;
    padding: 10px 20px !important;
    width: 100%;
}
.stButton button:hover { background: #333 !important; }

.stFileUploader {
    background: #fff !important;
    border: 1.5px dashed #ccc !important;
    border-radius: 4px !important;
    padding: 8px !important;
}

.result-box {
    background: #1e2029;
    border: 1.5px solid #4f8ef7;
    border-radius: 4px;
    padding: 18px 20px;
    margin-top: 16px;
}
.result-label {
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #888;
    margin-bottom: 6px;
}
.result-text {
    font-size: 0.95rem;
    color: #ffffff;
    line-height: 1.6;
}
.chunk-box {
    background: #16181f;
    border-left: 3px solid #4f8ef7;
    padding: 10px 14px;
    margin-top: 10px;
    border-radius: 0 4px 4px 0;
    font-family: 'DM Mono', monospace;
    font-size: 0.78rem;
    color: #c9d1d9;
    line-height: 1.5;
}
.chunk-meta {
    font-size: 0.65rem;
    color: #aaa;
    margin-bottom: 4px;
    text-transform: uppercase;
    letter-spacing: 1px;
}
.tag {
    display: inline-block;
    background: #1a1a1a;
    color: #f5f0e8;
    font-size: 0.65rem;
    padding: 2px 8px;
    border-radius: 2px;
    font-family: 'DM Mono', monospace;
    margin-right: 6px;
}
div[data-testid="stSuccess"] {
    background: #e8f5e9 !important;
    border: 1px solid #4caf50 !important;
    border-radius: 4px !important;
}
div[data-testid="stWarning"] {
    background: #fff8e1 !important;
    border: 1px solid #ffc107 !important;
    border-radius: 4px !important;
}
</style>
""", unsafe_allow_html=True)

if "pipeline" not in st.session_state:
    st.session_state.pipeline = RAGPipeline(top_k=3)
if "ready" not in st.session_state:
    st.session_state.ready = False
if "history" not in st.session_state:
    st.session_state.history = []

left, right = st.columns([1, 1.6], gap="large")

with left:
    st.title("DocQuery")
    st.caption("paste text or upload a file → ask questions → get answers from your own docs")

    st.markdown("### Input")

    uploaded = st.file_uploader("upload a file", type=["txt", "md", "pdf"], label_visibility="collapsed")

    pasted = st.text_area(
        "or paste text here",
        height=160,
        placeholder="paste any research paper, notes, article...",
        label_visibility="visible"
    )

    if st.button("INDEX DOCUMENTS"):
        pipeline = st.session_state.pipeline
        total = 0
        if uploaded:
            tmp = Path(f"/tmp/{uploaded.name}")
            tmp.write_bytes(uploaded.read())
            total += pipeline.ingest([tmp])
            tmp.unlink()
        if pasted.strip():
            total += pipeline.ingest_text(pasted, source_name="pasted-text")
        if total > 0:
            st.session_state.ready = True
            st.success(f"indexed {total} chunk(s) — ready to query")
        else:
            st.warning("nothing to index. paste some text or upload a file.")

    if st.session_state.ready:
        st.markdown("### Query")
        question = st.text_input("ask something", placeholder="what is dice loss?", label_visibility="collapsed")
        if st.button("SEARCH"):
            if question.strip():
                try:
                    resp = st.session_state.pipeline.query(question)
                    st.session_state.history.insert(0, resp)
                except Exception as e:
                    st.error(str(e))

with right:
    if not st.session_state.history:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("""
<div style='color:#aaa; font-size:0.85rem; margin-top:60px; line-height:2;'>
← index a document on the left<br>
then ask a question<br>
results will appear here
</div>
""", unsafe_allow_html=True)

    for resp in st.session_state.history:
        st.markdown(f"""
<div class='result-box'>
<div class='result-label'>question</div>
<div class='result-text' style='font-weight:600; margin-bottom:12px;'>{resp.query}</div>
<div class='result-label'>retrieved context</div>
""", unsafe_allow_html=True)

        for r in resp.retrieved_chunks:
            score_pct = f"{r.score * 100:.0f}%"
            snippet = r.document.text[:350]
            st.markdown(f"""
<div class='chunk-box'>
<div class='chunk-meta'>
    <span class='tag'>rank {r.rank+1}</span>
    <span class='tag'>sim {score_pct}</span>
    {r.document.source}
</div>
{snippet}{'...' if len(r.document.text) > 350 else ''}
</div>
""", unsafe_allow_html=True)

        st.markdown("</div><br>", unsafe_allow_html=True)
