import os
import io
import time
import tempfile
import streamlit as st

from functions import create_rag_system, quick_setup  # your functions.py

# Page config
st.set_page_config(
    page_title="Local RAG Demo (Ollama + Chroma)",
    page_icon="📚",
    layout="wide",
)

# Session state init
if "rag" not in st.session_state:
    st.session_state.rag = create_rag_system(
        ollama_base=os.getenv("OLLAMA_BASE", "http://localhost:11434")
    )
if "history" not in st.session_state:
    st.session_state.history = []   # [{"role": "user"/"assistant", "content": "..."}]
if "uploaded_pdf_path" not in st.session_state:
    st.session_state.uploaded_pdf_path = None

rag = st.session_state.rag

# Sidebar: settings
with st.sidebar:
    st.header("⚙️ Settings")
    st.caption("Set up models first, then upload and process a PDF.")

    # Ollama base
    ollama_base = st.text_input(
        "Ollama Base URL",
        value=rag.ollama_base,
        help="e.g., http://localhost:11434",
    )
    if ollama_base != rag.ollama_base:
        rag.ollama_base = ollama_base  # update in RAGSystem

    # Model selection
    emb_model = st.text_input(
        "Embedding model",
        value="nomic-embed-text",
        help="Must be pulled in Ollama",
    )
    chat_model = st.text_input(
        "Chat model",
        value="llama3.2",
        help="e.g., llama3.2 / mistral / qwen2.5",
    )
    temperature = st.slider("Temperature (creative ↔ stable)", 0.0, 1.0, 0.2, 0.1)

    # Initialize models
    if st.button("🚀 Initialize models", use_container_width=True):
        with st.spinner("Initializing embedding & chat models..."):
            ok, msg = quick_setup(rag, embedding_model=emb_model, chat_model=chat_model)
            if not ok:
                st.error(f"Failed to initialize: {msg}")
            else:
                # apply temperature
                _ok2, _ = rag.setup_chat_model(chat_model, temperature=temperature)
                st.success("Models ready")

    st.divider()

    # PDF upload & params
    st.subheader("📄 Document")
    uploaded = st.file_uploader("Upload PDF", type=["pdf"])
    chunk_size    = st.number_input("Chunk Size", min_value=200, max_value=4000, value=1000, step=100)
    chunk_overlap = st.number_input("Chunk Overlap", min_value=0, max_value=1000, value=200, step=50)

    if st.button("📚 Process PDF", use_container_width=True, disabled=(uploaded is None)):
        if uploaded is None:
            st.warning("Please upload a PDF first.")
        else:
            # Save to temp file; pass path to PyPDFLoader
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded.getbuffer())
                st.session_state.uploaded_pdf_path = tmp.name

            # Progress bar (for demo)
            prog = st.progress(0)
            with st.spinner("Embedding & building index..."):
                ok, msg, pages_cnt, chunks_cnt = rag.process_pdf(
                    st.session_state.uploaded_pdf_path,
                    chunk_size=int(chunk_size),
                    chunk_overlap=int(chunk_overlap),
                )
                for p in range(1, 101, 10):
                    time.sleep(0.03)
                    prog.progress(min(p, 100))

            if ok:
                st.success(f"Processed: {pages_cnt} pages → {chunks_cnt} chunks")
            else:
                st.error(f"Processing failed: {msg}")

    st.divider()

    st.subheader("🔍 Retrieval parameters")
    top_k = st.slider("Top-K", 1, 20, 8)
    st.caption("Number of chunks to retrieve per question.")


# Main area: title + status + chat
st.title("📚 Retrieval-Augmented Generation (Local)")

# Status bar
status = rag.get_system_status()
c1, c2, c3, c4 = st.columns(4)
c1.metric("Embedding", "✅ Ready" if status["embedding_ready"] else "❌ Not set")
c2.metric("Chat Model", "✅ Ready" if status["chat_ready"] else "❌ Not set")
c3.metric("Vector DB", "✅ Ready" if status["vector_db_ready"] else "❌ Not built")
c4.metric("System", "✅ OK" if status["fully_ready"] else "❌ Missing parts")

st.divider()

# Chat layout
left, right = st.columns([2, 1])

with left:
    st.subheader("💬 Chat")
    # History
    for msg in st.session_state.history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Input box
    user_q = st.chat_input("Type your question (e.g., Who is the author?)")
    if user_q:
        # record user message
        st.session_state.history.append({"role": "user", "content": user_q, "ts": time.time()})
        ans, docs = rag.ask_question(user_q, k=int(top_k))
        st.session_state.history.append({"role": "assistant", "content": ans, "ts": time.time()})
        st.session_state.last_docs = docs
        st.rerun()

with right:
    st.subheader("📂 Evidence")
    docs_to_show = st.session_state.get("last_docs", [])
    if not docs_to_show:
        st.caption("Relevant snippets will appear here after you ask a question.")
    else:
        for i, d in enumerate(docs_to_show, 1):
            meta = d.metadata or {}
            page0 = meta.get("page")
            page = (page0 + 1) if isinstance(page0, int) else page0
            src  = os.path.basename(meta.get("source", ""))
            with st.expander(f"[{i}] {src} - p{page}"):
                st.write(d.page_content[:1000])        # Evidence panel on the right
        

st.divider()

# Utility buttons
col_a, col_b = st.columns(2)
with col_a:
    if st.button("🧹 Clear chat", use_container_width=True):
        st.session_state.history = []
        st.success("Chat cleared")
        st.rerun()
        
        
with col_b:
    if st.button("♻️ Reset system", use_container_width=True):
        st.session_state.rag = create_rag_system(ollama_base=ollama_base)
        st.session_state.history = []
        st.session_state.uploaded_pdf_path = None
        st.success("System reset (please re-initialize and process a PDF)")
        st.rerun()

#Format st.session_state.history -> plain text transcript.
def history_to_text(history: list[dict]) -> str:
    if not history:
        return "No messages."

    lines = ["# Chat Transcript", ""]
    for i, m in enumerate(history, 1):
        role = m.get("role", "assistant").upper()
        content = (m.get("content") or "").strip()
        ts = m.get("ts")
        prefix = f"[{i}] {role}"
        if ts:
            from datetime import datetime
            dt = datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
            prefix += f" @ {dt}"
        lines += [prefix, content, ""]
    return "\n".join(lines)


# ---- Export chat as .txt ----
txt_data = history_to_text(st.session_state.get("history", []))
st.download_button(
    label="📝 Download chat (.txt)",
    data=txt_data.encode("utf-8"),
    file_name="chat_transcript.txt",
    mime="text/plain",
    use_container_width=True,
    disabled=not bool(st.session_state.get("history"))
)

