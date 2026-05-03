import streamlit as st
import os
import tempfile
from rag_engine import (
    process_document_and_build_index,
    load_vector_store,
    build_qa_chain,
    ask_question
)

# ─── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Document Q&A System",
    page_icon="📄",
    layout="wide"
)

st.title("📄 AI-Powered Document Q&A")
st.caption("Upload a document and ask questions about it.")

# ─── Session state (keeps data between reruns) ────────────────────────────────
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ─── Sidebar: Upload document ─────────────────────────────────────────────────
with st.sidebar:
    st.header("Upload Document")
    uploaded_file = st.file_uploader(
        "Choose a PDF or TXT file",
        type=["pdf", "txt"]
    )

    if uploaded_file and st.button("Process Document", type="primary"):
        with st.spinner("Processing... this may take a moment."):
            # Save uploaded file to a temp path
            suffix = ".pdf" if uploaded_file.name.endswith(".pdf") else ".txt"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            # Run ingestion pipeline
            vector_store = process_document_and_build_index(tmp_path)
            st.session_state.qa_chain = build_qa_chain(vector_store)
            st.session_state.chat_history = []
            os.unlink(tmp_path)  # clean up temp file

        st.success(f"✅ '{uploaded_file.name}' processed!")

    # Load existing index if available
    if st.session_state.qa_chain is None and os.path.exists("faiss_index"):
        if st.button("Load Existing Index"):
            with st.spinner("Loading..."):
                vs = load_vector_store()
                st.session_state.qa_chain = build_qa_chain(vs)
            st.success("Index loaded!")

    st.divider()
    st.caption("Built with LangChain + FAISS + OpenAI")

# ─── Main: Chat interface ─────────────────────────────────────────────────────
if st.session_state.qa_chain is None:
    st.info("👈 Upload a document in the sidebar to get started.")
else:
    # Show chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg["role"] == "assistant" and "sources" in msg:
                with st.expander("📎 Source chunks used"):
                    for i, src in enumerate(msg["sources"], 1):
                        st.caption(f"**Chunk {i}:** {src}...")

    # Input box at the bottom
    user_question = st.chat_input("Ask a question about your document...")

    if user_question:
        # Show user message
        with st.chat_message("user"):
            st.write(user_question)
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_question
        })

        # Get answer
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = ask_question(user_question, st.session_state.qa_chain)
            st.write(result["answer"])
            with st.expander("📎 Source chunks used"):
                for i, src in enumerate(result["sources"], 1):
                    st.caption(f"**Chunk {i}:** {src}...")

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": result["answer"],
            "sources": result["sources"]
        })