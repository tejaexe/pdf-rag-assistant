"""Streamlit entry point for the PDF RAG Assistant."""
import streamlit as st

from src.config import AppSettings
from src.rag_service import RAGService

st.set_page_config(page_title="PDF RAG Assistant", page_icon="📚", layout="wide")


@st.cache_resource(show_spinner=False)
def get_service(settings: AppSettings) -> RAGService:
    return RAGService(settings)


def main() -> None:
    st.title("📚 PDF RAG Assistant")
    st.caption("Upload PDFs, then ask questions grounded only in their contents.")
    settings = AppSettings()
    service = get_service(settings)

    with st.sidebar:
        st.header("Document index")
        chunk_size = st.slider("Chunk size (characters)", 400, 1800, settings.chunk_size, 100)
        chunk_overlap = st.slider("Chunk overlap", 0, 400, settings.chunk_overlap, 25)
        top_k = st.slider("Sources to retrieve", 1, 8, settings.top_k)
        uploads = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)
        if st.button("Process documents", type="primary", disabled=not uploads):
            with st.spinner("Extracting, chunking, and indexing documents..."):
                result = service.ingest(uploads, chunk_size, chunk_overlap)
            st.success(f"Indexed {result.documents} document(s) into {result.chunks} chunks.")
            if result.skipped:
                st.warning("Skipped: " + "; ".join(result.skipped))
        if st.button("Clear session index"):
            service.clear()
            st.session_state.messages = []
            st.rerun()
        st.divider()
        st.caption("API keys are read from `.env` and never displayed.")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                with st.expander("Sources used"):
                    for source in message["sources"]:
                        st.markdown(f"- **{source['document']}**, page {source['page']} - {source['excerpt']}")

    question = st.chat_input("Ask about the uploaded PDFs")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("Searching the documents and drafting an answer..."):
                answer = service.answer(question, st.session_state.messages[:-1], top_k)
            st.markdown(answer.text)
            if answer.sources:
                with st.expander("Sources used", expanded=True):
                    for source in answer.sources:
                        st.markdown(f"- **{source['document']}**, page {source['page']} - {source['excerpt']}")
        st.session_state.messages.append({"role": "assistant", "content": answer.text, "sources": answer.sources})


if __name__ == "__main__":
    main()
