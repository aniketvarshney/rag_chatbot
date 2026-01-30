"""Streamlit UI for Agentic RAG System (Groq + LangGraph)"""

import os
import sys
import time
from pathlib import Path

import streamlit as st
from langchain_groq import ChatGroq

# -------------------------------------------------------------------
# Streamlit page config (must be first Streamlit command)
# -------------------------------------------------------------------
st.set_page_config(
    page_title="🤖 RAG Search",
    page_icon="🔍",
    layout="centered",
)

# -------------------------------------------------------------------
# Path setup
# -------------------------------------------------------------------
sys.path.append(str(Path(__file__).parent))

# -------------------------------------------------------------------
# Imports from project
# -------------------------------------------------------------------
from src.config.config import Config
from src.document_ingestion.document_processor import DocumentProcessor
from src.vectorstore.vectorstore import VectorStore
from src.graph_builder.graph_builder import GraphBuilder

# -------------------------------------------------------------------
# Safety check: API key
# -------------------------------------------------------------------
if not os.environ.get("GROQ_API_KEY"):
    st.error("❌ GROQ_API_KEY is missing. Please add it in Streamlit Secrets.")
    st.stop()

# -------------------------------------------------------------------
# Styling
# -------------------------------------------------------------------
st.markdown(
    """
    <style>
    .stButton > button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------------------------------------------------
# Session state initialization
# -------------------------------------------------------------------
def init_session_state():
    if "rag_system" not in st.session_state:
        st.session_state.rag_system = None
    if "initialized" not in st.session_state:
        st.session_state.initialized = False
    if "history" not in st.session_state:
        st.session_state.history = []


# -------------------------------------------------------------------
# RAG system initialization (cached)
# -------------------------------------------------------------------
@st.cache_resource
def initialize_rag():
    try:
        # ✅ Groq LLM (HARDCODED MODEL — FIXES ERROR)
        llm = ChatGroq(
            model="llama-3.1-8b-instant",
            groq_api_key=os.environ["GROQ_API_KEY"],
            temperature=0.2,
        )

        # Document processing
        doc_processor = DocumentProcessor(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
        )

        vector_store = VectorStore()

        # Load default documents
        urls = Config.DEFAULT_URLS
        documents = doc_processor.process_urls(urls)

        # Create vector store
        vector_store.create_vectorstore(documents)

        # Build LangGraph
        graph_builder = GraphBuilder(
            retriever=vector_store.get_retriever(),
            llm=llm,
        )
        graph_builder.build()

        return graph_builder, len(documents)

    except Exception as e:
        st.error(f"❌ Failed to initialize system: {e}")
        return None, 0


# -------------------------------------------------------------------
# Main app
# -------------------------------------------------------------------
def main():
    init_session_state()

    st.title("🔍 RAG Document Search")
    st.markdown("Ask questions about the loaded documents")

    # Initialize system
    if not st.session_state.initialized:
        with st.spinner("Loading system..."):
            rag_system, num_chunks = initialize_rag()
            if rag_system:
                st.session_state.rag_system = rag_system
                st.session_state.initialized = True
                st.success(f"✅ System ready! ({num_chunks} document chunks loaded)")

    st.markdown("---")

    # Search UI
    with st.form("search_form"):
        question = st.text_input(
            "Enter your question:",
            placeholder="What would you like to know?",
        )
        submit = st.form_submit_button("🔍 Search")

    # Run query
    if submit and question:
        if st.session_state.rag_system:
            with st.spinner("Searching..."):
                start_time = time.time()

                result = st.session_state.rag_system.run(question)

                elapsed_time = time.time() - start_time

                st.session_state.history.append(
                    {
                        "question": question,
                        "answer": result["answer"],
                        "time": elapsed_time,
                    }
                )

                # Answer
                st.markdown("### 💡 Answer")
                st.success(result["answer"])

                # Sources
                with st.expander("📄 Source Documents"):
                    for i, doc in enumerate(result["retrieved_docs"], 1):
                        st.text_area(
                            f"Document {i}",
                            doc.page_content[:300] + "...",
                            height=100,
                            disabled=True,
                        )

                st.caption(f"⏱️ Response time: {elapsed_time:.2f} seconds")

    # History
    if st.session_state.history:
        st.markdown("---")
        st.markdown("### 📜 Recent Searches")

        for item in reversed(st.session_state.history[-3:]):
            st.markdown(f"**Q:** {item['question']}")
            st.markdown(f"**A:** {item['answer'][:200]}...")
            st.caption(f"Time: {item['time']:.2f}s")
            st.markdown("")


# -------------------------------------------------------------------
# Entry point
# -------------------------------------------------------------------
if __name__ == "__main__":
    main()
