"""
📄 RAG-Powered Document Intelligence Assistant
Streamlit UI for document upload, processing, and intelligent Q&A.
"""

import os
import sys
import tempfile
import streamlit as st
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.document_processor import DocumentProcessor
from src.vector_store import EmbeddingEngine, FAISSVectorStore
from src.rag_pipeline import (
    RAGPipeline,
    OpenAIProvider,
    AnthropicProvider,
    OllamaProvider,
)


# ──────────────────────────────────────────────
# Page Configuration
# ──────────────────────────────────────────────

st.set_page_config(
    page_title="Document Intelligence Assistant",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #888;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
    }
    .source-card {
        background: #f8f9fa;
        border-left: 4px solid #667eea;
        padding: 12px 16px;
        margin: 8px 0;
        border-radius: 0 8px 8px 0;
    }
    .stChatMessage {
        border-radius: 12px;
    }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Session State Initialization
# ──────────────────────────────────────────────

def init_session_state():
    defaults = {
        "vector_store": None,
        "rag_pipeline": None,
        "processed_docs": [],
        "chat_history": [],
        "embedding_engine": None,
        "is_initialized": False,
        "total_chunks": 0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_session_state()


# ──────────────────────────────────────────────
# Sidebar: Configuration
# ──────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Configuration")

    # LLM Provider Selection
    st.markdown("### 🤖 LLM Provider")
    provider = st.selectbox(
        "Select Provider",
        ["OpenAI", "Anthropic", "Ollama (Local)"],
        help="Choose which LLM to use for generating answers.",
    )

    if provider == "OpenAI":
        api_key = st.text_input("OpenAI API Key", type="password", value=os.getenv("OPENAI_API_KEY", ""))
        model = st.selectbox("Model", ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"])
    elif provider == "Anthropic":
        api_key = st.text_input("Anthropic API Key", type="password", value=os.getenv("ANTHROPIC_API_KEY", ""))
        model = st.selectbox("Model", ["claude-sonnet-4-20250514", "claude-haiku-4-5-20251001"])
    else:
        api_key = ""
        model = st.text_input("Ollama Model", value="llama3.2")
        ollama_url = st.text_input("Ollama URL", value="http://localhost:11434")

    st.markdown("---")

    # Chunking Configuration
    st.markdown("### 📑 Document Processing")
    chunk_size = st.slider("Chunk Size (chars)", 256, 2048, 512, 64)
    chunk_overlap = st.slider("Chunk Overlap (chars)", 0, 200, 50, 10)
    chunking_strategy = st.selectbox(
        "Chunking Strategy",
        ["semantic", "sentence", "paragraph", "fixed"],
        help="How to split documents into chunks.",
    )

    st.markdown("---")

    # Retrieval Configuration
    st.markdown("### 🔍 Retrieval Settings")
    top_k = st.slider("Top K Results", 1, 20, 5)
    score_threshold = st.slider("Score Threshold", 0.0, 1.0, 0.1, 0.05)

    st.markdown("---")

    # Embedding Model
    st.markdown("### 🧠 Embedding Model")
    embedding_model = st.selectbox(
        "Model",
        ["all-MiniLM-L6-v2", "all-mpnet-base-v2"],
        help="MiniLM is faster; mpnet is more accurate.",
    )

    st.markdown("---")

    # Stats
    if st.session_state.processed_docs:
        st.markdown("### 📊 Knowledge Base Stats")
        st.metric("Documents", len(st.session_state.processed_docs))
        st.metric("Total Chunks", st.session_state.total_chunks)
        if st.session_state.vector_store:
            st.metric("Vectors", st.session_state.vector_store.size)


# ──────────────────────────────────────────────
# Helper Functions
# ──────────────────────────────────────────────

def initialize_pipeline():
    """Initialize or reinitialize the RAG pipeline."""
    with st.spinner("🔧 Initializing embedding engine..."):
        embedding_engine = EmbeddingEngine(model_name=embedding_model)
        vector_store = FAISSVectorStore(
            embedding_engine=embedding_engine,
            persist_dir="./vectorstore",
        )
        st.session_state.embedding_engine = embedding_engine
        st.session_state.vector_store = vector_store

    # Initialize LLM provider
    if provider == "OpenAI":
        if not api_key:
            st.error("Please provide your OpenAI API key.")
            return False
        llm = OpenAIProvider(model=model, api_key=api_key)
    elif provider == "Anthropic":
        if not api_key:
            st.error("Please provide your Anthropic API key.")
            return False
        llm = AnthropicProvider(model=model, api_key=api_key)
    else:
        llm = OllamaProvider(model=model, base_url=ollama_url)

    st.session_state.rag_pipeline = RAGPipeline(
        vector_store=vector_store,
        llm_provider=llm,
        top_k=top_k,
        score_threshold=score_threshold,
    )
    st.session_state.is_initialized = True
    return True


def process_uploaded_files(files):
    """Process uploaded files and add to vector store."""
    processor = DocumentProcessor(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        chunking_strategy=chunking_strategy,
    )

    for file in files:
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=Path(file.name).suffix
        ) as tmp:
            tmp.write(file.read())
            tmp_path = tmp.name

        try:
            with st.spinner(f"📄 Processing {file.name}..."):
                doc = processor.process(tmp_path)
                st.session_state.processed_docs.append({
                    "name": file.name,
                    "doc_id": doc.doc_id,
                    "chunks": len(doc.chunks),
                    "size": file.size,
                    "type": doc.file_type,
                })

            with st.spinner(f"🧮 Generating embeddings for {file.name}..."):
                added = st.session_state.vector_store.add_chunks(doc.chunks)
                st.session_state.total_chunks += added

            st.success(f"✅ {file.name}: {len(doc.chunks)} chunks indexed")

        except Exception as e:
            st.error(f"❌ Error processing {file.name}: {e}")
        finally:
            os.unlink(tmp_path)


# ──────────────────────────────────────────────
# Main Content Area
# ──────────────────────────────────────────────

st.markdown('<p class="main-header">📄 Document Intelligence Assistant</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">Upload documents and ask questions — powered by RAG</p>',
    unsafe_allow_html=True,
)

# Tabs
tab_upload, tab_chat, tab_explore = st.tabs(["📤 Upload Documents", "💬 Chat", "🔍 Explore Knowledge Base"])


# ─── Tab 1: Upload ───
with tab_upload:
    st.markdown("### Upload Your Documents")
    st.markdown("Supported formats: **PDF**, **DOCX**, **TXT**, **MD**, **CSV**")

    uploaded_files = st.file_uploader(
        "Drop files here",
        type=["pdf", "docx", "txt", "md", "csv"],
        accept_multiple_files=True,
        help="Upload one or more documents to build your knowledge base.",
    )

    col1, col2 = st.columns([1, 3])
    with col1:
        process_btn = st.button("🚀 Process Documents", type="primary", use_container_width=True)

    if process_btn and uploaded_files:
        if not st.session_state.is_initialized:
            success = initialize_pipeline()
            if not success:
                st.stop()

        process_uploaded_files(uploaded_files)
        st.balloons()

    # Show processed documents
    if st.session_state.processed_docs:
        st.markdown("---")
        st.markdown("### 📚 Processed Documents")
        for doc in st.session_state.processed_docs:
            with st.expander(f"📄 {doc['name']}", expanded=False):
                c1, c2, c3 = st.columns(3)
                c1.metric("Type", doc["type"])
                c2.metric("Chunks", doc["chunks"])
                c3.metric("Size", f"{doc['size'] / 1024:.1f} KB")


# ─── Tab 2: Chat ───
with tab_chat:
    if not st.session_state.processed_docs:
        st.info("📤 Please upload and process documents first to start chatting.")
    else:
        # Query mode selection
        col1, col2 = st.columns([2, 1])
        with col2:
            mode = st.selectbox(
                "Query Mode",
                ["answer", "summarize", "compare", "extract"],
                help="'answer' for Q&A, 'summarize' for summaries, 'compare' for comparisons, 'extract' for structured extraction.",
            )

        # Display chat history
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg.get("sources"):
                    with st.expander("📎 Sources", expanded=False):
                        for src in msg["sources"]:
                            st.markdown(
                                f'<div class="source-card">'
                                f'<strong>{src["relevance"]}</strong> '
                                f'(score: {src["score"]:.3f})<br>'
                                f'<small>Source: {src["source"]}</small><br>'
                                f'<em>{src["preview"]}</em>'
                                f'</div>',
                                unsafe_allow_html=True,
                            )
                if msg.get("latency"):
                    st.caption(f"⏱️ {msg['latency']:.0f}ms | 🔗 {msg.get('num_sources', 0)} sources")

        # Chat input
        if question := st.chat_input("Ask a question about your documents..."):
            # Display user message
            st.session_state.chat_history.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)

            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("🔍 Searching & generating..."):
                    if not st.session_state.is_initialized:
                        initialize_pipeline()

                    response = st.session_state.rag_pipeline.query(
                        question=question,
                        mode=mode,
                        top_k=top_k,
                    )

                st.markdown(response.answer)

                # Format sources for storage and display
                sources_data = []
                for src in response.sources:
                    sources_data.append({
                        "relevance": src.relevance_label,
                        "score": src.score,
                        "source": src.metadata.get("source", "Unknown"),
                        "preview": src.content[:200],
                    })

                if sources_data:
                    with st.expander("📎 Sources", expanded=False):
                        for src in sources_data:
                            st.markdown(
                                f'<div class="source-card">'
                                f'<strong>{src["relevance"]}</strong> '
                                f'(score: {src["score"]:.3f})<br>'
                                f'<small>Source: {src["source"]}</small><br>'
                                f'<em>{src["preview"]}</em>'
                                f'</div>',
                                unsafe_allow_html=True,
                            )

                st.caption(
                    f"⏱️ {response.latency_ms:.0f}ms | "
                    f"🔗 {len(response.sources)} sources | "
                    f"🤖 {response.model}"
                )

            # Save to history
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": response.answer,
                "sources": sources_data,
                "latency": response.latency_ms,
                "num_sources": len(response.sources),
            })


# ─── Tab 3: Explore ───
with tab_explore:
    if not st.session_state.processed_docs:
        st.info("📤 Upload documents first to explore the knowledge base.")
    else:
        st.markdown("### 🔍 Semantic Search Explorer")
        st.markdown("Search your document knowledge base directly to see what the retriever finds.")

        search_query = st.text_input("Search query", placeholder="Enter a search term...")
        search_k = st.slider("Number of results", 1, 20, 5, key="explore_k")

        if search_query and st.session_state.vector_store:
            results = st.session_state.vector_store.search(
                query=search_query,
                top_k=search_k,
            )

            if results:
                st.markdown(f"**Found {len(results)} results:**")
                for i, result in enumerate(results, 1):
                    score_color = "🟢" if result.score >= 0.8 else "🟡" if result.score >= 0.5 else "🔴"
                    source = result.metadata.get("source", "Unknown")

                    with st.expander(
                        f"{score_color} Result {i} — Score: {result.score:.4f} — {source}",
                        expanded=(i <= 3),
                    ):
                        st.markdown(result.content)
                        st.json(result.metadata)
            else:
                st.warning("No results found. Try a different query.")

        # Vector Store Stats
        st.markdown("---")
        st.markdown("### 📊 Vector Store Statistics")
        if st.session_state.vector_store:
            stats = st.session_state.vector_store.get_stats()
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Vectors", stats["total_vectors"])
            c2.metric("Total Chunks", stats["total_chunks"])
            c3.metric("Documents", stats["total_documents"])
            c4.metric("Embedding Dim", stats["embedding_dimension"])


# ──────────────────────────────────────────────
# Footer
# ──────────────────────────────────────────────

st.markdown("---")
st.markdown(
    "<center><small>Built with ❤️ using Streamlit By Adarsh Hule, FAISS, Sentence-Transformers & LLMs</small></center>",
    unsafe_allow_html=True,
)
