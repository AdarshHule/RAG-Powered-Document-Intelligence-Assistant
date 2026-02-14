# 📄 RAG-Powered Document Intelligence Assistant

A production-ready Retrieval-Augmented Generation (RAG) system that lets you upload documents and ask intelligent questions about their content. Features multiple chunking strategies, FAISS/ChromaDB vector stores, and support for OpenAI, Anthropic, and local Ollama models.

---

## ✨ Features

- **Multi-Format Document Ingestion** — PDF, DOCX, TXT, Markdown, CSV
- **4 Chunking Strategies** — Semantic, sentence-based, paragraph-based, and fixed-size
- **Dual Vector Store Support** — FAISS (fast, local) and ChromaDB (persistent, feature-rich)
- **Multiple LLM Providers** — OpenAI, Anthropic Claude, Ollama (fully local)
- **Interactive Streamlit UI** — Upload, chat, and explore your knowledge base
- **CLI Interface** — For scripting and terminal-based workflows
- **Source Attribution** — Every answer cites the exact document chunks used
- **Semantic Search Explorer** — Browse and inspect what the retriever finds

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────┐
│                  User Interface                  │
│          (Streamlit UI  /  CLI  /  API)          │
└─────────────────────┬───────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────┐
│               RAG Pipeline                       │
│  ┌──────────┐  ┌──────────┐  ┌───────────────┐  │
│  │ Retriever│→ │ Context  │→ │  LLM Generate │  │
│  │ (Top-K)  │  │ Builder  │  │  (Answer)     │  │
│  └──────────┘  └──────────┘  └───────────────┘  │
└─────────────────────┬───────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────┐
│             Vector Store (FAISS/Chroma)           │
│  ┌──────────────┐  ┌─────────────────────────┐  │
│  │  Embeddings  │  │  Document Chunks + Meta  │  │
│  │  (384/768d)  │  │                         │  │
│  └──────────────┘  └─────────────────────────┘  │
└─────────────────────┬───────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────┐
│           Document Processor                     │
│  ┌────┐ ┌─────┐ ┌─────┐ ┌────┐ ┌─────┐        │
│  │PDF │ │DOCX │ │ TXT │ │ MD │ │ CSV │        │
│  └────┘ └─────┘ └─────┘ └────┘ └─────┘        │
│         ↓ Text Extraction ↓ Chunking            │
└─────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd rag-assistant
pip install -r requirements.txt
```

### 2. Set Up Environment

```bash
cp .env.example .env
# Edit .env and add your API key (OpenAI, Anthropic, or use Ollama)
```

### 3. Run the Streamlit App

```bash
streamlit run app.py
```

### 4. Or Use the CLI

```bash
# Ingest documents
python cli.py ingest --path ./data

# Ask a question
python cli.py query "What are the key findings?"

# Interactive mode
python cli.py interactive

# View stats
python cli.py stats
```

---

## 📁 Project Structure

```
rag-assistant/
├── app.py                  # Streamlit web UI
├── cli.py                  # Command-line interface
├── requirements.txt        # Python dependencies
├── .env.example            # Environment variable template
├── src/
│   ├── __init__.py
│   ├── document_processor.py   # Document loading & chunking
│   ├── vector_store.py         # FAISS & ChromaDB vector stores
│   └── rag_pipeline.py         # RAG orchestration & LLM integration
├── data/                   # Place documents here for CLI ingestion
├── uploads/                # Streamlit upload directory
└── vectorstore/            # Persisted vector indices
```

---

## ⚙️ Configuration Guide

### Chunking Strategies

| Strategy      | Best For                              | Description                                      |
|---------------|---------------------------------------|--------------------------------------------------|
| `semantic`    | General documents                     | Splits by headers/sections, then by sentences     |
| `sentence`    | Narrative text, articles              | Respects sentence boundaries with overlap          |
| `paragraph`   | Well-structured docs with paragraphs  | Splits by paragraph breaks, merges short ones      |
| `fixed`       | Uniform processing                    | Fixed character windows with overlap               |

### Embedding Models

| Model                  | Dimensions | Speed  | Quality |
|------------------------|-----------|--------|---------|
| `all-MiniLM-L6-v2`    | 384       | ⚡ Fast | Good    |
| `all-mpnet-base-v2`   | 768       | 🐢 Slower | Better |

### LLM Providers

| Provider   | Setup                                     | Cost       |
|------------|-------------------------------------------|------------|
| OpenAI     | Set `OPENAI_API_KEY` in `.env`            | Pay-per-use |
| Anthropic  | Set `ANTHROPIC_API_KEY` in `.env`         | Pay-per-use |
| Ollama     | Install Ollama + `ollama pull llama3.2`   | Free/local  |

---

## 🔧 Query Modes

- **`answer`** — Standard Q&A with source citations
- **`summarize`** — Generate a summary of retrieved context
- **`compare`** — Compare information across different document sections
- **`extract`** — Extract structured data (returns JSON)

---

## 🧪 Example Usage (Python API)

```python
from src.document_processor import DocumentProcessor
from src.vector_store import EmbeddingEngine, FAISSVectorStore
from src.rag_pipeline import RAGPipeline, OpenAIProvider

# 1. Process documents
processor = DocumentProcessor(chunk_size=512, chunking_strategy="semantic")
doc = processor.process("./data/report.pdf")

# 2. Build vector store
engine = EmbeddingEngine()
store = FAISSVectorStore(embedding_engine=engine)
store.add_chunks(doc.chunks)

# 3. Query
pipeline = RAGPipeline(
    vector_store=store,
    llm_provider=OpenAIProvider(model="gpt-4o-mini"),
)

response = pipeline.query("What are the key findings?")
print(response.answer)
print(response.format_sources())
```

---

## 📝 License

MIT — use freely for personal and commercial projects.
