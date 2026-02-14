"""
📄 RAG Assistant — CLI Interface
Use from the command line for quick document Q&A.

Usage:
    python cli.py ingest --dir ./data
    python cli.py query "What is the main topic?"
    python cli.py interactive
"""

import os
import sys
import argparse
from pathlib import Path
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt

sys.path.insert(0, str(Path(__file__).parent))

from src.document_processor import DocumentProcessor
from src.vector_store import EmbeddingEngine, FAISSVectorStore
from src.rag_pipeline import RAGPipeline, OpenAIProvider, AnthropicProvider, OllamaProvider

console = Console()

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────

DEFAULT_CONFIG = {
    "chunk_size": 512,
    "chunk_overlap": 50,
    "chunking_strategy": "semantic",
    "embedding_model": "all-MiniLM-L6-v2",
    "top_k": 5,
    "score_threshold": 0.1,
    "vectorstore_dir": "./vectorstore",
    "vectorstore_name": "default",
}


def get_llm_provider():
    """Detect and return the appropriate LLM provider."""
    if os.getenv("OPENAI_API_KEY"):
        console.print("[green]Using OpenAI (gpt-4o-mini)[/green]")
        return OpenAIProvider(model="gpt-4o-mini")
    elif os.getenv("ANTHROPIC_API_KEY"):
        console.print("[green]Using Anthropic (Claude Sonnet)[/green]")
        return AnthropicProvider()
    else:
        console.print("[yellow]No API keys found. Trying Ollama (local)...[/yellow]")
        return OllamaProvider(model="llama3.2")


# ──────────────────────────────────────────────
# Commands
# ──────────────────────────────────────────────

def cmd_ingest(args):
    """Ingest documents into the vector store."""
    console.print(Panel("📄 [bold]Document Ingestion[/bold]", style="blue"))

    # Initialize
    processor = DocumentProcessor(
        chunk_size=DEFAULT_CONFIG["chunk_size"],
        chunk_overlap=DEFAULT_CONFIG["chunk_overlap"],
        chunking_strategy=DEFAULT_CONFIG["chunking_strategy"],
    )
    engine = EmbeddingEngine(model_name=DEFAULT_CONFIG["embedding_model"])
    store = FAISSVectorStore(
        embedding_engine=engine,
        persist_dir=DEFAULT_CONFIG["vectorstore_dir"],
    )

    # Try to load existing store
    store.load(DEFAULT_CONFIG["vectorstore_name"])

    # Process files
    target = Path(args.path)
    if target.is_file():
        files = [target]
    elif target.is_dir():
        files = [
            f for f in target.iterdir()
            if f.suffix.lower() in DocumentProcessor.LOADERS
        ]
    else:
        console.print(f"[red]Path not found: {args.path}[/red]")
        return

    console.print(f"\nFound [bold]{len(files)}[/bold] files to process.\n")

    total_chunks = 0
    for file_path in files:
        try:
            doc = processor.process(str(file_path))
            added = store.add_chunks(doc.chunks)
            total_chunks += added
            console.print(f"  ✅ {file_path.name}: [green]{added} chunks[/green]")
        except Exception as e:
            console.print(f"  ❌ {file_path.name}: [red]{e}[/red]")

    # Save
    store.save(DEFAULT_CONFIG["vectorstore_name"])

    # Summary
    table = Table(title="Ingestion Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Files Processed", str(len(files)))
    table.add_row("Total Chunks", str(total_chunks))
    table.add_row("Vector Store Size", str(store.size))
    console.print(table)


def cmd_query(args):
    """Run a single query against the knowledge base."""
    console.print(Panel("🔍 [bold]RAG Query[/bold]", style="blue"))

    engine = EmbeddingEngine(model_name=DEFAULT_CONFIG["embedding_model"])
    store = FAISSVectorStore(
        embedding_engine=engine,
        persist_dir=DEFAULT_CONFIG["vectorstore_dir"],
    )

    if not store.load(DEFAULT_CONFIG["vectorstore_name"]):
        console.print("[red]No vector store found. Run 'ingest' first.[/red]")
        return

    llm = get_llm_provider()
    pipeline = RAGPipeline(
        vector_store=store,
        llm_provider=llm,
        top_k=DEFAULT_CONFIG["top_k"],
        score_threshold=DEFAULT_CONFIG["score_threshold"],
    )

    with console.status("Searching & generating..."):
        response = pipeline.query(args.question, mode=args.mode)

    # Display answer
    console.print(Panel(
        Markdown(response.answer),
        title="💡 Answer",
        border_style="green",
    ))

    # Display sources
    console.print(f"\n📎 [bold]Sources[/bold] ({len(response.sources)} retrieved):")
    for i, src in enumerate(response.sources, 1):
        source_name = src.metadata.get("source", "Unknown")
        console.print(
            f"  [{i}] {src.relevance_label} — Score: {src.score:.3f} — {source_name}"
        )

    console.print(f"\n⏱️ Latency: {response.latency_ms:.0f}ms | 🤖 Model: {response.model}")


def cmd_interactive(args):
    """Start an interactive Q&A session."""
    console.print(Panel(
        "📄 [bold]Document Intelligence Assistant[/bold]\n"
        "Interactive Mode — Type 'quit' to exit, 'history' to view past Q&A",
        style="blue",
    ))

    engine = EmbeddingEngine(model_name=DEFAULT_CONFIG["embedding_model"])
    store = FAISSVectorStore(
        embedding_engine=engine,
        persist_dir=DEFAULT_CONFIG["vectorstore_dir"],
    )

    if not store.load(DEFAULT_CONFIG["vectorstore_name"]):
        console.print("[red]No vector store found. Run 'ingest' first.[/red]")
        return

    llm = get_llm_provider()
    pipeline = RAGPipeline(
        vector_store=store,
        llm_provider=llm,
        top_k=DEFAULT_CONFIG["top_k"],
        score_threshold=DEFAULT_CONFIG["score_threshold"],
    )

    console.print(f"\n📊 Knowledge base: [green]{store.size} vectors[/green] from [green]{len(store.doc_ids)} documents[/green]\n")

    while True:
        try:
            question = Prompt.ask("\n[bold cyan]You[/bold cyan]")
        except (KeyboardInterrupt, EOFError):
            break

        if question.lower() in ("quit", "exit", "q"):
            console.print("[yellow]Goodbye! 👋[/yellow]")
            break

        if question.lower() == "history":
            for entry in pipeline.get_conversation_history():
                console.print(f"  Q: {entry['question']}")
                console.print(f"  A: {entry['answer'][:100]}...")
                console.print()
            continue

        if not question.strip():
            continue

        with console.status("🔍 Thinking..."):
            response = pipeline.query(question)

        console.print(Panel(
            Markdown(response.answer),
            title="💡 Assistant",
            border_style="green",
        ))
        console.print(
            f"  ⏱️ {response.latency_ms:.0f}ms | "
            f"📎 {len(response.sources)} sources | "
            f"🤖 {response.model}"
        )


def cmd_stats(args):
    """Show vector store statistics."""
    engine = EmbeddingEngine(model_name=DEFAULT_CONFIG["embedding_model"])
    store = FAISSVectorStore(
        embedding_engine=engine,
        persist_dir=DEFAULT_CONFIG["vectorstore_dir"],
    )

    if not store.load(DEFAULT_CONFIG["vectorstore_name"]):
        console.print("[red]No vector store found.[/red]")
        return

    stats = store.get_stats()
    table = Table(title="📊 Vector Store Statistics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    for key, value in stats.items():
        table.add_row(key.replace("_", " ").title(), str(value))

    console.print(table)


# ──────────────────────────────────────────────
# Main Entry Point
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="📄 RAG-Powered Document Intelligence Assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Ingest
    ingest_parser = subparsers.add_parser("ingest", help="Ingest documents into the knowledge base")
    ingest_parser.add_argument("--path", "-p", default="./data", help="File or directory path")

    # Query
    query_parser = subparsers.add_parser("query", help="Run a single query")
    query_parser.add_argument("question", help="Your question")
    query_parser.add_argument("--mode", "-m", default="answer", choices=["answer", "summarize", "compare", "extract"])

    # Interactive
    subparsers.add_parser("interactive", help="Start interactive Q&A session")

    # Stats
    subparsers.add_parser("stats", help="Show vector store statistics")

    args = parser.parse_args()

    commands = {
        "ingest": cmd_ingest,
        "query": cmd_query,
        "interactive": cmd_interactive,
        "stats": cmd_stats,
    }

    if args.command in commands:
        commands[args.command](args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
