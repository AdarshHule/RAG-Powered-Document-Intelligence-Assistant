"""
Vector Store Module
Handles embedding generation, vector storage, and similarity search.
Supports FAISS (local) and ChromaDB backends.
"""

import os
import json
import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field

from src.document_processor import DocumentChunk


# ──────────────────────────────────────────────
# Data Models
# ──────────────────────────────────────────────

@dataclass
class SearchResult:
    """Represents a single search result."""
    content: str
    score: float
    metadata: Dict = field(default_factory=dict)
    chunk_id: str = ""

    @property
    def relevance_label(self) -> str:
        if self.score >= 0.8:
            return "🟢 Highly Relevant"
        elif self.score >= 0.5:
            return "🟡 Relevant"
        else:
            return "🔴 Marginally Relevant"


# ──────────────────────────────────────────────
# Embedding Engine
# ──────────────────────────────────────────────

class EmbeddingEngine:
    """
    Generates embeddings using sentence-transformers.
    Default model: all-MiniLM-L6-v2 (fast, 384-dim)
    Alternative: all-mpnet-base-v2 (better quality, 768-dim)
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            print(f"  Loading embedding model: {self.model_name}...")
            self._model = SentenceTransformer(self.model_name)
            print(f"  ✓ Model loaded (dim={self._model.get_sentence_embedding_dimension()})")
        return self._model

    @property
    def dimension(self) -> int:
        return self.model.get_sentence_embedding_dimension()

    def embed(self, texts: List[str], batch_size: int = 64, show_progress: bool = True) -> np.ndarray:
        """Generate embeddings for a list of texts."""
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,  # For cosine similarity
        )
        return np.array(embeddings, dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """Generate embedding for a single query."""
        return self.embed([query], show_progress=False)[0]


# ──────────────────────────────────────────────
# FAISS Vector Store
# ──────────────────────────────────────────────

class FAISSVectorStore:
    """
    FAISS-based vector store for fast similarity search.
    Uses IndexFlatIP (inner product on normalized vectors = cosine similarity).
    """

    def __init__(
        self,
        embedding_engine: Optional[EmbeddingEngine] = None,
        persist_dir: str = "./vectorstore",
    ):
        import faiss

        self.embedding_engine = embedding_engine or EmbeddingEngine()
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        self.index: Optional[faiss.IndexFlatIP] = None
        self.chunks: List[DocumentChunk] = []
        self.doc_ids: set = set()

    @property
    def size(self) -> int:
        return self.index.ntotal if self.index else 0

    def add_chunks(self, chunks: List[DocumentChunk]) -> int:
        """Add document chunks to the vector store."""
        import faiss

        if not chunks:
            return 0

        texts = [c.content for c in chunks]
        embeddings = self.embedding_engine.embed(texts)

        if self.index is None:
            dim = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dim)

        self.index.add(embeddings)
        self.chunks.extend(chunks)

        for c in chunks:
            self.doc_ids.add(c.doc_id)

        return len(chunks)

    def search(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.0,
        filter_doc_id: Optional[str] = None,
    ) -> List[SearchResult]:
        """Search for similar chunks given a query."""
        if self.index is None or self.index.ntotal == 0:
            return []

        query_embedding = self.embedding_engine.embed_query(query)
        query_embedding = query_embedding.reshape(1, -1)

        # Search more than top_k to account for filtering
        search_k = min(top_k * 3, self.index.ntotal)
        scores, indices = self.index.search(query_embedding, search_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.chunks):
                continue
            if score < score_threshold:
                continue

            chunk = self.chunks[idx]

            if filter_doc_id and chunk.doc_id != filter_doc_id:
                continue

            results.append(SearchResult(
                content=chunk.content,
                score=float(score),
                metadata=chunk.metadata,
                chunk_id=chunk.chunk_id,
            ))

            if len(results) >= top_k:
                break

        return results

    def save(self, name: str = "default"):
        """Persist the vector store to disk."""
        import faiss

        if self.index is None:
            print("  ⚠ Nothing to save — index is empty.")
            return

        save_dir = self.persist_dir / name
        save_dir.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(save_dir / "index.faiss"))
        with open(save_dir / "chunks.pkl", "wb") as f:
            pickle.dump(self.chunks, f)
        with open(save_dir / "metadata.json", "w") as f:
            json.dump({
                "total_chunks": len(self.chunks),
                "doc_ids": list(self.doc_ids),
                "embedding_model": self.embedding_engine.model_name,
            }, f, indent=2)

        print(f"  ✓ Vector store saved to {save_dir}")

    def load(self, name: str = "default") -> bool:
        """Load vector store from disk."""
        import faiss

        save_dir = self.persist_dir / name
        index_path = save_dir / "index.faiss"
        chunks_path = save_dir / "chunks.pkl"

        if not index_path.exists() or not chunks_path.exists():
            return False

        self.index = faiss.read_index(str(index_path))
        with open(chunks_path, "rb") as f:
            self.chunks = pickle.load(f)
        with open(save_dir / "metadata.json", "r") as f:
            meta = json.load(f)
            self.doc_ids = set(meta.get("doc_ids", []))

        print(f"  ✓ Loaded vector store: {self.index.ntotal} vectors, {len(self.doc_ids)} documents")
        return True

    def delete_document(self, doc_id: str):
        """Remove all chunks for a document (requires rebuild)."""
        import faiss

        remaining = [c for c in self.chunks if c.doc_id != doc_id]
        if len(remaining) == len(self.chunks):
            return  # Nothing to delete

        self.chunks = remaining
        self.doc_ids.discard(doc_id)

        # Rebuild index
        if remaining:
            texts = [c.content for c in remaining]
            embeddings = self.embedding_engine.embed(texts, show_progress=False)
            dim = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dim)
            self.index.add(embeddings)
        else:
            self.index = None

    def get_stats(self) -> Dict:
        """Return statistics about the vector store."""
        return {
            "total_vectors": self.size,
            "total_chunks": len(self.chunks),
            "total_documents": len(self.doc_ids),
            "embedding_model": self.embedding_engine.model_name,
            "embedding_dimension": self.embedding_engine.dimension if self.index else 0,
        }


# ──────────────────────────────────────────────
# ChromaDB Vector Store (Alternative)
# ──────────────────────────────────────────────

class ChromaVectorStore:
    """
    ChromaDB-based vector store.
    Simpler API, built-in persistence, but slightly slower for large datasets.
    """

    def __init__(
        self,
        collection_name: str = "documents",
        persist_dir: str = "./vectorstore/chroma",
        embedding_engine: Optional[EmbeddingEngine] = None,
    ):
        import chromadb

        self.embedding_engine = embedding_engine or EmbeddingEngine()
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    @property
    def size(self) -> int:
        return self.collection.count()

    def add_chunks(self, chunks: List[DocumentChunk]) -> int:
        if not chunks:
            return 0

        texts = [c.content for c in chunks]
        embeddings = self.embedding_engine.embed(texts)
        ids = [c.chunk_id for c in chunks]
        metadatas = [
            {k: str(v) for k, v in c.metadata.items()}
            for c in chunks
        ]

        self.collection.add(
            documents=texts,
            embeddings=embeddings.tolist(),
            ids=ids,
            metadatas=metadatas,
        )
        return len(chunks)

    def search(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.0,
        filter_doc_id: Optional[str] = None,
    ) -> List[SearchResult]:
        query_embedding = self.embedding_engine.embed_query(query)

        where_filter = None
        if filter_doc_id:
            where_filter = {"doc_id": filter_doc_id}

        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            where=where_filter,
        )

        search_results = []
        for i in range(len(results["documents"][0])):
            score = 1 - results["distances"][0][i]  # Convert distance to similarity
            if score < score_threshold:
                continue

            search_results.append(SearchResult(
                content=results["documents"][0][i],
                score=score,
                metadata=results["metadatas"][0][i] if results["metadatas"] else {},
                chunk_id=results["ids"][0][i],
            ))

        return search_results
