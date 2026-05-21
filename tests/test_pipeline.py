"""
tests/test_pipeline.py
----------------------
Unit tests for the RAG Research Assistant pipeline.
Run: pytest tests/ -v
"""

import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from rag_pipeline import (
    Document, TextChunker, EmbeddingEngine, VectorStore,
    DocumentLoader, RAGPipeline,
)

# ── TextChunker ────────────────────────────────────────────────────────────────

class TestTextChunker:
    def test_basic_chunking(self):
        chunker = TextChunker(chunk_size=50, overlap=10)
        text = "This is sentence one. " * 50
        chunks = chunker.chunk(text, source="test")
        assert len(chunks) > 1
        for c in chunks:
            assert isinstance(c, Document)
            assert c.source == "test"
            assert len(c.text) > 0

    def test_short_text_single_chunk(self):
        chunker = TextChunker(chunk_size=512, overlap=64)
        text = "Short text."
        chunks = chunker.chunk(text, source="short")
        assert len(chunks) == 1

    def test_chunk_ids_sequential(self):
        chunker = TextChunker(chunk_size=30, overlap=5)
        text = "Word. " * 100
        chunks = chunker.chunk(text, source="ids")
        ids = [c.chunk_id for c in chunks]
        assert ids == list(range(len(ids)))


# ── EmbeddingEngine ────────────────────────────────────────────────────────────

class TestEmbeddingEngine:
    def test_encode_returns_numpy(self):
        engine = EmbeddingEngine.__new__(EmbeddingEngine)
        engine.model_name = "mock"
        engine._model = None
        engine._use_tfidf = True
        from sklearn.feature_extraction.text import TfidfVectorizer
        engine._tfidf = TfidfVectorizer(max_features=64)
        engine._dim = 64

        texts = ["hello world", "deep learning is great", "U-Net segmentation"]
        vecs = engine.encode(texts)
        assert isinstance(vecs, np.ndarray)
        assert vecs.shape[0] == 3
        assert vecs.shape[1] == 64

    def test_encode_query_shape(self):
        engine = EmbeddingEngine.__new__(EmbeddingEngine)
        engine.model_name = "mock"
        engine._model = None
        engine._use_tfidf = True
        from sklearn.feature_extraction.text import TfidfVectorizer
        tfidf = TfidfVectorizer(max_features=64)
        tfidf.fit(["train data for vocabulary building"])
        engine._tfidf = tfidf
        engine._dim = 64

        vec = engine.encode_query("what is U-Net?")
        assert isinstance(vec, np.ndarray)
        assert vec.ndim == 1


# ── VectorStore ────────────────────────────────────────────────────────────────

class TestVectorStore:
    def _make_store(self, n=10, dim=32):
        store = VectorStore(dim=dim)
        docs = [Document(text=f"document {i}", source="test", chunk_id=i) for i in range(n)]
        embeddings = np.random.rand(n, dim).astype(np.float32)
        # Normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings /= norms
        store.add(docs, embeddings)
        return store, embeddings, docs

    def test_len(self):
        store, _, _ = self._make_store(n=7)
        assert len(store) == 7

    def test_search_returns_top_k(self):
        store, embeddings, _ = self._make_store(n=10, dim=32)
        query = embeddings[0]
        results = store.search(query, top_k=3)
        assert len(results) == 3

    def test_search_scores_descending(self):
        store, embeddings, _ = self._make_store(n=10, dim=32)
        query = embeddings[2]
        results = store.search(query, top_k=5)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_exact_match_is_top(self):
        store, embeddings, docs = self._make_store(n=5, dim=32)
        # Query with an exact copy of doc[1]'s embedding
        query = embeddings[1]
        results = store.search(query, top_k=1)
        assert results[0].document.chunk_id == docs[1].chunk_id


# ── DocumentLoader ─────────────────────────────────────────────────────────────

class TestDocumentLoader:
    def test_load_txt(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("Hello, world!", encoding="utf-8")
        text = DocumentLoader.load(f)
        assert text == "Hello, world!"

    def test_load_md(self, tmp_path):
        f = tmp_path / "test.md"
        f.write_text("# Title\n\nSome content.", encoding="utf-8")
        text = DocumentLoader.load(f)
        assert "Title" in text

    def test_unsupported_raises(self, tmp_path):
        f = tmp_path / "test.xlsx"
        f.write_bytes(b"fake")
        with pytest.raises(ValueError, match="Unsupported"):
            DocumentLoader.load(f)


# ── RAGPipeline integration ────────────────────────────────────────────────────

class TestRAGPipelineIntegration:
    SAMPLE = (
        "Deep learning is a subfield of machine learning. "
        "Convolutional neural networks (CNNs) excel at image recognition tasks. "
        "U-Net is a CNN architecture designed for biomedical image segmentation. "
        "Dice loss is used to optimize segmentation models against class imbalance. "
        "Transformers use self-attention mechanisms and have transformed NLP. "
    )

    def test_ingest_text(self):
        pipeline = RAGPipeline(top_k=2)
        n = pipeline.ingest_text(self.SAMPLE)
        assert n >= 1

    def test_query_returns_rag_response(self):
        from rag_pipeline import RAGResponse
        pipeline = RAGPipeline(top_k=2)
        pipeline.ingest_text(self.SAMPLE)
        resp = pipeline.query("What is Dice loss?")
        assert isinstance(resp, RAGResponse)
        assert resp.query == "What is Dice loss?"
        assert len(resp.retrieved_chunks) <= 2
        assert isinstance(resp.answer, str)

    def test_empty_index_raises(self):
        pipeline = RAGPipeline()
        with pytest.raises(RuntimeError, match="empty"):
            pipeline.query("anything")

    def test_repr(self):
        pipeline = RAGPipeline()
        assert "RAGPipeline" in repr(pipeline)
