import os
import re
import numpy as np
import logging
from pathlib import Path
from dataclasses import dataclass, field

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)


@dataclass
class Document:
    text: str
    source: str
    chunk_id: int
    metadata: dict = field(default_factory=dict)


@dataclass
class RetrievalResult:
    document: Document
    score: float
    rank: int


@dataclass
class RAGResponse:
    query: str
    answer: str
    retrieved_chunks: list
    num_tokens_used: int = 0


class DocumentLoader:
    @classmethod
    def load(cls, path):
        path = Path(path)
        if path.suffix.lower() == ".pdf":
            try:
                import pdfplumber
                parts = []
                with pdfplumber.open(path) as pdf:
                    for page in pdf.pages:
                        t = page.extract_text()
                        if t:
                            parts.append(t)
                return "\n\n".join(parts)
            except ImportError:
                raise ImportError("pip install pdfplumber")
        return path.read_text(encoding="utf-8", errors="ignore")


class TextChunker:
    def __init__(self, chunk_size=300, overlap=50):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.char_limit = chunk_size * 4
        self.overlap_chars = overlap * 4

    def chunk(self, text, source="unknown"):
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        sentences = [s.strip() for s in sentences if s.strip()]
        chunks = []
        buffer = []
        buf_len = 0
        chunk_id = 0

        for sent in sentences:
            if buf_len + len(sent) > self.char_limit and buffer:
                chunk_text = " ".join(buffer).strip()
                if chunk_text:
                    chunks.append(Document(text=chunk_text, source=source, chunk_id=chunk_id))
                    chunk_id += 1
                overlap_text = chunk_text[-self.overlap_chars:]
                buffer = [overlap_text]
                buf_len = len(overlap_text)
            buffer.append(sent)
            buf_len += len(sent)

        if buffer:
            chunk_text = " ".join(buffer).strip()
            if chunk_text:
                chunks.append(Document(text=chunk_text, source=source, chunk_id=chunk_id))

        log.info(f"Created {len(chunks)} chunks from '{source}'")
        return chunks


class EmbeddingEngine:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None
        self._tfidf = None
        self._use_tfidf = False
        self._dim = None

    def _load(self):
        if self._model or self._use_tfidf:
            return
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
            self._dim = self._model.get_sentence_embedding_dimension()
        except Exception:
            log.warning("sentence-transformers failed, using TF-IDF fallback")
            from sklearn.feature_extraction.text import TfidfVectorizer
            self._tfidf = TfidfVectorizer(max_features=256)
            self._use_tfidf = True
            self._dim = 256

    def encode(self, texts):
        self._load()
        if self._use_tfidf:
            try:
                vecs = self._tfidf.transform(texts).toarray().astype(np.float32)
            except Exception:
                vecs = self._tfidf.fit_transform(texts).toarray().astype(np.float32)
            self._dim = vecs.shape[1]
            return vecs
        vecs = self._model.encode(texts, normalize_embeddings=True)
        return np.array(vecs, dtype=np.float32)

    def encode_query(self, query):
        self._load()
        if self._use_tfidf:
            try:
                vec = self._tfidf.transform([query]).toarray().astype(np.float32)
            except Exception:
                vec = self._tfidf.fit_transform([query]).toarray().astype(np.float32)
            return vec[0]
        return self._model.encode([query], normalize_embeddings=True)[0]

    @property
    def dim(self):
        self._load()
        return self._dim


class VectorStore:
    def __init__(self):
        self._docs = []
        self._embeddings = None
        self._dim = None

    def add(self, docs, embeddings):
        self._docs.extend(docs)
        if self._dim is None:
            self._dim = embeddings.shape[1]
        if self._embeddings is None:
            self._embeddings = embeddings
        else:
            self._embeddings = np.vstack([self._embeddings, embeddings])
        log.info(f"Store has {len(self._docs)} chunks now")

    def search(self, query_vec, top_k=5):
        if not self._docs or self._embeddings is None:
            return []
        query_vec = query_vec.reshape(1, -1).astype(np.float32)
        q_norm = np.linalg.norm(query_vec)
        if q_norm > 0:
            query_vec = query_vec / q_norm
        emb = self._embeddings.copy()
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms[norms == 0] = 1
        emb = emb / norms
        sims = (emb @ query_vec.T).flatten()
        top_idxs = np.argsort(sims)[::-1][:top_k]
        return [
            RetrievalResult(document=self._docs[i], score=float(sims[i]), rank=r)
            for r, i in enumerate(top_idxs)
        ]

    def __len__(self):
        return len(self._docs)


class RAGPipeline:
    def __init__(self, embedding_model="all-MiniLM-L6-v2", chunk_size=300, overlap=50, top_k=3):
        self.top_k = top_k
        self._chunker = TextChunker(chunk_size=chunk_size, overlap=overlap)
        self._embedder = EmbeddingEngine(model_name=embedding_model)
        self._store = VectorStore()

    def ingest_text(self, text, source_name="pasted"):
        docs = self._chunker.chunk(text, source=source_name)
        if not docs:
            return 0
        embeddings = self._embedder.encode([d.text for d in docs])
        self._store.add(docs, embeddings)
        return len(docs)

    def ingest(self, files):
        total = 0
        for f in files:
            text = DocumentLoader.load(f)
            docs = self._chunker.chunk(text, source=Path(f).name)
            if docs:
                embeddings = self._embedder.encode([d.text for d in docs])
                self._store.add(docs, embeddings)
                total += len(docs)
        return total

    def query(self, question):
        if len(self._store) == 0:
            raise RuntimeError("Index is empty. Call ingest() first.")
        q_vec = self._embedder.encode_query(question)
        retrieved = self._store.search(q_vec, top_k=self.top_k)
        context = "\n\n".join(r.document.text for r in retrieved)
        answer = f"Based on your documents:\n\n{context}"
        return RAGResponse(
            query=question,
            answer=answer,
            retrieved_chunks=retrieved,
            num_tokens_used=len(context.split())
        )
