# 🔬 RAG Research Assistant

**Retrieval-Augmented Generation pipeline for domain-specific Q&A over research documents.**

Built with sentence-transformers, FAISS, and Hugging Face Transformers. Designed for medical imaging, scientific literature, and any long-form document corpus. Includes a Streamlit web UI and a full CLI.

---

## Architecture

```
Documents (.txt / .pdf / .md)
        │
        ▼
 ┌─────────────────┐
 │  DocumentLoader │  pdfplumber / plain text
 └────────┬────────┘
          │
          ▼
 ┌─────────────────┐
 │   TextChunker   │  sliding-window + sentence-boundary (512 tok, 64 overlap)
 └────────┬────────┘
          │
          ▼
 ┌─────────────────┐
 │ EmbeddingEngine │  sentence-transformers all-MiniLM-L6-v2 (384-dim)
 │                 │  ─ fallback: TF-IDF sparse vectors
 └────────┬────────┘
          │
          ▼
 ┌─────────────────┐
 │   VectorStore   │  FAISS IndexFlatIP (cosine on L2-normalised vectors)
 │                 │  ─ fallback: numpy brute-force
 └────────┬────────┘
          │  top-K retrieval
          ▼
 ┌─────────────────┐
 │    Generator    │  google/flan-t5-base (text2text)
 │                 │  prompt: context + question → grounded answer
 └────────┬────────┘
          │
          ▼
      RAGResponse
  (answer + retrieved chunks + token count)
```

## Features

- **Sliding-window chunker** with sentence-boundary awareness and configurable overlap
- **Dense retrieval** via `all-MiniLM-L6-v2` (384-dim) + FAISS inner-product index
- **Graceful fallbacks**: TF-IDF + numpy when sentence-transformers / faiss-cpu unavailable
- **Pluggable generator**: swap `flan-t5-base` for any Hugging Face `text2text-generation` model
- **Multi-format ingestion**: `.txt`, `.md`, `.pdf` (via pdfplumber)
- **Streamlit UI**: upload docs, ask questions, inspect retrieved chunks with similarity scores
- **CLI**: `ingest`, `query`, `eval` sub-commands with full argument control
- **Retrieval evaluation**: hit-rate scoring against JSON Q&A pairs
- **Unit tests** (pytest) with 90%+ coverage of core components
- **CI/CD**: GitHub Actions matrix testing on Python 3.10 & 3.11

## Quickstart

```bash
git clone https://github.com/<your-username>/rag-research-assistant
cd rag-research-assistant
pip install -r requirements.txt
```

### Streamlit UI

```bash
streamlit run app.py
```

Open `http://localhost:8501`, upload documents, and start querying.

### CLI

```bash
# Index files and start interactive session
python cli.py ingest --files data/*.txt --interactive

# Single query
python cli.py query --files paper.pdf --question "What is Dice loss?"

# Evaluate retrieval hit-rate
python cli.py eval --files corpus.txt --qna data/qa_pairs.json
```

### Python API

```python
from src.rag_pipeline import RAGPipeline

pipeline = RAGPipeline(
    embedding_model="all-MiniLM-L6-v2",
    generator_model="google/flan-t5-base",
    chunk_size=512,
    overlap=64,
    top_k=5,
)

# Ingest documents
pipeline.ingest(["paper.pdf", "notes.txt"])

# Or ingest raw text
pipeline.ingest_text("U-Net is a CNN for biomedical segmentation...", source_name="notes")

# Query
response = pipeline.query("What is the role of skip connections in U-Net?")
print(response.answer)

# Inspect retrieved chunks
for chunk in response.retrieved_chunks:
    print(f"[{chunk.rank+1}] score={chunk.score:.3f} | {chunk.document.text[:100]}")
```

## Evaluation

```json
// data/qa_pairs.json
[
  {
    "question": "What loss function handles class imbalance in segmentation?",
    "answer_keywords": ["dice", "dice loss", "f1"]
  },
  {
    "question": "What is the encoder in U-Net responsible for?",
    "answer_keywords": ["context", "features", "contracting"]
  }
]
```

```bash
python cli.py eval --files corpus.txt --qna data/qa_pairs.json
# → Retrieval hit-rate: 9/10 = 90.0%
```

## Running Tests

```bash
pip install pytest
pytest tests/ -v
```

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `embedding_model` | `all-MiniLM-L6-v2` | HF sentence-transformer model |
| `generator_model` | `google/flan-t5-base` | HF text2text model |
| `chunk_size` | `512` | Target tokens per chunk |
| `overlap` | `64` | Overlap tokens between chunks |
| `top_k` | `5` | Chunks retrieved per query |
| `max_new_tokens` | `256` | Max generation length |

## Project Structure

```
rag-research-assistant/
├── src/
│   └── rag_pipeline.py       # Core pipeline (loader, chunker, embedder, store, generator)
├── app.py                    # Streamlit web UI
├── cli.py                    # Command-line interface
├── notebooks/
│   └── demo.ipynb            # Interactive walkthrough
├── tests/
│   └── test_pipeline.py      # Unit + integration tests (pytest)
├── .github/
│   └── workflows/ci.yml      # CI matrix (Python 3.10 & 3.11)
├── requirements.txt
└── README.md
```

## Tech Stack

| Layer | Library |
|-------|---------|
| Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`) |
| Vector Index | `faiss-cpu` (IndexFlatIP) |
| Generation | `transformers` (`flan-t5-base`) |
| PDF Parsing | `pdfplumber` |
| Web UI | `streamlit` |
| Fallback | `scikit-learn` TF-IDF + `numpy` |
| Testing | `pytest` |
| CI | GitHub Actions |

## Author

**Aravind V** · [LinkedIn](https://linkedin.com/in/) · [GitHub](https://github.com/)  
Research Intern, BioMedia Lab, IISc Bengaluru  
B.Tech CSE, Manipal University Jaipur

---

*Built as part of ongoing research in AI/ML and medical image analysis.*
