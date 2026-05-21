# rag-research-assistant

A document Q&A tool I built to understand how RAG (Retrieval-Augmented Generation) works under the hood. You paste or upload a document, ask a question, and it finds the most relevant chunks from your document and returns them.

No LLM API keys needed , runs fully offline.

---

## what it does

* upload a `.txt`, `.pdf`, or `.md` file (or just paste text)
* it splits the text into chunks, converts them to vectors using TF-IDF
* when you ask a question, it finds the most similar chunks using cosine similarity
* shows you the results with similarity scores

built this mostly to understand the retrieval side of RAG chunking, embedding, vector search. the generation part is just returning the retrieved text for now.

---

## how to run it

```bash
git clone https://github.com/arvindv0119/rag-research-assistant.git
cd rag-research-assistant
pip install -r requirements.txt
python -m streamlit run app.py
```

opens at `http://localhost:8501`

---

## stack

- `streamlit` for the UI
- `scikit-learn` for TF-IDF embeddings
- `numpy` for cosine similarity search
- `pdfplumber` for PDF parsing

---

## background

built this to get hands-on experience with how RAG systems work — something I kept reading about but wanted to actually implement from scratch. covers the core ideas: text chunking, vector search, and retrieval — without relying on frameworks like LangChain that abstract everything away.


---

Aravind V  B.Tech CSE, Manipal University Jaipur  
