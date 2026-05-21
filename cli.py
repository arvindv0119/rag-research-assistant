"""
cli.py
------
Command-line interface for the RAG Research Assistant.

Usage examples
--------------
# Index a folder of PDFs and start an interactive session
python cli.py ingest --files papers/*.pdf --interactive

# Single query (non-interactive)
python cli.py query --files paper.txt --question "What is U-Net?"

# Evaluate top-K retrieval
python cli.py eval --files data/corpus.txt --qna data/qa_pairs.json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
from rag_pipeline import RAGPipeline


# ── Helpers ───────────────────────────────────────────────────────────────────

def print_response(resp):
    print("\n" + "=" * 60)
    print(f"Q: {resp.query}")
    print("-" * 60)
    print(f"A: {resp.answer}")
    print("-" * 60)
    print(f"Retrieved {len(resp.retrieved_chunks)} chunks | ~{resp.num_tokens_used} tokens")
    for r in resp.retrieved_chunks:
        score_pct = f"{r.score * 100:.1f}%"
        snippet = r.document.text[:200].replace("\n", " ")
        print(f"  [{r.rank+1}] ({score_pct}) [{r.document.source}] {snippet}…")
    print("=" * 60 + "\n")


def build_pipeline(args) -> RAGPipeline:
    return RAGPipeline(
        embedding_model=args.embed_model,
        generator_model=args.gen_model,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        top_k=args.top_k,
        max_new_tokens=args.max_tokens,
    )


# ── Sub-commands ──────────────────────────────────────────────────────────────

def cmd_ingest(args):
    pipeline = build_pipeline(args)
    files = []
    for pattern in args.files:
        files.extend(Path(".").glob(pattern))
    if not files:
        print("No files matched. Check your --files argument.")
        sys.exit(1)

    total = pipeline.ingest(files)
    print(f"Indexed {total} chunks from {len(files)} file(s).")

    if args.interactive:
        print("Type your question (or 'exit' to quit).\n")
        while True:
            try:
                q = input(">>> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if q.lower() in {"exit", "quit", "q"}:
                break
            if q:
                print_response(pipeline.query(q))


def cmd_query(args):
    pipeline = build_pipeline(args)
    files = []
    for pattern in args.files:
        files.extend(Path(".").glob(pattern))
    pipeline.ingest(files)
    print_response(pipeline.query(args.question))


def cmd_eval(args):
    """
    Evaluate retrieval hit-rate on a JSON file of Q&A pairs.
    Format: [{"question": "...", "answer_keywords": ["...", "..."]}]
    """
    pipeline = build_pipeline(args)
    files = []
    for pattern in args.files:
        files.extend(Path(".").glob(pattern))
    pipeline.ingest(files)

    qa_pairs = json.loads(Path(args.qna).read_text())
    hits = 0
    for pair in qa_pairs:
        resp = pipeline.query(pair["question"])
        context = " ".join(r.document.text for r in resp.retrieved_chunks).lower()
        hit = any(kw.lower() in context for kw in pair["answer_keywords"])
        hits += int(hit)
        print(f"{'✓' if hit else '✗'} {pair['question'][:60]}")

    hit_rate = hits / len(qa_pairs) * 100
    print(f"\nRetrieval hit-rate: {hits}/{len(qa_pairs)} = {hit_rate:.1f}%")


# ── Argument parsing ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        prog="rag-cli",
        description="RAG Research Assistant — command-line interface",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # Shared args
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument("--files", nargs="+", default=[], metavar="FILE")
    shared.add_argument("--embed-model", default="all-MiniLM-L6-v2")
    shared.add_argument("--gen-model", default="google/flan-t5-base")
    shared.add_argument("--chunk-size", type=int, default=512)
    shared.add_argument("--overlap", type=int, default=64)
    shared.add_argument("--top-k", type=int, default=5)
    shared.add_argument("--max-tokens", type=int, default=256)

    p_ingest = sub.add_parser("ingest", parents=[shared])
    p_ingest.add_argument("--interactive", action="store_true")
    p_ingest.set_defaults(func=cmd_ingest)

    p_query = sub.add_parser("query", parents=[shared])
    p_query.add_argument("--question", required=True)
    p_query.set_defaults(func=cmd_query)

    p_eval = sub.add_parser("eval", parents=[shared])
    p_eval.add_argument("--qna", required=True)
    p_eval.set_defaults(func=cmd_eval)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
