"""Command-line interface: ``mmrag <ingest|index|query|eval|serve|ui>``."""

from __future__ import annotations

import argparse
import json
import sys


def main(argv: list[str] | None = None) -> int:
    # Windows cp1252 consoles crash on doc text (Greek/math symbols in PDFs)
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    parser = argparse.ArgumentParser(prog="mmrag",
                                     description="Multimodal Hybrid RAG Engine")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("ingest", help="parse PDFs -> chunks (tables/figures/text)")
    p.add_argument("--limit", type=int, default=None, help="only first N PDFs")
    p.add_argument("--dpi", type=int, default=144, help="page render DPI")
    p.add_argument("--from-items", action="store_true",
                   help="re-chunk from saved parsed_items.jsonl (skip PDF parsing)")

    sub.add_parser("index", help="embed chunks + build BM25 + upsert store")

    p = sub.add_parser("query", help="ask a question against the index")
    p.add_argument("question")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--json", action="store_true", help="raw JSON output")

    p = sub.add_parser("eval", help="run the 40-question RAGAS/hit-rate benchmark")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--no-generation", action="store_true",
                   help="skip LLM-judged generative metrics")

    p = sub.add_parser("serve", help="start the FastAPI backend")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)

    sub.add_parser("ui", help="start the Streamlit demo portal")

    args = parser.parse_args(argv)

    if args.cmd == "ingest":
        from .pipeline import cmd_ingest
        cmd_ingest(limit=args.limit, render_dpi=args.dpi, from_items=args.from_items)
    elif args.cmd == "index":
        from .pipeline import cmd_index
        cmd_index()
    elif args.cmd == "query":
        from .config import load_settings
        from .rag import LLMClient, answer_question
        from .retrieval import HybridRetriever
        settings = load_settings()
        retriever = HybridRetriever.load(settings)
        results, tel = retriever.retrieve(args.question, top_k=args.top_k)
        out = answer_question(args.question, results, LLMClient(settings), settings)
        out["telemetry"] = tel.__dict__
        if args.json:
            print(json.dumps(out, indent=2, ensure_ascii=False))
        else:
            print(f"\n=== ANSWER ({out['generator']}) ===\n{out['answer']}\n")
            print("=== EVIDENCE ===")
            for c in out["contexts"]:
                print(f"[{c['rank']}] p.{c['page_number']} {c['chunk_type']} "
                      f"rrf={c['rrf_score']:.4f} conf={c['rerank_confidence']:.3f} "
                      f"dense_rank={c['dense_rank']} sparse_rank={c['sparse_rank']}")
                print(f"    {c['text'][:160].replace(chr(10), ' ')}...")
            t = out["telemetry"]
            print(f"\nlatency={t['latency_ms']:.1f}ms embedder={t['embedder']} "
                  f"reranker={t['reranker']} store={t['store']}")
    elif args.cmd == "eval":
        from .eval import run_benchmark
        run_benchmark(k=args.k, run_generation=not args.no_generation)
    elif args.cmd == "serve":
        import uvicorn
        uvicorn.run("mmrag.api.app:app", host=args.host, port=args.port)
    elif args.cmd == "ui":
        import subprocess
        import streamlit
        from pathlib import Path
        app = Path(__file__).resolve().parent / "ui" / "streamlit_app.py"
        sys.exit(subprocess.call([sys.executable, "-m", "streamlit", "run", str(app)]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
