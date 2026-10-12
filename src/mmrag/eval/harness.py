"""Benchmark harness: hybrid system vs naive-dense baseline.

Runs the full 40-question suite, computes deterministic retrieval metrics for
both systems, adds LLM-judged (or RAGAS) generative metrics when an LLM is
configured, checks the spec's >30% Context-Recall hypothesis on
tabular/chart questions, and writes ``data/results/benchmark.md`` +
``metrics.json`` + ``per_question.jsonl``.
"""

from __future__ import annotations

import json
import statistics
import time
from collections import defaultdict
from pathlib import Path

from ..config import Settings, load_settings
from ..eval.baseline import BaselineDenseRetriever
from ..eval.dataset import build_dataset, load_dataset, save_dataset
from ..eval.metrics import (context_precision, context_recall, hit_rate_at_k,
                            llm_answer_relevance, llm_faithfulness, try_ragas_faithfulness)
from ..ingestion.chunker import Chunk, load_chunks
from ..rag.llm import LLMClient
from ..retrieval.retriever import HybridRetriever
from .harness_io import load_parsed_items, load_docs_meta

CATEGORIES = ["factual", "tabular", "visual"]


# --------------------------------------------------------------------------- #
# baseline gold alignment
# --------------------------------------------------------------------------- #

def _norm(s: str) -> str:
    return " ".join(s.lower().split())


def _align(retrieved_texts: list[str], gold: list[Chunk]) -> list[str]:
    """Map structure-blind baseline windows onto gold chunk ids by text overlap."""
    gold_norm = {g.chunk_id: _norm(g.text) for g in gold}
    out = []
    for i, t in enumerate(retrieved_texts):
        nt = _norm(t)
        matched = None
        for gid, gt in gold_norm.items():
            probe = gt[:200]
            if probe and probe in nt:
                matched = gid
                break
            if gt in nt or nt in gt:
                matched = gid
                break
        out.append(matched or f"miss:{i}")
    return out


# --------------------------------------------------------------------------- #
# aggregation
# --------------------------------------------------------------------------- #

def _mean(vals: list[float]) -> float:
    return statistics.fmean(vals) if vals else 0.0


def _aggregate(rows: list[dict], system: str, category: str | None = None) -> dict:
    sel = [r for r in rows if r["system"] == system
           and (category is None or r["category"] == category)]
    if not sel:
        return {}
    out = {
        "n": len(sel),
        "context_precision": _mean([r["context_precision"] for r in sel]),
        "context_recall": _mean([r["context_recall"] for r in sel]),
        "hit_rate_at_5": _mean([r["hit_rate_at_5"] for r in sel]),
    }
    faith = [r["faithfulness"] for r in sel if r.get("faithfulness") is not None]
    rel = [r["answer_relevance"] for r in sel if r.get("answer_relevance") is not None]
    out["faithfulness"] = _mean(faith) if faith else None
    out["answer_relevance"] = _mean(rel) if rel else None
    lat = [r["latency_ms"] for r in sel if r.get("latency_ms") is not None]
    out["mean_latency_ms"] = _mean(lat) if lat else None
    return out


def _fmt(v, pct: bool = True) -> str:
    if v is None:
        return "n/a"
    return f"{v * 100:.1f}%" if pct else f"{v:.1f}"


def _md_table(aggregates: dict, systems: list[str]) -> str:
    header = ("| System | Context Precision | Context Recall | Hit Rate @5 | "
              "Faithfulness | Answer Relevance | Mean latency (ms) |")
    sep = "|---|---|---|---|---|---|---|"
    lines = [header, sep]
    for sys in systems:
        a = aggregates.get((sys, "overall"), {})
        lines.append(
            f"| {sys} | {_fmt(a.get('context_precision'))} | "
            f"{_fmt(a.get('context_recall'))} | {_fmt(a.get('hit_rate_at_5'))} | "
            f"{_fmt(a.get('faithfulness'))} | {_fmt(a.get('answer_relevance'))} | "
            f"{_fmt(a.get('mean_latency_ms'), pct=False)} |")
    return "\n".join(lines)


def _md_categories(aggregates: dict, systems: list[str]) -> str:
    lines = ["| System | Category | n | Context Recall | Hit Rate @5 |",
             "|---|---|---|---|---|"]
    for sys in systems:
        for cat in CATEGORIES:
            a = aggregates.get((sys, cat), {})
            if not a:
                continue
            lines.append(f"| {sys} | {cat} | {a['n']} | {_fmt(a.get('context_recall'))} | "
                         f"{_fmt(a.get('hit_rate_at_5'))} |")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# main entry
# --------------------------------------------------------------------------- #

def run_benchmark(settings: Settings | None = None, k: int = 5,
                  run_generation: bool = True) -> dict:
    settings = settings or load_settings()
    results_dir = settings.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    chunks = load_chunks(settings.chunks_path)
    docs_meta = load_docs_meta(settings)
    if not chunks:
        raise SystemExit("No indexed chunks found — run `mmrag ingest` first.")

    ds_path = results_dir / "dataset.json"
    if ds_path.exists():
        dataset = load_dataset(ds_path)
    else:
        dataset = build_dataset(chunks, docs_meta)
        save_dataset(dataset, ds_path)
        print(f"[eval] built dataset: {dataset['counts']}")

    parsed_items = load_parsed_items(settings)
    hybrid = HybridRetriever.load(settings)
    baseline = BaselineDenseRetriever.build(parsed_items, settings) if parsed_items else None
    llm = LLMClient(settings)
    gold_by_id = {c.chunk_id: c for c in chunks}
    can_judge = llm.available

    rows: list[dict] = []
    gen_samples: list[dict] = []
    hybrid_latencies: list[float] = []

    for item in dataset["items"]:
        gold_ids = [g for g in item["gold_chunk_ids"] if g in gold_by_id]
        gold_chunks = [gold_by_id[g] for g in gold_ids]

        # --- hybrid ----------------------------------------------------------
        results, tel = hybrid.retrieve(item["question"], top_k=k)
        h_ids = [r.chunk.chunk_id for r in results]
        hybrid_latencies.append(tel.latency_ms)
        row = {
            "question_id": item["id"], "category": item["category"],
            "system": "hybrid", "question": item["question"],
            "retrieved": h_ids,
            "hit_rate_at_5": hit_rate_at_k(h_ids, gold_ids, k),
            "context_recall": context_recall(h_ids, gold_ids, k),
            "context_precision": context_precision(h_ids, gold_ids, k),
            "faithfulness": None, "answer_relevance": None,
            "latency_ms": tel.latency_ms,
            "embedder": tel.embedder, "reranker": tel.reranker,
        }

        if run_generation and can_judge:
            from ..rag.generate import answer_question
            ans = answer_question(item["question"], results, llm, settings)
            ctx_block = "\n".join(r.chunk.text for r in results)[:6000]
            row["faithfulness"] = llm_faithfulness(llm, item["question"], ctx_block, ans["answer"])
            row["answer_relevance"] = llm_answer_relevance(llm, item["question"], ans["answer"])
            gen_samples.append({
                "question": item["question"],
                "contexts": [r.chunk.text for r in results],
                "answer": ans["answer"],
                "ground_truth": item["ground_truth"],
            })
        rows.append(row)

        # --- baseline ---------------------------------------------------------
        if baseline is not None:
            b_ids_raw = baseline.retrieve_ids(item["question"], top_k=k)
            b_texts = [baseline.chunks[i]["text"] for i in b_ids_raw]
            b_ids = _align(b_texts, gold_chunks)
            brow = {
                "question_id": item["id"], "category": item["category"],
                "system": "baseline", "question": item["question"],
                "retrieved": b_ids,
                "hit_rate_at_5": hit_rate_at_k(b_ids, gold_ids, k),
                "context_recall": context_recall(b_ids, gold_ids, k),
                "context_precision": context_precision(b_ids, gold_ids, k),
                "faithfulness": None, "answer_relevance": None,
                "latency_ms": None,
            }
            rows.append(brow)

    # optional RAGAS delegation for faithfulness
    metric_backend = "internal-prompted-judge" if can_judge else "deterministic-only"
    if can_judge and gen_samples:
        ragas_scores = try_ragas_faithfulness(llm, gen_samples)
        if ragas_scores:
            metric_backend = "ragas"
            h_rows = [r for r in rows if r["system"] == "hybrid"]
            for r, s in zip(h_rows, ragas_scores):
                r["faithfulness"] = s

    systems = ["hybrid"] + (["baseline"] if baseline is not None else [])
    aggregates = {(s, "overall"): _aggregate(rows, s) for s in systems}
    for cat in CATEGORIES:
        for s in systems:
            a = _aggregate(rows, s, cat)
            if a:
                aggregates[(s, cat)] = a

    # --- hypothesis check (spec: >30% Context Recall gain on tabular/chart) --
    tabvis = [r for r in rows if r["category"] in {"tabular", "visual"}]
    h_rec = _aggregate([r for r in tabvis if r["system"] == "hybrid"], "hybrid").get("context_recall", 0.0)
    b_rows = [r for r in tabvis if r["system"] == "baseline"]
    b_rec = _aggregate(b_rows, "baseline").get("context_recall", 0.0) if b_rows else None
    if b_rec:
        improvement = (h_rec - b_rec) / b_rec if b_rec > 0 else float("inf")
        hypothesis = {
            "hybrid_recall_tabvis": h_rec,
            "baseline_recall_tabvis": b_rec,
            "relative_improvement": improvement,
            "threshold": 0.30,
            "supported": improvement > 0.30,
        }
    else:
        hypothesis = {"hybrid_recall_tabvis": h_rec, "supported": None,
                      "note": "baseline unavailable"}

    backend = {
        "embedder": hybrid.embedder.name,
        "reranker": hybrid.reranker.name,
        "store": hybrid.store.name,
        "llm": llm.model if llm.available else None,
        "generative_metrics": metric_backend,
    }

    payload = {
        "backend": backend,
        "dataset_counts": dataset["counts"],
        "aggregates": {f"{s}|{c}": v for (s, c), v in aggregates.items()},
        "hypothesis": hypothesis,
        "n_questions": len(dataset["items"]),
        "hybrid_mean_latency_ms": _mean(hybrid_latencies) or None,
    }

    # --- write outputs --------------------------------------------------------
    (results_dir / "metrics.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    with (results_dir / "per_question.jsonl").open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    md = [
        "# Multimodal Hybrid RAG — Benchmark Results",
        "",
        f"- Questions: **{len(dataset['items'])}** {dataset['counts']}",
        f"- Backends: embedder=`{backend['embedder']}`, reranker=`{backend['reranker']}`, "
        f"store=`{backend['store']}`, llm=`{backend['llm']}`, "
        f"generative metrics=`{metric_backend}`",
        "",
        "## Overall",
        "",
        _md_table(aggregates, systems),
        "",
        "## Per category",
        "",
        _md_categories(aggregates, systems),
        "",
        "## Hypothesis check (>30% Context Recall gain on tabular/chart questions)",
        "",
        json.dumps(hypothesis, indent=2),
        "",
    ]
    (results_dir / "benchmark.md").write_text("\n".join(md), encoding="utf-8")
    print(f"[eval] wrote {results_dir / 'benchmark.md'}")
    return payload
