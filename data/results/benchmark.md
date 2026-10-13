# Multimodal Hybrid RAG — Benchmark Results

- Questions: **30** {'factual': 15, 'tabular': 5, 'visual': 10}
- Backends: embedder=`hashing-tfidf`, reranker=`heuristic-rrf-blend`, store=`local-numpy`, llm=`None`, generative metrics=`deterministic-only`

## Overall

| System | Context Precision | Context Recall | Hit Rate @5 | Faithfulness | Answer Relevance | Mean latency (ms) |
|---|---|---|---|---|---|---|
| hybrid | 74.9% | 90.0% | 90.0% | n/a | n/a | 308.2 |
| baseline | 39.6% | 50.0% | 50.0% | n/a | n/a | n/a |

## Per category

| System | Category | n | Context Recall | Hit Rate @5 |
|---|---|---|---|---|
| hybrid | factual | 15 | 86.7% | 86.7% |
| hybrid | tabular | 5 | 80.0% | 80.0% |
| hybrid | visual | 10 | 100.0% | 100.0% |
| baseline | factual | 15 | 100.0% | 100.0% |
| baseline | tabular | 5 | 0.0% | 0.0% |
| baseline | visual | 10 | 0.0% | 0.0% |

## Hypothesis check (>30% Context Recall gain on tabular/chart questions)

{
  "hybrid_recall_tabvis": 0.9333333333333333,
  "supported": null,
  "note": "baseline unavailable"
}
