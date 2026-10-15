# Multimodal Hybrid RAG — Benchmark Results

- Questions: **40** {'factual': 15, 'tabular': 15, 'visual': 10}
- Backends: embedder=`bge-m3`, reranker=`D:\multimodal_hybrid_rag\models\bge-reranker-large`, store=`local-numpy`, llm=`None`, generative metrics=`deterministic-only`

## Overall

| System | Context Precision | Context Recall | Hit Rate @5 | Faithfulness | Answer Relevance | Mean latency (ms) |
|---|---|---|---|---|---|---|
| hybrid | 67.8% | 90.0% | 90.0% | n/a | n/a | 31402.5 |
| baseline | 13.3% | 17.5% | 17.5% | n/a | n/a | n/a |

## Per category

| System | Category | n | Context Recall | Hit Rate @5 |
|---|---|---|---|---|
| hybrid | factual | 15 | 100.0% | 100.0% |
| hybrid | tabular | 15 | 86.7% | 86.7% |
| hybrid | visual | 10 | 80.0% | 80.0% |
| baseline | factual | 15 | 46.7% | 46.7% |
| baseline | tabular | 15 | 0.0% | 0.0% |
| baseline | visual | 10 | 0.0% | 0.0% |

## Hypothesis check (>30% Context Recall gain on tabular/chart questions)

{
  "hybrid_recall_tabvis": 0.84,
  "baseline_recall_tabvis": 0.0,
  "relative_improvement": null,
  "threshold": 0.3,
  "supported": true
}
