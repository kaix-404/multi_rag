"""Evaluation metrics.

Deterministic (no-LLM) retrieval metrics:
- Hit Rate @ k
- Context Recall      = |retrieved ∩ gold| / |gold|
- Context Precision   = rank-weighted precision over the top-k

LLM-judged generative metrics (require a configured LLM):
- Faithfulness        — fraction of answer claims supported by the contexts
- Answer Relevance    — how directly the answer addresses the question

When ``ragas`` is installed and an LLM is configured the generative metrics
are delegated to RAGAS's reference-free evaluators; otherwise the built-in
prompted judge is used, and ``metric_backend`` records which ran.
"""

from __future__ import annotations

import json
import re

from ..rag.llm import LLMClient


def hit_rate_at_k(retrieved: list[str], gold: list[str], k: int = 5) -> float:
    top = set(retrieved[:k])
    return 1.0 if top & set(gold) else 0.0


def context_recall(retrieved: list[str], gold: list[str], k: int = 5) -> float:
    if not gold:
        return 0.0
    return len(set(retrieved[:k]) & set(gold)) / len(set(gold))


def context_precision(retrieved: list[str], gold: list[str], k: int = 5) -> float:
    """RAGAS-style rank-weighted precision: relevant contexts early rank higher."""
    gold = set(gold)
    hits = [(1.0 if cid in gold else 0.0) for cid in retrieved[:k]]
    n_rel = sum(hits)
    if n_rel == 0:
        return 0.0
    weighted = sum(h / (i + 1) for i, h in enumerate(hits) if h)
    return weighted / n_rel


# --------------------------------------------------------------------------- #
# LLM-judged metrics
# --------------------------------------------------------------------------- #

_FAITH_PROMPT = """You are a strict evaluation judge.
Question: {question}
Contexts: {contexts}
Answer: {answer}

Break the answer into atomic claims. Return ONLY JSON:
{{"supported": <number of claims fully supported by the contexts>,
  "total": <number of claims>}}"""

_RELEVANCE_PROMPT = """Rate how directly and completely the answer addresses the question.
Question: {question}
Answer: {answer}
Return ONLY JSON: {{"score": <0.0-1.0>}}"""


def _parse_json_num(text: str, *keys: str) -> float | None:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except Exception:
        return None
    if "score" in obj and "score" in keys:
        try:
            return max(0.0, min(1.0, float(obj["score"])))
        except Exception:
            return None
    if "supported" in obj and "total" in obj:
        try:
            t = float(obj["total"])
            return float(obj["supported"]) / t if t else 0.0
        except Exception:
            return None
    return None


def llm_faithfulness(llm: LLMClient, question: str, contexts: str, answer: str) -> float | None:
    if not llm.available:
        return None
    try:
        out = llm.chat([{"role": "user", "content": _FAITH_PROMPT.format(
            question=question, contexts=contexts[:6000], answer=answer[:3000])}])
        return _parse_json_num(out, "supported", "total")
    except Exception:
        return None


def llm_answer_relevance(llm: LLMClient, question: str, answer: str) -> float | None:
    if not llm.available:
        return None
    try:
        out = llm.chat([{"role": "user", "content": _RELEVANCE_PROMPT.format(
            question=question, answer=answer[:3000])}])
        return _parse_json_num(out, "score")
    except Exception:
        return None


def try_ragas_faithfulness(llm: LLMClient, samples: list[dict]) -> list[float] | None:
    """Delegate to RAGAS when the package + LLM are available.

    Returns None when RAGAS cannot run (not installed / API mismatch / no LLM)
    so callers fall back to the built-in judge. RAGAS APIs shift between
    releases, hence the defensive import.
    """
    if not llm.available:
        return None
    try:
        from ragas import evaluate  # noqa
        from ragas.metrics import Faithfulness  # noqa
        from datasets import Dataset  # noqa
    except Exception:
        return None
    try:
        # RAGAS >= 0.2 wants its own LLM wrapper; without one configured inside
        # RAGAS this raises, and we fall back cleanly.
        ds = Dataset.from_list(samples)
        res = evaluate(ds, metrics=[Faithfulness()])
        col = res.get("faithfulness")
        return [float(v) for v in col] if col is not None else None
    except Exception:
        return None
