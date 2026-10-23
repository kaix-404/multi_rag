"""Streamlit demonstration portal (dual-pane workspace + document upload).

Tab 1 — Query & Synthesis / Evidence & Inspection: question entry, streamed
        answer, ranked retrieved passages with full fusion telemetry, and the
        original page render with precise bounding-box overlays for the
        paragraph / table / figure used.
Tab 2 — Add documents: upload a PDF; the backend parses, chunks and embeds
        just that document and hot-swaps the retriever so it is immediately
        searchable.

Run: ``streamlit run src/mmrag/ui/streamlit_app.py`` (talks to the FastAPI
backend at MMRAG_API_URL).
"""

from __future__ import annotations

import io
import json

import httpx
import streamlit as st
from PIL import Image, ImageDraw

try:
    from ..config import load_settings
except ImportError:  # ``streamlit run`` executes this file outside its package
    from mmrag.config import load_settings

settings = load_settings()
API = st.sidebar.text_input("FastAPI URL", settings.api_url)
TOP_K = st.sidebar.slider("Contexts (top-k)", 3, 10, settings.answer_top_k)

# mirror of the backend's upload guards (api/app.py)
MAX_UPLOAD_MB = 30
MAX_UPLOAD_PAGES = 80

TYPE_COLOR = {"text": (52, 152, 219), "table": (46, 204, 113), "figure": (231, 76, 60)}


@st.cache_data(ttl=10)
def api_get(path: str):
    r = httpx.get(f"{API}{path}", timeout=30)
    r.raise_for_status()
    return r.content


def health() -> dict | None:
    try:
        r = httpx.get(f"{API}/health", timeout=15)
        return r.json()
    except Exception as exc:
        st.error(f"Backend unreachable at {API}: {exc}")
        return None


def api_upload(filename: str, data: bytes, timeout: float = 1800) -> dict:
    """POST a PDF to the backend ingest endpoint; returns JSON (or an error dict)."""
    try:
        r = httpx.post(f"{API}/documents/upload",
                       files={"file": (filename, data, "application/pdf")},
                       timeout=timeout)
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as exc:
        try:
            detail = exc.response.json().get("detail", str(exc))
        except Exception:
            detail = str(exc)
        return {"status": "error", "detail": detail}
    except Exception as exc:
        return {"status": "error", "detail": str(exc)}


def run_query(question: str):
    """Stream SSE answer; returns (text, payload)."""
    chunks_text: list[str] = []
    payload: dict = {}
    try:
        with httpx.stream("POST", f"{API}/query",
                          json={"question": question, "top_k": TOP_K, "stream": True},
                          timeout=120) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line.startswith("data: "):
                    continue
                ev = json.loads(line[6:])
                if ev.get("type") == "delta":
                    chunks_text.append(ev["text"])
                elif ev.get("type") == "done":
                    payload = ev
    except Exception as exc:
        st.error(f"Query failed: {exc}")
    return "".join(chunks_text), payload


def draw_overlays(page_png: bytes, contexts: list[dict]) -> Image.Image:
    img = Image.open(io.BytesIO(page_png)).convert("RGB")
    for ctx in contexts:
        bbox, (pw, ph) = ctx.get("bbox"), ctx.get("page_size")
        if not bbox or not pw:
            continue
        sx, sy = img.width / pw, img.height / ph
        box = [bbox[0] * sx, bbox[1] * sy, bbox[2] * sx, bbox[3] * sy]
        color = TYPE_COLOR.get(ctx.get("chunk_type", "text"), (241, 196, 15))
        rank = ctx.get("rank", 0)
        ImageDraw.Draw(img).rectangle(box, outline=color, width=3)
        ImageDraw.Draw(img).text((box[0] + 4, max(box[1] - 16, 2)),
                                 f"[{rank}]", fill=color)
    return img


# --------------------------------------------------------------------------- #

st.set_page_config(page_title="Multimodal Hybrid RAG", layout="wide")
st.title("Multimodal Hybrid RAG Engine")

hb = health()
if hb is None or hb.get("status") != "ok":
    st.stop()

with st.expander("System backends", expanded=False):
    st.json(hb)

tab_query, tab_add = st.tabs(["Query & Synthesis", "Add documents"])

# ---- tab 1: query & evidence ------------------------------------------------
with tab_query:
    left, right = st.columns([1, 1])

    # ---- left: query & synthesis ---------------------------------------------
    with left:
        st.subheader("Query & Synthesis")
        question = st.text_area("Question", height=80,
                                placeholder="e.g. What was the percentage change in "
                                            "R&D expenditure between FY2022 and FY2023?")
        submitted = st.button("Ask", type="primary")

        if submitted and question.strip():
            st.session_state["question"] = question.strip()

        q = st.session_state.get("question")
        if q:
            with st.spinner("Retrieving & synthesizing..."):
                text, payload = run_query(q)
            if payload:
                st.session_state["payload"] = payload
                st.markdown(f"*Generator:* `{payload.get('generator')}`")
                st.write(text or "_empty answer_")

                # --- answer-side evidence: which contexts actually supported this answer ---
                ctxs = payload.get("contexts", [])
                if ctxs:
                    st.subheader("Evidence behind this answer")
                    st.caption(
                        "Each generated answer is grounded in the ranked passages below. "
                        "Select a passage to open the original page with a bounding-box "
                        "overlay for the exact evidence used."
                    )
                    for c in ctxs[:3]:
                        st.markdown(
                            f"**#{c['rank']} · {c['chunk_type']} · p.{c['page_number']}** · "
                            f"{c.get('doc_id','unknown')[:24]} · "
                            f"RRF {c['rrf_score']:.4f} · conf {c['rerank_confidence']:.3f}"
                        )
                        st.markdown(c.get('section') or '')
                        st.code(c['text'][:1200], language=None)
                        st.divider()
                st.markdown("**Ranked retrieved passages**")
                for ctx in payload.get("contexts", []):
                    with st.expander(
                        f"#{ctx['rank']} · {ctx['chunk_type']} · p.{ctx['page_number']} · "
                        f"RRF {ctx['rrf_score']:.4f} · conf {ctx['rerank_confidence']:.3f}"
                    ):
                        st.caption(ctx.get("section") or "")
                        st.write(ctx["text"][:2000])

        payload = st.session_state.get("payload")
        if payload:
            st.subheader("Metric telemetry")
            t = payload.get("telemetry", {})
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Latency (ms)", f"{t.get('latency_ms', 0):.1f}")
            c2.metric("Dense search (ms)", f"{t.get('dense_ms', 0):.1f}")
            c3.metric("BM25 search (ms)", f"{t.get('sparse_ms', 0):.1f}")
            c4.metric("Rerank (ms)", f"{t.get('rerank_ms', 0):.1f}")
            st.caption(
                f"embedder=`{t.get('embedder')}` · reranker=`{t.get('reranker')}` · "
                f"store=`{t.get('store')}` · fused pool={t.get('fused_pool')} · "
                f"RRF k={settings.rrf_k}"
            )
            rows = [{
                "rank": c["rank"], "type": c["chunk_type"], "page": c["page_number"],
                "dense_rank": c["dense_rank"], "sparse_rank": c["sparse_rank"],
                "rrf": round(c["rrf_score"], 5),
                "conf": round(c["rerank_confidence"], 3),
            } for c in payload.get("contexts", [])]
            st.dataframe(rows, use_container_width=True)

    # ---- right: evidence & inspection ----------------------------------------
    with right:
        st.subheader("Evidence & Inspection")
        payload = st.session_state.get("payload")
        contexts = (payload or {}).get("contexts", [])
        if not contexts:
            st.info("Ask a question to inspect evidence with bounding-box overlays.")
        else:
            options = {f"[{c['rank']}] {c['chunk_type']} · p.{c['page_number']} · "
                       f"{c['doc_id'][:30]}": c for c in contexts}
            choice = st.selectbox("Highlighted evidence", list(options.keys()))
            ctx = options[choice]
            try:
                png = api_get(f"/documents/{ctx['doc_id']}/pages/{ctx['page_number']}")
                img = draw_overlays(png, [ctx] + [
                    c for c in contexts
                    if c["doc_id"] == ctx["doc_id"] and c["page_number"] == ctx["page_number"]
                    and c["rank"] != ctx["rank"]
                ])
                st.image(img, caption=f"page {ctx['page_number']} — {ctx['chunk_type']} "
                                      f"bbox {tuple(round(v) for v in ctx['bbox'])}",
                         use_container_width=True)
            except Exception as exc:
                st.warning(f"Could not render page: {exc}")
            st.markdown("**Exact passage used**")
            st.code(ctx["text"][:3000], language=None)
            if ctx.get("asset_path"):
                name = ctx["asset_path"].replace("\\", "/").split("/")[-1]
                try:
                    st.image(api_get(f"/figures/{name}"), caption="cropped figure asset",
                             use_container_width=True)
                except Exception:
                    pass

# ---- tab 2: add documents ----------------------------------------------------
with tab_add:
    st.subheader("Add documents")
    st.caption(
        f"Upload a PDF (≤ {MAX_UPLOAD_MB} MB, ≤ {MAX_UPLOAD_PAGES} pages). Parsing and "
        "embedding run on the backend host — a small PDF takes about a minute, larger "
        "ones several. Only the new file is embedded; existing documents keep their "
        "index. Re-uploading the same file replaces it."
    )
    up = st.file_uploader("PDF file", type=["pdf"])
    run_ingest = st.button("Ingest & index", type="primary", disabled=up is None)

    if run_ingest and up is not None:
        data = up.getvalue()
        with st.spinner("Uploading, parsing, embedding — keep this tab open..."):
            res = api_upload(up.name, data)
        if res.get("status") == "ok":
            st.success(
                f"Indexed **{res['filename']}** — +{res['chunks_added']} chunks from "
                f"{res['pages']} pages ({res['tables']} tables / {res['figures']} figures / "
                f"{res['text']} text) in {res['seconds']} s; "
                f"store now holds {res['chunks_total']} chunks."
            )
            with st.expander("Raw ingest stats"):
                st.json(res)
            st.info("Switch to the query tab and ask something about the new document.")
        else:
            st.error(f"Ingest failed: {res.get('detail', res)}")

    try:
        docs = json.loads(api_get("/documents").decode("utf-8")).get("documents", [])
        st.markdown(f"**Indexed documents ({len(docs)})**")
        st.dataframe(
            [{"doc_id": d.get("doc_id"), "filename": d.get("filename"),
              "pages": d.get("num_pages")} for d in docs],
            use_container_width=True, hide_index=True,
        )
    except Exception as exc:
        st.warning(f"Could not list documents: {exc}")
