# Flowcharts

End-to-end architecture and the live query path. Both diagrams are written in
[Markdown](https://daringfireball.net/projects/markdown/) and rendered to PNG
with [Mermaid](https://mermaid.js.org/). The `.png` files are checked in so
GitHub renders them in the project README without a JS dependency.

## 1. Pipeline architecture

```mermaid
flowchart TD
  A[Input PDF] --> B((Dual-Stream Parser))
  B --> C{Hybrid Index Store}
  C --> D[BM25 sparse vectors]
  C --> E[Dense vectors]
  C --> F[Vector store]
  D --> G((Fusion & Reranking))
  E --> G
  F --> G
  G --> H((Multimodal Synthesis))
  H --> I[Grounded answer + citations]
```

## 2. Live query path

```mermaid
flowchart TD
  A[Streamlit UI] --> B[FastAPI /query]
  B --> C{Dense search}
  B --> D[BM25 sparse search]
  C --> E((RRF k=60 fuse))
  D --> E
  E --> F{Rerank}
  F --> G[Top-N passage set]
  G --> H((LLM / VLM))
  G --> I[Citations + telemetry]
  H --> J[Streaming answer]
```

## Rendering

```bash
# Fast:
npx @mermaid-js/mermaid-cli -i docs/flowchart.md -o docs/flowchart.png

# Slower but more deterministic:
npm i -g @mermaid-js/mermaid-cli
mmdc -i docs/flowchart.md -o docs/flowchart.png
```
