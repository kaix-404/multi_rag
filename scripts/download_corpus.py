#!/usr/bin/env python
"""Download the evaluation corpus.

Per spec: 20 corporate Form 10-K annual reports (Apple, Tesla, Microsoft) +
5 arXiv technical papers, into ``data/corpus/``.

Three provenance-tracked sources (annualreports.com is unreachable from some
networks, so official sources are used):

1. **Direct PDF**  — Apple investor-relations CDN (verified application/pdf).
2. **SEC EDGAR**   — official filing HTML rendered to PDF with headless Edge
   using a declared SEC user-agent (Tesla + Microsoft + Apple edge years).
3. **arXiv**       — direct PDF downloads.

Everything is skipped when already present, so the script is re-runnable.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data" / "corpus"
TMP = ROOT / "tmp"

BROWSER_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/126.0 Safari/537.36")
SEC_UA = "CapstoneResearch/1.0 (research@example.com)"   # SEC requires a declared UA
EDGE_CANDIDATES = [
    Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
    Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
    Path.home() / "AppData/Local/Microsoft/Edge/Application/msedge.exe",
]

# --------------------------------------------------------------------------- #
# source 1: direct Apple IR PDFs (verified 206 application/pdf)
# --------------------------------------------------------------------------- #
DIRECT = [
    ("AAPL_2019.pdf", "https://s2.q4cdn.com/470004039/files/doc_financials/2019/ar/_10-K-2019-(As-Filed).pdf"),
    ("AAPL_2020.pdf", "https://s2.q4cdn.com/470004039/files/doc_financials/2020/ar/_10-K-2020-(As-Filed).pdf"),
    ("AAPL_2021.pdf", "https://s2.q4cdn.com/470004039/files/doc_financials/2021/q4/_10-K-2021-%28As-Filed%29.pdf"),
    ("AAPL_2022.pdf", "https://s2.q4cdn.com/470004039/files/doc_financials/2022/q4/_10-K-2022-(As-Filed).pdf"),
    ("AAPL_2023.pdf", "https://s2.q4cdn.com/470004039/files/doc_earnings/2023/q4/filing/_10-K-Q4-2023-As-Filed.pdf"),
]

# --------------------------------------------------------------------------- #
# source 2: SEC EDGAR -> headless Edge PDF
# --------------------------------------------------------------------------- #
EDGAR_COMPANIES = {
    # ticker: (cik, fiscal-year extraction from filing date)
    "TSLA": ("0001318605", lambda y, m: y - 1 if m <= 6 else y),   # filed Jan/Feb
    "MSFT": ("0000789019", lambda y, m: y),                        # filed Jul
    "AAPL": ("0000320193", lambda y, m: y),                        # filed Oct/Nov
}
# (name, ticker, fiscal year) rendered from EDGAR to reach 20 10-Ks
EDGAR_TARGETS = [
    ("TSLA_2018.pdf", "TSLA", 2018), ("TSLA_2019.pdf", "TSLA", 2019),
    ("TSLA_2020.pdf", "TSLA", 2020), ("TSLA_2021.pdf", "TSLA", 2021),
    ("TSLA_2022.pdf", "TSLA", 2022), ("TSLA_2023.pdf", "TSLA", 2023),
    ("TSLA_2024.pdf", "TSLA", 2024),
    ("MSFT_2019.pdf", "MSFT", 2019), ("MSFT_2020.pdf", "MSFT", 2020),
    ("MSFT_2021.pdf", "MSFT", 2021), ("MSFT_2022.pdf", "MSFT", 2022),
    ("MSFT_2023.pdf", "MSFT", 2023), ("MSFT_2024.pdf", "MSFT", 2024),
    ("AAPL_2018.pdf", "AAPL", 2018), ("AAPL_2024.pdf", "AAPL", 2024),
]

# --------------------------------------------------------------------------- #
# source 3: arXiv
# --------------------------------------------------------------------------- #
PAPERS = [
    ("arXiv_2005.11401_RAG.pdf", "https://arxiv.org/pdf/2005.11401"),
    ("arXiv_2312.10997_RAG_Survey.pdf", "https://arxiv.org/pdf/2312.10997"),
    ("arXiv_2407.01449_ColPali.pdf", "https://arxiv.org/pdf/2407.01449"),
    ("arXiv_1906.00037_ColBERT.pdf", "https://arxiv.org/pdf/1906.00037"),
    ("arXiv_1706.03762_Transformer.pdf", "https://arxiv.org/pdf/1706.03762"),
]


def _http_get(url: str, ua: str, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": ua})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def looks_like_pdf(path: Path, min_bytes: int = 100_000) -> bool:
    if not path.exists() or path.stat().st_size < min_bytes:
        return False
    with path.open("rb") as fh:
        return fh.read(5) == b"%PDF-"


def fetch_direct(name: str, url: str) -> bool:
    dest = CORPUS / name
    if looks_like_pdf(dest):
        print(f"[skip] {name}")
        return True
    try:
        data = _http_get(url, BROWSER_UA)
        if not data.startswith(b"%PDF-"):
            print(f"[fail] {name}: not a PDF ({len(data)} bytes)")
            return False
        dest.write_bytes(data)
        print(f"[ok]   {name} ({len(data) // 1024} KiB, direct)")
        return True
    except Exception as exc:
        print(f"[fail] {name}: {exc.__class__.__name__}: {exc}")
        return False


# ---- EDGAR helpers ----------------------------------------------------------

def edgar_index(cik: str) -> list[dict]:
    """All 10-K filings for a CIK (recent + overflow files)."""
    out: list[dict] = []
    base = f"https://data.sec.gov/submissions/CIK{cik}.json"
    urls = [base]
    data = json.loads(_http_get(base, SEC_UA))
    for extra in data.get("filings", {}).get("files", []):
        urls.append(f"https://data.sec.gov/submissions/{extra['name']}")
    for u in urls[1:]:
        data2 = json.loads(_http_get(u, SEC_UA))
        rec = data2 if "form" in data2 else data2.get("filings", {}).get("recent", data2)
        data.setdefault("_extra", []).append(rec)
    recs = [data["filings"]["recent"]] + list(data.get("_extra", []))
    for rec in recs:
        for form, acc, dt, doc in zip(rec.get("form", []), rec.get("accessionNumber", []),
                                      rec.get("filingDate", []), rec.get("primaryDocument", [])):
            if form == "10-K":
                out.append({"accession": acc, "date": dt, "doc": doc})
    return out


def edgar_url(cik: str, accession: str, doc: str) -> str:
    return (f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/"
            f"{accession.replace('-', '')}/{doc}")


def find_edge() -> Path | None:
    for p in EDGE_CANDIDATES:
        if p.exists():
            return p
    return None


def edge_render(url: str, dest: Path, timeout: int = 240) -> bool:
    edge = find_edge()
    if edge is None:
        print("[fail] Edge browser not found; cannot render EDGAR HTML")
        return False
    TMP.mkdir(exist_ok=True)
    # Unique profile per call: a leftover Edge instance holding a profile lock
    # would otherwise silently absorb our command and nothing would render.
    profile = TMP / f"edge-profile-{dest.stem}"
    cmd = [
        str(edge), "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
        f"--user-agent={SEC_UA}",
        f"--user-data-dir={profile.as_posix()}",
        "--virtual-time-budget=60000",
        f"--print-to-pdf={dest.as_posix()}",
        url,
    ]
    dest.unlink(missing_ok=True)
    try:
        res = subprocess.run(cmd, capture_output=True, timeout=timeout)
        (TMP / f"{dest.stem}.edge.log").write_bytes(
            (res.stdout or b"") + b"\n---stderr---\n" + (res.stderr or b""))
    except subprocess.TimeoutExpired:
        print(f"[warn] {name_of(dest)}: Edge timeout, waiting for file...")
    # Edge writes asynchronously — poll for a stable file
    deadline = time.time() + timeout
    last = -1
    while time.time() < deadline:
        if dest.exists():
            size = dest.stat().st_size
            if size == last and looks_like_pdf(dest, 100_000):
                return True
            last = size
        time.sleep(2)
    ok = looks_like_pdf(dest)
    # best-effort cleanup of the throwaway profile (may be briefly locked)
    import shutil
    try:
        shutil.rmtree(profile, ignore_errors=True)
    except Exception:
        pass
    return ok


def name_of(p: Path) -> str:
    return p.name


def verify_pdf(path: Path) -> str | None:
    """Return an error string when the PDF is invalid or an SEC block page."""
    if not looks_like_pdf(path):
        return "not a valid PDF"
    try:
        import pymupdf
        doc = pymupdf.open(str(path))
        if doc.page_count < 5:
            return f"too few pages ({doc.page_count})"
        head = doc[0].get_text()[:600]
        if "Undeclared Automated Tool" in head:
            return "SEC block page"
        return None
    except Exception as exc:
        return f"unreadable: {exc}"


def fetch_edgar(name: str, ticker: str, fiscal_year: int, index_cache: dict) -> bool:
    dest = CORPUS / name
    if looks_like_pdf(dest) and verify_pdf(dest) is None:
        print(f"[skip] {name}")
        return True
    cik, fy_fn = EDGAR_COMPANIES[ticker]
    if cik not in index_cache:
        index_cache[cik] = edgar_index(cik)
        time.sleep(0.5)
    match = None
    for f in index_cache[cik]:
        y, m = int(f["date"][:4]), int(f["date"][5:7])
        if fy_fn(y, m) == fiscal_year:
            match = f
    if match is None:
        print(f"[fail] {name}: no EDGAR 10-K found for FY{fiscal_year}")
        return False
    url = edgar_url(cik, match["accession"], match["doc"])
    print(f"[....] {name}: rendering {url}")
    if not edge_render(url, dest):
        print(f"[fail] {name}: Edge render timed out")
        return False
    err = verify_pdf(dest)
    if err:
        print(f"[fail] {name}: {err}")
        dest.unlink(missing_ok=True)
        return False
    print(f"[ok]   {name} ({dest.stat().st_size // 1024} KiB, EDGAR render)")
    return True


def fetch_paper(name: str, url: str) -> bool:
    return fetch_direct(name, url)


def main() -> int:
    CORPUS.mkdir(parents=True, exist_ok=True)
    ok = fail = 0
    index_cache: dict[str, list[dict]] = {}

    print("=== 1/3 direct PDFs (Apple IR CDN) ===")
    for name, url in DIRECT:
        if fetch_direct(name, url):
            ok += 1
        else:
            fail += 1
        time.sleep(1.0)

    print("=== 2/3 SEC EDGAR -> headless Edge PDF ===")
    for name, ticker, fy in EDGAR_TARGETS:
        if fetch_edgar(name, ticker, fy, index_cache):
            ok += 1
        else:
            fail += 1
        time.sleep(1.5)

    print("=== 3/3 arXiv papers ===")
    for name, url in PAPERS:
        if fetch_paper(name, url):
            ok += 1
        else:
            fail += 1
        time.sleep(1.5)

    tenks = sorted(p.name for p in CORPUS.glob("*.pdf")
                   if re.match(r"^(AAPL|TSLA|MSFT)_\d{4}\.pdf$", p.name))
    papers = sorted(p.name for p in CORPUS.glob("arXiv_*.pdf"))
    print(f"\n10-Ks: {len(tenks)} / 20   papers: {len(papers)} / 5   "
          f"ok={ok} fail={fail}")
    return 0 if len(tenks) >= 20 and len(papers) >= 5 else 1


if __name__ == "__main__":
    sys.exit(main())
