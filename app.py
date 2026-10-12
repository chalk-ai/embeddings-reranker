"""Two-stage semantic search over the Chalk docs: embedding retrieval, then cross-encoder reranking.

Runs in the CPU scaling group that deploy.py creates. The two model servers are separate GPU
scaling groups running text-embeddings-inference:
  EMBEDDER_URL  gte-modernbert-base, turns text into vectors (`POST /embed`)
  RERANKER_URL  gte-reranker-modernbert-base, scores (query, passage) pairs (`POST /rerank`)

HTTP API:
  GET  /             the search page (index.html)
  GET  /api/status   corpus size and model names
  POST /api/search   {"query", "candidates"?, "top"?} ->
                     {"embedding": [...], "reranked": [...], "timings_ms": {...}}
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator

import httpx
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

EMBEDDER_URL = os.environ["EMBEDDER_URL"].rstrip("/")
RERANKER_URL = os.environ["RERANKER_URL"].rstrip("/")
EMBEDDER_MODEL = os.environ.get("EMBEDDER_MODEL", "Alibaba-NLP/gte-modernbert-base")
RERANKER_MODEL = os.environ.get("RERANKER_MODEL", "Alibaba-NLP/gte-reranker-modernbert-base")
CORPUS_DIR = Path(os.environ.get("CORPUS_DIR", "/corpus"))
DOCS_BASE_URL = "https://docs.chalk.ai/docs/"
PORT = int(os.environ.get("PORT", "8080"))
INDEX_HTML = Path(__file__).with_name("index.html")

EMBED_BATCH = 64
MAX_CHUNK_CHARS = 1200
MAX_QUERY_CHARS = 500
MAX_CANDIDATES = 100
# Bounds concurrent reranks so a burst of visitors queues here instead of on the GPUs.
RERANK_CONCURRENCY = 8

log = logging.getLogger("search")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


@dataclass(frozen=True)
class Chunk:
    slug: str
    title: str
    heading: str
    text: str

    @property
    def url(self) -> str:
        return DOCS_BASE_URL + self.slug

    def passage(self) -> str:
        """What the models see: the page and section titles give a short chunk its context."""
        head = f"{self.title} — {self.heading}" if self.heading else self.title
        return f"{head}\n{self.text}"


# --- corpus ---------------------------------------------------------------------------------

_FRONTMATTER = re.compile(r"\A---\n(.*?)\n---\n", re.S)
_JSX_LINE = re.compile(r"^\s*(import |export |<[A-Za-z/][^>]*>\s*$)")
_TAG = re.compile(r"</?[A-Za-z][^>]*>")
_HEADING = re.compile(r"^(#{2,3})\s+(.*)$")
# Markdown table separator rows (| --- | :--: |) and horizontal rules carry no text.
_RULE = re.compile(r"^\s*\|?[\s:|-]*-{3,}[\s:|-]*\|?\s*$")


def _title(frontmatter: str, fallback: str) -> str:
    m = re.search(r"^title:\s*(.+)$", frontmatter, re.M)
    return m.group(1).strip().strip("'\"") if m else fallback


def _split(text: str) -> list[str]:
    """Packs paragraphs into chunks of at most MAX_CHUNK_CHARS; a longer paragraph is cut."""
    out: list[str] = []
    cur = ""
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if not para:
            continue
        while len(para) > MAX_CHUNK_CHARS:
            out.append(para[:MAX_CHUNK_CHARS])
            para = para[MAX_CHUNK_CHARS:]
        if cur and len(cur) + len(para) + 2 > MAX_CHUNK_CHARS:
            out.append(cur)
            cur = ""
        cur = f"{cur}\n\n{para}" if cur else para
    if cur:
        out.append(cur)
    return out


def load_corpus(root: Path) -> list[Chunk]:
    # A GitHub tarball unpacks into one <repo>-<sha>/ directory; slugs are relative to it.
    entries = list(root.iterdir())
    if len(entries) == 1 and entries[0].is_dir():
        root = entries[0]
    chunks: list[Chunk] = []
    for path in sorted(root.rglob("*.mdx")):
        raw = path.read_text(encoding="utf-8", errors="replace")
        fm = _FRONTMATTER.match(raw)
        if fm and re.search(r"^published:\s*false", fm.group(1), re.M):
            continue
        body = raw[fm.end() :] if fm else raw
        slug = path.relative_to(root).with_suffix("").as_posix()
        title = _title(fm.group(1) if fm else "", slug.replace("-", " ").title())
        lines = [_TAG.sub("", ln) for ln in body.splitlines() if not _JSX_LINE.match(ln)]
        for heading, section in _sections(lines):
            for piece in _split("\n".join(section)):
                if len(piece) >= 80:
                    chunks.append(Chunk(slug, title, heading, piece))
    return chunks


def _sections(lines: list[str]) -> list[tuple[str, list[str]]]:
    """Splits a page at its ## and ### headings; text before the first heading has heading ""."""
    sections: list[tuple[str, list[str]]] = [("", [])]
    for ln in lines:
        m = _HEADING.match(ln)
        if m:
            sections.append((m.group(2).strip(), []))
        elif not _RULE.match(ln):
            sections[-1][1].append(ln)
    return sections


# --- model clients --------------------------------------------------------------------------


def _auth_headers() -> dict[str, str]:
    # The model groups are authenticated; this replica's workload identity token rotates on
    # disk, so it is re-read per request rather than cached.
    path = os.environ.get("CHALK_WEB_IDENTITY_TOKEN_FILE")
    if not path:
        return {}
    return {"Authorization": f"Bearer {Path(path).read_text().strip()}"}


class ModelError(RuntimeError):
    def __init__(self, r: httpx.Response) -> None:
        super().__init__(f"{r.request.url} returned {r.status_code}: {r.text[:300]}")
        self.status_code = r.status_code


def _check(r: httpx.Response) -> httpx.Response:
    if r.status_code >= 400:
        raise ModelError(r)
    return r


async def embed(client: httpx.AsyncClient, texts: list[str]) -> np.ndarray:
    r = await client.post(
        f"{EMBEDDER_URL}/embed",
        json={"inputs": texts, "normalize": True, "truncate": True},
        headers=_auth_headers(),
    )
    return np.asarray(_check(r).json(), dtype=np.float32)


async def rerank(client: httpx.AsyncClient, query: str, texts: list[str]) -> list[dict[str, Any]]:
    r = await client.post(
        f"{RERANKER_URL}/rerank",
        json={"query": query, "texts": texts, "truncate": True, "raw_scores": False},
        headers=_auth_headers(),
    )
    return _check(r).json()


async def build_index(client: httpx.AsyncClient, chunks: list[Chunk]) -> np.ndarray:
    t0 = time.perf_counter()
    batches = [chunks[i : i + EMBED_BATCH] for i in range(0, len(chunks), EMBED_BATCH)]
    for attempt in range(60):
        try:
            vecs = [await embed(client, [c.passage() for c in b]) for b in batches]
            break
        except (httpx.TransportError, ModelError) as exc:
            # Only "not up yet" is worth waiting out: the embedder may still be loading when this
            # replica boots. A 4xx means the request itself is wrong, so fail the boot loudly.
            if isinstance(exc, ModelError) and exc.status_code < 500:
                raise
            log.info("embedder not ready (%s); retry %d", exc, attempt + 1)
            await asyncio.sleep(10)
    else:
        raise RuntimeError("embedder never became ready")
    matrix = np.concatenate(vecs)
    log.info("embedded %d chunks in %.1fs", len(chunks), time.perf_counter() - t0)
    return matrix


# --- app ------------------------------------------------------------------------------------

state: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    client = httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=10.0))
    chunks = load_corpus(CORPUS_DIR)
    log.info("loaded %d chunks from %s", len(chunks), CORPUS_DIR)
    state.update(
        client=client,
        chunks=chunks,
        matrix=await build_index(client, chunks),
        rerank_slots=asyncio.Semaphore(RERANK_CONCURRENCY),
    )
    yield
    await client.aclose()


app = FastAPI(lifespan=lifespan)


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=MAX_QUERY_CHARS)
    candidates: int = Field(default=50, ge=1, le=MAX_CANDIDATES)
    top: int = Field(default=10, ge=1, le=25)


def _hit(chunk: Chunk, **extra: Any) -> dict[str, Any]:
    return {
        "title": chunk.title,
        "heading": chunk.heading,
        "url": chunk.url,
        "snippet": chunk.text[:360],
        **extra,
    }


@app.get("/")
async def index() -> HTMLResponse:
    return HTMLResponse(INDEX_HTML.read_text())


@app.get("/api/status")
async def status() -> JSONResponse:
    return JSONResponse({"chunks": len(state["chunks"]), "embedder": EMBEDDER_MODEL, "reranker": RERANKER_MODEL})


@app.post("/api/search")
async def search(req: SearchRequest) -> JSONResponse:
    client: httpx.AsyncClient = state["client"]
    chunks: list[Chunk] = state["chunks"]
    matrix: np.ndarray = state["matrix"]
    query = req.query.strip()
    if not query:
        raise HTTPException(400, "empty query")

    t0 = time.perf_counter()
    qvec = (await embed(client, [query]))[0]
    t1 = time.perf_counter()
    sims = matrix @ qvec
    k = min(req.candidates, len(chunks))
    cand = np.argpartition(-sims, k - 1)[:k]
    cand = cand[np.argsort(-sims[cand])]
    t2 = time.perf_counter()
    async with state["rerank_slots"]:
        scored = await rerank(client, query, [chunks[i].passage() for i in cand])
    t3 = time.perf_counter()

    embedding_hits = [
        _hit(chunks[i], rank=r + 1, similarity=round(float(sims[i]), 4)) for r, i in enumerate(cand[: req.top])
    ]
    reranked_hits = [
        _hit(
            chunks[cand[s["index"]]],
            rank=r + 1,
            score=round(float(s["score"]), 4),
            embedding_rank=int(s["index"]) + 1,
        )
        for r, s in enumerate(sorted(scored, key=lambda s: -s["score"])[: req.top])
    ]
    return JSONResponse(
        {
            "query": query,
            "corpus_chunks": len(chunks),
            "candidates": k,
            "embedding": embedding_hits,
            "reranked": reranked_hits,
            "timings_ms": {
                "embed_query": round((t1 - t0) * 1000, 1),
                "vector_search": round((t2 - t1) * 1000, 1),
                "rerank": round((t3 - t2) * 1000, 1),
                "total": round((t3 - t0) * 1000, 1),
            },
        }
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=PORT)
