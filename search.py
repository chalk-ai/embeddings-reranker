#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12"
# dependencies = ["httpx"]
# ///
"""Query the deployed docs search from the command line, or load-test it.

./search.py https://<web_url> "how do I backfill a feature"
./search.py https://<web_url> --bench --concurrency 16 --requests 200
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import time

import httpx

BENCH_QUERIES = [
    "how do I backfill a feature after changing a resolver",
    "count failed logins in the last 10 minutes",
    "connect Chalk to Snowflake",
    "what happens when an online query times out",
    "run a resolver on a schedule",
    "test features locally before deploying",
    "stream features from Kafka",
    "cache feature values in the online store",
]


def show(res: dict) -> None:
    print(f"\n{res['corpus_chunks']} chunks, {res['candidates']} candidates, {res['timings_ms']}\n")
    print(f"{'embedding':<52} reranked")
    for e, r in zip(res["embedding"], res["reranked"]):
        left = f"{e['rank']:>2}. {e['title'][:30]} · {e['heading'][:12]}"
        right = f"{r['rank']:>2}. {r['title'][:30]} · {r['heading'][:12]} (was #{r['embedding_rank']})"
        print(f"{left:<52} {right}")


async def bench(url: str, concurrency: int, requests: int) -> None:
    latencies: list[float] = []
    rerank_ms: list[float] = []
    queue: asyncio.Queue[int] = asyncio.Queue()
    for i in range(requests):
        queue.put_nowait(i)

    async def worker(client: httpx.AsyncClient) -> None:
        while not queue.empty():
            i = queue.get_nowait()
            t0 = time.perf_counter()
            r = await client.post(f"{url}/api/search", json={"query": BENCH_QUERIES[i % len(BENCH_QUERIES)]})
            r.raise_for_status()
            latencies.append((time.perf_counter() - t0) * 1000)
            rerank_ms.append(r.json()["timings_ms"]["rerank"])

    async with httpx.AsyncClient(timeout=120) as client:
        await client.post(f"{url}/api/search", json={"query": "warm up"})
        t0 = time.perf_counter()
        await asyncio.gather(*(worker(client) for _ in range(concurrency)))
        wall = time.perf_counter() - t0

    q = statistics.quantiles(latencies, n=20)
    print(f"{requests} searches, concurrency {concurrency}, {wall:.1f}s wall")
    print(f"  throughput   {requests / wall:.1f} searches/s ({requests * 50 / wall:.0f} pairs reranked/s)")
    print(f"  end-to-end   p50 {statistics.median(latencies):.0f} ms  p95 {q[18]:.0f} ms")
    print(f"  rerank only  p50 {statistics.median(rerank_ms):.0f} ms")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("query", nargs="?")
    parser.add_argument("--bench", action="store_true")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--requests", type=int, default=100)
    args = parser.parse_args()
    url = args.url.rstrip("/")
    if args.bench:
        asyncio.run(bench(url, args.concurrency, args.requests))
    else:
        r = httpx.post(f"{url}/api/search", json={"query": args.query}, timeout=120)
        r.raise_for_status()
        show(r.json())


if __name__ == "__main__":
    main()
