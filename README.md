# Embeddings + reranker search on Chalk GPUs

Semantic search over the [Chalk docs](https://github.com/chalk-ai/docs) in two stages:

1. **Retrieve.** [`gte-modernbert-base`](https://huggingface.co/Alibaba-NLP/gte-modernbert-base)
   embeds about 1,800 doc chunks once at startup. A query is embedded the same way, and cosine
   similarity picks the 50 closest chunks in a few milliseconds.
2. **Rerank.** [`gte-reranker-modernbert-base`](https://huggingface.co/Alibaba-NLP/gte-reranker-modernbert-base),
   a cross-encoder, reads the query and each candidate together and scores how well the passage
   answers it. The reranker runs on **two L4 GPUs**, so concurrent requests are spread across both.

The page shows both rankings side by side, so you can see which results the reranker promoted
or demoted.

```
query ─▶ embedder (1× L4) ─▶ cosine top-50 over 1.8k chunks ─▶ reranker (2× L4) ─▶ top 10
```

## Files

| File | Role |
| --- | --- |
| `deploy.py` | Deploys three `chalkcompute.ScalingGroup`s: the embedder, the reranker and the search app. |
| `app.py` | The search app: chunks the docs, builds the vector index, serves `/api/search` and the page. |
| `index.html` | The side-by-side search page. |
| `search.py` | Command-line client: one query, or `--bench` for a concurrent load test. |

Both models are served by Hugging Face's
[text-embeddings-inference](https://github.com/huggingface/text-embeddings-inference) (TEI). TEI
batches concurrent requests on the GPU and exposes `/embed` and `/rerank`. Each TEI process owns
one GPU, so the reranker is a scaling group with two single-L4 replicas behind one URL. Add
replicas, or set `max_replicas` above `min_replicas`, to scale it out.

## Run it

You need [`uv`](https://docs.astral.sh/uv/) and a Chalk environment that can schedule L4 GPUs.
Credentials come from the SDK’s ambient credentials from `chalk login`, or from `CHALK_API_SERVER`,
`CHALK_CLIENT_ID`, `CHALK_CLIENT_SECRET` and `CHALK_ENVIRONMENT_ID`.

```bash
./deploy.py
./search.py https://<web_url> "how do I backfill a feature"
./search.py https://<web_url> --bench --concurrency 16 --requests 200
```

The model servers are `authenticated=True`, so they are not reachable from the internet. The
search app runs with `chalk_identity=True` and calls them with its own workload token, which it
re-reads from `CHALK_WEB_IDENTITY_TOKEN_FILE` on every request because the token rotates.

## API

`POST /api/search` with `{"query": "...", "candidates": 50, "top": 10}` returns:

```json
{
  "embedding": [{"rank": 1, "title": "...", "heading": "...", "url": "...", "snippet": "...", "similarity": 0.71}],
  "reranked":  [{"rank": 1, "title": "...", "heading": "...", "url": "...", "snippet": "...", "score": 0.93, "embedding_rank": 7}],
  "timings_ms": {"embed_query": 9.1, "vector_search": 0.4, "rerank": 38.2, "total": 47.9}
}
```

## Notes

- TEI caps how many texts one request may carry (`--max-client-batch-size`, default 32). A
  rerank sends every candidate in one request, so both servers raise the cap to 128.
- Chunks are split at `##`/`###` headings and packed to about 1,200 characters. Each chunk is
  prefixed with its page and section title, which helps both models on short sections.
- Clean up with `ScalingGroup.from_name("<name>").delete()` for `docs-search`,
  `docs-search-embedder` and `docs-search-reranker`.

## License

Apache-2.0. Both models are Apache-2.0, released by Alibaba-NLP.

## Local authentication

The SDK reads credentials, API host, and the selected environment from your existing
`chalk login` in `~/.chalk.yml`, including project-scoped and default logins.
For unattended runs, `CHALK_CLIENT_ID` and `CHALK_CLIENT_SECRET` remain supported.
The examples use the current environment selected by your Chalk login.
