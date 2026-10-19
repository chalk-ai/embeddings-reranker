#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = ["chalkcompute==2.13.4"]
# ///
"""Deploy two-stage docs search: an embedding model, a reranker on two L4s, and the search app.

Three Chalk ScalingGroups:
  <prefix>-embedder  text-embeddings-inference serving gte-modernbert-base on one L4
  <prefix>-reranker  text-embeddings-inference serving gte-reranker-modernbert-base, two
                     replicas on one L4 each; requests are load-balanced across both GPUs
  <prefix>           the search app (CPU): embeds the corpus at startup, then for each query
                     retrieves candidates by cosine similarity and reranks them

The two model groups are authenticated; the app reaches them with its own workload identity.

Credentials come from your existing `chalk login` (or the CHALK_* env vars):
    ./deploy.py [--env-id <environment>] [--prefix docs-search]
"""

from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from chalkcompute import ScalingGroup

HERE = Path(__file__).resolve().parent

# text-embeddings-inference images are built per GPU architecture; 89 is Ada (L4, L40S).
TEI_IMAGE = "ghcr.io/huggingface/text-embeddings-inference:89-1.9.4"
EMBEDDER_MODEL = "Alibaba-NLP/gte-modernbert-base"
EMBEDDER_REVISION = "e7f32e3c00f91d699e8c43b53106206bcc72bb22"
RERANKER_MODEL = "Alibaba-NLP/gte-reranker-modernbert-base"
RERANKER_REVISION = "f7481e6055501a30fb19d090657df9ec1f79ab2c"
DOCS_TARBALL = "https://codeload.github.com/chalk-ai/docs/tar.gz/b6a3a3c5c0dfad31f454a601483a422bc903fba7"


def tei_group(name: str, model: str, revision: str, gpu: str, replicas: int) -> ScalingGroup:
    from chalkcompute import Image, ReadinessProbe, ScalingGroup

    return ScalingGroup(
        image=Image.base(TEI_IMAGE),
        name=name,
        gpu=gpu,
        cpu="4",
        memory="16Gi",
        entrypoint=[
            "text-embeddings-router",
            "--model-id",
            model,
            "--revision",
            revision,
            "--port",
            "8080",
            "--auto-truncate",
            # The default (32) is below one rerank request's 50 candidates.
            "--max-client-batch-size",
            "128",
        ],
        port=8080,
        authenticated=True,
        # /health answers only once the model is loaded and warmed up.
        readiness_probe=ReadinessProbe.http("/health"),
        min_replicas=replicas,
        max_replicas=replicas,
    )


def deploy(sg: ScalingGroup, ready_timeout: float) -> str:
    t0 = time.monotonic()
    sg.deploy(build_timeout=1800.0, ready_timeout=ready_timeout)
    url = sg.web_url
    if url is None:
        raise RuntimeError(f"{sg!r} is ready but has no web_url")
    print(f"{sg!r}: ready in {time.monotonic() - t0:.0f}s at {url}")
    return url


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", default="docs-search")
    parser.add_argument("--env-id", default=None)
    parser.add_argument("--gpu", default="nvidia-l4")
    parser.add_argument("--reranker-gpus", type=int, default=2)
    parser.add_argument("--ready-timeout", type=float, default=1800.0)
    args = parser.parse_args()

    if args.env_id:
        os.environ["CHALK_ENVIRONMENT"] = args.env_id
    from chalkcompute import Image, ReadinessProbe, ScalingGroup

    embedder = tei_group(f"{args.prefix}-embedder", EMBEDDER_MODEL, EMBEDDER_REVISION, args.gpu, 1)
    reranker = tei_group(
        f"{args.prefix}-reranker",
        RERANKER_MODEL,
        RERANKER_REVISION,
        args.gpu,
        args.reranker_gpus,
    )
    with ThreadPoolExecutor(max_workers=2) as pool:
        embedder_url, reranker_url = pool.map(lambda sg: deploy(sg, args.ready_timeout), [embedder, reranker])

    image = (
        Image.debian_slim(python_version="3.12")
        .pip_install(["fastapi==0.116.1", "uvicorn[standard]==0.35.0", "httpx==0.28.1", "numpy<3"])
        # The corpus is the public Chalk docs, pinned to one commit and baked into the image.
        .run_commands(
            'mkdir -p /corpus && python -c "import io, tarfile, urllib.request; '
            + f"tarfile.open(fileobj=io.BytesIO(urllib.request.urlopen('{DOCS_TARBALL}').read())).extractall('/corpus')\""
        )
        .add_local_file(str(HERE / "app.py"), "/app/app.py", strategy="copy")
        .add_local_file(str(HERE / "index.html"), "/app/index.html", strategy="copy")
        .workdir("/app")
    )
    app = ScalingGroup(
        image=image,
        name=args.prefix,
        cpu="1",
        memory="2Gi",
        env={
            "EMBEDDER_URL": embedder_url,
            "RERANKER_URL": reranker_url,
            "EMBEDDER_MODEL": EMBEDDER_MODEL,
            "RERANKER_MODEL": RERANKER_MODEL,
        },
        entrypoint=["python", "/app/app.py"],
        port=8080,
        chalk_identity=True,
        readiness_probe=ReadinessProbe.http("/api/status"),
        min_replicas=1,
        max_replicas=1,
    )
    print("web_url:", deploy(app, args.ready_timeout))


if __name__ == "__main__":
    main()
