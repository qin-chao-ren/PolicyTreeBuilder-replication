#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import os

from run_v10_1_reference_normalization import ROOT, embed_text_batch, load_env_file, rerank_documents


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manual embedding/rerank connectivity test for v10.1")
    parser.add_argument("--env", default=str(ROOT / "configs" / "roundC_v4.env"))
    parser.add_argument("--embed-model", default=None)
    parser.add_argument("--rerank-model", default=None)
    parser.add_argument("--skip-rerank", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_env_file(args.env)

    embed_model = args.embed_model or os.getenv("EMBED_MODEL_NAME")
    if not embed_model:
        raise ValueError("Missing embedding model. Provide --embed-model or set EMBED_MODEL_NAME in env.")

    embed_base_url = os.getenv("EMBED_MODEL_BASE_URL") or os.getenv("OPENAI_BASE_URL")
    embed_api_key = os.getenv("EMBED_MODEL_API_KEY") or os.getenv("OPENAI_API_KEY")
    vectors = embed_text_batch(
        ["航空货运网络", "异地货站一次安检机制", "国际货运航线奖励"],
        model=embed_model,
        base_url=str(embed_base_url or ""),
        api_key=str(embed_api_key or ""),
        timeout_s=float(os.getenv("TIMEOUT", "120")),
    )

    output = {
        "embed_model": embed_model,
        "embedding_count": int(vectors.shape[0]),
        "embedding_dim": int(vectors.shape[1]),
    }

    if not args.skip_rerank:
        rerank_model = args.rerank_model or os.getenv("RERANK_MODEL_NAME")
        rerank_endpoint = os.getenv("RERANK_API_ENDPOINT") or os.getenv("RERANK_API_URL")
        rerank_api_key = os.getenv("RERANK_API_KEY")
        if not rerank_model:
            raise ValueError("Missing rerank model. Provide --rerank-model or set RERANK_MODEL_NAME in env.")
        scores = rerank_documents(
            endpoint=str(rerank_endpoint or ""),
            api_key=str(rerank_api_key or ""),
            model=rerank_model,
            query="异地货站 一次安检",
            documents=["异地货站一次安检机制", "国际货运航线奖励", "空空中转业务"],
            timeout_s=float(os.getenv("TIMEOUT", "120")),
        )
        output["rerank_model"] = rerank_model
        output["rerank_scores"] = scores

    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
