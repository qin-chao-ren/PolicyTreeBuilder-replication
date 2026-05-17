#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import os
from typing import Dict


def _read_env(key: str) -> str:
    return (os.getenv(key) or "").strip()


def activate_openai_compatible_env(
    *,
    model_env_key: str,
    base_url_env_key: str,
    api_key_env_key: str,
    model_override: str | None = None,
    overwrite: bool = True,
) -> Dict[str, str]:
    model = (model_override or "").strip() or _read_env(model_env_key)
    base_url = _read_env(base_url_env_key)
    api_key = _read_env(api_key_env_key)

    if not base_url:
        raise ValueError(f"Missing base URL: env {base_url_env_key} is not set")
    if not api_key:
        raise ValueError(f"Missing API key: env {api_key_env_key} is not set")

    if overwrite or not _read_env("OPENAI_BASE_URL"):
        os.environ["OPENAI_BASE_URL"] = base_url
    if overwrite or not _read_env("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = api_key
    if model and (overwrite or not _read_env("OPENAI_MODEL")):
        os.environ["OPENAI_MODEL"] = model

    return {
        "model": model,
        "model_env_key": model_env_key,
        "base_url_env_key": base_url_env_key,
        "api_key_env_key": api_key_env_key,
        "base_url": base_url,
        "api_key_set": "true" if bool(api_key) else "false",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Activate an OpenAI-compatible runtime env mapping")
    parser.add_argument("--model-env-key", default="PRIMARY_LLM_MODEL")
    parser.add_argument("--base-url-env-key", default="PRIMARY_LLM_BASE_URL")
    parser.add_argument("--api-key-env-key", default="PRIMARY_LLM_API_KEY")
    parser.add_argument("--model-override", default=None)
    parser.add_argument("--no-overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = activate_openai_compatible_env(
        model_env_key=args.model_env_key,
        base_url_env_key=args.base_url_env_key,
        api_key_env_key=args.api_key_env_key,
        model_override=args.model_override,
        overwrite=not args.no_overwrite,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
