#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve()
SIM_DIR = SCRIPT_PATH.parent
ROOT = SIM_DIR.parent
SCRIPTS_DIR = ROOT / "scripts"
UTILS_DIR = SCRIPTS_DIR / "utils"

for path in (SIM_DIR, SCRIPTS_DIR, UTILS_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from llm_client import chat_json, load_env_file  # noqa: E402
from openai_compatible_env import activate_openai_compatible_env  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manual LLM connectivity test for v10.1")
    parser.add_argument("--env", default=str(ROOT / "configs" / "roundC_v4.env"))
    parser.add_argument("--llm-model", default=None)
    parser.add_argument("--llm-model-env-key", default="PRIMARY_LLM_MODEL")
    parser.add_argument("--llm-base-url-env-key", default="PRIMARY_LLM_BASE_URL")
    parser.add_argument("--llm-api-key-env-key", default="PRIMARY_LLM_API_KEY")
    parser.add_argument("--max-tokens", type=int, default=128)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_env_file(args.env)
    runtime_env = activate_openai_compatible_env(
        model_env_key=args.llm_model_env_key,
        base_url_env_key=args.llm_base_url_env_key,
        api_key_env_key=args.llm_api_key_env_key,
        model_override=args.llm_model,
        overwrite=True,
    )

    model = args.llm_model or runtime_env["model"] or os.getenv(args.llm_model_env_key)
    if not model:
        raise ValueError(
            "Missing LLM model. Provide --llm-model or set the env referenced by --llm-model-env-key."
        )

    connectivity_result, connectivity_payload = chat_json(
        system="You are an LLM connectivity test assistant. Return JSON only.",
        user='Return {"ok": true, "message": "v10.1 llm connectivity ok"}.',
        model=model,
        temperature=0.0,
        max_tokens=args.max_tokens,
        timeout_s=float(os.getenv("TIMEOUT", "120")),
        retries=int(os.getenv("RETRIES", "3")),
        backoff=float(os.getenv("BACKOFF", "1.5")),
    )

    identity_result, identity_payload = chat_json(
        system="You are an LLM identity test assistant. Return JSON only.",
        user=(
            f'The configured API model name for this request is "{model}". '
            "State the exact model identifier and exact version you are currently running as. "
            "If you cannot verify the exact version from runtime information, do not guess. "
            "In that case, set exact_version_known to false and explain why. "
            "Answer with JSON in this exact shape: "
            '{"self_report_model":"...",'
            '"self_report_model_version":"...",'
            '"self_report_provider":"...",'
            '"exact_version_known":true,'
            '"matches_configured_model":true,'
            '"notes":"..."}'
        ),
        model=model,
        temperature=0.0,
        max_tokens=args.max_tokens,
        timeout_s=float(os.getenv("TIMEOUT", "120")),
        retries=int(os.getenv("RETRIES", "3")),
        backoff=float(os.getenv("BACKOFF", "1.5")),
    )

    print(
        json.dumps(
            {
                "configured_model": model,
                "env_keys": {
                    "model": args.llm_model_env_key,
                    "base_url": args.llm_base_url_env_key,
                    "api_key": args.llm_api_key_env_key,
                },
                "runtime_env": {
                    "base_url": runtime_env["base_url"],
                    "api_key_set": runtime_env["api_key_set"] == "true",
                },
                "connectivity": {
                    "ok": connectivity_result.ok,
                    "status": connectivity_result.status,
                    "payload": connectivity_payload,
                    "raw": connectivity_result.raw,
                },
                "identity": {
                    "ok": identity_result.ok,
                    "status": identity_result.status,
                    "payload": identity_payload,
                    "raw": identity_result.raw,
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
