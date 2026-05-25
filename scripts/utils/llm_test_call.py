#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM 调用测试脚本（V4）

示例（PowerShell）：
  python scripts/utils/llm_test_call.py `
    --env configs/.env `
    --role primary `
    --user "请用不超过 8 个字回答：你好世界"

用途：
  - 快速验证 env 配置与模型连通性
  - 默认读取 PRIMARY_LLM_* 环境变量，也可以通过 --role 切换 secondary/judge*
  - 直接复用 scripts/common_llm.call_json
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPTS_DIR = HERE.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from utils.llm_client import load_env_file  # noqa: E402
from common_llm import call_json  # noqa: E402


def _pick_env(keys: tuple[str, ...]) -> str | None:
    for key in keys:
        val = os.getenv(key)
        if val:
            return val
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description="LLM quick test caller (V4)")
    ap.add_argument("--env", type=str, default="configs/.env")
    ap.add_argument(
        "--role",
        type=str,
        choices=["primary", "secondary", "judge1", "judge2", "judge3"],
        default="primary",
        help="选择要测试的模型角色",
    )
    ap.add_argument("--model", type=str, help="覆盖模型名，不再读取角色默认值")
    ap.add_argument("--system", type=str, default="你是一个谨慎、简洁的助手。")
    ap.add_argument("--user", type=str, default="请用不超过 12 个字回答：你好，世界。")
    args = ap.parse_args()

    if args.env:
        try:
            load_env_file(args.env)
        except FileNotFoundError as exc:
            print(f"[WARN] env file not found: {exc}")

    role_map = {
        "primary": ("PRIMARY_LLM_MODEL", "PRIMARY_LLM_BASE_URL", "PRIMARY_LLM_API_KEY"),
        "secondary": ("SECONDARY_LLM_MODEL", "SECONDARY_LLM_BASE_URL", "SECONDARY_LLM_API_KEY"),
        "judge1": ("JUDGE_LLM_1_MODEL", "JUDGE_LLM_1_BASE_URL", "JUDGE_LLM_1_API_KEY"),
        "judge2": ("JUDGE_LLM_2_MODEL", "JUDGE_LLM_2_BASE_URL", "JUDGE_LLM_2_API_KEY"),
        "judge3": ("JUDGE_LLM_3_MODEL", "JUDGE_LLM_3_BASE_URL", "JUDGE_LLM_3_API_KEY"),
    }
    mk, bk, kk = role_map[args.role]
    model = args.model or os.getenv(mk)
    if not model:
        raise ValueError(f"model not specified and env {mk} is empty")

    base_url = _pick_env((bk, "OPENAI_BASE_URL", "OPENAI_BASE"))
    api_key = _pick_env((kk, "OPENAI_API_KEY"))
    if base_url:
        os.environ["OPENAI_BASE_URL"] = base_url
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

    print(f"[INFO] Calling role={args.role} model={model}")
    resp = call_json(
        model=model,
        system_text=args.system,
        user_text=args.user,
        temperature=float(os.getenv("LLM_TEMPERATURE", "0.2")),
        max_tokens=int(os.getenv("LLM_MAX_TOKENS", "512")),
        response_format=os.getenv("LLM_RESPONSE_FORMAT", "json_object"),
        timeout=float(os.getenv("LLM_TIMEOUT", "60")),
        retries=int(os.getenv("LLM_RETRIES", "2")),
        backoff=float(os.getenv("LLM_BACKOFF", "1.5")),
    )
    ok = bool(resp.get("ok"))
    status = resp.get("status")
    latency = resp.get("latency_ms")
    raw = (resp.get("raw") or "")[:400]
    print(f"[RESULT] ok={ok} status={status} latency_ms={latency}")
    if ok:
        print(f"[RAW] {raw}")
        sys.exit(0)
    else:
        print(f"[ERROR] {raw}")
        sys.exit(1)


if __name__ == "__main__":
    main()

