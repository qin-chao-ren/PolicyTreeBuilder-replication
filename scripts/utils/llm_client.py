#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一的 LLM 调用辅助（v4）
- 提供 load_env_file：读取 env 并注入 OPENAI_BASE_URL / OPENAI_API_KEY 等。
- 提供 chat_json：基于 scripts/common_llm.call_json 的轻量封装，返回 (ChatResult, json_obj)。
"""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Optional, Tuple

from common_llm import call_json


@dataclass
class ChatResult:
    ok: bool
    raw: str
    latency_ms: Optional[int] = None
    status: Optional[int] = None
    attempts: int = 1
    error: Optional[str] = None


def load_env_file(path: str | Path) -> None:
    """
    读取 env 文件，按 “KEY=VALUE” 写入 os.environ（若变量不存在）。
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Env file not found: {p}")
    for line in p.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        k, v = s.split("=", 1)
        key = k.strip()
        val = v.strip().strip('"')
        if key and os.getenv(key) is None:
            os.environ[key] = val


def chat_json(
    *,
    system: str,
    user: str,
    model: str,
    temperature: float = 0.2,
    max_tokens: int = 512,
    timeout_s: float = 60.0,
    retries: int = 2,
    backoff: float = 1.5,
) -> Tuple[ChatResult, Optional[dict]]:
    """
    便捷的 JSON 对话封装。
    返回 (ChatResult, json_obj)。当 ok=False 时 json_obj 可能为 None。
    """
    resp = call_json(
        model=model,
        system_text=system,
        user_text=user,
        temperature=temperature,
        max_tokens=max_tokens,
        response_format="json_object",
        timeout=timeout_s,
        retries=retries,
        backoff=backoff,
    )
    ok = bool(resp.get("ok"))
    raw = resp.get("raw") or ""
    result = ChatResult(
        ok=ok,
        raw=raw,
        latency_ms=resp.get("latency_ms"),
        status=resp.get("status"),
        error=None if ok else (resp.get("fix_err") or raw or "unknown"),
    )
    return result, resp.get("json")


__all__ = ["ChatResult", "chat_json", "load_env_file"]
