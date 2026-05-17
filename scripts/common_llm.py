#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import requests

_ENV_LOADED = False


def _load_env_file_once():
    global _ENV_LOADED
    if _ENV_LOADED:
        return
    # 简易 .env 加载器（不覆盖已存在变量）
    for candidate in ["configs/roundC.env", "roundC.env", ".env"]:
        p = Path(candidate)
        if p.exists():
            try:
                for line in p.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if "=" in line:
                        k, v = line.split("=", 1)
                        k = k.strip(); v = v.strip()
                        if k and (os.getenv(k) is None):
                            os.environ[k] = v
                break
            except Exception:
                pass
    _ENV_LOADED = True


def _now_ms() -> int:
    return int(time.time() * 1000)


def _prompt_hash(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()[:10]


@dataclass
class LLMConfig:
    primary: str
    secondary: str
    temperature: float = 0.2
    max_tokens: int = 1500
    response_format: str = "json_object"
    workers: int = 1
    tie_breaker: str = "score_margin_or_conservative"


def _default_base_for_model(model: str) -> str:
    m = (model or "").lower()
    # 兼容 OPENAI_BASE_URL 与 OPENAI_BASE
    openai_base = os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_BASE") or "https://api.openai.com/v1"
    if "qwen" in m:
        # Qwen 兼容OpenAI接口，但优先QWEN_BASE；否则使用 OPENAI_BASE_URL（如配置了DashScope兼容端点）
        return os.getenv("QWEN_BASE") or openai_base or "https://dashscope.aliyuncs.com/compatible-mode/v1"
    if "deepseek" in m:
        # DeepSeek 使用其官方端点，除非显式设置 DEEPSEEK_BASE
        return os.getenv("DEEPSEEK_BASE") or "https://api.deepseek.com/v1"
    return openai_base


def _key_for_model(model: str) -> str:
    m = (model or "").lower()
    # 允许通用 OPENAI_API_KEY 作为回退
    if "qwen" in m:
        return os.getenv("QWEN_API_KEY") or os.getenv("OPENAI_API_KEY", "")
    if "deepseek" in m:
        return os.getenv("DEEPSEEK_API_KEY", "")
    return os.getenv("OPENAI_API_KEY", "")


def call_json(model: str, system_text: str, user_text: str,
              temperature: float = 0.2, max_tokens: int = 1500,
              response_format: str = "json_object",
              timeout: float = 60.0, retries: int = 2, backoff: float = 1.5) -> Dict[str, Any]:
    _load_env_file_once()
    base = _default_base_for_model(model)
    key = _key_for_model(model)
    url = base.rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_text},
            {"role": "user", "content": user_text},
        ],
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
        "stream": False,
    }
    if response_format == "json_object":
        payload["response_format"] = {"type": "json_object"}

    last_err = None
    for attempt in range(retries + 1):
        t0 = _now_ms()
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
            lat = _now_ms() - t0
            if resp.status_code == 200:
                data = resp.json()
                content = (data.get("choices") or [{}])[0].get("message", {}).get("content", "")
                ok, obj, fix_err = _parse_json_strict(content)
                return {"ok": ok, "json": obj, "raw": content, "latency_ms": lat, "status": 200,
                        "fix_err": (None if ok else str(fix_err))}
            elif resp.status_code in (429, 500, 502, 503, 504):
                last_err = f"HTTP {resp.status_code} {resp.text[:200]}"
            else:
                return {"ok": False, "json": None, "raw": resp.text, "latency_ms": lat, "status": resp.status_code}
        except Exception as e:
            last_err = str(e)
        time.sleep(backoff ** attempt)
    return {"ok": False, "json": None, "raw": str(last_err), "latency_ms": None, "status": None}


def _parse_json_strict(s: str) -> Tuple[bool, Optional[dict], Optional[Exception]]:
    try:
        return True, json.loads(s), None
    except Exception as e1:
        # 尝试从最外层花括号截取
        try:
            l = s.find("{"); r = s.rfind("}")
            if l >= 0 and r > l:
                return True, json.loads(s[l:r+1]), None
        except Exception as e2:
            return False, None, e2
        return False, None, e1


def write_jsonl(path: str | Path, obj: Dict[str, Any]):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def adjudicate(cfg: LLMConfig, prompt_sys: str, prompt_user: str,
               input_meta: Dict[str, Any] | None, log_path: str | None,
               tie_support_margin: float = 0.10,
               evidence_score_primary: float | None = None,
               evidence_score_secondary: float | None = None) -> Dict[str, Any]:
    _load_env_file_once()
    """双模型仲裁：一致采纳；不一致则看结构证据差距≥margin是否支持primary，否则保守。
    evidence_score_*：可传候选得分或 p75 差距等结构证据（数值越大越支持）。
    返回：{"final": obj, "primary": obj1, "secondary": obj2, "decision": "primary|secondary|conservative"}
    """
    p = call_json(cfg.primary, prompt_sys, prompt_user, cfg.temperature, cfg.max_tokens, cfg.response_format)
    s = call_json(cfg.secondary, prompt_sys, prompt_user, cfg.temperature, cfg.max_tokens, cfg.response_format)

    # 日志
    if log_path:
        now = int(time.time())
        ph = _prompt_hash(prompt_sys + "\n" + prompt_user)
        write_jsonl(log_path, {
            "ts": now,
            "prompt_hash": ph,
            "input_meta": input_meta,
            "primary_model": cfg.primary,
            "primary": p,
            "secondary_model": cfg.secondary,
            "secondary": s,
        })

    pobj = p.get("json") if p.get("ok") else None
    sobj = s.get("json") if s.get("ok") else None
    if pobj is not None and sobj is not None:
        if _norm(pobj) == _norm(sobj):
            return {"final": pobj, "primary": pobj, "secondary": sobj, "decision": "agree"}
        # 不一致：看证据是否明显支持 primary
        if evidence_score_primary is not None and evidence_score_secondary is not None:
            if (evidence_score_primary - evidence_score_secondary) >= tie_support_margin:
                return {"final": pobj, "primary": pobj, "secondary": sobj, "decision": "primary"}
        return {"final": {"needs_review": True}, "primary": pobj, "secondary": sobj, "decision": "conservative"}
    # 单边成功
    if pobj is not None:
        return {"final": pobj, "primary": pobj, "secondary": sobj, "decision": "primary_only"}
    if sobj is not None:
        return {"final": sobj, "primary": pobj, "secondary": sobj, "decision": "secondary_only"}
    # 均失败
    return {"final": {"needs_review": True}, "primary": pobj, "secondary": sobj, "decision": "both_failed"}


def _norm(obj: Any) -> Any:
    try:
        return json.dumps(obj, sort_keys=True, ensure_ascii=False)
    except Exception:
        return str(obj)


__all__ = [
    "LLMConfig",
    "call_json",
    "adjudicate",
]
