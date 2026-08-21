#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import hashlib
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import requests
import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROFILE_PATHS = [
    ROOT / "configs" / "llm_profiles.yaml",
    ROOT / "configs" / "llm_profiles.yaml.example",
]

_ENV_LOADED = False
_PROFILE_CACHE: Dict[str, Any] | None = None


BUILTIN_PROFILE_DATA: Dict[str, Any] = {
    "version": "public-llm-runtime-v1",
    "defaults": {
        "provider": "openai_compatible",
        "temperature": 0.2,
        "max_tokens": 1500,
        "timeout_s": 60,
        "retries": 2,
        "backoff": 1.5,
        "json_mode": True,
    },
    "profiles": {
        "pipeline_primary": {
            "provider": "openai_compatible",
            "model_env": "PRIMARY_LLM_MODEL",
            "model": "qwen3-max",
            "base_url_env": "PRIMARY_LLM_BASE_URL",
            "api_key_env": "PRIMARY_LLM_API_KEY",
            "timeout_s": 120,
            "max_tokens": 3000,
        },
        "pipeline_secondary": {
            "provider": "openai_compatible",
            "model_env": "SECONDARY_LLM_MODEL",
            "model": "deepseek-r1-0528",
            "base_url_env": "SECONDARY_LLM_BASE_URL",
            "api_key_env": "SECONDARY_LLM_API_KEY",
            "timeout_s": 120,
            "max_tokens": 3000,
        },
        "judge_A_kimi": {
            "provider": "openai_compatible",
            "model_env": "A_KIMI_MODEL",
            "model": "kimi-k2.5",
            "base_url_env": "A_KIMI_BASE_URL",
            "api_key_env": "A_KIMI_API_KEY",
            "temperature": 0.0,
            "max_tokens": 1024,
        },
        "judge_B_claude": {
            "provider": "openai_compatible",
            "model_env": "B_CLAUDE_MODEL",
            "model": "claude-opus-4-6",
            "base_url_env": "B_CLAUDE_BASE_URL",
            "api_key_env": "B_CLAUDE_API_KEY",
            "temperature": 0.0,
            "max_tokens": 1024,
        },
        "judge_C_gemini": {
            "provider": "openai_compatible",
            "model_env": "C_GEMINI_MODEL",
            "model": "gemini-pro",
            "base_url_env": "C_GEMINI_BASE_URL",
            "api_key_env": "C_GEMINI_API_KEY",
            "temperature": 0.0,
            "max_tokens": 16384,
            "timeout_s": 90,
        },
        "embedding_default": {
            "provider": "openai_compatible_embedding",
            "model_env": "EMBED_MODEL_NAME",
            "model": "text-embedding-v4",
            "base_url_env": "EMBED_MODEL_BASE_URL",
            "api_key_env": "EMBED_MODEL_API_KEY",
            "timeout_s": 60,
            "retries": 3,
        },
        "rerank_default": {
            "provider": "generic_rerank",
            "model_env": "RERANK_MODEL_NAME",
            "model": "gte-rerank-v2",
            "endpoint_env": "RERANK_API_ENDPOINT",
            "base_url_env": "RERANK_BASE_URL",
            "api_key_env": "RERANK_API_KEY",
            "timeout_s": 60,
            "retries": 3,
        },
    },
}


@dataclass
class ResolvedProfile:
    name: str
    provider: str
    model: str
    api_key: str
    base_url: str = ""
    endpoint: str = ""
    temperature: float = 0.2
    max_tokens: int = 1500
    timeout_s: float = 60.0
    retries: int = 2
    backoff: float = 1.5
    json_mode: bool = True
    raw: Dict[str, Any] | None = None


def load_env_file(path: str | Path, *, required: bool = False) -> None:
    p = Path(path)
    if not p.exists():
        if required:
            raise FileNotFoundError(f"Env file not found: {p}")
        return
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key.startswith("export "):
            key = key[len("export "):].strip()
        if key and key not in os.environ:
            os.environ[key] = val


def load_default_env_files_once() -> None:
    global _ENV_LOADED
    if _ENV_LOADED:
        return
    for candidate in [
        ROOT / "configs" / ".env",
        ROOT / "evaluation" / ".env",
        ROOT / ".env",
        Path.cwd() / ".env",
    ]:
        load_env_file(candidate, required=False)
    _ENV_LOADED = True


def _read_profile_data() -> Dict[str, Any]:
    global _PROFILE_CACHE
    if _PROFILE_CACHE is not None:
        return _PROFILE_CACHE
    path_from_env = os.getenv("LLM_PROFILES_CONFIG")
    paths = [Path(path_from_env)] if path_from_env else []
    paths.extend(DEFAULT_PROFILE_PATHS)
    for path in paths:
        if path.exists():
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            _PROFILE_CACHE = data
            return data
    _PROFILE_CACHE = BUILTIN_PROFILE_DATA
    return BUILTIN_PROFILE_DATA


def load_profiles() -> Dict[str, Dict[str, Any]]:
    data = _read_profile_data()
    return dict(data.get("profiles") or {})


def _profile_defaults() -> Dict[str, Any]:
    data = _read_profile_data()
    defaults = dict(BUILTIN_PROFILE_DATA.get("defaults") or {})
    defaults.update(data.get("defaults") or {})
    return defaults


def judge_profile_name(judge_key: str) -> str:
    profiles = load_profiles()
    if judge_key in profiles:
        return judge_key
    prefixed = f"judge_{judge_key}"
    if prefixed in profiles:
        return prefixed
    return judge_key


def available_judges() -> List[str]:
    out: List[str] = []
    for name in load_profiles():
        if name.startswith("judge_"):
            out.append(name[len("judge_"):])
    return sorted(out)


def _env_value(name: str | None) -> str:
    return os.getenv(name or "", "")


def resolve_profile(
    profile: str,
    *,
    model_override: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    timeout_s: float | None = None,
    retries: int | None = None,
    backoff: float | None = None,
) -> ResolvedProfile:
    load_default_env_files_once()
    profiles = load_profiles()
    defaults = _profile_defaults()
    if profile not in profiles:
        raise KeyError(f"LLM/service profile not found: {profile}")
    raw = dict(defaults)
    raw.update(profiles[profile] or {})
    provider = str(raw.get("provider") or "openai_compatible")
    model = model_override or _env_value(raw.get("model_env")) or str(raw.get("model") or "")
    base_url = _env_value(raw.get("base_url_env")) or str(raw.get("base_url") or "")
    endpoint = _env_value(raw.get("endpoint_env")) or str(raw.get("endpoint") or "")
    api_key = _env_value(raw.get("api_key_env")) or str(raw.get("api_key") or "")
    if not base_url and provider in {"openai_compatible", "openai_compatible_embedding"}:
        base_url = "https://api.openai.com/v1"
    return ResolvedProfile(
        name=profile,
        provider=provider,
        model=model,
        api_key=api_key,
        base_url=base_url,
        endpoint=endpoint,
        temperature=float(temperature if temperature is not None else raw.get("temperature", 0.2)),
        max_tokens=int(max_tokens if max_tokens is not None else raw.get("max_tokens", 1500)),
        timeout_s=float(timeout_s if timeout_s is not None else raw.get("timeout_s", 60)),
        retries=int(retries if retries is not None else raw.get("retries", 2)),
        backoff=float(backoff if backoff is not None else raw.get("backoff", 1.5)),
        json_mode=bool(raw.get("json_mode", True)),
        raw=raw,
    )


def profiles_from_config(lcfg: Dict[str, Any] | None) -> Tuple[str, str]:
    lcfg = lcfg or {}
    primary = str(
        lcfg.get("primary_profile")
        or os.getenv("PRIMARY_LLM_PROFILE")
        or "pipeline_primary"
    )
    secondary = str(
        lcfg.get("secondary_profile")
        or os.getenv("SECONDARY_LLM_PROFILE")
        or "pipeline_secondary"
    )
    return primary, secondary


def _now_ms() -> int:
    return int(time.time() * 1000)


# C13RF27: transport-failure backoff.
#
# The pre-RF27 delay was ``backoff ** (attempt - 1)`` for every retry class.
# With the shipped default ``backoff=1.5`` that is 1.0s / 1.5s / 2.25s, i.e.
# ~4.75s of spacing in total.  For a *content* failure (the model returned
# unparseable JSON) that is fine: the request itself came back promptly, and
# retrying immediately is what we want.  For a *transport* failure it is not:
# C13RF26's 14d run lost its only call touching the duplicate parent/child pair
# because all four attempts were read timeouts of 120.1s each, so the whole
# logical call was 4x120s of waiting plus 4.75s of "backoff" -- 1% of the
# window.  Four attempts against one ~8-minute network stall is effectively one
# attempt, and the model therefore never got to propose the merge that would
# have cleared the E0 blocker.
#
# So transport failures get their own schedule: a real floor, exponential
# growth, a cap so a single logical call cannot run away with the wall clock,
# and jitter so parallel/consecutive calls do not re-synchronise onto the same
# stall.  The attempts budget itself is untouched (still ``retries + 1``, still
# every attempt counted) -- this changes *when* attempts happen, never how many.
TRANSPORT_BACKOFF_BASE_S = 5.0
TRANSPORT_BACKOFF_FACTOR = 3.0
TRANSPORT_BACKOFF_CAP_S = 60.0
TRANSPORT_BACKOFF_JITTER = 0.25


def transport_retry_delay_seconds(
    attempt: int,
    *,
    jitter_source: Callable[[], float] | None = None,
) -> float:
    """Seconds to wait before retrying after a *transport* failure.

    ``attempt`` is 1-based and names the attempt that just failed.  The
    schedule is ``base * factor ** (attempt - 1)``, clamped to
    ``TRANSPORT_BACKOFF_CAP_S``, then spread by up to
    ``TRANSPORT_BACKOFF_JITTER`` in *both* directions.  With the constants
    above the nominal spacing for a 4-attempt call is 5s / 15s / 45s, so the
    retries straddle roughly a minute of network trouble instead of a
    rounding error of it.

    Jitter is symmetric so the expected delay still equals the nominal one;
    the floor keeps a pathological ``random()`` from collapsing the wait.
    """
    if attempt < 1:
        raise ValueError("attempt must be >= 1")
    nominal = TRANSPORT_BACKOFF_BASE_S * (TRANSPORT_BACKOFF_FACTOR ** (attempt - 1))
    nominal = min(nominal, TRANSPORT_BACKOFF_CAP_S)
    source = jitter_source if jitter_source is not None else random.random
    # source() in [0, 1) -> spread in [-JITTER, +JITTER)
    spread = 1.0 + TRANSPORT_BACKOFF_JITTER * (2.0 * float(source()) - 1.0)
    return max(TRANSPORT_BACKOFF_BASE_S / 2.0, nominal * spread)


def _prompt_hash(system: str, user: str) -> str:
    return hashlib.md5((system + "\n" + user).encode("utf-8")).hexdigest()[:10]


def append_jsonl(path: str | Path, obj: Dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _extract_balanced_json(text: str) -> Optional[str]:
    n = len(text)
    i = 0
    while i < n:
        if text[i] == "{":
            depth = 0
            in_str = False
            escape = False
            for j in range(i, n):
                ch = text[j]
                if in_str:
                    if escape:
                        escape = False
                    elif ch == "\\":
                        escape = True
                    elif ch == '"':
                        in_str = False
                else:
                    if ch == '"':
                        in_str = True
                    elif ch == "{":
                        depth += 1
                    elif ch == "}":
                        depth -= 1
                        if depth == 0:
                            return text[i:j + 1]
            break
        i += 1
    return None


def _try_fix_unescaped_inner_quotes(text: str) -> str:
    out: List[str] = []
    n = len(text)
    in_string = False
    i = 0
    while i < n:
        ch = text[i]
        if not in_string:
            out.append(ch)
            if ch == '"':
                in_string = True
            i += 1
            continue
        if ch == "\\":
            out.append(ch)
            if i + 1 < n:
                out.append(text[i + 1])
                i += 2
            else:
                i += 1
            continue
        if ch == '"':
            j = i + 1
            while j < n and text[j] in " \t\r\n":
                j += 1
            next_meaningful = text[j] if j < n else ""
            if next_meaningful in (",", ":", "}", "]", "") or j >= n:
                out.append(ch)
                in_string = False
            else:
                out.extend(['\\', '"'])
            i += 1
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def parse_json_safe(text: str) -> Tuple[bool, Optional[dict], Optional[str]]:
    t = (text or "").strip()
    if not t:
        return False, None, "empty response text"
    if t.startswith("```"):
        first_nl = t.find("\n")
        if first_nl != -1:
            t = t[first_nl + 1:]
        if t.endswith("```"):
            t = t[:-3]
        t = t.strip()
    candidates = [t]
    balanced = _extract_balanced_json(t)
    if balanced:
        candidates.append(balanced)
    fixed = _try_fix_unescaped_inner_quotes(t)
    if fixed != t:
        candidates.append(fixed)
        fixed_balanced = _extract_balanced_json(fixed)
        if fixed_balanced:
            candidates.append(fixed_balanced)
    if "{" in t and "}" in t:
        i = t.find("{")
        j = t.rfind("}")
        if j > i:
            candidates.append(t[i:j + 1])
    last_err = ""
    for candidate in candidates:
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return True, obj, None
            return False, None, f"JSON root is {type(obj).__name__}, expected object"
        except Exception as exc:
            last_err = str(exc)
    return False, None, last_err or "unable to parse JSON object"


def _result_dict(
    *,
    ok: bool,
    profile: ResolvedProfile,
    raw: str = "",
    json_obj: Optional[dict] = None,
    error: Optional[str] = None,
    status: Optional[int] = None,
    latency_ms: Optional[int] = None,
    attempts: int = 0,
    attempt_history: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    return {
        "ok": bool(ok),
        "json": json_obj,
        "raw": raw,
        "error": error,
        "status": status,
        "latency_ms": latency_ms,
        "profile": profile.name,
        "provider": profile.provider,
        "model": profile.model,
        "attempts": int(attempts),
        "attempt_history": list(attempt_history or []),
    }


def call_llm_json(
    *,
    profile: str,
    system: str,
    user: str,
    task: str,
    model_override: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    timeout_s: float | None = None,
    retries: int | None = None,
    backoff: float | None = None,
    log_path: str | Path | None = None,
) -> Dict[str, Any]:
    cfg = resolve_profile(
        profile,
        model_override=model_override,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout_s=timeout_s,
        retries=retries,
        backoff=backoff,
    )
    if cfg.provider != "openai_compatible":
        return _result_dict(ok=False, profile=cfg, error=f"unsupported chat provider: {cfg.provider}")
    if not cfg.model:
        return _result_dict(ok=False, profile=cfg, error=f"profile {profile} has no model")
    if not cfg.api_key:
        return _result_dict(ok=False, profile=cfg, error=f"profile {profile} has no API key")

    url = cfg.base_url.rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {cfg.api_key}", "Content-Type": "application/json"}
    payload: Dict[str, Any] = {
        "model": cfg.model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": cfg.temperature,
        "max_tokens": cfg.max_tokens,
        "stream": False,
    }
    if cfg.json_mode:
        payload["response_format"] = {"type": "json_object"}

    last_error = ""
    last_status: Optional[int] = None
    last_latency: Optional[int] = None
    attempt_history: List[Dict[str, Any]] = []
    attempts = max(1, cfg.retries + 1)
    # C13RF27: two retry classes with deliberately different schedules.
    #   content failures (HTTP 200 arrived, body unparseable) keep the original
    #     `cfg.backoff ** (attempt - 1)` schedule -- the far side answered
    #     promptly, so there is nothing to wait out and retrying fast is right;
    #   transport failures (retryable status / network exception) use
    #     transport_retry_delay_seconds -- see the loop tail.
    # Do not collapse these onto one schedule: making content retries slow buys
    # nothing, and making transport retries fast is the RF26 failure mode.
    for attempt in range(1, attempts + 1):
        started = _now_ms()
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=cfg.timeout_s)
            last_latency = _now_ms() - started
            last_status = resp.status_code
            if resp.status_code == 200:
                try:
                    data = resp.json()
                except Exception as exc:
                    body = resp.text[:500]
                    last_error = f"{type(exc).__name__}: {exc}"
                    attempt_history.append({
                        "attempt": attempt,
                        "status": 200,
                        "latency_ms": last_latency,
                        "raw": body,
                        "parse_ok": False,
                        "error": last_error,
                    })
                    if attempt < attempts:
                        time.sleep(cfg.backoff ** (attempt - 1))
                        continue
                    break
                try:
                    if not isinstance(data, dict):
                        raise TypeError("HTTP 200 response JSON must be an object")
                    choices = data.get("choices") or [{}]
                    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
                        raise TypeError("HTTP 200 response choices must be a non-empty object list")
                    message = choices[0].get("message", {})
                    if not isinstance(message, dict):
                        raise TypeError("HTTP 200 response message must be an object")
                    content = message.get("content", "")
                    if not isinstance(content, str):
                        raise TypeError("HTTP 200 response content must be a string")
                except Exception as exc:
                    body = resp.text[:500]
                    last_error = f"{type(exc).__name__}: {exc}"
                    attempt_history.append({
                        "attempt": attempt,
                        "status": 200,
                        "latency_ms": last_latency,
                        "raw": body,
                        "parse_ok": False,
                        "error": last_error,
                    })
                    if attempt < attempts:
                        time.sleep(cfg.backoff ** (attempt - 1))
                        continue
                    break
                ok, obj, parse_err = parse_json_safe(content)
                attempt_history.append({
                    "attempt": attempt,
                    "status": 200,
                    "latency_ms": last_latency,
                    "raw": content,
                    "parse_ok": bool(ok),
                    "error": parse_err,
                })
                if not ok:
                    last_error = parse_err or "unable to parse JSON object"
                    if attempt < attempts:
                        time.sleep(cfg.backoff ** (attempt - 1))
                        continue
                result = _result_dict(
                    ok=ok,
                    profile=cfg,
                    raw=content,
                    json_obj=obj,
                    error=parse_err,
                    status=200,
                    latency_ms=last_latency,
                    attempts=attempt,
                    attempt_history=attempt_history,
                )
                if log_path:
                    append_jsonl(log_path, {"ts": int(time.time()), "task": task, "result": result})
                return result
            body = resp.text[:500]
            last_error = f"HTTP {resp.status_code}: {body}"
            attempt_history.append({
                "attempt": attempt,
                "status": resp.status_code,
                "latency_ms": last_latency,
                "raw": body,
                "parse_ok": False,
                "error": last_error,
            })
            if resp.status_code not in (408, 409, 425, 429, 500, 502, 503, 504):
                break
        except Exception as exc:
            last_latency = _now_ms() - started
            last_status = None
            last_error = f"{type(exc).__name__}: {exc}"
            attempt_history.append({
                "attempt": attempt,
                "status": None,
                "latency_ms": last_latency,
                "raw": "",
                "parse_ok": False,
                "error": last_error,
            })
        if attempt < attempts:
            # C13RF27: the transport tail -- reached by a retryable HTTP status
            # (408/409/425/429/5xx, which fell through the break above) or by a
            # network-level exception (ReadTimeout, connection reset, ...).
            # Both mean "the far side did not answer", so space the retries out
            # instead of re-hammering the same stall.  RF26's 14d loss was
            # exactly this: 4 read timeouts of 120.1s with 4.75s of spacing
            # between them.  See transport_retry_delay_seconds.
            time.sleep(transport_retry_delay_seconds(attempt))

    actual_attempts = len(attempt_history)
    result = _result_dict(
        ok=False,
        profile=cfg,
        raw=last_error,
        error=last_error or "request failed",
        status=last_status,
        latency_ms=last_latency,
        attempts=actual_attempts,
        attempt_history=attempt_history,
    )
    if log_path:
        append_jsonl(log_path, {"ts": int(time.time()), "task": task, "result": result})
    return result


def adjudicate_llm_json(
    *,
    primary_profile: str,
    secondary_profile: str,
    system: str,
    user: str,
    task: str,
    temperature: float | None = None,
    max_tokens: int | None = None,
    log_path: str | Path | None = None,
    tie_support_margin: float = 0.10,
    evidence_score_primary: float | None = None,
    evidence_score_secondary: float | None = None,
) -> Dict[str, Any]:
    primary = call_llm_json(
        profile=primary_profile,
        system=system,
        user=user,
        task=f"{task}:primary",
        temperature=temperature,
        max_tokens=max_tokens,
    )
    secondary = call_llm_json(
        profile=secondary_profile,
        system=system,
        user=user,
        task=f"{task}:secondary",
        temperature=temperature,
        max_tokens=max_tokens,
    )
    pobj = primary.get("json") if primary.get("ok") else None
    sobj = secondary.get("json") if secondary.get("ok") else None
    if pobj is not None and sobj is not None:
        if _norm(pobj) == _norm(sobj):
            decision = "agree"
            final = pobj
        elif (
            evidence_score_primary is not None
            and evidence_score_secondary is not None
            and (evidence_score_primary - evidence_score_secondary) >= tie_support_margin
        ):
            decision = "primary"
            final = pobj
        else:
            decision = "conservative"
            final = {"needs_review": True}
    elif pobj is not None:
        decision = "primary_only"
        final = pobj
    elif sobj is not None:
        decision = "secondary_only"
        final = sobj
    else:
        decision = "both_failed"
        final = {"needs_review": True}
    out = {"final": final, "primary": primary, "secondary": secondary, "decision": decision}
    if log_path:
        append_jsonl(
            log_path,
            {
                "ts": int(time.time()),
                "task": task,
                "prompt_hash": _prompt_hash(system, user),
                "primary_profile": primary_profile,
                "secondary_profile": secondary_profile,
                **out,
            },
        )
    return out


def _norm(obj: Any) -> str:
    try:
        return json.dumps(obj, sort_keys=True, ensure_ascii=False)
    except Exception:
        return str(obj)


def call_embedding(
    *,
    profile: str = "embedding_default",
    texts: List[str],
    model_override: str | None = None,
    timeout_s: float | None = None,
    retries: int | None = None,
) -> Dict[str, Any]:
    cfg = resolve_profile(profile, model_override=model_override, timeout_s=timeout_s, retries=retries)
    if cfg.provider != "openai_compatible_embedding":
        return {"ok": False, "vectors": [], "error": f"unsupported embedding provider: {cfg.provider}"}
    if not cfg.api_key:
        return {"ok": False, "vectors": [], "error": f"profile {profile} has no API key"}
    base = cfg.base_url.rstrip("/")
    url = f"{base}/embeddings" if base.endswith("/v1") else f"{base}/v1/embeddings"
    headers = {"Authorization": f"Bearer {cfg.api_key}", "Content-Type": "application/json"}
    payload = {"model": cfg.model, "input": texts}
    last_error = ""
    attempts = max(1, cfg.retries + 1)
    for attempt in range(1, attempts + 1):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=cfg.timeout_s)
            if resp.status_code == 200:
                data = resp.json()
                items = data.get("data", [])
                if not items and isinstance(data.get("output"), dict):
                    items = data["output"].get("embeddings", data["output"].get("data", [])) or []
                vectors = []
                for item in items:
                    if isinstance(item, dict) and "embedding" in item:
                        vectors.append(item["embedding"])
                    else:
                        vectors.append(item)
                return {"ok": True, "vectors": vectors, "raw": data, "attempts": attempt}
            last_error = f"HTTP {resp.status_code}: {resp.text[:500]}"
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        if attempt < attempts:
            # C13RF27: transport-only loop (any non-200 or exception lands here).
            time.sleep(transport_retry_delay_seconds(attempt))
    return {"ok": False, "vectors": [], "error": last_error or "embedding request failed", "attempts": attempts}


def call_rerank(
    *,
    profile: str = "rerank_default",
    query: str,
    documents: List[str],
    model_override: str | None = None,
    endpoint_override: str | None = None,
    timeout_s: float | None = None,
    retries: int | None = None,
) -> Dict[str, Any]:
    cfg = resolve_profile(profile, model_override=model_override, timeout_s=timeout_s, retries=retries)
    endpoint = endpoint_override or cfg.endpoint
    if not endpoint and cfg.base_url:
        endpoint = cfg.base_url.rstrip("/") + "/api/v1/services/rerank/text-rerank/text-rerank"
    if not endpoint:
        return {"ok": False, "scores": [], "error": f"profile {profile} has no rerank endpoint"}
    if not cfg.api_key:
        return {"ok": False, "scores": [], "error": f"profile {profile} has no API key"}
    headers = {"Authorization": f"Bearer {cfg.api_key}", "Content-Type": "application/json"}
    is_dashscope = "dashscope.aliyuncs.com" in endpoint or cfg.provider == "dashscope_rerank"
    last_error = ""
    attempts = max(1, cfg.retries + 1)
    for attempt in range(1, attempts + 1):
        try:
            if is_dashscope:
                docs = [str(d)[:1000] for d in documents]
                body = {
                    "model": cfg.model,
                    "input": {"query": str(query or "")[:1000], "documents": docs},
                    "parameters": {"top_n": max(1, min(len(docs), 20)), "return_documents": False},
                }
                resp = requests.post(endpoint, headers=headers, json=body, timeout=cfg.timeout_s)
                resp.raise_for_status()
                items = resp.json().get("output", {}).get("results", [])
                scores = [0.0] * len(docs)
                for item in items:
                    idx = item.get("index")
                    sc = item.get("relevance_score", item.get("score", 0.0))
                    if isinstance(idx, int) and 0 <= idx < len(scores):
                        scores[idx] = float(sc)
                return {"ok": True, "scores": scores, "attempts": attempt}
            body = {"model": cfg.model, "query": query, "documents": documents, "top_n": len(documents)}
            resp = requests.post(endpoint, headers=headers, json=body, timeout=cfg.timeout_s)
            resp.raise_for_status()
            data = resp.json()
            items = data.get("results", []) or data.get("data", [])
            scores = [0.0] * len(documents)
            has_index = any(isinstance(item, dict) and "index" in item for item in items)
            if has_index:
                for item in items:
                    if not isinstance(item, dict):
                        continue
                    idx = item.get("index")
                    sc = item.get("relevance_score", item.get("score", 0.0))
                    if isinstance(idx, int) and 0 <= idx < len(scores):
                        scores[idx] = float(sc)
            else:
                for pos, item in enumerate(items[: len(scores)]):
                    if isinstance(item, dict):
                        sc = item.get("relevance_score", item.get("score", 0.0))
                    elif isinstance(item, (list, tuple)) and len(item) >= 2:
                        sc = item[1]
                    else:
                        sc = float(item) if isinstance(item, (int, float)) else 0.0
                    scores[pos] = float(sc)
            return {"ok": True, "scores": scores, "attempts": attempt}
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        if attempt < attempts:
            # C13RF27: transport-only loop (any non-200 or exception lands here).
            time.sleep(transport_retry_delay_seconds(attempt))
    return {"ok": False, "scores": [], "error": last_error or "rerank request failed", "attempts": attempts}


__all__ = [
    "ResolvedProfile",
    "adjudicate_llm_json",
    "append_jsonl",
    "available_judges",
    "call_embedding",
    "call_llm_json",
    "call_rerank",
    "judge_profile_name",
    "load_default_env_files_once",
    "load_env_file",
    "load_profiles",
    "parse_json_safe",
    "profiles_from_config",
    "resolve_profile",
    "transport_retry_delay_seconds",
]
