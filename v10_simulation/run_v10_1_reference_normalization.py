#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import copy
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd
import requests

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


COMPONENT_KEYS = ("M", "A", "A_type", "O", "S_scope", "S_focus", "S_stage", "S_type")
SEMANTIC_A_TYPES = {"substantive", "operational"}
GUARD_TERMS = ("试点", "新开", "加密", "存量", "示范", "战略性")
STABLE_CHUNKS = ("空空中转", "一次安检", "稳定运行", "口岸一体化营运费用", "异地货站", "转运分拨中心")
PROTECTED_SUBSTANTIVE = {"建设", "改造", "管理"}
SETTLEMENT_TOOL_TERMS = ("落户奖励", "落户支持", "落户补贴")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class RunLogger:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, message: str) -> None:
        line = f"[{now_iso()}] {message}"
        print(line, flush=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")


def ensure_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def ensure_str_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return [ensure_text(item) for item in value if ensure_text(item)]
    text = ensure_text(value)
    return [text] if text else []


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, data: Dict[str, Any]) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def join_parts(parts: Iterable[str], sep: str = " ") -> str:
    return sep.join(part for part in (ensure_text(x) for x in parts) if part)


def build_s_text(components: Dict[str, Any]) -> str:
    return join_parts(
        ensure_str_list(components.get("S_scope"))
        + ensure_str_list(components.get("S_focus"))
        + ensure_str_list(components.get("S_stage"))
        + ensure_str_list(components.get("S_type"))
    )


def build_reference_text(node: Dict[str, Any]) -> str:
    lines = [
        f"node_id={node['node_id']}",
        f"track={node['track']}",
        f"tool_nature={node.get('tool_nature') or ''}",
        f"parent_title={node.get('parent_title') or ''}",
        f"pau_final={node['pau_final']}",
        f"O={node['O_text']}",
        f"S={node['S_text']}",
    ]
    if node["A_text"]:
        lines.append(f"A={node['A_text']}")
    return "\n".join(lines)


def coerce_components(candidate: Any, original: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(candidate, dict):
        candidate = {}
    result: Dict[str, Any] = {}
    for key in COMPONENT_KEYS:
        if key in {"M", "S_scope", "S_focus", "S_stage", "S_type"}:
            result[key] = ensure_str_list(candidate.get(key, original.get(key)))
        else:
            result[key] = ensure_text(candidate.get(key, original.get(key)))
    return result


def render_pau(components: Dict[str, Any], fallback: str) -> str:
    prefix = "".join(
        ensure_str_list(components.get("S_scope"))
        + ensure_str_list(components.get("S_focus"))
        + ensure_str_list(components.get("S_stage"))
        + ensure_str_list(components.get("S_type"))
    )
    body = ensure_text(components.get("O"))
    action = ensure_text(components.get("A"))
    action_type = ensure_text(components.get("A_type"))
    if action_type in SEMANTIC_A_TYPES and action:
        rendered = f"{prefix}{body}{action}"
    else:
        rendered = f"{prefix}{body}"
    rendered = ensure_text(rendered)
    return rendered or fallback


def clean_settlement_text(text: str) -> str:
    cleaned = ensure_text(text)
    for phrase in ("引进落户", "引入落户"):
        cleaned = cleaned.replace(phrase, "引进")
    cleaned = re.sub(r"落户(?!奖励|支持|补贴)", "", cleaned)
    cleaned = re.sub(r"\s+", "", cleaned)
    cleaned = cleaned.replace("引入", "引进")
    return cleaned


def detect_guard_tags(source_text: str, target_text: str) -> List[str]:
    tags: List[str] = []
    merged = f"{ensure_text(source_text)} {ensure_text(target_text)}"
    for term in GUARD_TERMS:
        if term in merged:
            tags.append(term)
    return tags


def fallback_result(node: Dict[str, Any], reason: str) -> Dict[str, Any]:
    return {
        "components_v10": copy.deepcopy(node["components"]),
        "pau_std_v10": node["pau_final"],
        "guard_tags_v10": detect_guard_tags(node["pau_final"], node["pau_final"]),
        "normalization_log_v10": ensure_text(reason) or "fallback to original",
        "consistency_review_flag_v10": True,
    }


def validate_and_repair_result(node: Dict[str, Any], raw_result: Dict[str, Any]) -> Dict[str, Any]:
    components_v10 = coerce_components(raw_result.get("components_v10"), node["components"])
    pau_std_v10 = ensure_text(raw_result.get("pau_std_v10"))
    if not pau_std_v10:
        pau_std_v10 = render_pau(components_v10, node["pau_final"])

    guard_tags = ensure_str_list(raw_result.get("guard_tags_v10"))
    if not guard_tags:
        guard_tags = detect_guard_tags(node["pau_final"], pau_std_v10)

    log = ensure_text(raw_result.get("normalization_log_v10"))
    review_flag = bool(raw_result.get("consistency_review_flag_v10", False))
    extra_logs: List[str] = []

    is_independent_settlement_tool = any(term in node["pau_final"] for term in SETTLEMENT_TOOL_TERMS)
    if (
        node.get("tool_nature") != "monetary"
        and ensure_text(node["components"].get("A")) in {"引进", "引入"}
        and "落户" in ensure_text(node["components"].get("O"))
        and not is_independent_settlement_tool
    ):
        components_v10["A"] = "引进"
        components_v10["O"] = clean_settlement_text(components_v10.get("O") or node["components"].get("O", ""))
        if not components_v10["O"]:
            components_v10["O"] = clean_settlement_text(ensure_text(node["components"].get("O")))
        pau_std_v10 = render_pau(components_v10, clean_settlement_text(node["pau_final"]))
        extra_logs.append("applied intro-settlement rule: keep 引进, drop 落户 from canonical output")

    source_text = f"{node['pau_final']} {ensure_text(node['components'].get('O'))}"
    for term in GUARD_TERMS + STABLE_CHUNKS:
        if term in source_text and term not in pau_std_v10:
            return fallback_result(node, f"protected token dropped: {term}")

    source_action = ensure_text(node["components"].get("A"))
    target_action = ensure_text(components_v10.get("A"))
    if source_action in PROTECTED_SUBSTANTIVE and target_action in PROTECTED_SUBSTANTIVE and source_action != target_action:
        return fallback_result(node, f"protected substantive drift: {source_action} -> {target_action}")

    if node.get("tool_nature") == "monetary" and node.get("track") == "track_c":
        if not any(term in pau_std_v10 for term in ("奖励", "补贴", "减免", "补助", "支持")):
            return fallback_result(node, "monetary boundary unclear in normalized output")

    if node.get("tool_nature") == "non_monetary" and node.get("track") == "track_c":
        if any(term in pau_std_v10 for term in ("奖励", "补贴", "减免", "补助")) and not any(
            term in node["pau_final"] for term in ("奖励", "补贴", "减免", "补助")
        ):
            return fallback_result(node, "non_monetary node drifted into monetary wording")

    if extra_logs:
        log = " | ".join([part for part in [log] + extra_logs if part])
        review_flag = True

    return {
        "components_v10": components_v10,
        "pau_std_v10": pau_std_v10,
        "guard_tags_v10": guard_tags,
        "normalization_log_v10": log,
        "consistency_review_flag_v10": review_flag,
    }


def flatten_nodes(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    nodes: List[Dict[str, Any]] = []
    counters = {"TA": 0, "TB": 0, "TC": 0}

    for idx, item in enumerate(data.get("track_a", [])):
        if not item.get("components") or not item.get("pau_final"):
            continue
        counters["TA"] += 1
        components = copy.deepcopy(item["components"])
        node = {
            "node_id": f"TA_{counters['TA']:03d}",
            "track": "track_a",
            "doc_id": item.get("doc_id"),
            "semantic_block_id": item.get("semantic_block_id"),
            "parent_title": ensure_text(item.get("block_text")),
            "section_type": "",
            "leaf_index": None,
            "tool_nature": "",
            "source_path_v9": f"track_a[{idx}]",
            "source_index": idx,
            "group_index": None,
            "components": components,
            "pau_final": ensure_text(item.get("pau_final")),
            "leaf_name": ensure_text(item.get("block_text")),
        }
        node["O_text"] = ensure_text(components.get("O"))
        node["S_text"] = build_s_text(components)
        node["A_text"] = ensure_text(components.get("A")) if ensure_text(components.get("A_type")) in SEMANTIC_A_TYPES else ""
        node["reference_text_v10_internal"] = build_reference_text(node)
        nodes.append(node)

    for group_idx, group in enumerate(data.get("track_b", [])):
        for leaf_idx, leaf in enumerate(group.get("leaves", [])):
            if not leaf.get("components") or not leaf.get("pau_final"):
                continue
            counters["TB"] += 1
            components = copy.deepcopy(leaf["components"])
            node = {
                "node_id": f"TB_{counters['TB']:03d}",
                "track": "track_b",
                "doc_id": group.get("doc_id"),
                "semantic_block_id": group.get("semantic_block_id"),
                "parent_title": ensure_text(group.get("parent_title")),
                "section_type": "",
                "leaf_index": leaf_idx,
                "tool_nature": "",
                "source_path_v9": f"track_b[{group_idx}].leaves[{leaf_idx}]",
                "source_index": group_idx,
                "group_index": group_idx,
                "components": components,
                "pau_final": ensure_text(leaf.get("pau_final")),
                "leaf_name": ensure_text(leaf.get("leaf_name")),
            }
            node["O_text"] = ensure_text(components.get("O"))
            node["S_text"] = build_s_text(components)
            node["A_text"] = ensure_text(components.get("A")) if ensure_text(components.get("A_type")) in SEMANTIC_A_TYPES else ""
            node["reference_text_v10_internal"] = build_reference_text(node)
            nodes.append(node)

    for block_idx, block in enumerate(data.get("track_c", [])):
        for leaf_idx, leaf in enumerate(block.get("leaves", [])):
            if not leaf.get("components") or not leaf.get("pau_final"):
                continue
            counters["TC"] += 1
            components = copy.deepcopy(leaf["components"])
            node = {
                "node_id": f"TC_{counters['TC']:03d}",
                "track": "track_c",
                "doc_id": block.get("doc_id"),
                "semantic_block_id": block.get("semantic_block_id"),
                "parent_title": ensure_text(block.get("section_type")),
                "section_type": ensure_text(block.get("section_type")),
                "leaf_index": leaf_idx,
                "tool_nature": ensure_text(leaf.get("tool_nature")),
                "source_path_v9": f"track_c[{block_idx}].leaves[{leaf_idx}]",
                "source_index": block_idx,
                "group_index": block_idx,
                "components": components,
                "pau_final": ensure_text(leaf.get("pau_final")),
                "leaf_name": ensure_text(leaf.get("leaf_name")),
            }
            node["O_text"] = ensure_text(components.get("O"))
            node["S_text"] = build_s_text(components)
            node["A_text"] = ensure_text(components.get("A")) if ensure_text(components.get("A_type")) in SEMANTIC_A_TYPES else ""
            node["reference_text_v10_internal"] = build_reference_text(node)
            nodes.append(node)

    return nodes


def resolve_model_name(cli_value: str | None, env_key: str) -> str:
    model = ensure_text(cli_value) or ensure_text(os.getenv(env_key))
    if not model:
        raise ValueError(f"Missing model name: CLI override empty and env {env_key} not set")
    return model


def resolve_runtime_config(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "llm_model": resolve_model_name(args.llm_model, args.llm_model_env_key),
        "embed_model": resolve_model_name(args.embed_model, "EMBED_MODEL_NAME"),
        "rerank_model": resolve_model_name(args.rerank_model, "RERANK_MODEL_NAME"),
        "embed_base_url": ensure_text(os.getenv("EMBED_MODEL_BASE_URL") or os.getenv("OPENAI_BASE_URL")),
        "embed_api_key": ensure_text(os.getenv("EMBED_MODEL_API_KEY") or os.getenv("OPENAI_API_KEY")),
        "rerank_endpoint": ensure_text(os.getenv("RERANK_API_ENDPOINT") or os.getenv("RERANK_API_URL")),
        "rerank_api_key": ensure_text(os.getenv("RERANK_API_KEY")),
        "llm_temperature": float(os.getenv("LLM_TEMPERATURE", "0.2")),
        "llm_max_tokens": int(os.getenv("LLM_MAX_TOKENS", "2000")),
        "timeout_s": float(os.getenv("TIMEOUT", "120")),
        "retries": int(os.getenv("RETRIES", "3")),
        "backoff": float(os.getenv("BACKOFF", "1.5")),
        "top_k": int(args.top_k),
        "target_window_size": int(args.target_window_size),
        "min_window_size": int(args.min_window_size),
        "max_window_size": int(args.max_window_size),
        "weights": {"O": 1.0, "S": 0.8, "A": 0.6},
    }


def embed_text_batch(
    texts: List[str],
    *,
    model: str,
    base_url: str,
    api_key: str,
    timeout_s: float = 60.0,
    batch_size: int = 10,
) -> np.ndarray:
    if not base_url or not api_key:
        raise RuntimeError("Embedding base URL or API key is not configured")

    try:
        from openai import OpenAI  # type: ignore

        client = OpenAI(base_url=base_url, api_key=api_key).with_options(timeout=timeout_s)
        mode = "v1"
    except Exception:
        client = None
        mode = "http"

    vectors: List[List[float]] = []
    for start in range(0, len(texts), max(1, batch_size)):
        batch = texts[start : start + max(1, batch_size)]
        if mode == "v1" and client is not None:
            response = client.embeddings.create(model=model, input=batch)
            vectors.extend(item.embedding for item in response.data)
            continue

        base = base_url.rstrip("/")
        url = f"{base}/embeddings" if base.endswith("/v1") else f"{base}/v1/embeddings"
        response = requests.post(
            url,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"model": model, "input": batch},
            timeout=timeout_s,
        )
        response.raise_for_status()
        payload = response.json()
        items = payload.get("data", [])
        if not items and isinstance(payload.get("output"), dict):
            items = payload["output"].get("embeddings", payload["output"].get("data", [])) or []
        for item in items:
            if isinstance(item, dict) and "embedding" in item:
                vectors.append(item["embedding"])
            else:
                vectors.append(item)
    return np.asarray(vectors, dtype=np.float32)


def rerank_documents(
    *,
    endpoint: str,
    api_key: str,
    model: str,
    query: str,
    documents: List[str],
    timeout_s: float = 60.0,
) -> List[float] | None:
    if not endpoint or not api_key or not documents:
        return None

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    is_dashscope = "dashscope.aliyuncs.com" in endpoint

    if is_dashscope:
        payload = {
            "model": model,
            "input": {"query": query[:1000], "documents": [doc[:1000] for doc in documents]},
            "parameters": {"top_n": max(1, min(len(documents), 20)), "return_documents": False},
        }
    else:
        payload = {"model": model, "query": query, "documents": documents, "top_n": len(documents)}

    response = requests.post(endpoint, headers=headers, json=payload, timeout=timeout_s)
    response.raise_for_status()
    data = response.json()

    if is_dashscope:
        items = data.get("output", {}).get("results", [])
    else:
        items = data.get("results", []) or data.get("data", [])

    scores = [0.0] * len(documents)
    for idx, item in enumerate(items):
        if isinstance(item, dict) and isinstance(item.get("index"), int):
            pos = item["index"]
            if 0 <= pos < len(scores):
                scores[pos] = float(item.get("relevance_score", item.get("score", 0.0)))
        elif idx < len(scores):
            if isinstance(item, dict):
                scores[idx] = float(item.get("relevance_score", item.get("score", 0.0)))
            elif isinstance(item, (int, float)):
                scores[idx] = float(item)
    return scores


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector if norm == 0.0 else vector / norm


def attach_embeddings(nodes: List[Dict[str, Any]], cfg: Dict[str, Any], logger: RunLogger | None = None) -> None:
    unique_texts = {
        text
        for node in nodes
        for text in (node["O_text"], node["S_text"], node["A_text"])
        if ensure_text(text)
    }
    text_list = sorted(unique_texts)
    if logger:
        logger.log(f"embedding start | unique_texts={len(text_list)}")
    vectors = embed_text_batch(
        text_list,
        model=cfg["embed_model"],
        base_url=cfg["embed_base_url"],
        api_key=cfg["embed_api_key"],
        timeout_s=cfg["timeout_s"],
    )
    text_to_vector = {text: np.asarray(vector, dtype=np.float32) for text, vector in zip(text_list, vectors)}
    dim = int(vectors.shape[1])
    zero = np.zeros(dim, dtype=np.float32)

    for node in nodes:
        o_vec = text_to_vector.get(node["O_text"], zero)
        s_vec = text_to_vector.get(node["S_text"], zero)
        a_vec = text_to_vector.get(node["A_text"], zero)
        weighted = cfg["weights"]["O"] * o_vec + cfg["weights"]["S"] * s_vec
        if node["A_text"]:
            weighted = weighted + cfg["weights"]["A"] * a_vec
        node["retrieval_vec"] = normalize_vector(weighted.astype(np.float32))
    if logger:
        logger.log(f"embedding complete | vector_dim={dim}")


def build_pool_membership(nodes: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    policy_pool = [
        node["node_id"]
        for node in nodes
        if node["track"] in {"track_a", "track_b"} or (node["track"] == "track_c" and node.get("tool_nature") == "non_monetary")
    ]
    tool_pool_monetary = [
        node["node_id"] for node in nodes if node["track"] == "track_c" and node.get("tool_nature") == "monetary"
    ]
    return {"policy_pool": policy_pool, "tool_pool_monetary": tool_pool_monetary}


def choose_pool_name(node: Dict[str, Any]) -> str:
    if node["track"] == "track_c" and node.get("tool_nature") == "monetary":
        return "tool_pool_monetary"
    return "policy_pool"


def build_rerank_query(node: Dict[str, Any]) -> str:
    parts = [
        f"O={node['O_text']}",
        f"S={node['S_text']}",
        f"pau_final={node['pau_final']}",
        f"parent_title={node.get('parent_title') or ''}",
        f"tool_nature={node.get('tool_nature') or ''}",
    ]
    if node["A_text"]:
        parts.append(f"A={node['A_text']}")
    return "\n".join(parts)


def shape_window(candidate_ids: List[str], cfg: Dict[str, Any]) -> List[str]:
    candidate_count = len(candidate_ids)
    if candidate_count <= 0:
        return []
    target_neighbors = max(0, cfg["target_window_size"] - 1)
    min_neighbors = max(0, cfg["min_window_size"] - 1)
    max_neighbors = max(0, cfg["max_window_size"] - 1)

    if candidate_count >= target_neighbors:
        keep = target_neighbors
    elif candidate_count >= min_neighbors:
        keep = candidate_count
    else:
        keep = candidate_count
    return candidate_ids[: min(max_neighbors, keep)]


def retrieve_owner_window(
    node: Dict[str, Any],
    nodes_by_id: Dict[str, Dict[str, Any]],
    pools: Dict[str, List[str]],
    cfg: Dict[str, Any],
    fallback_flags: Dict[str, int],
    logger: RunLogger | None = None,
) -> List[str]:
    pool_ids = pools[choose_pool_name(node)]
    candidates: List[tuple[str, float]] = []
    for candidate_id in pool_ids:
        if candidate_id == node["node_id"]:
            continue
        score = float(np.dot(node["retrieval_vec"], nodes_by_id[candidate_id]["retrieval_vec"]))
        candidates.append((candidate_id, score))

    candidates.sort(key=lambda item: (-item[1], item[0]))
    top_candidates = candidates[: min(cfg["top_k"], len(candidates))]
    ordered_ids = [candidate_id for candidate_id, _ in top_candidates]
    documents = [nodes_by_id[candidate_id]["reference_text_v10_internal"] for candidate_id in ordered_ids]
    rerank_scores: List[float] | None = None

    if documents and cfg["rerank_endpoint"] and cfg["rerank_api_key"]:
        try:
            rerank_scores = rerank_documents(
                endpoint=cfg["rerank_endpoint"],
                api_key=cfg["rerank_api_key"],
                model=cfg["rerank_model"],
                query=build_rerank_query(node),
                documents=documents,
                timeout_s=cfg["timeout_s"],
            )
        except Exception as exc:
            rerank_scores = None
            fallback_flags["rerank_fallback_count"] += 1
            if logger:
                logger.log(f"rerank fallback | node={node['node_id']} | reason={exc}")
    elif documents:
        fallback_flags["rerank_fallback_count"] += 1
        if logger:
            logger.log(f"rerank skipped/fallback | node={node['node_id']} | reason=missing endpoint or api key")

    if rerank_scores and len(rerank_scores) == len(ordered_ids):
        coarse_map = {candidate_id: score for candidate_id, score in top_candidates}
        reranked = [(candidate_id, float(score), coarse_map[candidate_id]) for candidate_id, score in zip(ordered_ids, rerank_scores)]
        reranked.sort(key=lambda item: (-item[1], -item[2], item[0]))
        ordered_ids = [candidate_id for candidate_id, _, _ in reranked]

    neighbor_ids = shape_window(ordered_ids, cfg)
    if logger:
        logger.log(
            f"window ready | node={node['node_id']} | pool={choose_pool_name(node)} "
            f"| candidates={len(candidates)} | selected_neighbors={len(neighbor_ids)}"
        )
    return neighbor_ids


def build_user_prompt(target: Dict[str, Any], references: List[Dict[str, Any]]) -> str:
    payload = {
        "target_node": {
            "node_id": target["node_id"],
            "track": target["track"],
            "tool_nature": target.get("tool_nature") or "",
            "parent_title": target.get("parent_title") or "",
            "leaf_name": target.get("leaf_name") or "",
            "components": target["components"],
            "pau_final": target["pau_final"],
        },
        "reference_nodes": [
            {
                "node_id": node["node_id"],
                "track": node["track"],
                "tool_nature": node.get("tool_nature") or "",
                "parent_title": node.get("parent_title") or "",
                "components": node["components"],
                "pau_final": node["pau_final"],
            }
            for node in references
        ],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def run_pass1_normalization(
    nodes: List[Dict[str, Any]],
    cfg: Dict[str, Any],
    prompt_path: Path,
    *,
    logger: RunLogger | None = None,
    log_every: int = 5,
) -> tuple[Dict[str, Dict[str, Any]], Dict[str, int]]:
    prompt_system = prompt_path.read_text(encoding="utf-8")
    nodes_by_id = {node["node_id"]: node for node in nodes}
    pools = build_pool_membership(nodes)
    attach_embeddings(nodes, cfg, logger=logger)

    fallback_flags = {"rerank_fallback_count": 0, "llm_fallback_count": 0}
    results: Dict[str, Dict[str, Any]] = {}

    for idx, node in enumerate(nodes, start=1):
        if logger and (idx == 1 or idx % log_every == 0):
            logger.log(f"pass1 progress | {idx}/{len(nodes)} | node={node['node_id']} | pau_final={node['pau_final']}")
        neighbor_ids = retrieve_owner_window(node, nodes_by_id, pools, cfg, fallback_flags, logger=logger)
        references = [nodes_by_id[neighbor_id] for neighbor_id in neighbor_ids]
        owner_window_size = 1 + len(neighbor_ids)

        try:
            llm_result, llm_json = chat_json(
                system=prompt_system,
                user=build_user_prompt(node, references),
                model=cfg["llm_model"],
                temperature=cfg["llm_temperature"],
                max_tokens=cfg["llm_max_tokens"],
                timeout_s=cfg["timeout_s"],
                retries=cfg["retries"],
                backoff=cfg["backoff"],
            )
            if not llm_result.ok or not isinstance(llm_json, dict):
                fallback_flags["llm_fallback_count"] += 1
                if logger:
                    logger.log(f"llm fallback | node={node['node_id']} | reason={ensure_text(llm_result.error) or 'LLM call failed'}")
                normalized = fallback_result(node, ensure_text(llm_result.error) or "LLM call failed")
            else:
                normalized = validate_and_repair_result(node, llm_json)
        except Exception as exc:
            fallback_flags["llm_fallback_count"] += 1
            if logger:
                logger.log(f"llm exception | node={node['node_id']} | reason={exc}")
            normalized = fallback_result(node, f"LLM exception: {exc}")

        normalized["owner_window_id_v10"] = f"OW_{node['node_id']}"
        normalized["reference_neighbor_ids_v10"] = neighbor_ids
        normalized["owner_window_size_v10"] = owner_window_size
        results[node["node_id"]] = normalized
        if logger:
            logger.log(
                f"node done | node={node['node_id']} | owner_window_size={owner_window_size} "
                f"| review_flag={normalized['consistency_review_flag_v10']}"
            )

    return results, fallback_flags


def apply_results_to_structure(output_data: Dict[str, Any], nodes: List[Dict[str, Any]], results: Dict[str, Dict[str, Any]]) -> None:
    for node in nodes:
        result = results[node["node_id"]]
        if node["track"] == "track_a":
            target = output_data["track_a"][node["source_index"]]
        elif node["track"] == "track_b":
            target = output_data["track_b"][node["group_index"]]["leaves"][node["leaf_index"]]
        else:
            target = output_data["track_c"][node["group_index"]]["leaves"][node["leaf_index"]]

        target["owner_window_id_v10"] = result["owner_window_id_v10"]
        target["reference_neighbor_ids_v10"] = result["reference_neighbor_ids_v10"]
        target["owner_window_size_v10"] = result["owner_window_size_v10"]
        target["components_v10"] = result["components_v10"]
        target["pau_std_v10"] = result["pau_std_v10"]
        target["guard_tags_v10"] = result["guard_tags_v10"]
        target["normalization_log_v10"] = result["normalization_log_v10"]
        target["consistency_review_flag_v10"] = result["consistency_review_flag_v10"]


def build_flat_output(nodes: List[Dict[str, Any]], results: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    flattened: List[Dict[str, Any]] = []
    for node in nodes:
        result = results[node["node_id"]]
        flattened.append(
            {
                "node_id": node["node_id"],
                "track": node["track"],
                "tool_nature": node.get("tool_nature") or "",
                "doc_id": node["doc_id"],
                "semantic_block_id": node["semantic_block_id"],
                "parent_title": node.get("parent_title") or "",
                "leaf_index": node["leaf_index"],
                "source_path_v9": node["source_path_v9"],
                "components": node["components"],
                "pau_final": node["pau_final"],
                "owner_window_id_v10": result["owner_window_id_v10"],
                "reference_neighbor_ids_v10": result["reference_neighbor_ids_v10"],
                "owner_window_size_v10": result["owner_window_size_v10"],
                "components_v10": result["components_v10"],
                "pau_std_v10": result["pau_std_v10"],
                "guard_tags_v10": result["guard_tags_v10"],
                "normalization_log_v10": result["normalization_log_v10"],
                "consistency_review_flag_v10": result["consistency_review_flag_v10"],
            }
        )
    return flattened


def write_review_csv(path: Path, flat_nodes: List[Dict[str, Any]]) -> None:
    rows = []
    for node in flat_nodes:
        rows.append(
            {
                "node_id": node["node_id"],
                "track": node["track"],
                "tool_nature": node.get("tool_nature") or "",
                "parent_title": node.get("parent_title") or "",
                "pau_final": node["pau_final"],
                "pau_std_v10": node["pau_std_v10"],
                "guard_tags_v10": "|".join(ensure_str_list(node["guard_tags_v10"])),
                "owner_window_size_v10": node["owner_window_size_v10"],
                "normalization_log_v10": node["normalization_log_v10"],
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")


def add_meta(
    output_data: Dict[str, Any],
    input_path: Path,
    prompt_path: Path,
    cfg: Dict[str, Any],
    nodes: List[Dict[str, Any]],
    flat_nodes: List[Dict[str, Any]],
    fallback_flags: Dict[str, int],
) -> None:
    counts = {
        "effective_nodes": len(nodes),
        "track_a": sum(1 for node in nodes if node["track"] == "track_a"),
        "track_b": sum(1 for node in nodes if node["track"] == "track_b"),
        "track_c": sum(1 for node in nodes if node["track"] == "track_c"),
        "track_c_monetary": sum(1 for node in nodes if node["track"] == "track_c" and node.get("tool_nature") == "monetary"),
        "track_c_non_monetary": sum(1 for node in nodes if node["track"] == "track_c" and node.get("tool_nature") == "non_monetary"),
    }
    output_data.setdefault("meta", {})
    output_data["meta"]["v10_1"] = {
        "version": "v10.1-mvp",
        "run_time_utc": now_iso(),
        "input_file": str(input_path),
        "prompt_file": str(prompt_path),
        "effective_node_counts": counts,
        "pool_strategy": {
            "policy_pool": "Track A + Track B + Track C non_monetary",
            "tool_pool_monetary": "Track C monetary",
        },
        "window_params": {
            "top_k": cfg["top_k"],
            "target_window_size": cfg["target_window_size"],
            "min_window_size": cfg["min_window_size"],
            "max_window_size": cfg["max_window_size"],
        },
        "weights": cfg["weights"],
        "models": {
            "llm_model": cfg["llm_model"],
            "embed_model": cfg["embed_model"],
            "rerank_model": cfg["rerank_model"],
        },
        "fallback_summary": fallback_flags,
        "notes": [
            "pass1 only",
            "snapshot frozen to v9.1",
            "no automatic network validation was executed by this implementation turn",
        ],
        "flat_node_count": len(flat_nodes),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Step 2 v10.1 reference-guided local normalization MVP")
    parser.add_argument("--input", default=str(SIM_DIR / "v5_step2_simulation_full_v9_1.json"))
    parser.add_argument("--env", default=str(ROOT / "configs" / "roundC_v4.env"))
    parser.add_argument("--prompt", default=str(SIM_DIR / "prompt_v10_1_reference_normalization.md"))
    parser.add_argument("--output-json", default=str(SIM_DIR / "v10_1_reference_window_simulation.json"))
    parser.add_argument("--output-csv", default=str(SIM_DIR / "v10_1_reference_window_review.csv"))
    parser.add_argument("--log-file", default=str(SIM_DIR / "v10_1_reference_normalization.log"))
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--llm-model", default=None)
    parser.add_argument("--llm-model-env-key", default="PRIMARY_LLM_MODEL")
    parser.add_argument("--llm-base-url-env-key", default="PRIMARY_LLM_BASE_URL")
    parser.add_argument("--llm-api-key-env-key", default="PRIMARY_LLM_API_KEY")
    parser.add_argument("--embed-model", default=None)
    parser.add_argument("--rerank-model", default=None)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--target-window-size", type=int, default=20)
    parser.add_argument("--min-window-size", type=int, default=15)
    parser.add_argument("--max-window-size", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = RunLogger(Path(args.log_file))
    logger.log(f"starting v10.1 normalization | input={args.input}")
    logger.log(f"env={args.env} prompt={args.prompt}")
    load_env_file(args.env)
    runtime_env = activate_openai_compatible_env(
        model_env_key=args.llm_model_env_key,
        base_url_env_key=args.llm_base_url_env_key,
        api_key_env_key=args.llm_api_key_env_key,
        model_override=args.llm_model,
        overwrite=True,
    )
    logger.log("env loaded")
    logger.log(
        "llm env activated | "
        f"model_key={args.llm_model_env_key} "
        f"base_url_key={args.llm_base_url_env_key} "
        f"api_key_key={args.llm_api_key_env_key} "
        f"base_url={runtime_env['base_url']}"
    )

    input_path = Path(args.input)
    prompt_path = Path(args.prompt)
    output_json_path = Path(args.output_json)
    output_csv_path = Path(args.output_csv)

    logger.log("loading input json")
    data = load_json(input_path)
    output_data = copy.deepcopy(data)
    nodes = flatten_nodes(data)
    logger.log(
        "flatten complete | total_nodes="
        f"{len(nodes)} | TA={sum(1 for n in nodes if n['track']=='track_a')} "
        f"| TB={sum(1 for n in nodes if n['track']=='track_b')} "
        f"| TC={sum(1 for n in nodes if n['track']=='track_c')}"
    )
    cfg = resolve_runtime_config(args)
    logger.log(
        f"models resolved | llm={cfg['llm_model']} embed={cfg['embed_model']} "
        f"rerank={cfg['rerank_model']} | top_k={cfg['top_k']} "
        f"| window(target/min/max)={cfg['target_window_size']}/{cfg['min_window_size']}/{cfg['max_window_size']}"
    )

    results, fallback_flags = run_pass1_normalization(nodes, cfg, prompt_path, logger=logger, log_every=max(1, int(args.log_every)))
    logger.log(
        f"pass1 complete | llm_fallbacks={fallback_flags['llm_fallback_count']} "
        f"| rerank_fallbacks={fallback_flags['rerank_fallback_count']}"
    )
    logger.log("writing structured output")
    apply_results_to_structure(output_data, nodes, results)
    flat_nodes = build_flat_output(nodes, results)
    output_data["nodes_flat_v10"] = flat_nodes
    add_meta(output_data, input_path, prompt_path, cfg, nodes, flat_nodes, fallback_flags)

    save_json(output_json_path, output_data)
    logger.log(f"json written -> {output_json_path}")
    write_review_csv(output_csv_path, flat_nodes)
    logger.log(f"csv written -> {output_csv_path}")
    print(json.dumps({"output_json": str(output_json_path), "output_csv": str(output_csv_path), "node_count": len(nodes)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
