#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 3 · 纵向装配（v4 · 最终修复版）

修复记录：
1. [严重Bug修复] 修复了在 L1 分批处理时，新节点写入会导致其他 L1 类别节点被覆盖删除的问题。
   现在采用 _append_with_dedup 增量写入模式，确保数据安全。
2. [取值保险] 优先读取 adjudicate 的 final，如果为空，强制读取 primary.json。
3. [格式清洗] 增加 .strip() 容错。
"""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import yaml
import sys

from common_id import make_node_id  # type: ignore
from common_utils import (  # type: ignore
    ensure_outdir,
    read_corpus,
    read_pairs,
    read_embeddings,
    vector_centroid,
    cosine_sim,
    jaccard_overlap,
    safe_write_csv,
)
from common_llm import LLMConfig, adjudicate  # type: ignore
from step3_edge_contract import contract_edges  # type: ignore
from pandas.errors import EmptyDataError

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
LOG_DIR = ROOT / "data" / "intermediate_outputs" / "logs"
OPS_LOG = ROOT / "data" / "intermediate_outputs" / "v4_operations_log.jsonl"
PROMPT_CLASSIFY = ROOT / "prompts" / "step3_classify_assign.md"
ENV_FILE = ROOT / "configs" / ".env"

ALLOWED = {("L4", "L3"), ("L3", "L2")}
HARD_ASSIGN_THRESHOLD = 0.92

def _load_env():
    if not ENV_FILE.exists():
        return
    for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        k, v = s.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"'))


def _mk_llm_config(lcfg: dict) -> LLMConfig:
    primary = str(lcfg.get("primary") or os.getenv("PRIMARY_LLM_MODEL") or "qwen3-max")
    secondary = primary
    os.environ.setdefault("OPENAI_BASE_URL", os.getenv("PRIMARY_LLM_BASE_URL", "https://api.openai.com/v1"))
    if os.getenv("PRIMARY_LLM_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = os.getenv("PRIMARY_LLM_API_KEY", "")
    return LLMConfig(
        primary=primary,
        secondary=secondary,
        temperature=float(lcfg.get("temperature", 0.2)),
        max_tokens=int(lcfg.get("max_tokens", 1500)),
        response_format=str(lcfg.get("response_format", "json_object")),
        workers=int(lcfg.get("workers", 1)),
        tie_breaker=str(lcfg.get("tie_breaker", "score_margin_or_conservative")),
    )


def _load_assignments(outdir: Path) -> Dict[str, str]:
    path = outdir / "v4_l1_node_assignments.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path, dtype=str)
    if "sample_id" not in df.columns or "assigned_l1_id" not in df.columns:
        return {}
    return {str(r["sample_id"]): str(r["assigned_l1_id"]) for _, r in df.iterrows() if str(r["sample_id"])}


def _load_discards(outdir: Path) -> set[str]:
    path = outdir / "v4_l1_classification_review.csv"
    if not path.exists():
        return set()
    try:
        df = pd.read_csv(path, dtype=str)
    except EmptyDataError:
        return set()
    if "sample_id" not in df.columns or "human_decision" not in df.columns:
        return set()
    return set(df[df["human_decision"].astype(str) == "discard"]["sample_id"].astype(str).tolist())


def _write_jsonl(path: Path, obj: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _log_line(path: Path, msg: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(msg.rstrip() + "\n")


def _level_to_T(level: str) -> str:
    level = level.strip().upper()
    if level.startswith("L"):
        return "T" + level[1:]
    return level


def _load_layer(outdir: Path, level: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    nodes_path = outdir / f"v4_nodes_{level}.csv"
    mem_path = outdir / f"v4_membership_{level}.csv"
    if not nodes_path.exists() or not mem_path.exists():
        return pd.DataFrame(), pd.DataFrame()
    nodes = pd.read_csv(nodes_path, dtype=str)
    mem = pd.read_csv(mem_path, dtype=str)
    return nodes, mem


def _infer_l1(members: Iterable[str], assign: Dict[str, str]) -> str:
    hits = [assign.get(str(s)) for s in members if assign.get(str(s))]
    if not hits:
        return ""
    cnt = Counter(hits)
    return cnt.most_common(1)[0][0]


def _filter_membership(mem: pd.DataFrame, discards: set[str]) -> pd.DataFrame:
    if mem.empty: return mem
    if discards:
        return mem[~mem["member_id"].astype(str).isin(discards)].copy()
    return mem.copy()


def _append_with_dedup(df_new: pd.DataFrame, path: Path, subset: List[str]) -> pd.DataFrame:
    if path.exists():
        old = pd.read_csv(path, dtype=str)
        combined = pd.concat([old, df_new], ignore_index=True)
    else:
        combined = df_new.copy()
    combined = combined.drop_duplicates(subset=subset, keep="last")
    safe_write_csv(combined, path)
    return combined


def run_vertical_link(config_path: Path, from_level: str, to_level: str, l1_category: str | None):
    if (from_level, to_level) not in ALLOWED:
        raise ValueError(f"仅支持 {sorted(ALLOWED)}，收到 {from_level}->{to_level}")

    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    paths = cfg.get("paths", {})
    outdir = Path(paths.get("outdir", ROOT / "roundC_v4" / "outputs"))
    corpus_path = Path(paths.get("corpus", outdir / "v4_corpus_calibrated.csv"))
    emb_path = Path(paths.get("embeddings", outdir / "v4_embeddings.parquet"))
    pairs_path = Path(paths.get("pairs", outdir / "v4_rerank_edges.csv"))

    ensure_outdir(outdir)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    _load_env()

    vcfg = cfg.get("vertical", {})
    topk = int(vcfg.get("topk_candidates", 5))
    w_cos = float(vcfg.get("weights", {}).get("cosine", 0.7))
    w_jac = float(vcfg.get("weights", {}).get("jaccard", 0.3))

    llm_cfg = _mk_llm_config(cfg.get("llm", {}))

    corpus = read_corpus(corpus_path)
    sample_title = {str(r["sample_id"]): str(r["cleaned_title"]) for _, r in corpus.iterrows()}
    emb = read_embeddings(emb_path)
    pairs = read_pairs(pairs_path)

    assign = _load_assignments(outdir)
    discards = _load_discards(outdir)

    f_nodes, f_mem = _load_layer(outdir, from_level)
    t_nodes, t_mem = _load_layer(outdir, to_level)

    if f_nodes.empty:
        print(f"[WARN] {from_level} 节点为空，跳过。", file=sys.stderr)
        return

    f_mem = _filter_membership(f_mem, discards)
    t_mem = _filter_membership(t_mem, discards)

    f_members_map = {nid: grp["member_id"].astype(str).tolist() for nid, grp in f_mem.groupby("node_id")}
    t_members_map = {nid: grp["member_id"].astype(str).tolist() for nid, grp in t_mem.groupby("node_id")}

    f_nodes = f_nodes.copy()
    t_nodes = t_nodes.copy()
    f_nodes["l1_category"] = f_nodes.get("l1_category", "")
    t_nodes["l1_category"] = t_nodes.get("l1_category", "")

    for df, members_map in [(f_nodes, f_members_map), (t_nodes, t_members_map)]:
        if df.empty: continue
        mask = (df["l1_category"].astype(str).fillna("") == "")
        if mask.any():
            df.loc[mask, "l1_category"] = df.loc[mask, "node_id"].map(lambda nid: _infer_l1(members_map.get(str(nid), []), assign))

    if l1_category:
        f_nodes = f_nodes[f_nodes["l1_category"] == l1_category]
        if not t_nodes.empty:
            t_nodes = t_nodes[t_nodes["l1_category"] == l1_category]

    child_ids = f_nodes["node_id"].astype(str).tolist()
    parent_ids = t_nodes["node_id"].astype(str).tolist() if not t_nodes.empty else []

    if not child_ids:
        print(
            f"[WARN] 缺少 {from_level} (child) 节点，跳过：L1={l1_category or 'ALL'}",
            file=sys.stderr,
        )
        return

    f_centroids = {nid: vector_centroid(f_members_map.get(nid, []), emb) for nid in child_ids}
    t_centroids = {nid: vector_centroid(t_members_map.get(nid, []), emb) for nid in parent_ids}
    f_labels = {r["node_id"]: r.get("label", "") for _, r in f_nodes.iterrows()}
    t_labels = {r["node_id"]: r.get("label", "") for _, r in t_nodes.iterrows()}

    log_classify = LOG_DIR / "llm_step3_classify_decisions.jsonl"

    links_rows: List[Dict[str, object]] = []
    new_parents: List[Dict[str, object]] = []
    new_parent_members: Dict[str, set[str]] = {}

    print(f"[INFO] 开始处理 {len(child_ids)} 个子节点，现有父节点 {len(parent_ids)} 个...")

    for child_id in child_ids:
        members = f_members_map.get(child_id, [])
        cvec = f_centroids.get(child_id)
        clabel = f_labels.get(child_id, "")
        if not members:
            continue

        # 准备候选
        cand_rows = []
        for pid in parent_ids:
            sim_c = cosine_sim(cvec, t_centroids.get(pid))
            sim_j = jaccard_overlap(clabel, t_labels.get(pid, ""))
            score = w_cos * sim_c + w_jac * sim_j
            cand_rows.append((pid, score, sim_c, sim_j))
        cand_rows.sort(key=lambda x: x[1], reverse=True)
        cands = cand_rows[:topk] or cand_rows

        assign_to = None
        create_new = None
        conf = 0.0
        reason = ""
        decision_type = ""

        # --- 决策逻辑 ---

        # 1. [直通车] 候选池为空 -> 自动新建
        if not parent_ids:
            create_new = {"label": clabel}
            conf = 1.0
            reason = "No existing parents available (Auto-init)."
            decision_type = "auto_create_first"

        # 2. [直通车] Hard Assign (Top1 > 0.92) -> 直接分配
        elif cands and cands[0][1] >= HARD_ASSIGN_THRESHOLD:
            assign_to = cands[0][0]
            conf = 1.0
            reason = f"Hard Assign (Score {cands[0][1]:.2f} >= {HARD_ASSIGN_THRESHOLD})"
            decision_type = "hard_assign"

        # 3. [慢车道] 调用 LLM 裁决
        else:
            child_titles = [sample_title.get(sid, "") for sid in members[:8]]
            desc = [
                f"{idx+1}) {pid} · {t_labels.get(pid, '')} | 综合={sc:.2f} cos={sc1:.2f} jac={sc2:.2f}"
                for idx, (pid, sc, sc1, sc2) in enumerate(cands[:3])
            ]
            user = (
                f"子节点：{child_id} · {clabel}\n"
                f"成员示例：\n" + "\n".join([f"- {t}" for t in child_titles]) +
                f"\n候选父 Top-{len(cands)}：\n" + "\n".join(desc) +
                "\n返回 JSON：{\"assign_to\": \"<parent_id>\"|null, \"create_new\": {\"label\": \"...\"}|null, \"confidence\": 0-1, \"reason\": \"...\"}"
            )
            margin = cands[0][1] - cands[1][1] if len(cands) >= 2 else (cands[0][1] if cands else 0.0)

            adj = adjudicate(
                llm_cfg,
                PROMPT_CLASSIFY.read_text(encoding="utf-8"),
                user,
                input_meta={"step": "vertical_link", "child": child_id, "from": from_level, "to": to_level},
                log_path=str(log_classify),
                evidence_score_primary=margin,
                evidence_score_secondary=0.0,
            )

            obj = adj.get("final")
            if not obj:
                obj = adj.get("primary", {}).get("json") or {}

            assign_to = obj.get("assign_to")
            create_new = obj.get("create_new")
            conf = float(obj.get("confidence") or 0.0)
            reason = str(obj.get("reason") or "")
            decision_type = "llm_assign"

        # --- 执行结果 ---
        applied = False

        if assign_to and str(assign_to).strip() in parent_ids:
            parent_id = str(assign_to).strip()
            links_rows.append(
                {
                    "child_id": child_id,
                    "parent_id": parent_id,
                    "decision": decision_type,
                    "confidence": conf,
                    "reason": reason,
                    "human_parent_id": parent_id,
                    "human_action": "keep",
                    "human_notes": "",
                    "is_new_parent": False,
                }
            )
            applied = True

        elif create_new:
            parent_id = make_node_id(_level_to_T(to_level), members)
            label = str(create_new.get("label") or clabel or f"NEW_{to_level}")
            child_row = f_nodes[f_nodes["node_id"] == child_id]
            child_l1 = child_row["l1_category"].iloc[0] if not child_row.empty else (l1_category or "")

            new_parents.append(
                {
                    "node_id": parent_id,
                    "level": to_level,
                    "label": label,
                    "provenance": "classify-llm" if decision_type == "llm_assign" else "auto-create",
                    "size": len(set(members)),
                    "confidence": round(max(conf, 0.6), 4),
                    "human_label": label,
                    "human_action": "keep",
                    "human_notes": reason,
                    "locked": False,
                    "l1_category": child_l1,
                    "is_new": True,
                }
            )
            new_parent_members[parent_id] = set(members)
            parent_ids.append(parent_id)
            t_labels[parent_id] = label
            t_centroids[parent_id] = vector_centroid(members, emb)

            links_rows.append(
                {
                    "child_id": child_id,
                    "parent_id": parent_id,
                    "decision": "llm_create_parent" if decision_type == "llm_assign" else decision_type,
                    "confidence": conf,
                    "reason": reason,
                    "human_parent_id": parent_id,
                    "human_action": "keep",
                    "human_notes": "",
                    "is_new_parent": True,
                }
            )
            applied = True

        if not applied and cands and decision_type != "auto_create_first":
            fallback_parent = cands[0][0]
            links_rows.append(
                {
                    "child_id": child_id,
                    "parent_id": fallback_parent,
                    "decision": "fallback_top1",
                    "confidence": conf,
                    "reason": f"LLM未明确或匹配失败(raw_assign={assign_to}), 采用Top1",
                    "human_parent_id": fallback_parent,
                    "human_action": "review",
                    "human_notes": "",
                    "is_new_parent": False,
                }
            )

    # --- 1. 写入链接关系 (Append) ---
    links_path = outdir / f"v4_links_{from_level}_to_{to_level}.csv"
    links_df = pd.DataFrame(links_rows)
    _append_with_dedup(links_df, links_path, ["child_id"])

    # --- 2. 写入新生成的父节点 (Append + Dedup) ---
    # [修复点]：这里不再覆盖写入，而是将新父节点追加到现有文件中，防止其他 L1 类别的节点被删除。
    if new_parents:
        new_nodes_df = pd.DataFrame(new_parents)
        _append_with_dedup(new_nodes_df, outdir / f"v4_nodes_{to_level}.csv", ["node_id"])

        mem_rows = [{"node_id": pid, "member_id": sid} for pid, sids in new_parent_members.items() for sid in sorted(sids)]
        new_mem_df = pd.DataFrame(mem_rows)
        _append_with_dedup(new_mem_df, outdir / f"v4_membership_{to_level}.csv", ["node_id", "member_id"])

        # 为了保证 Edges 的完整性，我们需要重新计算（或者追加）Edges
        # 这里为了安全起见，我们将新父节点构成的子图的 Edges 计算出来并追加。
        # 这是一个折中方案：确保新节点有边，且不删除旧边。

        # 重新加载（包含了刚才写入的新节点）的全量 membership，确保计算准确（可选，为了速度只计算新增）
        # 为了速度，我们只计算新节点相关的边
        node_members_new = {nid: set(sids) for nid, sids in new_parent_members.items()}
        # 注意：这里只计算了新节点内部的边。如果需要新节点和旧节点的边，需要加载旧节点。
        # 考虑到 L1 隔离，旧节点也在 t_mem 中。
        # 我们合并一下：
        for nid, sids in t_members_map.items():
            node_members_new[nid] = set(sids)

        new_edges_df = contract_edges(node_members_new, pairs)
        new_edges_df["human_action"] = "keep"
        new_edges_df["human_notes"] = ""

        _append_with_dedup(new_edges_df, outdir / f"v4_edges_{to_level}.csv", ["node_a", "node_b"])

    log_path = LOG_DIR / f"step3_vertical_link_{from_level}_to_{to_level}{'_' + l1_category if l1_category else ''}.log"
    _log_line(log_path, f"[VERT LINK] {from_level}->{to_level} l1={l1_category or 'ALL'} links={len(links_rows)} new_parents={len(new_parents)}")

    now = int(time.time())
    for link in links_rows:
        _write_jsonl(
            OPS_LOG,
            {
                "ts": now,
                "step": "step3_vertical_link",
                "from": from_level,
                "to": to_level,
                "child_id": link["child_id"],
                "parent_id": link["parent_id"],
                "decision": link["decision"],
                "l1_category": l1_category,
                "op": "link_child_parent",
            },
        )
    for parent in new_parents:
        _write_jsonl(
            OPS_LOG,
            {
                "ts": now,
                "step": "step3_vertical_link",
                "from": from_level,
                "to": to_level,
                "node_id": parent["node_id"],
                "op": "create_parent_node",
                "l1_category": parent.get("l1_category"),
            },
        )


def main():
    parser = argparse.ArgumentParser(description="Round C v4 · Step3 Vertical Link (L4→L3 / L3→L2)")
    parser.add_argument("--config", required=True)
    parser.add_argument("--from-level", required=True, choices=["L4", "L3"])
    parser.add_argument("--to-level", required=True, choices=["L3", "L2"])
    parser.add_argument("--l1-category", default=None, help="仅处理指定 L1（缺省遍历全部 L1）")
    args = parser.parse_args()

    if (args.from_level, args.to_level) not in ALLOWED:
        raise SystemExit(f"仅支持 {sorted(ALLOWED)}，收到 {args.from_level}->{args.to_level}")

    config_path = Path(args.config)
    if args.l1_category:
        run_vertical_link(config_path, args.from_level, args.to_level, args.l1_category)
        return

    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    outdir = Path(cfg.get("paths", {}).get("outdir", ROOT / "roundC_v4" / "outputs"))
    assign = _load_assignments(outdir)
    if not assign:
        run_vertical_link(config_path, args.from_level, args.to_level, None)
        return
    for lid in sorted(set(assign.values())):
        run_vertical_link(config_path, args.from_level, args.to_level, lid)


if __name__ == "__main__":
    main()