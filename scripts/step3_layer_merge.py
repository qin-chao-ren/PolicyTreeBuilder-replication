#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 3 · 层内合并（v4 · 单模型极速版）

修改记录：
1. [单模型] 强制 Secondary Model = Primary Model，消除思路不一致，提升速度。
2. [极速] 只有一个成员的节点，直接复用原标题，不再调用 LLM。
3. [输入] 依赖修复后的 Step 2.8 (v4_l1_node_assignments.csv)。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import yaml
from pandas.errors import EmptyDataError

from common_id import make_node_id
from common_utils import (
    ensure_outdir,
    read_corpus,
    read_pairs,
    safe_write_csv,
)
from common_llm import LLMConfig, adjudicate
from step3_edge_contract import contract_edges

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
LOG_DIR = ROOT / "data" / "intermediate_outputs" / "logs"
OPS_LOG = ROOT / "data" / "intermediate_outputs" / "v4_operations_log.jsonl"
PROMPT_MERGE = ROOT / "prompts" / "step3_merge_decision.md"
ENV_FILE = ROOT / "configs" / ".env"


def build_high_graph(atom_ids: List[str], pairs: pd.DataFrame, high_thr: float, mutual_required: bool) -> Dict[str, set]:
    atoms = set(atom_ids)
    sub = pairs[(pairs["id_a"].astype(str).isin(atoms)) & (pairs["id_b"].astype(str).isin(atoms))]
    sub = sub[sub["rerank_score"].astype(float) >= float(high_thr)]
    if mutual_required and "is_mutual" in sub.columns:
        sub = sub[sub["is_mutual"] == True]
    graph: Dict[str, set] = {a: set() for a in atom_ids}
    for _, row in sub.iterrows():
        a = str(row["id_a"])
        b = str(row["id_b"])
        graph.setdefault(a, set()).add(b)
        graph.setdefault(b, set()).add(a)
    return graph


def connected_components(graph: Dict[str, set]) -> List[List[str]]:
    seen = set()
    comps: List[List[str]] = []
    for v in graph:
        if v in seen:
            continue
        stack = [v]
        cur = []
        seen.add(v)
        while stack:
            x = stack.pop()
            cur.append(x)
            for y in graph.get(x, ()):  # pragma: no mutate
                if y not in seen:
                    seen.add(y)
                    stack.append(y)
        comps.append(cur)
    return comps


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
    # [修改 1] 强制单模型逻辑
    # 优先读取 config 中的 primary，其次读取环境变量，最后兜底
    primary = str(lcfg.get("primary") or os.getenv("PRIMARY_LLM_MODEL") or "qwen3-max")

    # 将 secondary 强制设置为与 primary 相同
    # 这样 common_llm 中的 adjudicate 即使执行双重检查，也是同一模型，避免思路打架
    secondary = primary

    os.environ.setdefault("OPENAI_BASE_URL", os.getenv("PRIMARY_LLM_BASE_URL", "https://api.openai.com/v1"))
    if os.getenv("PRIMARY_LLM_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = os.getenv("PRIMARY_LLM_API_KEY", "")

    return LLMConfig(
        primary=primary,
        secondary=secondary, # 此时 secondary == primary
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
        print(
            f"[WARN] discards file {path} is empty; continuing without discards",
            file=sys.stderr,
        )
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


def choose_label_llm(level_str: str, titles: List[str], cfg: LLMConfig, out_log: Path) -> Tuple[str, float]:
    prompt_sys = PROMPT_MERGE.read_text(encoding="utf-8")
    sample = titles[:10]
    stats_line = f"样本数={len(titles)}"
    user = f"层级：{level_str}\n候选成员：\n" + "\n".join([f"- {t}" for t in sample]) + f"\n{stats_line}\n"
    res = adjudicate(cfg, prompt_sys, user, input_meta={"step": "layer_merge", "level": level_str, "count": len(titles)}, log_path=str(out_log))
    obj = res.get("final") or {}
    lbl = str(obj.get("canonical_label") or obj.get("label") or (sample[0] if sample else ""))
    conf = float(obj.get("confidence") or 0.8)
    return lbl, conf


def _append_with_dedup(df_new: pd.DataFrame, path: Path, subset: List[str]) -> pd.DataFrame:
    if path.exists():
        old = pd.read_csv(path, dtype=str)
        combined = pd.concat([old, df_new], ignore_index=True)
    else:
        combined = df_new.copy()
    combined = combined.drop_duplicates(subset=subset, keep="last")
    safe_write_csv(combined, path)
    return combined


def run_layer_merge_for_partition(config_path: Path, level_L: str, l1_category: str | None):
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    paths = cfg.get("paths", {})
    outdir = Path(paths.get("outdir", ROOT / "data" / "intermediate_outputs"))
    corpus_path = Path(paths.get("corpus", outdir / "v4_corpus_calibrated.csv"))
    pairs_path = Path(paths.get("pairs", outdir / "v4_rerank_edges.csv"))

    ensure_outdir(outdir)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    _load_env()

    mcfg = cfg.get("merge", {})
    high_thr = float(mcfg.get("high_thr", 0.90))
    mid_low = float(mcfg.get("mid_thr_low", 0.80))
    mutual_required = bool(mcfg.get("mutual_required", False))

    corpus = read_corpus(corpus_path)
    level_T = _level_to_T(level_L)
    sub = corpus[corpus["calibrated_level"].astype(str) == level_T].copy()

    assign = _load_assignments(outdir)
    discards = _load_discards(outdir)
    if discards:
        sub = sub[~sub["sample_id"].astype(str).isin(discards)]
    if l1_category:
        if not assign:
            raise ValueError("无法按 L1 切分：缺少 v4_l1_node_assignments.csv")
        sub = sub[sub["sample_id"].astype(str).map(lambda sid: assign.get(str(sid)) == l1_category)]
    atom_ids = sub["sample_id"].astype(str).tolist()

    if not atom_ids:
        print(
            f"[WARN] No atoms found for level={level_L} L1={l1_category or 'ALL'}. Skipping this combination.",
            file=sys.stderr,
        )
        return

    pairs = read_pairs(pairs_path)

    graph = build_high_graph(atom_ids, pairs, high_thr, mutual_required)
    comps = connected_components(graph)

    clusters: List[Dict] = []
    id2title = {str(r["sample_id"]): str(r["cleaned_title"]) for _, r in sub.iterrows()}
    llm_cfg = _mk_llm_config(cfg.get("llm", {}))
    log_merge = LOG_DIR / "llm_step3_merge_decisions.jsonl"

    for comp in comps:
        members = sorted(set(comp))
        titles = [id2title.get(a, "") for a in members]

        label: str = ""
        conf: float = 1.0
        provenance: str = ""

        # [修改 2] 只有一个成员时，跳过 LLM，加速！
        if len(members) == 1:
            label = titles[0] if titles else ""
            conf = 1.0
            provenance = "seed-existing"
        else:
            label, conf = choose_label_llm(level_L, titles, llm_cfg, log_merge)
            conf = max(0.9, min(1.0, conf))
            provenance = "merge-high"

        clusters.append(
            {
                "members": members,
                "label": label,
                "confidence": conf,
                "provenance": provenance,
            }
        )

    node_members_tmp = {f"C{i}": set(c["members"]) for i, c in enumerate(clusters)}
    agg = contract_edges(node_members_tmp, pairs)
    cand = agg[(agg["score_p75"] >= mid_low) & (agg["score_p75"] < high_thr)]

    parent = list(range(len(clusters)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for _, row in cand.sort_values("score_p75", ascending=False).iterrows():
        a = int(str(row["node_a"]).replace("C", ""))
        b = int(str(row["node_b"]).replace("C", ""))
        ra, rb = find(a), find(b)
        if ra == rb:
            continue
        mems = sorted(set(clusters[ra]["members"]) | set(clusters[rb]["members"]))
        titles = [id2title.get(x, "") for x in mems[:12]]
        stats_line = f"样本数={len(mems)}; 边p75={row['score_p75']:.2f}, mean={row['score_mean']:.2f}, max={row['score_max']:.2f}"
        user = f"层级：{level_L}\n候选成员：\n" + "\n".join([f"- {t}" for t in titles]) + f"\n{stats_line}\n"

        # 这里 adjudicate 会使用 llm_cfg，其中 secondary 已被强制设为 primary
        adj = adjudicate(
            llm_cfg,
            PROMPT_MERGE.read_text(encoding="utf-8"),
            user,
            input_meta={"step": "layer_merge_mid", "level": level_L, "p75": float(row["score_p75"])},
            log_path=str(log_merge),
            evidence_score_primary=float(row["score_p75"]),
            evidence_score_secondary=float(row["score_mean"]),
        )
        obj = adj.get("final") or {}
        if obj.get("merge") is True:
            union(ra, rb)

    groups: Dict[int, List[int]] = defaultdict(list)
    for idx in range(len(clusters)):
        groups[find(idx)].append(idx)

    final_nodes = []
    final_members = []
    default_l1 = l1_category or ""
    for ridx, idxs in groups.items():
        all_members: List[str] = []
        confs = []
        provs = []
        for i in idxs:
            all_members.extend(clusters[i]["members"])
            confs.append(clusters[i]["confidence"])
            provs.append(clusters[i]["provenance"])

        unique_members = sorted(set(all_members))
        node_id = make_node_id(_level_to_T(level_L), unique_members)

        # [修改 3] 最终生成节点时，若只有1个成员，再次跳过 LLM 起名
        if len(unique_members) == 1:
            label = id2title.get(unique_members[0], "")
            lconf = 1.0
            provenance_final = "seed-existing" # 或继承之前的
        else:
            label, lconf = choose_label_llm(level_L, [id2title.get(x, "") for x in unique_members[:15]], llm_cfg, log_merge)
            provenance_final = "merge-llm" if any(p == "merge-llm" for p in provs) else ("merge-high" if any(p == "merge-high" for p in provs) else "seed-existing")

        confidence = float(np.clip(np.nanmean(confs) if confs else lconf, 0.6, 1.0))

        node_l1 = default_l1
        if assign and not node_l1 and all_members:
            node_l1 = assign.get(all_members[0], "")

        final_nodes.append(
            {
                "node_id": node_id,
                "level": level_L,
                "label": label,
                "provenance": provenance_final,
                "size": len(unique_members),
                "confidence": round(confidence, 4),
                "human_label": label,
                "human_action": "keep",
                "human_notes": "",
                "locked": False,
                "l1_category": node_l1,
            }
        )
        for sid in unique_members:
            final_members.append({"node_id": node_id, "member_id": sid})

    nodes_df = pd.DataFrame(final_nodes)
    members_df = pd.DataFrame(final_members)
    nodes_path = outdir / f"v4_nodes_{level_L}.csv"
    mem_path = outdir / f"v4_membership_{level_L}.csv"
    nodes_df = _append_with_dedup(nodes_df, nodes_path, ["node_id"])
    members_df = _append_with_dedup(members_df, mem_path, ["node_id", "member_id"])

    node_members = {nid: set(grp["member_id"].astype(str).tolist()) for nid, grp in members_df.groupby("node_id")}
    edges_df = contract_edges(node_members, pairs)
    edges_df["human_action"] = "keep"
    edges_df["human_notes"] = ""
    edges_path = outdir / f"v4_edges_{level_L}.csv"
    safe_write_csv(edges_df, edges_path)

    log_path = LOG_DIR / f"step3_layer_merge_{level_L}{'_' + l1_category if l1_category else ''}.log"
    _log_line(log_path, f"[LAYER MERGE] level={level_L} l1={l1_category or 'ALL'} nodes={len(nodes_df)} edges={len(edges_df)}")

    now = int(time.time())
    for node in final_nodes:
        _write_jsonl(
            OPS_LOG,
            {
                "ts": now,
                "step": "step3_layer_merge",
                "level": level_L,
                "l1_category": l1_category,
                "node_id": node["node_id"],
                "size": node["size"],
                "op": "create_or_update_node",
            },
        )


def main():
    parser = argparse.ArgumentParser(description="Round C v4 · Step3 Layer Merge (L4/L3/L2)")
    parser.add_argument("--config", required=True)
    parser.add_argument("--level", required=True, choices=["L4", "L3", "L2"])
    parser.add_argument("--l1-category", default=None, help="仅处理指定 L1（缺省遍历全部 L1）")
    args = parser.parse_args()

    config_path = Path(args.config)
    if args.l1_category:
        run_layer_merge_for_partition(config_path, args.level, args.l1_category)
        return

    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    outdir = Path(cfg.get("paths", {}).get("outdir", ROOT / "data" / "intermediate_outputs"))
    assign = _load_assignments(outdir)
    if not assign:
        run_layer_merge_for_partition(config_path, args.level, None)
        return
    for lid in sorted(set(assign.values())):
        run_layer_merge_for_partition(config_path, args.level, lid)


if __name__ == "__main__":
    main()