#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 3 · T2 -> L1 全局挂接（public · 三轮漏斗裁决版 · 配置增强）

修改点：
1. 增加配置读取优先级：CLI参数 > YAML配置 > 默认值。
2. 支持从 YAML 中读取 global_link.hard_threshold 和 auto_accept_conf。
3. [Fix] 补充缺失的 _append_with_dedup 函数。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
import pandas as pd
import yaml
from pandas.errors import EmptyDataError

from common_utils import (  # type: ignore
    ensure_outdir,
    read_corpus,
    jaccard_overlap,
    safe_write_csv,
)
from llm_runtime import call_llm_json, load_env_file, profiles_from_config  # type: ignore

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
LOG_DIR = ROOT / "data" / "intermediate_outputs" / "logs"
OPS_LOG = ROOT / "data" / "intermediate_outputs" / "tree_edit_operations.jsonl"
PROMPT_LINK = ROOT / "prompts" / "link_second_level_to_top_level.md"
ENV_FILE = ROOT / "configs" / ".env"

# --- 默认配置 ---
DEFAULT_HARD_THRESHOLD = 0.50
DEFAULT_AUTO_ACCEPT_CONF = 0.85

def _load_env():
    load_env_file(ENV_FILE)

def _write_jsonl(path: Path, obj: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(obj, ensure_ascii=False) + "\n")

def _log_line(path: Path, msg: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(msg.rstrip() + "\n")
    print(msg)

def _md5_8(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()[:8]

def _bootstrap_l1_nodes(outdir: Path) -> pd.DataFrame:
    nodes_path = outdir / "tree_nodes_L1.csv"
    if nodes_path.exists():
        return pd.read_csv(nodes_path, dtype=str)
    def_path = outdir / "top_level_categories.json"
    if not def_path.exists():
        raise FileNotFoundError("Critical: top_level_categories.json not found.")
    data = json.loads(def_path.read_text(encoding="utf-8"))
    cats = data.get("categories") or []
    rows = []
    for c in cats:
        rows.append({
            "node_id": str(c.get("id")),
            "level": "L1",
            "label": str(c.get("name")),
            "keywords": ",".join(c.get("keywords") or []),
            "definition": c.get("definition", ""),
            "provenance": "top_level_category_assignment",
            "size": 0,
            "confidence": 1.0,
            "human_label": str(c.get("name")),
            "human_action": "keep",
            "human_notes": "",
            "locked": False,
            "is_new": False,
        })
    df = pd.DataFrame(rows)
    safe_write_csv(df, nodes_path)
    return df

def _load_l3_context(outdir: Path) -> Dict[str, List[str]]:
    links_path = outdir / "tree_parent_links_L3_to_L2.csv"
    nodes_path = outdir / "tree_nodes_L3.csv"
    if not links_path.exists() or not nodes_path.exists():
        return {}
    try:
        links = pd.read_csv(links_path, dtype=str)
        nodes = pd.read_csv(nodes_path, dtype=str)
        l3_labels = {str(r["node_id"]): str(r["label"]) for _, r in nodes.iterrows()}
        context = {}
        for _, row in links.iterrows():
            parent = str(row["parent_id"])
            child = str(row["child_id"])
            if child in l3_labels:
                context.setdefault(parent, []).append(l3_labels[child])
        return context
    except Exception as e:
        print(f"[WARN] Failed to load L3 context: {e}")
        return {}

# --- [修复] 补充缺失的 helper 函数 ---
def _append_with_dedup(new_df: pd.DataFrame, path: Path, subset: List[str]):
    """
    将 new_df 追加到 path 指定的 CSV 中，并根据 subset 列去重（保留最后一条）。
    用于增量更新链接关系或节点表。
    """
    if new_df.empty:
        return

    if path.exists():
        try:
            existing = pd.read_csv(path, dtype=str)
            combined = pd.concat([existing, new_df], ignore_index=True)
        except EmptyDataError:
            combined = new_df
    else:
        combined = new_df

    # 去重：keep='last' 确保最新的决策覆盖旧的
    combined = combined.drop_duplicates(subset=subset, keep="last")
    safe_write_csv(combined, path)

def run_link_t2_l1(config_path: Path, cli_hard_threshold: Optional[float]):
    # 1. 加载配置
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    # 【逻辑修正】优先级：CLI参数 > YAML配置 > 默认值
    yaml_hard_thr = cfg.get("global_link", {}).get("hard_threshold")
    yaml_auto_conf = cfg.get("global_link", {}).get("auto_accept_conf")

    # 确定 hard_threshold
    if cli_hard_threshold is not None:
        hard_threshold = cli_hard_threshold
    elif yaml_hard_thr is not None:
        hard_threshold = float(yaml_hard_thr)
    else:
        hard_threshold = DEFAULT_HARD_THRESHOLD

    # 确定 auto_accept_conf
    auto_accept_conf = float(yaml_auto_conf) if yaml_auto_conf is not None else DEFAULT_AUTO_ACCEPT_CONF

    print(f"[CONFIG] 最终参数: Hard Threshold = {hard_threshold}, Auto Accept Conf = {auto_accept_conf}")

    paths = cfg.get("paths", {})
    outdir = Path(paths.get("outdir", ROOT / "data" / "intermediate_outputs"))
    ensure_outdir(outdir)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    _load_env()

    # 2. 准备数据
    corpus = read_corpus(paths.get("corpus", outdir / "policy_corpus_calibrated.csv"))
    sample_title = {str(r["sample_id"]): str(r["cleaned_title"]) for _, r in corpus.iterrows()}

    assign_df = pd.read_csv(outdir / "node_top_level_category_assignments.csv", dtype=str)
    valid_assign = assign_df[assign_df["assigned_l1_id"].notna() & (assign_df["assigned_l1_id"] != "")]
    physical_map = dict(zip(valid_assign["sample_id"], valid_assign["assigned_l1_id"]))

    l2_nodes = pd.read_csv(outdir / "tree_nodes_L2.csv", dtype=str)
    l2_mem = pd.read_csv(outdir / "tree_node_membership_L2.csv", dtype=str)
    l2_members_map = l2_mem.groupby("node_id")["member_id"].apply(list).to_dict()

    l1_nodes = _bootstrap_l1_nodes(outdir)
    l1_meta = {}
    for _, r in l1_nodes.iterrows():
        l1_meta[str(r["node_id"])] = {
            "label": str(r["label"]),
            "keywords": str(r.get("keywords", ""))
        }

    l3_context_map = _load_l3_context(outdir)
    llm_profile, _ = profiles_from_config(cfg.get("llm", {}))
    log_link = LOG_DIR / "llm_step3_link_t2_l1.jsonl"
    links_rows = []
    new_l1_nodes = []
    stats = {"R1_Hard": 0, "R2_Fast": 0, "R3_Deep": 0, "New_L1": 0}

    print(f"[INFO] Processing {len(l2_nodes)} L2 nodes via multi-pass strategy...")

    for _, row in l2_nodes.iterrows():
        node_id = str(row["node_id"])
        label = str(row["label"])
        members = l2_members_map.get(node_id, [])
        if not members: continue

        # --- R1: 物理硬链接 ---
        votes = {}
        for m in members:
            if m in physical_map:
                l1 = physical_map[m]
                if l1 in l1_meta:
                    votes[l1] = votes.get(l1, 0) + 1

        best_phy_l1 = None
        best_phy_ratio = 0.0
        if votes:
            best_phy_l1, count = max(votes.items(), key=lambda x: x[1])
            best_phy_ratio = count / len(members)

        if best_phy_ratio >= hard_threshold:
            links_rows.append({
                "child_id": node_id, "parent_id": best_phy_l1,
                "decision": "R1_Hard_Link", "confidence": 1.0,
                "reason": f"Physical support {best_phy_ratio:.1%} >= {hard_threshold}",
                "support_counts": json.dumps(votes),
                "is_new_parent": False, "human_action": "keep"
            })
            stats["R1_Hard"] += 1
            continue

        # --- 候选准备 ---
        cand_rows = []
        for l1_id, meta in l1_meta.items():
            sim = jaccard_overlap(label, meta["label"])
            phy_cnt = votes.get(l1_id, 0)
            score = (phy_cnt * 1.0) + sim
            cand_rows.append((l1_id, score, meta["label"], meta["keywords"]))
        cand_rows.sort(key=lambda x: x[1], reverse=True)
        top_cands = cand_rows[:5]

        # --- R2: 快速筛选 ---
        titles_sample = [sample_title.get(m, "") for m in members[:6]]
        cand_desc = [f"{i+1}) {c[2]} (ID:{c[0]})" for i, c in enumerate(top_cands)]

        user_prompt_r2 = (
            f"待归类 L2: {label}\n"
            f"样本示例: {'; '.join(titles_sample)}\n"
            f"候选 L1: {', '.join(cand_desc)}\n"
            "任务: 选择最佳 L1 ID。若必须新建请设 create_new_l1=true。\n"
            "JSON: {best_l1_id, confidence, create_new_l1, not_match_reason}"
        )

        resp_r2 = call_llm_json(
            profile=llm_profile,
            system=PROMPT_LINK.read_text(encoding="utf-8"),
            user=user_prompt_r2,
            task="link_second_level_to_top_level_r2",
        )
        obj_r2 = resp_r2.get("json") or {}

        r2_pass = False
        if obj_r2 and not obj_r2.get("create_new_l1"):
            conf = float(obj_r2.get("confidence") or 0)
            best_id = obj_r2.get("best_l1_id")
            if conf >= auto_accept_conf and best_id in l1_meta:
                links_rows.append({
                    "child_id": node_id, "parent_id": best_id,
                    "decision": "R2_Fast_Match", "confidence": conf,
                    "reason": obj_r2.get("analysis") or "High confidence fast match",
                    "is_new_parent": False, "human_action": "keep"
                })
                stats["R2_Fast"] += 1
                r2_pass = True

        if r2_pass:
            continue

        # --- R3: 深度审计 ---
        l3_subs = l3_context_map.get(node_id, [])
        titles_more = [sample_title.get(m, "") for m in members[:15]]
        cand_desc_full = [f"{i+1}) 【{c[2]}】(ID:{c[0]}) | kws:{c[3]} | 物理票数={votes.get(c[0],0)}" for i, c in enumerate(top_cands)]

        user_prompt_r3 = (
            f"!!! AUDIT MODE !!!\n"
            f"待裁决 L2 节点: {label} (ID: {node_id})\n"
            f"下属 L3 结构: {', '.join(l3_subs[:8])} ...\n"
            f"成员样本 ({len(members)}个): \n" + "\n".join([f"- {t}" for t in titles_more]) + "\n\n"
            f"候选 L1 列表:\n" + "\n".join(cand_desc_full) + "\n\n"
            "指令: 除非属于现有体系完全无法覆盖的全新领域，否则禁止新建。请优先归入现有类别。"
        )

        resp_r3 = call_llm_json(
            profile=llm_profile,
            system=PROMPT_LINK.read_text(encoding="utf-8"),
            user=user_prompt_r3,
            task="link_second_level_to_top_level_r3",
        )
        _write_jsonl(log_link, {"node": node_id, "round": 3, "resp": resp_r3})

        obj_r3 = resp_r3.get("json") or {}
        create_new = bool(obj_r3.get("create_new_l1"))
        best_id = obj_r3.get("best_l1_id")
        reason = str(obj_r3.get("analysis") or obj_r3.get("not_match_reason") or "")

        parent_id = None
        is_new_parent = False

        if create_new:
            new_label = str(obj_r3.get("new_l1_label") or label).strip()
            exists_id = next((k for k, v in l1_meta.items() if v["label"] == new_label), None)
            if exists_id:
                parent_id = exists_id
                reason += " (Name collision, merged to existing)"
            else:
                base_id = f"L1_N{_md5_8(new_label)}"
                parent_id = base_id
                new_meta = {
                    "label": new_label, "keywords": str(obj_r3.get("new_l1_keywords") or ""),
                    "definition": reason
                }
                l1_meta[parent_id] = new_meta
                new_l1_nodes.append({
                    "node_id": parent_id, "level": "L1", "label": new_label,
                    "keywords": new_meta["keywords"], "definition": new_meta["definition"],
                    "provenance": "R3_Create_New", "size": len(members),
                    "confidence": 0.7, "human_label": new_label,
                    "human_action": "review", "is_new": True, "locked": False
                })
                is_new_parent = True
                stats["New_L1"] += 1
        elif best_id and best_id in l1_meta:
            parent_id = best_id
        else:
            parent_id = top_cands[0][0] if top_cands else None
            reason += " (R3 Fallback)"

        if parent_id:
            links_rows.append({
                "child_id": node_id, "parent_id": parent_id,
                "decision": "R3_Deep_Audit" if not is_new_parent else "R3_New_Created",
                "confidence": float(obj_r3.get("confidence") or 0.5),
                "reason": reason,
                "is_new_parent": is_new_parent,
                "human_action": "review" if is_new_parent else "keep"
            })
            stats["R3_Deep"] += 1

    links_df = pd.DataFrame(links_rows)
    links_path = outdir / "tree_parent_links_L2_to_L1.csv"
    _append_with_dedup(links_df, links_path, ["child_id"])

    if new_l1_nodes:
        l1_df = pd.concat([l1_nodes, pd.DataFrame(new_l1_nodes)], ignore_index=True)
        l1_df = l1_df.drop_duplicates(subset=["node_id"], keep="last")
        safe_write_csv(l1_df, outdir / "tree_nodes_L1.csv")

    _log_line(LOG_DIR / "link_second_level_to_top_level.log", f"[COMPLETE] Stats: {stats}")

def main():
    parser = argparse.ArgumentParser(description="PolicyTreeBuilder final replication · Step3 L2->L1 (3-Round Funnel)")
    parser.add_argument("--config", required=True)
    # 将默认值设为 None，以便区分是否传入了参数
    parser.add_argument("--hard-threshold", type=float, default=None, help="覆盖 YAML 中的 hard_threshold 配置")
    args = parser.parse_args()

    run_link_t2_l1(Path(args.config), args.hard_threshold)

if __name__ == "__main__":
    main()
