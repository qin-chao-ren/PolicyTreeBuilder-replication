#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 4.1 · Skeleton (Fixed Version + Stable Paths)
功能：纵向坍缩 (Vertical Collapse)
核心修复：
1. 严格限制 promote 场景：仅当父节点只有1个子节点（单脉传）时才允许
2. 移除激进的 rehome_siblings 逻辑
3. 添加层级保护
"""

import argparse
import time
import json
from pathlib import Path
from typing import Dict, List, Tuple

from llm_runtime import call_llm_json
from common_utils import jaccard_overlap
from utils.step4_shared import (
    Step4Env, EmbeddingHelper, load_tree, dump_tree,
    append_jsonl, read_membership_map, read_title_map
)
from utils.tree_manager import TreeManager

# ==========================================
# 1. 路径锚点 (Path Anchors) - 保留你的配置
# ==========================================
HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent

# 2. 关键资源路径
ENV_PATH = PROJECT_ROOT / "configs" / ".env"

# 3. Prompt 路径
PROMPT_COLLAPSE = PROJECT_ROOT / "prompts" / "collapse_redundant_hierarchy.md"
# PROMPT_REHOME = PROJECT_ROOT / "prompts" / "rehome_sibling_nodes.md" # (本逻辑已移除，可注释)

# ==========================================

# 相似度阈值
SIMILARITY_THRESHOLD_JAC = 0.3
SIMILARITY_THRESHOLD_COS = 0.5

def collect_titles(sample_ids, title_map, top_k=5):
    return [title_map[sid] for sid in sample_ids if sid in title_map][:top_k]

def describe_node(node_id: str, manager: TreeManager, membership: Dict, title_map: Dict) -> str:
    node = manager.get_node(node_id)
    if not node: return "Node Not Found"

    level = node.get("level", "")
    members = membership.get(level, {}).get(node_id, [])
    titles = collect_titles(members, title_map)
    children = manager.get_children(node_id)
    children_count = len(children)

    # 增加子节点摘要，辅助 LLM 判断
    children_summary = ""
    if children_count > 0:
        child_labels = [c.get("label", "?")[:20] for c in children[:5]]
        children_summary = f"\n子节点示例: {', '.join(child_labels)}"
        if children_count > 5:
            children_summary += f" ... 等共 {children_count} 个"

    return (
        f"ID: {node_id} · level={level} · label={node.get('label','')}\n"
        f"成员数={len(members)} · 子节点数={children_count}"
        f"{children_summary}\n"
        f"示例标题: {', '.join(titles) if titles else '无'}"
    )

def get_level_depth(level_str: str) -> int:
    if not level_str: return 0
    s = str(level_str).upper()
    if s == "ROOT": return 0
    if s.startswith("L") and s[1:].isdigit(): return int(s[1:])
    return 99

def can_promote_safely(tm: TreeManager, child_id: str, parent_id: str) -> Tuple[bool, str]:
    """安全检查：是否允许子节点上位"""
    parent = tm.get_node(parent_id)
    child = tm.get_node(child_id)
    if not parent or not child: return False, "节点不存在"

    # 1. 必须是单脉传（父节点只有一个孩子）
    siblings = tm.get_children(parent_id)
    if len(siblings) > 1:
        return False, f"父节点有 {len(siblings)} 个子节点，非单脉传，禁止 Promote"

    # 2. 必须有爷爷
    grandparent_id = tm.get_parent_id(parent_id)
    if not grandparent_id: return False, "无爷爷节点"

    # 3. 层级跨度保护 (防止 L3 直接跳到 Root)
    grandparent = tm.get_node(grandparent_id)
    gp_level = get_level_depth(grandparent.get("level", ""))
    child_level = get_level_depth(child.get("level", ""))

    if gp_level == 0 and child_level > 2:
        return False, "跨度过大(L3+ -> ROOT)"

    return True, "OK"

class SkeletonRefiner:
    def __init__(self, env: Step4Env, tm: TreeManager, args):
        self.env = env
        self.llm_profile = env.primary_llm_profile()
        self.tm = tm
        self.args = args
        self.emb_helper = EmbeddingHelper(Path(env.config["paths"]["embeddings"]))
        self.title_map = read_title_map(Path(env.config["paths"]["corpus"]))
        self.membership = {lvl: read_membership_map(env.outdir, lvl) for lvl in ["L4", "L3", "L2", "L1"]}
        self.ops_log = env.outdir / "tree_edit_operations.jsonl"
        self.llm_log = env.log_dir / "llm_collapse_redundant_hierarchy.jsonl"
        self.redirect_map = {}

    def run(self):
        print("[Step 4.1] Starting Skeleton Refinement (Fixed)...")
        candidate_parents = [nid for nid in self.tm.get_all_node_ids() if self.tm.get_children(nid)]

        for parent_id in candidate_parents:
            if not self.tm.exists(parent_id): continue
            self._process_parent(parent_id)

        dump_tree(Path(self.args.output), self.tm.root)
        trace_path = self.env.outdir / "vertical_collapse_trace.json"
        trace_path.write_text(json.dumps(self.redirect_map, indent=2), encoding="utf-8")
        print(f"[DONE] Skeleton Refined. Tree saved to {self.args.output}")

    def _process_parent(self, parent_id):
        parent_node = self.tm.get_node(parent_id)
        children = list(self.tm.get_children(parent_id)) # Snapshot

        for child_node in children:
            child_id = child_node["node_id"]
            if not self.tm.exists(child_id): continue

            # 计算相似度
            p_mems = self.membership.get(parent_node.get("level"), {}).get(parent_id, [])
            c_mems = self.membership.get(child_node.get("level"), {}).get(child_id, [])
            vec_p = self.emb_helper.get_centroid(p_mems)
            vec_c = self.emb_helper.get_centroid(c_mems)
            jac = jaccard_overlap(parent_node.get("label", ""), child_node.get("label", ""))
            cos = self.emb_helper.cosine_sim(vec_p, vec_c)

            if jac < SIMILARITY_THRESHOLD_JAC and cos < SIMILARITY_THRESHOLD_COS:
                continue

            is_single_child = (len(self.tm.get_children(parent_id)) == 1)

            # LLM Call
            evidence = self._build_evidence(parent_id, child_id, jac, cos, is_single_child)
            if not is_single_child:
                evidence += "\n\n⚠️ 注意：父节点有多个子节点，请勿选择 promote_child，只能选择 keep/rename/absorb。"

            resp = call_llm_json(
                profile=self.llm_profile,
                system=PROMPT_COLLAPSE.read_text(encoding="utf-8"),
                user=evidence,
                task="collapse_redundant_hierarchy",
                temperature=0.0,
            )

            append_jsonl(self.llm_log, {
                "ts": int(time.time()), "parent": parent_id, "child": child_id,
                "is_single": is_single_child, "metrics": {"jac":jac, "cos":cos}, "resp": resp
            })

            self._execute_decision(parent_id, parent_node, child_id, resp.get("json", {}), is_single_child)

    def _build_evidence(self, pid, cid, jac, cos, single):
        note = "【单脉传场景】" if single else "【多子节点场景】"
        return (
            f"# 场景：父子语义重叠检测 {note}\n"
            f"# 父节点\n{describe_node(pid, self.tm, self.membership, self.title_map)}\n"
            f"# 子节点\n{describe_node(cid, self.tm, self.membership, self.title_map)}\n"
            f"相似度: Jaccard={jac:.2f}, Cosine={cos:.2f}\n"
        )

    def _execute_decision(self, pid, pnode, cid, res, is_single):
        dec = res.get("decision", "keep")
        new_lbl = res.get("new_label")
        rec = {"step":"vertical_collapse", "parent":pid, "child":cid, "op":dec}

        if dec == "rename_then_keep" and new_lbl:
            pnode["label"] = new_lbl
            append_jsonl(self.ops_log, {**rec, "detail": "rename"})

        elif dec == "absorb_child":
            if self.tm.absorb_node(pid, cid):
                if new_lbl: pnode["label"] = new_lbl
                self.redirect_map[cid] = pid
                append_jsonl(self.ops_log, {**rec, "detail": "absorb"})

        elif dec == "promote_child":
            # 核心修复点：安全检查
            ok, reason = can_promote_safely(self.tm, cid, pid)
            if not ok:
                append_jsonl(self.ops_log, {**rec, "status": "rejected", "reason": reason})
                return

            if self.tm.promote_child_safe(cid):
                if new_lbl: self.tm.get_node(cid)["label"] = new_lbl
                self.redirect_map[pid] = cid
                # 关键：移除了原来那个 "move all siblings to child" 的错误逻辑
                # 因为能进这里必定是 single child，根本没有 siblings
                self.tm.remove_node(pid)
                append_jsonl(self.ops_log, {**rec, "status": "success"})

def main():
    parser = argparse.ArgumentParser(description="PolicyTreeBuilder final replication · Step4.1 Skeleton (Fixed)")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    # 【修改点】使用锚点定义的 ENV_PATH
    env = Step4Env(args.config, str(ENV_PATH))

    raw = load_tree(Path(args.input))
    refiner = SkeletonRefiner(env, TreeManager(raw), args)
    refiner.run()

if __name__ == "__main__":
    main()
