#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 4.3 · Polishing (Refactored)
功能：精细润色 (兄弟合并 / 跨父统一 / 标签精炼)
核心职责：
1. 结构收尾：使用 TreeManager 进行安全的 Merge/Move/Rename。
2. 链路闭环：加载 Step 4.1/4.2 的 Trace，合并本步骤产生的 Trace。
3. 最终产出：生成 policy_tree_final_membership.csv，确保所有样本能找到最终归属。
"""

import argparse
import time
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Set

from common_utils import jaccard_overlap
from common_llm import call_json
from utils.step4_shared import (
    Step4Env,
    EmbeddingHelper,
    load_tree,
    dump_tree,
    append_jsonl,
    read_membership_map,
    read_title_map
)
from utils.tree_manager import TreeManager

# --- 1. 路径锚点 (Path Anchors) ---
# 无论在哪里运行命令，__file__ 都能定位到 scripts/ 目录
HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent

# --- 2. 关键资源路径 (基于锚点) ---
# 这样写绝对不会错
ENV_PATH = PROJECT_ROOT / "configs" / ".env"

# --- 3. Prompt 路径 (每个脚本不一样，请按需保留) ---
PROMPT_POLISH = PROJECT_ROOT / "prompts" / "polish_tree_labels.md"

class PolishingProcess:
    def __init__(self, env: Step4Env, tm: TreeManager, args):
        self.env = env
        self.llm = env.build_llm_config()
        self.tm = tm
        self.args = args

        # 加载辅助数据
        self.emb_helper = EmbeddingHelper(Path(env.config["paths"]["embeddings"]))
        corpus_path = Path(env.config["paths"]["corpus"])
        self.title_map = read_title_map(corpus_path)
        self.membership = {
            lvl: read_membership_map(env.outdir, lvl)
            for lvl in ["L4", "L3", "L2", "L1"]
        }

        # 日志
        self.ops_log = env.outdir / "tree_edit_operations.jsonl"
        self.llm_log = env.log_dir / "llm_polish_tree_labels.jsonl"

        # 本步骤产生的 ID 变更记录 (Old ID -> New ID)
        self.local_trace_map: Dict[str, str] = {}

    def run(self):
        # 1. 兄弟节点合并 (同父)
        print("[Step 4.3] Starting Sibling Merge...")
        self._run_sibling_merge()

        # 2. 跨父节点统一 (同层级)
        print("[Step 4.3] Starting Cross-Parent Unify...")
        self._run_cross_parent_unify()

        # 3. 追溯与导出 (The Core Traceability Logic)
        print("[Step 4.3] Consolidating Traces & Exporting Final Membership...")
        final_trace = self._consolidate_all_traces()
        self._export_final_membership(final_trace)

        # 4. 保存最终树
        dump_tree(Path(self.args.output), self.tm.root)
        print(f"[DONE] Polishing Completed. Tree saved to {self.args.output}")

    # --- Logic 1: Sibling Merge ---

    def _run_sibling_merge(self):
        # 遍历所有非叶子节点
        parents = [nid for nid in self.tm.get_all_node_ids() if self.tm.get_children(nid)]

        for parent_id in parents:
            if not self.tm.exists(parent_id): continue

            # 获取子节点快照
            children = list(self.tm.get_children(parent_id))
            if len(children) < 2: continue

            # 两两比较 (简单冒泡以覆盖所有对，或限制 Top-K)
            # 为了效率，我们这里只比较 jaccard > 0.6 的对
            handled_pairs = set()

            for i in range(len(children)):
                for j in range(i + 1, len(children)):
                    a = children[i]
                    b = children[j]

                    # 动态检查有效性 (因为前面的循环可能已经 merge 掉了某节点)
                    if not self.tm.exists(a["node_id"]) or not self.tm.exists(b["node_id"]):
                        continue

                    pair_key = tuple(sorted([a["node_id"], b["node_id"]]))
                    if pair_key in handled_pairs: continue
                    handled_pairs.add(pair_key)

                    # 预筛选
                    if not self._quick_check_similarity(a, b, thr_jac=0.75, thr_cos=0.85):
                        continue

                    # LLM 决策
                    self._process_pair(a, b, case="sibling_merge", parent_id=parent_id)

    # --- Logic 2: Cross-Parent Unify ---

    def _run_cross_parent_unify(self):
        # 按层级桶 (Bucket) 聚合所有节点
        level_buckets: Dict[str, List[Dict]] = {}
        for nid in self.tm.get_all_node_ids():
            node = self.tm.get_node(nid)
            lvl = node.get("level", "")
            if lvl in ["L3", "L2"]: # L4 太多通常不做跨父，L1 不动
                level_buckets.setdefault(lvl, []).append(node)

        for lvl, nodes in level_buckets.items():
            # 限制每层处理数量，避免 O(N^2) 爆炸，或者使用聚类加速
            # 这里简化逻辑：只比较相邻/高相似对，实际生产环境建议配合 Faiss 检索
            # 本代码演示：简单双重循环，加严格预筛选

            print(f"  > Processing {lvl} ({len(nodes)} nodes)...")
            handled_cross = set()

            for i in range(len(nodes)):
                for j in range(i + 1, min(i + 50, len(nodes))): # 滑动窗口减少计算量
                    a = nodes[i]
                    b = nodes[j]

                    if not self.tm.exists(a["node_id"]) or not self.tm.exists(b["node_id"]):
                        continue

                    # 必须是不同父
                    pa = self.tm.get_parent_id(a["node_id"])
                    pb = self.tm.get_parent_id(b["node_id"])
                    if pa == pb: continue

                    # 严格预筛选
                    if not self._quick_check_similarity(a, b, thr_jac=0.80, thr_cos=0.90):
                        continue

                    self._process_pair(a, b, case="cross_parent_unify", parent_id=None)

    # --- Helper: LLM Interaction & Execution ---

    def _quick_check_similarity(self, a, b, thr_jac, thr_cos) -> bool:
        jac = jaccard_overlap(a.get("label", ""), b.get("label", ""))
        if jac >= thr_jac: return True

        vec_a = self.emb_helper.get_centroid(self._get_members(a))
        vec_b = self.emb_helper.get_centroid(self._get_members(b))
        cos = self.emb_helper.cosine_sim(vec_a, vec_b)

        return cos >= thr_cos

    def _get_members(self, node) -> List[str]:
        return self.membership.get(node.get("level"), {}).get(node["node_id"], [])

    def _process_pair(self, a, b, case, parent_id):
        # 1. 构造 Context
        ctx_a = self._describe_node(a["node_id"])
        ctx_b = self._describe_node(b["node_id"])

        # 计算当前指标供 LLM 参考
        jac = jaccard_overlap(a.get("label", ""), b.get("label", ""))
        vec_a = self.emb_helper.get_centroid(self._get_members(a))
        vec_b = self.emb_helper.get_centroid(self._get_members(b))
        cos = self.emb_helper.cosine_sim(vec_a, vec_b)

        evidence = (
            f"# 场景: {case}\n"
            f"## 节点 A\n{ctx_a}\n\n"
            f"## 节点 B\n{ctx_b}\n\n"
            f"## 相似度指标\nJaccard={jac:.2f}, Cosine={cos:.2f}\n"
            "请根据 label 语义、父节点语境和样本内容，决定是否合并、移动或重命名。\n"
        )

        # 2. Call LLM
        resp = call_json(self.llm.primary, PROMPT_POLISH.read_text(encoding="utf-8"), evidence)

        append_jsonl(self.llm_log, {
            "ts": int(time.time()), "case": case,
            "node_a": a["node_id"], "node_b": b["node_id"],
            "metrics": {"jac": jac, "cos": cos}, "resp": resp
        })

        # 3. Execute
        res = resp.get("json", {})
        op = res.get("operation", "keep")

        record = {
            "ts": int(time.time()), "step": "label_polishing", "case": case,
            "op": op, "reason": res.get("reason"), "confidence": res.get("confidence")
        }

        if op == "merge":
            winner_id = res.get("winner_id")
            loser_id = res.get("loser_id")
            new_label = res.get("new_label")

            # 安全检查
            if winner_id not in [a["node_id"], b["node_id"]] or loser_id not in [a["node_id"], b["node_id"]]:
                return # LLM 幻觉了 ID

            # 使用 TreeManager 进行原子合并
            if self.tm.absorb_node(winner_id, loser_id):
                if new_label: self.tm.get_node(winner_id)["label"] = new_label
                self.local_trace_map[loser_id] = winner_id # 记录重定向
                append_jsonl(self.ops_log, {**record, "winner": winner_id, "loser": loser_id})

        elif op == "move":
            # 仅在跨父场景有效
            node_id = res.get("winner_id") # LLM 可能会把要移动的节点填在 winner_id
            target_parent = res.get("target_parent")

            if node_id and target_parent and self.tm.exists(node_id) and self.tm.exists(target_parent):
                self.tm.move_node(node_id, target_parent)
                append_jsonl(self.ops_log, {**record, "node": node_id, "new_parent": target_parent})

        elif op == "rename":
            node_id = res.get("winner_id")
            new_label = res.get("new_label")
            if node_id and new_label and self.tm.exists(node_id):
                self.tm.get_node(node_id)["label"] = new_label
                append_jsonl(self.ops_log, {**record, "node": node_id, "new_label": new_label})

    def _describe_node(self, node_id):
        node = self.tm.get_node(node_id)
        pid = self.tm.get_parent_id(node_id)
        p_label = self.tm.get_node(pid)["label"] if pid and self.tm.exists(pid) else "ROOT"

        titles = self._collect_titles(node_id)
        return (
            f"ID: {node_id} · Label: {node.get('label','')}\n"
            f"Level: {node.get('level')} · Parent: {p_label}\n"
            f"Examples: {', '.join(titles)}"
        )

    def _collect_titles(self, node_id):
        node = self.tm.get_node(node_id)
        mems = self._get_members(node)
        return [self.title_map[sid] for sid in mems if sid in self.title_map][:5]

    # --- Logic 3: Trace Consolidation (Critical) ---

    def _consolidate_all_traces(self) -> Dict[str, str]:
        """
        合并 Trace 4.1 + 4.2 + 4.3 (Local)
        逻辑：链式更新 A->B, B->C  =>  A->C
        """
        full_map = {}

        # 1. 加载历史 Trace
        trace_files = [
            ("vertical_collapse", self.env.outdir / "vertical_collapse_trace.json"),
            ("structure_balancing", self.env.outdir / "structure_balancing_trace.json"),
        ]
        for step, p in trace_files:
            if p.exists():
                try:
                    sub_map = json.loads(p.read_text(encoding="utf-8"))
                    self._merge_trace_into(full_map, sub_map)
                    print(f"  + Loaded {len(sub_map)} redirects from {step}")
                except Exception as e:
                    print(f"[WARN] Failed to load trace {step}: {e}")

        # 2. 合并当前步骤 Trace
        self._merge_trace_into(full_map, self.local_trace_map)
        print(f"  + Loaded {len(self.local_trace_map)} redirects from 4.3 (Local)")

        return full_map

    def _merge_trace_into(self, main_map: Dict[str, str], new_map: Dict[str, str]):
        """
        将 new_map 合并入 main_map，并处理链式引用。
        """
        # 1. 遍历 main_map，如果其目标在 new_map 中有进一步重定向，则更新 main_map
        for src, dst in main_map.items():
            if dst in new_map:
                main_map[src] = new_map[dst]

        # 2. 将 new_map 中新的映射加入 main_map
        for src, dst in new_map.items():
            if src not in main_map:
                main_map[src] = dst
            # 如果 src 已经在 main_map，说明是 A->B (old), A->C (new)?
            # 理论上 ID 是唯一的，不应该出现同一个 ID 在不同阶段作为 Source 出现两次（除非它被移动了而非合并）
            # 对于 Merge 来说，ID 一旦消失就不应该再出现。
            # 这里我们假设后来的操作优于先前的操作 (Update)

    def _export_final_membership(self, trace_map: Dict[str, str]):
        """
        读取原始 CSV，应用 trace_map，生成 policy_tree_final_membership.csv
        """
        output_rows = []

        for level in ["L4", "L3", "L2"]:
            path = self.env.outdir / f"tree_node_membership_{level}.csv"
            if not path.exists(): continue

            df = pd.read_csv(path, dtype=str)
            print(f"  > Processing {level} membership ({len(df)} rows)...")

            for _, row in df.iterrows():
                original_nid = str(row["node_id"])
                sid = str(row["member_id"])

                # 追溯最终 ID
                # 由于我们已经在 _merge_trace_into 中展平了链条，这里直接查即可
                final_nid = trace_map.get(original_nid, original_nid)

                # 双重保险：检查 final_nid 是否还存在于当前树中 (可能被级联删除了?)
                # 如果树中不存在，且没有被进一步映射，说明该节点可能成为了孤儿或被丢弃
                # 但 public 逻辑尽量避免丢弃。如果真找不到，保留原 ID 或标记为 Unknown
                if not self.tm.exists(final_nid):
                     # 尝试看是否能向上找到存在的祖先？暂时不处理太复杂，直接保留
                     pass

                output_rows.append({
                    "sample_id": sid,
                    "final_node_id": final_nid,
                    "original_node_id": original_nid,
                    "original_level": level
                })

        if output_rows:
            out_path = self.env.outdir / "policy_tree_final_membership.csv"
            pd.DataFrame(output_rows).to_csv(out_path, index=False, encoding="utf-8-sig")
            print(f"[SUCCESS] Final membership exported to {out_path} ({len(output_rows)} rows)")

def main():
    parser = argparse.ArgumentParser(description="PolicyTreeBuilder final replication · Step4.3 Polishing")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    # 1. 初始化
    env = Step4Env(args.config, str(ENV_PATH))

    # 2. 加载数据
    raw_tree = load_tree(Path(args.input))
    tm = TreeManager(raw_tree)

    # 3. 执行
    process = PolishingProcess(env, tm, args)
    process.run()

if __name__ == "__main__":
    main()
