#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Finalization · Overall Structure (Fixed Version)
功能：L1 级整体结构审计 (Semantic Audit) + 最终层级强制对齐

【修复说明】
1. _call_llm 添加防御性检查，防止 resp 为 None 时崩溃
2. _export_flat_csv 添加防御性检查，防止幽灵节点
3. 添加更详细的错误日志
"""

import argparse
import time
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Generator, Tuple
from difflib import SequenceMatcher

# 保持与你提供的一致
from llm_runtime import call_llm_json
from utils.step4_shared import (
    Step4Env,
    load_tree,
    dump_tree,
    append_jsonl
)
from utils.tree_manager import TreeManager

# --- 1. 路径锚点 (Path Anchors) ---
HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent

# --- 2. 关键资源路径 (基于锚点) ---
ENV_PATH = PROJECT_ROOT / "configs" / ".env"

# --- 3. Prompt 路径 ---
PROMPT_PATH = PROJECT_ROOT / "prompts" / "finalize_tree_structure.md"

# --- 4. 默认输出/输入路径 ---
DEFAULT_L1_DEF = PROJECT_ROOT / "data" / "intermediate_outputs" / "top_level_categories.json"
DEFAULT_OPS_OUT = PROJECT_ROOT / "data" / "intermediate_outputs" / "policy_tree_final_operations.jsonl"
DEFAULT_AUDIT = PROJECT_ROOT / "data" / "intermediate_outputs" / "policy_tree_final_audit.json"
DEFAULT_FLAT = PROJECT_ROOT / "data" / "intermediate_outputs" / "policy_tree_final_flat.csv"

BATCH_SIZE_LIMIT = 200

# --- Helper Functions ---

def get_l1_ancestor(tm: TreeManager, node_id: str) -> Optional[str]:
    curr = node_id
    while curr:
        node = tm.get_node(curr)
        if not node: return None
        if str(node.get("level", "")).upper() == "L1":
            return curr
        curr = tm.get_parent_id(curr)
    return None

def render_subtree(tm: TreeManager, node_id: str, prefix_path: List[str]) -> List[str]:
    node = tm.get_node(node_id)
    if not node: return []

    level = str(node.get("level", "")).upper()
    label = str(node.get("label", "") or "")
    path_str = ' / '.join(prefix_path + [label])

    children = tm.get_children(node_id)
    structural_children = [c for c in children if str(c.get("level", "")).upper().startswith("L")]

    markers = []
    if len(structural_children) == 1:
        markers.append('⚡SingleChild')

    parent_label = prefix_path[-1] if prefix_path else ''
    if parent_label:
        sim = SequenceMatcher(None, str(parent_label).lower(), label.lower()).ratio()
        if sim > 0.6:
            markers.append(f'⚡Repetitive({int(sim * 100)}%)')

    suffix = f" << {' '.join(markers)} >>" if markers else ''
    lines = [f"- [{level}] {path_str} ({node_id}){suffix}"]

    next_path = prefix_path + [label]
    for ch in structural_children:
        lines.extend(render_subtree(tm, ch["node_id"], next_path))

    return lines

def generate_l1_batches(tm: TreeManager, l1_id: str, definition: str) -> Generator[str, None, None]:
    l1_node = tm.get_node(l1_id)
    header_lines = [
        f"# L1 节点\nID={l1_id} · label={l1_node.get('label','')}\n定义：{definition or '未提供'}",
        "# Tree 展开 (L2-L4)",
    ]
    header_text = '\n'.join(header_lines)

    l2_nodes = [ch for ch in tm.get_children(l1_id) if str(ch.get("level", "")).upper() == "L2"]

    if not l2_nodes:
        yield header_text
        return

    current_batch_lines = []
    for l2 in l2_nodes:
        l2_block = render_subtree(tm, l2["node_id"], [])
        current_size = len(header_lines) + len(current_batch_lines)
        block_size = len(l2_block)

        if current_size + block_size > BATCH_SIZE_LIMIT and current_batch_lines:
            yield header_text + "\n" + "\n".join(current_batch_lines)
            current_batch_lines = []

        if block_size > BATCH_SIZE_LIMIT:
            if current_batch_lines:
                yield header_text + "\n" + "\n".join(current_batch_lines)
                current_batch_lines = []
            yield header_text + "\n" + "\n".join(l2_block[:BATCH_SIZE_LIMIT]) + "\n... (截断: 节点过大)"
        else:
            current_batch_lines.extend(l2_block)

    if current_batch_lines:
        yield header_text + "\n" + "\n".join(current_batch_lines)

# --- Main Logic Class ---

class OverallStructureAudit:
    def __init__(self, env: Step4Env, tm: TreeManager, args):
        self.env = env
        self.llm_profile = env.primary_llm_profile()
        self.tm = tm
        self.args = args

        self.ops_log = env.outdir / "policy_tree_final_operations.jsonl"
        self.llm_log = env.log_dir / "llm_finalize_policy_tree.jsonl"
        self.audit_log = env.outdir / "policy_tree_final_audit.json"

        self.redirect_map: Dict[str, str] = {}
        self.audit_entries = []

        # 统计
        self.stats = {
            "llm_calls": 0,
            "llm_failures": 0,
            "ops_applied": 0,
            "ops_skipped": 0,
        }

    def run(self):
        # 1. 解决挂在 Root 下的非 L1 游离节点 (Pending L1)
        self._resolve_pending_root_nodes()

        # 2. 遍历所有 L1 进行审计
        l1_nodes = [n for n in self.tm.get_children(self.tm.root["node_id"])
                    if str(n.get("level", "")).upper() == "L1"]

        l1_defs = self._load_l1_defs()
        print(f"[Finalization] Auditing {len(l1_nodes)} L1 categories...")

        with self.ops_log.open("w", encoding="utf-8") as f:
            pass

        for l1 in l1_nodes:
            l1_id = l1["node_id"]
            definition = l1_defs.get(l1_id, "")

            for batch_ctx in generate_l1_batches(self.tm, l1_id, definition):
                ops = self._call_llm(batch_ctx, l1_id)
                if not ops:
                    continue

                applied = self._apply_operations(ops, l1_id)
                self.audit_entries.append({
                    "l1_id": l1_id,
                    "label": l1.get("label"),
                    "raw_ops": ops,
                    "applied": applied
                })
                for item in applied:
                    append_jsonl(self.ops_log, item)

        # 关键修复：在导出前强制重算层级
        self._realign_tree_levels()

        # 3. 导出结果
        dump_tree(Path(self.args.output), self.tm.root)
        self._export_flat_csv()

        # 4. 更新 Membership
        if self.redirect_map:
            self._update_final_membership()

        # 5. 保存审计报告
        Path(self.args.audit_out).write_text(
            json.dumps({
                "ts": int(time.time()),
                "entries": self.audit_entries,
                "stats": self.stats
            }, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )

        # 打印统计
        print(f"\n[Finalization Stats]")
        print(f"  LLM calls: {self.stats['llm_calls']} (failures: {self.stats['llm_failures']})")
        print(f"  Operations: applied={self.stats['ops_applied']}, skipped={self.stats['ops_skipped']}")
        print(f"[DONE] Audit Completed. Final Tree: {self.args.output}")

    def _realign_tree_levels(self):
        """
        DFS 遍历树，根据物理深度强制重写 level 属性。
        """
        print("[Finalization] Re-aligning node levels based on physical topology...")

        def dfs(node_id, current_depth):
            node = self.tm.get_node(node_id)
            if not node: return

            new_level_str = f"L{current_depth}"

            if current_depth > 0:
                node["level"] = new_level_str

            children = self.tm.get_children(node_id)
            for child in children:
                dfs(child["node_id"], current_depth + 1)

        root_children = self.tm.get_children(self.tm.root["node_id"])
        for l1_node in root_children:
            dfs(l1_node["node_id"], 1)
        print("[Finalization] Level realignment completed.")

    def _load_l1_defs(self) -> Dict[str, str]:
        p = Path(self.args.l1_def)
        if not p.exists(): return {}
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            return {str(item.get("id")): item.get("definition", "")
                    for item in data.get("categories", [])}
        except:
            return {}

    def _resolve_pending_root_nodes(self):
        root_children = self.tm.get_children(self.tm.root["node_id"])
        pending = [ch for ch in root_children if ch.get("pending_as_l1")]

        if not pending: return
        print(f"[Finalization] Resolving {len(pending)} pending root nodes...")

        for node in pending:
            node["level"] = "L1"
            node.pop("pending_as_l1", None)
            node.pop("original_level", None)
            node.pop("pending_parent", None)
            append_jsonl(self.ops_log, {
                "op": "pending_auto_promote", "node_id": node["node_id"], "reason": "Finalization Auto Fix"
            })

    def _call_llm(self, context: str, l1_id: str) -> List[Dict]:
        """
        调用 LLM 进行审计

        【修复】添加防御性检查，防止 resp 为 None 时崩溃
        """
        self.stats["llm_calls"] += 1

        try:
            instruction = PROMPT_PATH.read_text(encoding="utf-8")
        except FileNotFoundError:
            print(f"[ERROR] Prompt file not found: {PROMPT_PATH}")
            self.stats["llm_failures"] += 1
            return []

        # 调用 LLM
        try:
            resp = call_llm_json(
                profile=self.llm_profile,
                system=instruction,
                user=context,
                task="finalize_policy_tree",
            )
        except Exception as e:
            print(f"[ERROR] LLM call raised exception: {e}")
            self.stats["llm_failures"] += 1
            append_jsonl(self.llm_log, {
                "ts": int(time.time()), "l1_id": l1_id,
                "error": str(e), "resp": None
            })
            return []

        # 记录日志
        append_jsonl(self.llm_log, {
            "ts": int(time.time()), "l1_id": l1_id, "resp": resp
        })

        # =====================================================
        # 【核心修复】防御性检查：防止 resp 为 None 时崩溃
        # =====================================================
        if resp is None:
            print(f"[WARN] LLM call returned None for L1: {l1_id}. Skipping this batch.")
            self.stats["llm_failures"] += 1
            return []

        if not isinstance(resp, dict):
            print(f"[WARN] LLM response is not a dict: {type(resp)}. Skipping.")
            self.stats["llm_failures"] += 1
            return []

        if "json" not in resp:
            # 可能 LLM 返回了纯文本或解析失败
            print(f"[WARN] LLM response missing 'json' key. Raw snippet: {str(resp)[:200]}")
            self.stats["llm_failures"] += 1
            return []

        json_data = resp.get("json")
        if json_data is None:
            print(f"[WARN] resp['json'] is None. Skipping.")
            self.stats["llm_failures"] += 1
            return []

        if not isinstance(json_data, dict):
            print(f"[WARN] resp['json'] is not a dict: {type(json_data)}. Skipping.")
            self.stats["llm_failures"] += 1
            return []

        operations = json_data.get("operations", [])
        if not isinstance(operations, list):
            print(f"[WARN] 'operations' is not a list: {type(operations)}. Skipping.")
            self.stats["llm_failures"] += 1
            return []

        return operations

    def _apply_operations(self, ops: List[Dict], l1_id: str) -> List[Dict]:
        applied = []
        for op in ops:
            typ = op.get("type", "").lower()
            node_id = op.get("node_id")

            if not node_id or not self.tm.exists(node_id):
                applied.append({**op, "status": "skipped", "msg": "Node not found"})
                self.stats["ops_skipped"] += 1
                continue

            status = "applied"
            msg = ""
            try:
                if typ == "rename":
                    new_label = op.get("new_label")
                    if new_label:
                        self.tm.get_node(node_id)["label"] = new_label
                        self.stats["ops_applied"] += 1
                    else:
                        status, msg = "skipped", "missing new_label"
                        self.stats["ops_skipped"] += 1

                elif typ == "move":
                    target = op.get("target_parent_id")
                    if not target or not self.tm.exists(target):
                        status, msg = "skipped", "target not found"
                        self.stats["ops_skipped"] += 1
                    elif self.tm.is_descendant(target, node_id) if hasattr(self.tm, 'is_descendant') else False:
                        status, msg = "skipped", "target is descendant"
                        self.stats["ops_skipped"] += 1
                    else:
                        t_l1 = get_l1_ancestor(self.tm, target)
                        if t_l1 != l1_id:
                            status, msg = "skipped", "cross-L1 move forbidden"
                            self.stats["ops_skipped"] += 1
                        else:
                            self.tm.move_node(node_id, target)
                            self.stats["ops_applied"] += 1

                elif typ == "merge":
                    target = op.get("merge_into")
                    if not target or not self.tm.exists(target):
                        status, msg = "skipped", "target not found"
                        self.stats["ops_skipped"] += 1
                    elif target == node_id:
                        status, msg = "skipped", "merge into self"
                        self.stats["ops_skipped"] += 1
                    else:
                        t_l1 = get_l1_ancestor(self.tm, target)
                        if t_l1 != l1_id:
                            status, msg = "skipped", "cross-L1 merge forbidden"
                            self.stats["ops_skipped"] += 1
                        else:
                            if self.tm.absorb_node(target, node_id):
                                self.redirect_map[node_id] = target
                                self.stats["ops_applied"] += 1
                            else:
                                status, msg = "failed", "absorb failed"
                                self.stats["ops_skipped"] += 1
                else:
                    status, msg = "skipped", "unknown type"
                    self.stats["ops_skipped"] += 1
            except Exception as e:
                status, msg = "error", str(e)
                self.stats["ops_skipped"] += 1
            applied.append({**op, "status": status, "message": msg})
        return applied

    def _update_final_membership(self):
        csv_path = self.env.outdir / "policy_tree_final_membership.csv"
        if not csv_path.exists():
            return
        print(f"[Finalization] Updating membership trace with {len(self.redirect_map)} ops...")
        df = pd.read_csv(csv_path, dtype=str)
        count = 0
        for idx, row in df.iterrows():
            curr = str(row.get("final_node_id", ""))
            if curr in self.redirect_map:
                while curr in self.redirect_map:
                    curr = self.redirect_map[curr]
                df.at[idx, "final_node_id"] = curr
                count += 1
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    def _export_flat_csv(self):
        rows = []
        cols = ["L1_id", "L1_label", "L2_id", "L2_label", "L3_id", "L3_label", "L4_id", "L4_label"]

        def dfs(node_id, path_stack):
            node = self.tm.get_node(node_id)

            # 【修复】防御性检查：防止幽灵节点
            if not node:
                return

            new_stack = path_stack + [node]

            children = [c for c in self.tm.get_children(node_id)
                        if str(c.get("level", "")).upper().startswith("L")]

            if not children:
                row = {c: "" for c in cols}
                for n in new_stack:
                    if not n: continue  # 再次防御
                    nlvl = str(n.get("level", "")).upper()
                    if f"{nlvl}_id" in row:
                        row[f"{nlvl}_id"] = n["node_id"]
                        row[f"{nlvl}_label"] = n.get("label", "")
                rows.append(row)

            for ch in children:
                dfs(ch["node_id"], new_stack)

        l1_nodes = [n for n in self.tm.get_children(self.tm.root["node_id"])
                    if str(n.get("level", "")).upper() == "L1"]
        for l1 in l1_nodes:
            dfs(l1["node_id"], [])

        pd.DataFrame(rows, columns=cols).to_csv(
            self.args.flat_csv, index=False, encoding="utf-8-sig"
        )

def main():
    parser = argparse.ArgumentParser(description="PolicyTreeBuilder final replication · Finalization Overall Structure (Fixed)")
    parser.add_argument("--input", required=True, help="policy_tree_refined.json")
    parser.add_argument("--output", required=True, help="policy_tree_final.json")
    parser.add_argument("--config", required=True, help="tree_refinement_config.yaml")

    parser.add_argument("--l1-def", default=str(DEFAULT_L1_DEF))
    parser.add_argument("--audit-out", default=str(DEFAULT_AUDIT))
    parser.add_argument("--flat-csv", default=str(DEFAULT_FLAT))

    args = parser.parse_args()

    # 1. 初始化
    env = Step4Env(args.config, str(ENV_PATH))

    # 2. 加载数据
    raw_tree = load_tree(Path(args.input))
    tm = TreeManager(raw_tree)

    # 3. 执行
    audit = OverallStructureAudit(env, tm, args)
    audit.run()

if __name__ == "__main__":
    main()
