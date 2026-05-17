#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 4.5 · Overall Structure (Refactored)
功能：L1 级整体结构审计 (Semantic Audit) + 最终层级强制对齐
架构：
  - 依赖 utils.tree_manager 维护动态拓扑
  - 依赖 utils.step4_shared 处理环境与IO
  - 核心逻辑：分批次 LLM 审计 -> TreeManager 执行 -> 全局层级重算 -> 导出
"""

import argparse
import time
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Generator, Tuple
from difflib import SequenceMatcher

# 保持与你提供的一致
from common_llm import call_json
from utils.step4_shared import (
    Step4Env, 
    load_tree, 
    dump_tree, 
    append_jsonl
)
from utils.tree_manager import TreeManager

# --- 1. 路径锚点 (Path Anchors) ---
# 无论在哪里运行命令，__file__ 都能定位到 scripts/ 目录
HERE = Path(__file__).resolve().parent 
PROJECT_ROOT = HERE.parent  # roundC_v4/

# --- 2. 关键资源路径 (基于锚点) ---
# 这样写绝对不会错
ENV_PATH = PROJECT_ROOT / "configs" / "roundC_v4.env"

# --- 3. Prompt 路径 ---
PROMPT_PATH = PROJECT_ROOT / "prompts" / "step4_5_override.md"

# --- 4. 默认输出/输入路径 ---
DEFAULT_L1_DEF = PROJECT_ROOT / "outputs" / "v4_l1_definition.json"
DEFAULT_OPS_OUT = PROJECT_ROOT / "outputs" / "v4_final_ops.jsonl"
DEFAULT_AUDIT = PROJECT_ROOT / "outputs" / "v4_final_audit.json"
DEFAULT_FLAT = PROJECT_ROOT / "outputs" / "v4_tree_final_flat.csv"

BATCH_SIZE_LIMIT = 200

# --- Helper Functions (保持不变) ---

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
        self.llm = env.build_llm_config()
        self.tm = tm
        self.args = args
        
        self.ops_log = env.outdir / "v4_final_ops.jsonl"
        self.llm_log = env.log_dir / "llm_step4_5_overall.jsonl"
        self.audit_log = env.outdir / "v4_final_audit.json"
        
        self.redirect_map: Dict[str, str] = {}
        self.audit_entries = []

    def run(self):
        # 1. 解决挂在 Root 下的非 L1 游离节点 (Pending L1)
        self._resolve_pending_root_nodes()
        
        # 2. 遍历所有 L1 进行审计
        l1_nodes = [n for n in self.tm.get_children(self.tm.root["node_id"]) 
                    if str(n.get("level", "")).upper() == "L1"]
        
        l1_defs = self._load_l1_defs()
        print(f"[Step 4.5] Auditing {len(l1_nodes)} L1 categories...")
        
        with self.ops_log.open("w", encoding="utf-8") as f:
            pass 

        for l1 in l1_nodes:
            l1_id = l1["node_id"]
            definition = l1_defs.get(l1_id, "")
            
            for batch_ctx in generate_l1_batches(self.tm, l1_id, definition):
                ops = self._call_llm(batch_ctx, l1_id)
                if not ops: continue
                
                applied = self._apply_operations(ops, l1_id)
                self.audit_entries.append({
                    "l1_id": l1_id, 
                    "label": l1.get("label"), 
                    "raw_ops": ops, 
                    "applied": applied
                })
                for item in applied:
                    append_jsonl(self.ops_log, item)

        # =========================================================
        # 【新增】关键修复：在导出前强制重算层级，解决 L3->L2 倒挂问题
        # =========================================================
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
                "entries": self.audit_entries
            }, ensure_ascii=False, indent=2), 
            encoding="utf-8"
        )
        print(f"[DONE] Audit Completed. Final Tree: {self.args.output}")

    def _realign_tree_levels(self):
        """
        DFS 遍历树，根据物理深度强制重写 level 属性。
        规则：ROOT 的子节点强制为 L1，其子节点为 L2，以此类推。
        """
        print("[Step 4.5] Re-aligning node levels based on physical topology...")
        
        def dfs(node_id, current_depth):
            node = self.tm.get_node(node_id)
            if not node: return
            
            # 计算新的 level 标签 (例如 Depth 1 -> L1)
            new_level_str = f"L{current_depth}"
            
            # 更新节点属性 (跳过 Root)
            if current_depth > 0:
                # 可选：如果 level 发生变化，可以打印日志
                # old_level = node.get("level", "unknown")
                # if old_level != new_level_str:
                #     print(f"  Fixing {node_id}: {old_level} -> {new_level_str}")
                node["level"] = new_level_str
            
            # 递归处理子节点
            children = self.tm.get_children(node_id)
            for child in children:
                dfs(child["node_id"], current_depth + 1)
        
        # 从 Root 开始
        # Root 的直接子节点是 L1 (Depth=1)
        root_children = self.tm.get_children(self.tm.root["node_id"])
        for l1_node in root_children:
            dfs(l1_node["node_id"], 1)
        print("[Step 4.5] Level realignment completed.")

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
        print(f"[Step 4.5] Resolving {len(pending)} pending root nodes...")
        
        for node in pending:
            node["level"] = "L1"
            node.pop("pending_as_l1", None)
            node.pop("original_level", None)
            node.pop("pending_parent", None)
            append_jsonl(self.ops_log, {
                "op": "pending_auto_promote", "node_id": node["node_id"], "reason": "Step 4.5 Auto Fix"
            })

    def _call_llm(self, context: str, l1_id: str) -> List[Dict]:
        instruction = PROMPT_PATH.read_text(encoding="utf-8")
        resp = call_json(self.llm.primary, instruction, context)
        append_jsonl(self.llm_log, {
            "ts": int(time.time()), "l1_id": l1_id, "resp": resp
        })
        return resp.get("json", {}).get("operations", [])

    def _apply_operations(self, ops: List[Dict], l1_id: str) -> List[Dict]:
        applied = []
        for op in ops:
            typ = op.get("type", "").lower()
            node_id = op.get("node_id")
            
            if not node_id or not self.tm.exists(node_id):
                applied.append({**op, "status": "skipped", "msg": "Node not found"})
                continue
                
            status = "applied"
            msg = ""
            try:
                if typ == "rename":
                    new_label = op.get("new_label")
                    if new_label:
                        self.tm.get_node(node_id)["label"] = new_label
                    else:
                        status, msg = "skipped", "missing new_label"
                        
                elif typ == "move":
                    target = op.get("target_parent_id")
                    if not target or not self.tm.exists(target):
                        status, msg = "skipped", "target not found"
                    elif self.tm.is_descendant(target, node_id) if hasattr(self.tm, 'is_descendant') else False:
                         status, msg = "skipped", "target is descendant"
                    else:
                        t_l1 = get_l1_ancestor(self.tm, target)
                        if t_l1 != l1_id:
                            status, msg = "skipped", "cross-L1 move forbidden"
                        else:
                            self.tm.move_node(node_id, target)
                            
                elif typ == "merge":
                    target = op.get("merge_into")
                    if not target or not self.tm.exists(target):
                        status, msg = "skipped", "target not found"
                    elif target == node_id:
                        status, msg = "skipped", "merge into self"
                    else:
                        t_l1 = get_l1_ancestor(self.tm, target)
                        if t_l1 != l1_id:
                            status, msg = "skipped", "cross-L1 merge forbidden"
                        else:
                            if self.tm.absorb_node(target, node_id):
                                self.redirect_map[node_id] = target
                            else:
                                status, msg = "failed", "absorb failed"
                else:
                    status, msg = "skipped", "unknown type"
            except Exception as e:
                status, msg = "error", str(e)
            applied.append({**op, "status": status, "message": msg})
        return applied

    def _update_final_membership(self):
        csv_path = self.env.outdir / "v4_final_membership.csv"
        if not csv_path.exists():
            return
        print(f"[Step 4.5] Updating membership trace with {len(self.redirect_map)} ops...")
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
        # 注意：这里的 cols 假设最多到 L4，如果层级超过 L4，csv 只会记录到 L4
        # 但这通常符合业务需求
        cols = ["L1_id", "L1_label", "L2_id", "L2_label", "L3_id", "L3_label", "L4_id", "L4_label"]
        
        def dfs(node_id, path_stack):
            node = self.tm.get_node(node_id)
            
            # 入栈
            new_stack = path_stack + [node]
            
            children = [c for c in self.tm.get_children(node_id) 
                        if str(c.get("level", "")).upper().startswith("L")]
            
            # 如果是叶子或者层级很深了，记录一行
            if not children:
                row = {c: "" for c in cols}
                for n in new_stack:
                    nlvl = str(n.get("level", "")).upper()
                    # 动态匹配 L1_id, L2_id 等
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
    parser = argparse.ArgumentParser(description="Round C v4 · Step4.5 Overall Structure")
    parser.add_argument("--input", required=True, help="v4_tree_refined.json")
    parser.add_argument("--output", required=True, help="v4_tree_final.json")
    parser.add_argument("--config", required=True, help="step4_config.yaml")
    
    # 使用基于锚点的默认路径
    parser.add_argument("--l1-def", default=str(DEFAULT_L1_DEF))
    parser.add_argument("--audit-out", default=str(DEFAULT_AUDIT))
    parser.add_argument("--flat-csv", default=str(DEFAULT_FLAT))
    
    args = parser.parse_args()

    # 1. 初始化 (传入绝对路径 ENV_PATH)
    env = Step4Env(args.config, str(ENV_PATH))
    
    # 2. 加载数据
    raw_tree = load_tree(Path(args.input))
    tm = TreeManager(raw_tree)
    
    # 3. 执行 (类内部逻辑包含了 _realign_tree_levels)
    audit = OverallStructureAudit(env, tm, args)
    audit.run()

if __name__ == "__main__":
    main()
