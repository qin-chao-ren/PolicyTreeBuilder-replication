#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trace ID Lineage Tool
功能：通过重放操作日志，建立从 [原始样本 ID] -> [最终树节点 ID] 的映射表。
作用：为 Step 5 挂接 T5 行动单元提供“寻址地图”。
"""
import json
import csv
import argparse
from pathlib import Path
from typing import Dict, List, Optional

# 默认路径配置
DEFAULT_OUTDIR = Path("data/intermediate_outputs")
CORPUS_FILE = DEFAULT_OUTDIR / "v4_cluster_corpus_cleaned.csv"
MEMBERSHIP_FILES = [
    DEFAULT_OUTDIR / "v4_membership_L4.csv",
    DEFAULT_OUTDIR / "v4_membership_L3.csv",
    DEFAULT_OUTDIR / "v4_membership_L2.csv"
]
OPS_LOG_FILE = DEFAULT_OUTDIR / "v4_operations_log.jsonl"
FINAL_TREE_FILE = DEFAULT_OUTDIR / "v4_tree_final.json"
OUTPUT_FILE = DEFAULT_OUTDIR / "v4_id_lineage_trace.csv"

class LineageTracker:
    def __init__(self):
        # redirects: source_id -> target_id
        self.redirects: Dict[str, str] = {}
        # final_paths: node_id -> readable_path_string
        self.final_paths: Dict[str, str] = {}
        # valid_final_ids: set of IDs that actually exist in the final tree
        self.valid_final_ids = set()

    def register_redirect(self, loser_id: str, winner_id: str):
        """
        注册重定向关系，并处理传递性 (Transitivity)。
        如果 A->B, 现在来了一个 B->C, 那么我们需要更新表里的 A->C。
        """
        if loser_id == winner_id:
            return

        # 1. 记录当前的直接映射
        self.redirects[loser_id] = winner_id

        # 2. 更新所有指向 loser_id 的旧映射 (链式更新)
        # 例如：之前有 X->loser_id，现在 loser_id 变成了 winner_id，
        # 那么 X 应该直接指向 winner_id
        for src, tgt in self.redirects.items():
            if tgt == loser_id:
                self.redirects[src] = winner_id

    def resolve_id(self, original_id: str) -> str:
        """查询 ID 的最终去向，如果没变则返回自身"""
        return self.redirects.get(original_id, original_id)

    def load_final_tree(self, tree_path: Path):
        """加载最终树，建立 ID -> Path 的索引，用于验证节点是否存活"""
        if not tree_path.exists():
            raise FileNotFoundError(f"Tree file not found: {tree_path}")
        
        with open(tree_path, "r", encoding="utf-8") as f:
            root = json.load(f)
        
        def dfs(node, path_stack):
            node_id = node.get("node_id")
            label = node.get("label", "ROOT")
            current_path = path_stack + [label]
            
            if node_id:
                self.valid_final_ids.add(node_id)
                self.final_paths[node_id] = " / ".join(current_path)
            
            for child in node.get("children", []):
                dfs(child, current_path)

        dfs(root, [])
        print(f"[INFO] Loaded final tree with {len(self.valid_final_ids)} nodes.")

    def replay_logs(self, log_path: Path):
        """核心逻辑：重放操作日志"""
        if not log_path.exists():
            print(f"[WARN] Log file not found: {log_path}. Assuming no structural changes.")
            return

        count = 0
        with open(log_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                try:
                    entry = json.loads(line)
                    self._process_log_entry(entry)
                    count += 1
                except json.JSONDecodeError:
                    continue
        print(f"[INFO] Replayed {count} operations from log.")

    def _process_log_entry(self, entry: Dict):
        op = entry.get("op", "").lower()
        
        # === Case A: Step 4.1/4.3 Standard Merge ===
        # 字段通常是: winner, loser
        if op in ["merge", "cross_merge", "absorb_child"]:
            # 尝试获取 loser
            loser = entry.get("loser") or entry.get("child") # absorb_child uses 'child'
            # 尝试获取 winner
            winner = entry.get("winner") or entry.get("parent") # absorb_child uses 'parent'
            
            # === Case B: Step 4.5 Structure Audit ===
            # 字段是: node_id (被合并者), merge_into (目标)
            if not loser and not winner:
                # Step 4.5 的 merge 逻辑
                if op == "merge":
                    loser = entry.get("node_id")
                    # 这里的 details 可能是 Step 4.5 记录 payload 的方式，也可能直接在顶层
                    # 根据之前的代码逻辑，可能是 entry.get('merge_into') 或者 entry['details']['merge_into']
                    # 我们做一个鲁棒性查找
                    winner = entry.get("merge_into")
                    if not winner and "details" in entry:
                        winner = entry["details"].get("merge_into")

            if loser and winner:
                self.register_redirect(str(loser), str(winner))

def main():
    # 1. 加载 Corpus (基准)
    print("[1/5] Loading Sample Corpus...")
    sample_map = {}
    if CORPUS_FILE.exists():
        with open(CORPUS_FILE, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                sid = row.get("sample_id")
                if sid:
                    sample_map[sid] = {
                        "title": row.get("title_text", ""),
                        "initial_node": None # 待填充
                    }
    else:
        print("[ERROR] Corpus file missing!")
        return

    # 2. 绑定初始归属 (Membership)
    print("[2/5] Binding Initial Membership...")
    for m_file in MEMBERSHIP_FILES:
        if not m_file.exists(): continue
        with open(m_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            # 假设 csv 有 node_id, sample_ids (分号分隔) 或直接 member_id
            # 根据 v3/v4 惯例，L4 membership 通常是 node_id, sample_id
            for row in reader:
                node_id = row.get("node_id")
                # 有些 membership 表可能是以 node_id 为主键，列出 members string
                # 有些是每一行一个 sample。这里假设是平铺的或者需要解析
                # 这里为了通用，假设如果有 member_id 列，则它是 sample_id
                # 如果没有，检查 members 列
                if "member_id" in row:
                    sid = row["member_id"]
                    if sid in sample_map:
                        sample_map[sid]["initial_node"] = node_id
                elif "sample_id" in row:
                    sid = row["sample_id"]
                    if sid in sample_map:
                        sample_map[sid]["initial_node"] = node_id
                elif "members" in row:
                    # 聚合格式
                    members = row["members"].split(";")
                    for m in members:
                        m = m.strip()
                        if m in sample_map:
                            sample_map[m]["initial_node"] = node_id

    # 3. 重放日志 (State Machine)
    print("[3/5] Replaying Operations Log...")
    tracker = LineageTracker()
    tracker.replay_logs(OPS_LOG_FILE)

    # 4. 加载最终树
    print("[4/5] Parsing Final Tree...")
    tracker.load_final_tree(FINAL_TREE_FILE)

    # 5. 生成最终映射表
    print("[5/5] Generating Lineage Report...")
    success_count = 0
    orphan_count = 0
    
    with open(OUTPUT_FILE, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "sample_id", 
            "original_title", 
            "initial_node_id", 
            "final_node_id", 
            "final_path_label", 
            "status"
        ])

        for sid, meta in sample_map.items():
            init_id = meta["initial_node"]
            
            if not init_id:
                # 可能是 Step 1.5 过滤掉的 T0
                writer.writerow([sid, meta["title"], "N/A", "N/A", "N/A", "FILTERED_EARLY"])
                continue

            # 核心：解析最终 ID
            final_id = tracker.resolve_id(init_id)
            
            # 检查状态
            if final_id in tracker.valid_final_ids:
                path_str = tracker.final_paths[final_id]
                status = "ACTIVE"
                success_count += 1
            else:
                # ID 存在，但在最终树里找不到 -> 孤儿 (被删除了或者合并逻辑有断层)
                path_str = "N/A"
                status = "ORPHAN"
                orphan_count += 1
            
            writer.writerow([sid, meta["title"], init_id, final_id, path_str, status])

    print("-" * 30)
    print(f"Lineage Trace Complete: {OUTPUT_FILE}")
    print(f"  - Successfully Linked: {success_count}")
    print(f"  - Orphans (Deleted/Lost): {orphan_count}")
    if orphan_count > 0:
        print("  [WARN] Orphans detected. This means some nodes were deleted or merge logic had gaps.")

if __name__ == "__main__":
    main()
    