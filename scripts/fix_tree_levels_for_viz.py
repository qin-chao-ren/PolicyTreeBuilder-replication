#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility Script · Fix Tree Levels for Visualization
强制重置树的 Level 属性：
- 根节点的子节点 -> L1
- L1 的子节点 -> L2
- L2 的子节点 -> L3
- 以此类推...
目的：解决可视化时因层级跳跃（Skip-level）导致的连线混乱问题。

python scripts/fix_tree_levels_for_viz.py `
--input "data/intermediate_outputs/v4_tree_refined.json" `
--output "data/intermediate_outputs/v4_tree_viz_ready.json"
"""


import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

def load_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))

def save_json(path: Path, data: Dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

def node_children(node: Dict) -> List[Dict]:
    return node.get("children", []) or []

def fix_recursive(node: Dict, parent_id: Optional[str], current_depth: int, stats: Dict[str, int]):
    """
    递归遍历：
    1. 修正当前节点的 parent_id
    2. 修正当前节点的 level
    3. 递归处理子节点
    """
    # --- 1. 修正 Root 节点 ---
    if current_depth == 0:
        node["level"] = "ROOT"
        node["parent_id"] = None  # 根节点没有父节点
        # 确保根节点 ID 存在，方便可视化软件识别
        if not node.get("node_id"):
            node["node_id"] = "TREE_ROOT"
            node["label"] = "全部分类"
    
    # --- 2. 修正普通节点 (L1 - L4) ---
    else:
        # 强制计算目标层级：Depth 1 -> L1, Depth 2 -> L2...
        target_level = f"L{current_depth}"
        
        # 更新 Level
        node["level"] = target_level
        
        # [关键] 更新 Parent ID
        # 很多可视化错误是因为 parent_id 还是旧的（指向了更上层的祖先），导致连线跨层。
        # 这里我们强制把它指向当前的物理父节点。
        if parent_id:
            node["parent_id"] = parent_id
            
        stats["processed"] += 1

    # --- 3. 递归 ---
    # 获取当前节点的 ID，作为下一层的 parent_id
    current_id = node.get("node_id")
    
    # 防御性检查：如果节点没有 children 字段，初始化为空列表
    if "children" not in node:
        node["children"] = []
        
    children = node_children(node)
    
    # 如果层级过深 (超过 L4)，虽然保留数据，但标记一下警告（可选）
    if current_depth >= 4 and children:
        stats["deep_nodes"] += len(children)
        
    for child in children:
        fix_recursive(child, current_id, current_depth + 1, stats)

def main():
    parser = argparse.ArgumentParser(description="Strictly fix tree hierarchy and parent_ids")
    parser.add_argument("--input", required=True, help="Input JSON (e.g., v4_tree_refined.json)")
    parser.add_argument("--output", required=True, help="Output JSON for Viz")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"[Error] Input file not found: {input_path}")
        return

    print(f"Reading tree from: {input_path}")
    root = load_json(input_path)

    stats = {"processed": 0, "deep_nodes": 0}
    
    # 启动递归：深度 0，无父节点
    fix_recursive(root, None, 0, stats)

    print("-" * 30)
    print(f"Processing Complete:")
    print(f" - Nodes Re-linked: {stats['processed']}")
    if stats['deep_nodes'] > 0:
        print(f" - [WARN] Nodes deeper than L4 found: {stats['deep_nodes']} (Converted to L5+)")
    print("-" * 30)
    
    save_json(output_path, root)
    print(f"Fixed tree saved to: {output_path}")

if __name__ == "__main__":
    main()