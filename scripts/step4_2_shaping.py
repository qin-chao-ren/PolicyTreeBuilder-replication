#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 4.2 · Shaping (Refactored)
功能：结构塑形（层级跳跃修复 + 扇出平衡）
架构：
  - 依赖 utils.tree_manager 维护动态拓扑
  - 依赖 utils.step4_shared 处理环境与IO
  - 迭代式修复：while changed -> loop，直到结构稳定
"""

import argparse
import time
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple

from common_utils import call_json
from utils.step4_shared import (
    Step4Env, 
    load_tree, 
    dump_tree, 
    append_jsonl
)
from utils.tree_manager import TreeManager

PROMPT_BALANCE = Path("prompts/step4_balance_structure.md")

LEVEL_MAP = {"L1": 1, "L2": 2, "L3": 3, "L4": 4}

def get_level_num(lvl: str) -> int:
    return LEVEL_MAP.get(str(lvl).upper(), 99)

def get_next_level(lvl: str) -> str:
    n = min(get_level_num(lvl) + 1, 4)
    return f"L{n}"

def generate_bridge_id(parent_id: str, label: str, suffix: str = "BR") -> str:
    """生成稳定的中间节点 ID"""
    raw = f"{parent_id}_{label}_{suffix}"
    # 使用 Hash 缩短 ID 长度，保持整洁
    h = hashlib.md5(raw.encode()).hexdigest()[:6]
    return f"{parent_id}_{suffix}_{h}"

def describe_node_simple(node: Dict, child_count: int) -> str:
    return (
        f"ID={node['node_id']} · level={node.get('level')} · label={node.get('label','')}\n"
        f"子节点数量={child_count}"
    )

class ShapingProcess:
    def __init__(self, env: Step4Env, tm: TreeManager):
        self.env = env
        self.llm = env.build_llm_config()
        self.tm = tm
        self.ops_log = env.outdir / "v4_operations_log.jsonl"
        self.llm_log = env.log_dir / "llm_step4_shaping.jsonl"
        
        # Trace Map 在 4.2 通常为空，但为了兼容性保留
        self.trace_map = {} 

    def run(self):
        # 1. 修复层级跳跃 (L2 -> L4)
        print("[Step 4.2] Starting Jump Fix...")
        self._iterate_fix(self._check_and_fix_jump, "jump_fix")

        # 2. 修复过大扇出 (>7)
        print("[Step 4.2] Starting Fanout Balance...")
        self._iterate_fix(self._check_and_fix_fanout, "fanout_balance")

    def _iterate_fix(self, fix_func, stage_name: str, max_rounds=5):
        """
        通用迭代修复循环
        由于修复操作（插入节点、移动节点）会改变树结构，
        我们需要多次扫描直到没有变更，或者达到最大轮数。
        """
        for round_idx in range(max_rounds):
            changed = False
            # 获取当前所有非叶子节点 ID 的快照
            candidates = [nid for nid in self.tm.get_all_node_ids() if self.tm.get_children(nid)]
            
            print(f"  > Round {round_idx+1}: Scanning {len(candidates)} nodes...")
            
            for parent_id in candidates:
                # 动态检查：节点可能在上一轮循环被改变
                if not self.tm.exists(parent_id): continue
                
                # 执行具体的修复逻辑
                if fix_func(parent_id):
                    changed = True
            
            if not changed:
                print(f"  > Round {round_idx+1}: Converged.")
                break

    def _format_context(self, parent_id: str, children: List[Dict], scenario: str) -> str:
        parent = self.tm.get_node(parent_id)
        child_desc = "\n".join([
            f"- {ch['node_id']} · {ch.get('label','')} · level={ch.get('level')}"
            for ch in children
        ])
        return (
            f"# 场景：{scenario}\n"
            f"# 父节点\n{describe_node_simple(parent, len(children))}\n"
            f"# 子节点列表 (Top {len(children)})\n{child_desc}\n"
        )

    # --- Logic: Jump Fix ---
    
    def _check_and_fix_jump(self, parent_id: str) -> bool:
        parent = self.tm.get_node(parent_id)
        p_level_num = get_level_num(parent.get("level", ""))
        
        # 找出跳跃的子节点 (Gap > 1)
        jump_children = []
        for ch in self.tm.get_children(parent_id):
            c_level_num = get_level_num(ch.get("level", ""))
            if c_level_num - p_level_num > 1:
                jump_children.append(ch)
        
        if not jump_children:
            return False

        # 构造 Prompt
        scenario = f"层级跳跃 level_gap > 1 (父{parent.get('level')} -> 子{[c.get('level') for c in jump_children]})"
        context = self._format_context(parent_id, jump_children, scenario)
        
        resp = call_json(self.llm.primary, PROMPT_BALANCE.read_text(encoding="utf-8"), context)
        
        append_jsonl(self.llm_log, {
            "ts": int(time.time()), "case": "jump_fix", "parent": parent_id,
            "children_count": len(jump_children), "resp": resp
        })
        
        return self._apply_llm_decision(parent_id, resp.get("json", {}), "jump_fix")

    # --- Logic: Fanout Balance ---

    def _check_and_fix_fanout(self, parent_id: str) -> bool:
        children = self.tm.get_children(parent_id)
        if len(children) <= 7:
            return False
            
        # 构造 Prompt
        scenario = f"扇出过大 (Children={len(children)} > 7)，考虑分组"
        # 仅取前 25 个子节点放入 Prompt，避免 Token 爆炸
        context = self._format_context(parent_id, children[:25], scenario)
        
        resp = call_json(self.llm.primary, PROMPT_BALANCE.read_text(encoding="utf-8"), context)
        
        append_jsonl(self.llm_log, {
            "ts": int(time.time()), "case": "fanout_balance", "parent": parent_id,
            "fanout": len(children), "resp": resp
        })
        
        return self._apply_llm_decision(parent_id, resp.get("json", {}), "fanout_balance")

    # --- Logic: Apply Actions ---

    def _apply_llm_decision(self, parent_id: str, result: Dict, stage: str) -> bool:
        action = result.get("action", "keep")
        parent_node = self.tm.get_node(parent_id)
        did_change = False
        
        # Action 1: Insert Bridge (单子节点桥接) or Create Groups (多子节点分组)
        if action in ["insert_bridge", "create_groups"]:
            groups = result.get("groups") or []
            
            for grp in groups:
                bridge_label = grp.get("bridge_label")
                target_child_ids = grp.get("child_ids") or []
                
                # 过滤：确保子节点确实在当前父节点下
                valid_children = [
                    cid for cid in target_child_ids 
                    if self.tm.get_parent_id(cid) == parent_id
                ]
                
                if not valid_children or not bridge_label:
                    continue
                
                # 1. 创建 Bridge 节点
                new_id = generate_bridge_id(parent_id, bridge_label)
                new_level = get_next_level(parent_node.get("level", "L1"))
                
                new_node = {
                    "node_id": new_id,
                    "label": bridge_label,
                    "level": new_level,
                    "children": [] # TreeManager 会处理
                }
                
                if self.tm.add_child_node(parent_id, new_node):
                    # 2. 将子节点移动到 Bridge 下
                    for cid in valid_children:
                        self.tm.move_node(cid, new_id)
                    
                    did_change = True
                    append_jsonl(self.ops_log, {
                        "ts": int(time.time()), "step": "step4_2", "stage": stage,
                        "op": "create_bridge", "parent": parent_id, "bridge": new_id, 
                        "children_moved": len(valid_children)
                    })

        # Action 2: Lift Children (提升子节点到爷爷下)
        lift_ids = result.get("lift_children") or []
        if lift_ids:
            for cid in lift_ids:
                if self.tm.get_parent_id(cid) != parent_id: continue
                
                # 使用 Safe Promote
                if self.tm.promote_child_safe(cid):
                    did_change = True
                    append_jsonl(self.ops_log, {
                        "ts": int(time.time()), "step": "step4_2", "stage": stage,
                        "op": "lift_child", "child": cid, "old_parent": parent_id
                    })

        return did_change

def main():
    parser = argparse.ArgumentParser(description="Round C v4 · Step4.2 Shaping")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    # 1. 初始化
    env = Step4Env(args.config, "configs/roundC_v4.env")
    
    # 2. 加载数据
    raw_tree = load_tree(Path(args.input))
    tm = TreeManager(raw_tree)
    
    # 3. 执行处理
    processor = ShapingProcess(env, tm)
    processor.run()
    
    # 4. 导出
    dump_tree(Path(args.output), tm.root)
    
    # 导出空 Trace 保持兼容 (4.2 主要是 Move/Add，没有 Merge 导致的 ID 消失)
    trace_path = env.outdir / "v4_trace_4_2.json"
    trace_path.write_text("{}", encoding="utf-8")
    
    print(f"[DONE] Shaping Completed. Tree saved to {args.output}")

if __name__ == "__main__":
    main()
