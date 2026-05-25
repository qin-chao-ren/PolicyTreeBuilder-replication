#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 4.2 · Shaping (Fixed Version + Stable Paths)
功能：结构塑形
修复内容：深度压平、提升为兄弟、层级跳跃修复
"""

import argparse
import time
import hashlib
from pathlib import Path
from typing import Dict, List

from common_llm import call_json
from utils.step4_shared import (
    Step4Env, load_tree, dump_tree, append_jsonl
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
PROMPT_BALANCE = PROJECT_ROOT / "prompts" / "balance_tree_structure.md"

# ==========================================

# 常量配置
LEVEL_MAP = {"L1": 1, "L2": 2, "L3": 3, "L4": 4}
MAX_DEPTH = 4
MAX_FANOUT = 7
MAX_ROUNDS = 10

def get_next_level(lvl):
    n = min(LEVEL_MAP.get(str(lvl).upper(), 99) + 1, 4)
    return f"L{n}"

def generate_bridge_id(pid, lbl):
    h = hashlib.md5(f"{pid}_{lbl}_BR".encode()).hexdigest()[:6]
    return f"{pid}_BR_{h}"

def describe_node(node, count):
    return f"ID={node['node_id']} · level={node.get('level')} · label={node.get('label','')}\n子节点数={count}"

def calc_depth(tm, nid):
    d = 0
    curr = nid
    while curr:
        p = tm.get_parent_id(curr)
        if not p: break
        d += 1
        curr = p
    return d

class ShapingProcess:
    def __init__(self, env: Step4Env, tm: TreeManager):
        self.env = env
        self.llm = env.build_llm_config()
        self.tm = tm
        self.ops_log = env.outdir / "tree_edit_operations.jsonl"
        self.llm_log = env.log_dir / "llm_balance_tree_structure.jsonl"
        self.trace_map = {}

    def run(self):
        print("[Step 4.2] Starting Jump Fix...")
        self._iterate(self._fix_jump, "jump_fix")

        print("[Step 4.2] Starting Fanout Balance...")
        self._iterate(self._fix_fanout, "fanout_balance")

        print("[Step 4.2] Starting Depth Flattening (New)...")
        self._iterate(self._fix_depth, "depth_flatten")

    def _iterate(self, func, name):
        for i in range(MAX_ROUNDS):
            changed = False
            candidates = [n for n in self.tm.get_all_node_ids() if self.tm.get_children(n)]
            print(f"  > Pass {i+1}: Scanning {len(candidates)} nodes...")
            for pid in candidates:
                if not self.tm.exists(pid): continue
                if func(pid): changed = True
            if not changed: break

    def _fmt_ctx(self, pid, children, scene):
        p = self.tm.get_node(pid)
        c_desc = "\n".join([f"- {c['node_id']} · {c.get('label','')} · level={c.get('level')}" for c in children])
        return f"# 场景：{scene}\n# 父节点\n{describe_node(p, len(children))}\n# 子节点\n{c_desc}\n"

    # 逻辑1：层级跳跃
    def _fix_jump(self, pid):
        p = self.tm.get_node(pid)
        p_lvl = LEVEL_MAP.get(str(p.get("level")).upper(), 99)
        jumps = [c for c in self.tm.get_children(pid)
                 if LEVEL_MAP.get(str(c.get("level")).upper(), 99) - p_lvl > 1]

        if not jumps: return False

        ctx = self._fmt_ctx(pid, jumps, "层级跳跃(Level Gap > 1)")
        resp = call_json(self.llm.primary, PROMPT_BALANCE.read_text(encoding="utf-8"), ctx)
        append_jsonl(self.llm_log, {"ts":int(time.time()), "case":"jump", "parent":pid, "resp":resp})
        return self._apply(pid, resp.get("json", {}), "jump")

    # 逻辑2：扇出过大
    def _fix_fanout(self, pid):
        children = self.tm.get_children(pid)
        if len(children) <= MAX_FANOUT: return False

        ctx = self._fmt_ctx(pid, children[:30], f"扇出过大({len(children)}>{MAX_FANOUT})")
        resp = call_json(self.llm.primary, PROMPT_BALANCE.read_text(encoding="utf-8"), ctx)
        append_jsonl(self.llm_log, {"ts":int(time.time()), "case":"fanout", "parent":pid, "resp":resp})
        return self._apply(pid, resp.get("json", {}), "fanout")

    # 逻辑3：深度压平 (New)
    def _fix_depth(self, pid):
        depth = calc_depth(self.tm, pid)
        if depth < MAX_DEPTH: return False

        children = self.tm.get_children(pid)
        gpid = self.tm.get_parent_id(pid)
        if not gpid: return False

        # 简单策略：如果子节点少，直接拍平给爷爷
        if len(children) <= 3:
            print(f"    [Flatten] Node {pid} (depth={depth}) -> lifting {len(children)} children to grandparent")
            for c in children: self.tm.move_node(c["node_id"], gpid)
            self.tm.remove_node(pid)
            append_jsonl(self.ops_log, {"op":"flatten", "node":pid, "depth":depth})
            return True
        return False

    def _apply(self, pid, res, stage):
        act = res.get("action", "keep")
        changed = False

        # 1. Bridge/Group
        if act in ["insert_bridge", "create_groups"]:
            # 深度保护：如果已经太深，禁止创建 Bridge
            if calc_depth(self.tm, pid) >= MAX_DEPTH - 1:
                print(f"    [Skip Bridge] Parent {pid} too deep.")
            else:
                groups = res.get("groups") or []
                for grp in groups:
                    lbl = grp.get("bridge_label")
                    cids = grp.get("child_ids", [])
                    valid = [c for c in cids if self.tm.get_parent_id(c) == pid]
                    if not valid or not lbl: continue

                    nid = generate_bridge_id(pid, lbl)
                    nlvl = get_next_level(self.tm.get_node(pid).get("level", "L1"))
                    if self.tm.add_child_node(pid, {"node_id":nid, "label":lbl, "level":nlvl, "children":[]}):
                        for c in valid: self.tm.move_node(c, nid)
                        changed = True
                        append_jsonl(self.ops_log, {"op":"create_bridge", "parent":pid, "bridge":lbl})

        # 2. Lift as Sibling (New)
        lift = res.get("lift_as_sibling") or res.get("lift_children") or []
        if lift:
            gpid = self.tm.get_parent_id(pid)
            if gpid:
                for c in lift:
                    if self.tm.get_parent_id(c) == pid:
                        self.tm.move_node(c, gpid)
                        changed = True
                        append_jsonl(self.ops_log, {"op":"lift_sibling", "child":c, "new_parent":gpid})

        return changed

def main():
    parser = argparse.ArgumentParser(description="PolicyTreeBuilder final replication · Step4.2 Shaping (Fixed)")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    # 【修改点】使用锚点定义的 ENV_PATH
    env = Step4Env(args.config, str(ENV_PATH))

    raw = load_tree(Path(args.input))
    proc = ShapingProcess(env, TreeManager(raw))
    proc.run()

    dump_tree(Path(args.output), proc.tm.root)
    # 兼容性 Trace
    (env.outdir / "structure_balancing_trace.json").write_text("{}", encoding="utf-8")
    print(f"[DONE] Shaping Completed. Tree saved to {args.output}")

if __name__ == "__main__":
    main()
