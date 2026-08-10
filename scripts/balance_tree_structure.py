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
import json
from pathlib import Path
from typing import Dict, List

from llm_runtime import call_llm_json
from utils.step4_shared import (
    Step4Env, load_tree, dump_tree, append_jsonl, read_membership_map
)
from utils.tree_manager import TreeManager
from utils.tree_manager import normalize_label
from utils.tree_integrity import atomic_write_json
from utils.tree_integrity import merge_lineage_maps
from utils.semantic_contract import (
    execute_semantic_decision,
    membership_counts_from_level_maps,
    parse_semantic_decisions,
    rejected_parse_record,
)

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
# ROOT 必须在表内：ROOT 扇出 > MAX_FANOUT 时会给 ROOT 造桥节点，
# 缺 ROOT 条目会让 get_next_level("ROOT") 取默认 99 → min(100,4) → "L4"，
# 造出标成 L4 却位于 L1 之上的桥节点，顶层结构错乱。
LEVEL_MAP = {"ROOT": 0, "L1": 1, "L2": 2, "L3": 3, "L4": 4}
MAX_DEPTH = 4
MAX_FANOUT = 7
MAX_ROUNDS = 10
BRIDGE_TARGET_PLACEHOLDER = "__NEW_BRIDGE__"

def get_next_level(lvl):
    n = min(LEVEL_MAP.get(str(lvl).upper(), 99) + 1, 4)
    return f"L{n}"

def generate_bridge_id(pid, lbl):
    h = hashlib.md5(f"{pid}_{normalize_label(lbl)}_BR".encode()).hexdigest()[:6]
    return f"{pid}_BR_{h}"

def describe_node(node, count):
    return f"ID={node['node_id']} · level={node.get('level')} · label={node.get('label','')}\n子节点数={count}"

def calc_depth(tm, nid):
    d = 0
    curr = nid
    seen = set()
    while curr:
        if curr in seen:
            return MAX_DEPTH + 1
        seen.add(curr)
        p = tm.get_parent_id(curr)
        if not p: break
        d += 1
        curr = p
    return d

class ShapingProcess:
    def __init__(self, env: Step4Env, tm: TreeManager):
        self.env = env
        self.llm_profile = env.primary_llm_profile()
        self.tm = tm
        self.ops_log = env.outdir / "tree_refinement_operations.jsonl"
        self.llm_log = env.log_dir / "llm_balance_tree_structure.jsonl"
        self.trace_map = {}
        self.membership = {
            level: read_membership_map(env.outdir, level)
            for level in ["L4", "L3", "L2", "L1"]
        }
        self.membership_counts = membership_counts_from_level_maps(self.membership)
        self.prior_lineage = self._load_prior_lineage()

    def _load_prior_lineage(self):
        path = self.env.outdir / "vertical_collapse_trace.json"
        if not path.exists():
            return {}
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"vertical collapse trace must be an object: {path}")
        return {str(source): str(target) for source, target in payload.items()}

    def _current_lineage(self):
        return merge_lineage_maps(self.prior_lineage, self.trace_map)

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
        parent_id = str(self.tm.get_parent_id(pid) or "")
        parent = self.tm.get_node(parent_id) if parent_id else None
        parent_desc = (
            describe_node(parent, len(self.tm.get_children(parent_id)))
            if parent else "none"
        )
        c_desc = "\n".join([
            f"- {c['node_id']} · {c.get('label','')} · level={c.get('level')}"
            f" · direct_membership={self.membership_counts.get(c['node_id'], 0)}"
            for c in children
        ])
        return (
            f"# 场景：{scene}\n"
            f"# 当前候选节点\n{describe_node(p, len(children))}\n"
            f"直属 membership={self.membership_counts.get(pid, 0)}\n"
            f"# 当前候选的父节点\n{parent_desc}\n"
            f"# 当前候选的直属子节点\n{c_desc}\n"
            f"create_bridge 的占位 target_id 固定为 {BRIDGE_TARGET_PLACEHOLDER}\n"
        )

    # 逻辑1：层级跳跃
    def _fix_jump(self, pid):
        p = self.tm.get_node(pid)
        p_lvl = LEVEL_MAP.get(str(p.get("level")).upper(), 99)
        jumps = [c for c in self.tm.get_children(pid)
                 if LEVEL_MAP.get(str(c.get("level")).upper(), 99) - p_lvl > 1]

        if not jumps: return False

        ctx = self._fmt_ctx(pid, jumps, "层级跳跃(Level Gap > 1)")
        resp = call_llm_json(
            profile=self.llm_profile,
            system=PROMPT_BALANCE.read_text(encoding="utf-8"),
            user=ctx,
            task="balance_tree_jump_fix",
        )
        append_jsonl(self.llm_log, {"ts":int(time.time()), "case":"jump", "parent":pid, "resp":resp})
        payload = resp.get("json") if isinstance(resp, dict) else None
        return self._apply(pid, payload, "jump")

    # 逻辑2：扇出过大
    def _fix_fanout(self, pid):
        # ROOT 的子节点是步 5 定义的顶层分类（top_level_categories.json），
        # 其数量由分类法本身决定，不是「扇出过大」需要整形的对象。
        # 若对 ROOT 造桥，LLM 临时起名的分组会取代 l1-def 里的类目、真 L1 被降级，
        # 14d 的 L1 审计随之只覆盖 1/9。基准运行恰好 7 个 L1（7<=7）未触发，故未暴露。
        if pid == "ROOT":
            return False
        children = self.tm.get_children(pid)
        if len(children) <= MAX_FANOUT: return False

        ctx = self._fmt_ctx(pid, children[:30], f"扇出过大({len(children)}>{MAX_FANOUT})")
        resp = call_llm_json(
            profile=self.llm_profile,
            system=PROMPT_BALANCE.read_text(encoding="utf-8"),
            user=ctx,
            task="balance_tree_fanout",
        )
        append_jsonl(self.llm_log, {"ts":int(time.time()), "case":"fanout", "parent":pid, "resp":resp})
        payload = resp.get("json") if isinstance(resp, dict) else None
        return self._apply(pid, payload, "fanout")

    # 逻辑3：深度压平 (New)
    def _fix_depth(self, pid):
        depth = calc_depth(self.tm, pid)
        if depth < MAX_DEPTH: return False

        children = self.tm.get_children(pid)
        gpid = self.tm.get_parent_id(pid)
        if not gpid: return False

        # Small deep branches are candidates, not automatic flatten approvals.
        if len(children) <= 3:
            source = self.tm.get_node(pid)
            parent = self.tm.get_node(gpid)
            ctx = (
                "# Scene: depth flatten candidate\n"
                f"# Candidate source\n{describe_node(source, len(children))}\n"
                f"# New parent if flattened\n{describe_node(parent, len(self.tm.get_children(gpid)))}\n"
                "# Direct children requiring an explicit plan\n"
                + "\n".join(
                    f"- {child['node_id']} · {child.get('label', '')} · level={child.get('level')}"
                    for child in children
                )
                + f"\nsource_direct_membership={self.membership_counts.get(pid, 0)}\n"
            )
            resp = call_llm_json(
                profile=self.llm_profile,
                system=PROMPT_BALANCE.read_text(encoding="utf-8"),
                user=ctx,
                task="balance_tree_depth_flatten",
            )
            append_jsonl(
                self.llm_log,
                {"ts": int(time.time()), "case": "depth", "parent": pid, "resp": resp},
            )
            payload = resp.get("json") if isinstance(resp, dict) else None
            return self._apply(pid, payload, "depth")
        return False

    def _apply(self, pid, payload, stage):
        decisions, errors = parse_semantic_decisions(payload)
        if errors:
            append_jsonl(
                self.ops_log,
                rejected_parse_record(f"structure_balancing_{stage}", errors),
            )
            return False

        changed = False

        for raw_decision in decisions:
            decision = dict(raw_decision)
            action = decision.get("action")
            source_id = str(decision.get("source_id") or "")
            target_id = str(decision.get("target_id") or "")
            allowed_actions = {
                "create_bridge", "move", "split_reparent", "flatten",
                "keep", "reject_merge", "uncertain",
            }
            scope_error = None
            new_node_level = None

            if action not in allowed_actions:
                scope_error = f"action {action!r} is not allowed in structure balancing"
            elif action in {"create_bridge", "flatten", "split_reparent"} and source_id != pid:
                scope_error = f"{action} source must be the current candidate {pid}"
            elif action == "create_bridge":
                if calc_depth(self.tm, pid) >= MAX_DEPTH - 1:
                    scope_error = "bridge parent is too deep"
                elif target_id != BRIDGE_TARGET_PLACEHOLDER:
                    scope_error = (
                        f"create_bridge target must be {BRIDGE_TARGET_PLACEHOLDER}"
                    )
                else:
                    direct_children = {
                        str(child.get("node_id")) for child in self.tm.get_children(pid)
                    }
                    plan_items = [
                        item for item in decision.get("child_plan", [])
                        if isinstance(item, dict)
                    ]
                    invalid_children = sorted({
                        str(item.get("child_id") or "") for item in plan_items
                    } - direct_children)
                    invalid_targets = sorted({
                        str(item.get("target_parent_id") or "") for item in plan_items
                    } - {BRIDGE_TARGET_PLACEHOLDER})
                    if invalid_children:
                        scope_error = (
                            "bridge plan contains non-direct children: "
                            + ", ".join(invalid_children)
                        )
                    elif invalid_targets:
                        scope_error = (
                            f"bridge child targets must be {BRIDGE_TARGET_PLACEHOLDER}"
                        )
                    label = decision.get("new_label")
                    if not scope_error and isinstance(label, str) and label.strip():
                        bridge_id = generate_bridge_id(pid, label)
                        decision["target_id"] = bridge_id
                        decision["child_plan"] = [
                            {**item, "target_parent_id": bridge_id}
                            if isinstance(item, dict) else item
                            for item in decision.get("child_plan", [])
                        ]
                    new_node_level = get_next_level(
                        self.tm.get_node(pid).get("level", "L1")
                    )
            elif action == "flatten" and target_id != str(self.tm.get_parent_id(pid) or ""):
                scope_error = "flatten target must be the current candidate parent"
            elif action == "split_reparent":
                if target_id != pid:
                    scope_error = "split_reparent target must equal the current candidate"
                else:
                    allowed_targets = {
                        pid,
                        str(self.tm.get_parent_id(pid) or ""),
                        *(
                            str(child.get("node_id"))
                            for child in self.tm.get_children(pid)
                        ),
                    }
                    invalid_targets = sorted({
                        str(item.get("target_parent_id") or "")
                        for item in decision.get("child_plan", [])
                        if isinstance(item, dict)
                    } - allowed_targets)
                    if invalid_targets:
                        scope_error = (
                            "split targets must be visible in the current candidate context: "
                            + ", ".join(invalid_targets)
                        )
            elif action == "move":
                current_children = {
                    str(child.get("node_id")) for child in self.tm.get_children(pid)
                }
                if source_id not in current_children:
                    scope_error = "move source must be a direct child of the current candidate"
                elif target_id != str(self.tm.get_parent_id(pid) or ""):
                    scope_error = "balancing move target must be the candidate grandparent"
            elif action in {"keep", "reject_merge", "uncertain"} and not (
                source_id == target_id == pid
            ):
                scope_error = "non-mutating decision must reference only the current candidate"

            if scope_error:
                append_jsonl(
                    self.ops_log,
                    rejected_parse_record(
                        f"structure_balancing_{stage}",
                        [{"code": "DECISION_SCOPE_VIOLATION", "message": scope_error}],
                    ),
                )
                continue

            before_structure = (
                set(self.tm.get_all_node_ids()),
                dict(self.tm.parent_map),
            )
            record = execute_semantic_decision(
                self.tm,
                decision,
                direct_membership_counts=self.membership_counts,
                membership_known=True,
                stage=f"structure_balancing_{stage}",
                lineage=self._current_lineage(),
                allow_cross_l1=False,
                new_node_level=new_node_level,
            )
            append_jsonl(self.ops_log, record)
            if record.get("status") == "applied":
                after_structure = (
                    set(self.tm.get_all_node_ids()),
                    dict(self.tm.parent_map),
                )
                changed = changed or before_structure != after_structure
                if record.get("action") == "flatten" and record.get("lineage_target"):
                    self.trace_map[record["source_id"]] = record["lineage_target"]

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
    atomic_write_json(env.outdir / "structure_balancing_trace.json", proc.trace_map)
    print(f"[DONE] Shaping Completed. Tree saved to {args.output}")

if __name__ == "__main__":
    main()
