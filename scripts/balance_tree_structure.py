#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 4.2 · Shaping (Fixed Version + Stable Paths)
功能：结构塑形
修复内容：深度压平、提升为兄弟、层级跳跃修复
"""

import argparse
import copy
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
from utils.tree_integrity import atomic_write_bytes, atomic_write_json
from utils.tree_integrity import merge_lineage_maps
from utils.semantic_contract import (
    deferred_restructure_record,
    execute_semantic_decision,
    membership_counts_from_level_maps,
    parse_semantic_decisions,
    rejected_parse_record,
)
from utils.local_reference_binding import (
    VOID_CHILD_PLAN_FLAG,
    assert_call_audit_payload,
    attach_operation_audit,
    canonical_payload_sha256,
    build_local_reference_context,
    call_local_reference_json,
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
BRIDGE_DEPTH_DEFERRED_MESSAGE = (
    "create_bridge is not expressible at the structure-balancing depth ceiling"
)
# Scope messages that describe a legitimate restructure this stage cannot
# express, as opposed to a factual misread the model must repair.  Membership is
# per message, not per whole list: a decision carrying one of these plus an
# ordinary violation must still report the deferrable half instead of dropping
# it (C13RF16 fix ②).
BRIDGE_TOO_DEEP_MESSAGE = "bridge parent is too deep"
INEXPRESSIBLE_SCOPE_MESSAGES = frozenset({BRIDGE_TOO_DEEP_MESSAGE})
VOID_CHILD_PLAN_MESSAGE = "source has no children; child_plan must be empty"
AUDITED_DECISION_FIELDS = (
    "relation",
    "action",
    "source_id",
    "target_id",
    "new_label",
    "confidence",
    "evidence",
    "child_plan",
)


def _atomic_append_jsonl_batch(path: Path, records) -> None:
    """Append a complete logical-call batch with one atomic replacement."""

    previous = path.read_bytes() if path.exists() else b""
    payload = "".join(
        json.dumps(dict(record), ensure_ascii=False) + "\n"
        for record in records
    ).encode("utf-8")
    atomic_write_bytes(path, previous + payload)


def _restore_file_snapshot(path: Path, existed: bool, payload: bytes) -> None:
    if existed:
        atomic_write_bytes(path, payload)
    elif path.exists():
        path.unlink()


def _retain_executed_decision(record, decision):
    """Make a non-applied audit row independently hash-reconstructable."""

    record.update({
        field: copy.deepcopy(decision.get(field))
        for field in AUDITED_DECISION_FIELDS
    })
    return record


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
        self.deferred_candidates = set()

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

    def _deferred_candidate_keys(self):
        if not hasattr(self, "deferred_candidates"):
            self.deferred_candidates = set()
        return self.deferred_candidates

    def _was_deferred(self, pid, stage):
        return (str(stage), str(pid)) in self._deferred_candidate_keys()

    def _mark_deferred(self, pid, stage):
        self._deferred_candidate_keys().add((str(stage), str(pid)))

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
        child_lines = []
        for child in children:
            child_id = str(child["node_id"])
            child_lines.append(
                f"- {child_id} · {child.get('label','')} · level={child.get('level')}"
                f" · direct_membership={self.membership_counts.get(child_id, 0)}"
            )
            for grandchild in self.tm.get_children(child_id):
                child_lines.append(
                    f"  - direct child of {child_id}: {grandchild['node_id']}"
                    f" · {grandchild.get('label','')} · level={grandchild.get('level')}"
                )
        c_desc = "\n".join(child_lines)
        return (
            f"# 场景：{scene}\n"
            f"# 当前候选节点\n{describe_node(p, len(children))}\n"
            f"直属 membership={self.membership_counts.get(pid, 0)}\n"
            f"# 当前候选的父节点\n{parent_desc}\n"
            f"# 当前候选的直属子节点\n{c_desc}\n"
            f"create_bridge 的虚拟 target ref 固定为 {BRIDGE_TARGET_PLACEHOLDER}\n"
        )

    def _call_local(self, pid, context_text, task):
        preferred_refs = [("CANDIDATE", str(pid))]
        parent_id = str(self.tm.get_parent_id(pid) or "")
        if parent_id and parent_id in context_text:
            preferred_refs.append(("PARENT", parent_id))
        virtual_refs = (
            (("NEW_BRIDGE", BRIDGE_TARGET_PLACEHOLDER),)
            if BRIDGE_TARGET_PLACEHOLDER in context_text else ()
        )
        system_prompt = PROMPT_BALANCE.read_text(encoding="utf-8")
        local_context = build_local_reference_context(
            task=task,
            user_text=context_text,
            candidate_node_ids=self.tm.get_all_node_ids(),
            preferred_refs=preferred_refs,
            virtual_refs=virtual_refs,
            contract_text=system_prompt,
            expected_count=None,
        )
        return call_local_reference_json(
            transport=call_llm_json,
            profile=self.llm_profile,
            system=system_prompt,
            context=local_context,
            task=task,
            protected_manager=self.tm,
            bound_validator=lambda payload: self._scope_issues(pid, payload),
        )

    # 逻辑1：层级跳跃
    def _fix_jump(self, pid):
        if self._was_deferred(pid, "jump"):
            return False
        p = self.tm.get_node(pid)
        p_lvl = LEVEL_MAP.get(str(p.get("level")).upper(), 99)
        jumps = [c for c in self.tm.get_children(pid)
                 if LEVEL_MAP.get(str(c.get("level")).upper(), 99) - p_lvl > 1]

        if not jumps: return False

        ctx = self._fmt_ctx(pid, jumps, "层级跳跃(Level Gap > 1)")
        resp = self._call_local(pid, ctx, "balance_tree_jump_fix")
        append_jsonl(self.llm_log, {"ts":int(time.time()), "case":"jump", "parent":pid, "resp":resp})
        payload = resp.get("json") if isinstance(resp, dict) else None
        return self._apply(pid, payload, "jump", call_audit=resp)

    # 逻辑2：扇出过大
    def _fix_fanout(self, pid):
        if self._was_deferred(pid, "fanout"):
            return False
        # ROOT 的子节点是步 5 定义的顶层分类（top_level_categories.json），
        # 其数量由分类法本身决定，不是「扇出过大」需要整形的对象。
        # 若对 ROOT 造桥，LLM 临时起名的分组会取代 l1-def 里的类目、真 L1 被降级，
        # 14d 的 L1 审计随之只覆盖 1/9。基准运行恰好 7 个 L1（7<=7）未触发，故未暴露。
        if pid == "ROOT":
            return False
        children = self.tm.get_children(pid)
        if len(children) <= MAX_FANOUT: return False

        ctx = self._fmt_ctx(pid, children, f"扇出过大({len(children)}>{MAX_FANOUT})")
        resp = self._call_local(pid, ctx, "balance_tree_fanout")
        append_jsonl(self.llm_log, {"ts":int(time.time()), "case":"fanout", "parent":pid, "resp":resp})
        payload = resp.get("json") if isinstance(resp, dict) else None
        return self._apply(pid, payload, "fanout", call_audit=resp)

    # 逻辑3：深度压平 (New)
    def _fix_depth(self, pid):
        if self._was_deferred(pid, "depth"):
            return False
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
            resp = self._call_local(pid, ctx, "balance_tree_depth_flatten")
            append_jsonl(
                self.llm_log,
                {"ts": int(time.time()), "case": "depth", "parent": pid, "resp": resp},
            )
            payload = resp.get("json") if isinstance(resp, dict) else None
            return self._apply(pid, payload, "depth", call_audit=resp)
        return False

    def _prepare_decision(self, pid, raw_decision):
        decision = dict(raw_decision)
        action = decision.get("action")
        source_id = str(decision.get("source_id") or "")
        target_id = str(decision.get("target_id") or "")
        allowed_actions = {
            "create_bridge", "move", "split_reparent", "flatten",
            "keep", "reject_merge", "uncertain",
        }
        scope_messages = []
        mutable_ref_paths = []
        terminal = False
        new_node_level = None
        void_child_plan = False

        if action not in allowed_actions:
            scope_messages.append("action is not allowed in structure balancing")
            terminal = True
        elif action in {"create_bridge", "flatten", "split_reparent"} and source_id != pid:
            scope_messages.append(f"{action} source must be the current candidate")
            mutable_ref_paths.append("source_ref")

        if action == "create_bridge":
            if calc_depth(self.tm, pid) >= MAX_DEPTH - 1:
                scope_messages.append(BRIDGE_TOO_DEEP_MESSAGE)
                terminal = True
            if target_id != BRIDGE_TARGET_PLACEHOLDER:
                scope_messages.append("create_bridge target must be NEW_BRIDGE")
                mutable_ref_paths.append("target_ref")
            direct_children = {
                str(child.get("node_id")) for child in self.tm.get_children(pid)
            }
            plan_items = [
                item for item in decision.get("child_plan", [])
                if isinstance(item, dict)
            ]
            for child_index, item in enumerate(plan_items):
                if str(item.get("child_id") or "") not in direct_children:
                    scope_messages.append("bridge plan contains a non-direct child")
                    mutable_ref_paths.append(f"child_plan[{child_index}].child_ref")
                if str(item.get("target_parent_id") or "") != BRIDGE_TARGET_PLACEHOLDER:
                    scope_messages.append("bridge child target must be NEW_BRIDGE")
                    mutable_ref_paths.append(
                        f"child_plan[{child_index}].target_parent_ref"
                    )
            label = decision.get("new_label")
            if not scope_messages and isinstance(label, str) and label.strip():
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
        elif action == "flatten":
            if target_id != str(self.tm.get_parent_id(pid) or ""):
                scope_messages.append("flatten target must be the current candidate parent")
                mutable_ref_paths.append("target_ref")
            direct_children = {
                str(child.get("node_id")) for child in self.tm.get_children(pid)
            }
            for child_index, item in enumerate(decision.get("child_plan", [])):
                if not isinstance(item, dict):
                    continue
                if str(item.get("child_id") or "") not in direct_children:
                    scope_messages.append("flatten plan contains a non-direct child")
                    mutable_ref_paths.append(
                        f"child_plan[{child_index}].child_ref"
                    )
                if str(item.get("target_parent_id") or "") != target_id:
                    scope_messages.append("flatten child target must equal the flatten target")
                    mutable_ref_paths.append(
                        f"child_plan[{child_index}].target_parent_ref"
                    )
        elif action == "split_reparent":
            if target_id != pid:
                scope_messages.append("split_reparent target must equal the current candidate")
                mutable_ref_paths.append("target_ref")
            direct_children = {
                str(child.get("node_id")) for child in self.tm.get_children(pid)
            }
            allowed_targets = {
                pid,
                str(self.tm.get_parent_id(pid) or ""),
                *direct_children,
            }
            for child_index, item in enumerate(decision.get("child_plan", [])):
                if not isinstance(item, dict):
                    continue
                if str(item.get("child_id") or "") not in direct_children:
                    scope_messages.append("split plan contains a non-direct child")
                    mutable_ref_paths.append(
                        f"child_plan[{child_index}].child_ref"
                    )
                if str(item.get("target_parent_id") or "") not in allowed_targets:
                    scope_messages.append("split target is outside the candidate context")
                    mutable_ref_paths.append(
                        f"child_plan[{child_index}].target_parent_ref"
                    )
                child_id = str(item.get("child_id") or "")
                plan_target = str(item.get("target_parent_id") or "")
                disposition = item.get("disposition")
                invalid_role = (
                    disposition == "keep" and plan_target != source_id
                ) or (
                    disposition == "move"
                    and (
                        plan_target in {source_id, child_id}
                        or self.tm.is_descendant(plan_target, child_id)
                    )
                )
                if invalid_role:
                    scope_messages.append("split child target does not match its disposition role")
                    mutable_ref_paths.append(
                        f"child_plan[{child_index}].target_parent_ref"
                    )
        elif action == "move":
            current_children = {
                str(child.get("node_id")) for child in self.tm.get_children(pid)
            }
            if source_id not in current_children:
                scope_messages.append("move source must be a direct candidate child")
                mutable_ref_paths.append("source_ref")
            if target_id != str(self.tm.get_parent_id(pid) or ""):
                scope_messages.append("move target must be the candidate parent")
                mutable_ref_paths.append("target_ref")
            source_children = {
                str(child.get("node_id")) for child in self.tm.get_children(source_id)
            }
            plan_items = decision.get("child_plan", [])
            plan_items = plan_items if isinstance(plan_items, list) else []
            # No children at all means no child_ref is nameable, so every item
            # below is reported and no per-item repair exists.  A move whose
            # source is childless legitimately relocates nothing, so the empty
            # plan is the correction (C13RF16 fix ①).  Only merge and move earn
            # this flag: create_bridge, flatten and split_reparent require at
            # least one child by contract, so for them an empty plan is never
            # legal and the proposal is void as a whole.
            void_child_plan = bool(plan_items) and not source_children
            for child_index, item in enumerate(plan_items):
                if not isinstance(item, dict):
                    continue
                if str(item.get("child_id") or "") not in source_children:
                    scope_messages.append("move plan contains a non-source child")
                    mutable_ref_paths.append(
                        f"child_plan[{child_index}].child_ref"
                    )
                if str(item.get("target_parent_id") or "") != source_id:
                    scope_messages.append("moved node children must remain under their source")
                    mutable_ref_paths.append(
                        f"child_plan[{child_index}].target_parent_ref"
                    )
        elif action in {"keep", "reject_merge", "uncertain"} and not (
            source_id == target_id == pid
        ):
            scope_messages.append("non-mutating decision must reference only the candidate")
            mutable_ref_paths.extend(["source_ref", "target_ref"])

        scope_issue = None
        if scope_messages:
            if void_child_plan:
                scope_messages.append(VOID_CHILD_PLAN_MESSAGE)
            unique_messages = list(dict.fromkeys(scope_messages))
            inexpressible_messages = [
                message for message in unique_messages
                if message in INEXPRESSIBLE_SCOPE_MESSAGES
            ]
            other_messages = [
                message for message in unique_messages
                if message not in INEXPRESSIBLE_SCOPE_MESSAGES
            ]
            inexpressible = (
                action == "create_bridge"
                and bool(inexpressible_messages)
                and not other_messages
            )
            extra_context = {}
            # Mixed decision: the depth ceiling carries no mutable ref path of
            # its own, so nothing has to be withheld here -- ``terminal`` already
            # blocks the repair round.  Recording the deferrable component is
            # what changes: it used to vanish silently (C13RF16 fix ②).
            if inexpressible_messages and other_messages:
                extra_context["inexpressible_components"] = inexpressible_messages
            if void_child_plan:
                extra_context[VOID_CHILD_PLAN_FLAG] = True
            scope_issue = {
                "code": (
                    "DECISION_SCOPE_INEXPRESSIBLE"
                    if inexpressible else "DECISION_SCOPE_VIOLATION"
                ),
                "message": (
                    BRIDGE_DEPTH_DEFERRED_MESSAGE
                    if inexpressible else "; ".join(unique_messages)
                ),
                "mutable_ref_paths": list(dict.fromkeys(mutable_ref_paths)),
                "repairable": False if inexpressible else not terminal,
                "extra_context": extra_context,
            }
        return decision, new_node_level, scope_issue

    def _scope_issues(self, pid, payload):
        decisions = payload.get("decisions", []) if isinstance(payload, dict) else []
        issues = []
        for index, raw_decision in enumerate(decisions):
            _, _, issue = self._prepare_decision(pid, raw_decision)
            if not issue:
                continue
            issues.append({
                "code": issue["code"],
                "message": issue["message"],
                "context": {
                    "decision_index": index,
                    "mutable_ref_paths": [
                        f"decisions[{index}].{path}"
                        for path in issue["mutable_ref_paths"]
                    ],
                    "repairable": issue["repairable"],
                    **issue.get("extra_context", {}),
                },
            })
        return issues

    @staticmethod
    def _is_deferrable_response(resp):
        if not isinstance(resp, dict):
            return False
        scope_errors = resp.get("initial_scope_errors")
        local_proposal = resp.get("final_local")
        decisions = (
            local_proposal.get("decisions")
            if isinstance(local_proposal, dict) else None
        )
        # A scope-error list only describes rejected decisions.  Requiring it
        # to cover every local proposal prevents a mixed batch (one legal
        # decision plus one depth-conflicted bridge) from being deferred as a
        # whole.  The deferred path is intentionally limited to homogeneous
        # create_bridge batches whose only issue is the depth ceiling.
        error_indexes = [
            item.get("context", {}).get("decision_index")
            for item in scope_errors
            if isinstance(item, dict)
            and isinstance(item.get("context"), dict)
        ] if isinstance(scope_errors, list) else []
        valid_error_indexes = all(
            isinstance(index, int) and not isinstance(index, bool)
            for index in error_indexes
        )
        all_decisions_are_depth_bridges = (
            isinstance(decisions, list)
            and bool(decisions)
            and all(
                isinstance(decision, dict)
                and decision.get("action") == "create_bridge"
                for decision in decisions
            )
            and valid_error_indexes
            and sorted(error_indexes) == list(range(len(decisions)))
            and len(set(error_indexes)) == len(error_indexes)
        )
        scope_errors_are_depth_only = (
            isinstance(scope_errors, list)
            and all(
                isinstance(item, dict)
                and item.get("code") == "DECISION_SCOPE_INEXPRESSIBLE"
                and item.get("message") == BRIDGE_DEPTH_DEFERRED_MESSAGE
                and isinstance(item.get("context"), dict)
                and item["context"].get("mutable_ref_paths") == []
                and item["context"].get("repairable") is False
                for item in scope_errors
            )
        )
        return (
            resp.get("ok") is not True
            and resp.get("final_disposition") == "scope_rejected"
            and resp.get("error") == "BOUND_SCOPE_VALIDATION_FAILED"
            and resp.get("mutation_before_validation") is False
            and resp.get("final_bound") is None
            and resp.get("repair_count") == 0
            and isinstance(scope_errors, list)
            and bool(scope_errors)
            and all_decisions_are_depth_bridges
            and scope_errors_are_depth_only
        )

    @staticmethod
    def _deferred_record(resp, stage):
        mapping = {
            str(item.get("ref")): str(item.get("node_id"))
            for item in resp.get("context", {}).get("ref_mapping", [])
            if isinstance(item, dict)
        }
        local_proposal = resp["final_local"]
        resolved_proposal = copy.deepcopy(local_proposal)
        for decision in resolved_proposal.get("decisions", []):
            decision["source_id"] = mapping[decision.pop("source_ref")]
            decision["target_id"] = mapping[decision.pop("target_ref")]
            for item in decision.get("child_plan", []):
                item["child_id"] = mapping[item.pop("child_ref")]
                item["target_parent_id"] = mapping[
                    item.pop("target_parent_ref")
                ]
        record = deferred_restructure_record(
            stage,
            logical_call_id=resp["logical_call_id"],
            local_proposal=local_proposal,
            resolved_proposal=resolved_proposal,
        )
        record["message"] = "structure-balancing create_bridge deferred before execution"
        record["semantic_contract"]["violations"][0][
            "message"
        ] = BRIDGE_DEPTH_DEFERRED_MESSAGE
        return record

    def _apply(self, pid, payload, stage, call_audit=None):
        if isinstance(call_audit, dict) and call_audit.get("ok") is True:
            assert_call_audit_payload(call_audit, payload)

        def execution_transform(index, executed_decision):
            if not isinstance(call_audit, dict):
                return None
            bound_decisions = call_audit.get("final_bound", {}).get(
                "decisions", []
            )
            if index >= len(bound_decisions):
                return None
            if executed_decision == bound_decisions[index]:
                return None
            if bound_decisions[index].get("action") == "create_bridge":
                return "materialize_new_bridge"
            return None

        decisions, errors = parse_semantic_decisions(payload)
        if errors:
            if self._is_deferrable_response(call_audit):
                append_jsonl(
                    self.ops_log,
                    self._deferred_record(
                        call_audit, f"structure_balancing_{stage}"
                    ),
                )
                self._mark_deferred(pid, stage)
                return False
            append_jsonl(
                self.ops_log,
                rejected_parse_record(f"structure_balancing_{stage}", errors),
            )
            return False

        prepared = [self._prepare_decision(pid, decision) for decision in decisions]
        scope_errors = [error for _, _, error in prepared if error]
        if scope_errors:
            aborted_records = []
            for index, (executed_decision, _level, own_error) in enumerate(prepared):
                error = own_error or {
                    "code": "BATCH_SCOPE_ABORTED",
                    "message": "another decision in the batch failed scope preflight",
                }
                aborted = rejected_parse_record(
                    f"structure_balancing_{stage}",
                    [
                        {
                            "code": error["code"],
                            "message": error["message"],
                            "context": error,
                        }
                    ],
                )
                aborted["batch_status"] = "aborted"
                aborted["batch_index"] = index
                aborted["batch_size"] = len(prepared)
                _retain_executed_decision(aborted, executed_decision)
                attach_operation_audit(
                    aborted,
                    call_audit=call_audit,
                    decision_index=index,
                    executed_decision=executed_decision,
                    execution_transform=execution_transform(
                        index, executed_decision
                    ),
                )
                aborted_records.append(aborted)
            _atomic_append_jsonl_batch(self.ops_log, aborted_records)
            return False

        original_root = copy.deepcopy(self.tm.root)
        candidate_tm = TreeManager(copy.deepcopy(original_root))
        candidate_trace = dict(getattr(self, "trace_map", {}))
        candidate_lineage = self._current_lineage()
        records = []
        failure = None
        mutating_actions = {
            "merge", "move", "move_across_l1", "split_reparent",
            "flatten", "create_bridge", "rename",
        }
        try:
            for index, (decision, new_node_level, _) in enumerate(prepared):
                record = execute_semantic_decision(
                    candidate_tm,
                    decision,
                    direct_membership_counts=self.membership_counts,
                    membership_known=True,
                    stage=f"structure_balancing_{stage}",
                    lineage=candidate_lineage,
                    allow_cross_l1=False,
                    new_node_level=new_node_level,
                )
                records.append(record)
                action = record.get("action")
                status = record.get("status")
                contract_passed = (
                    isinstance(record.get("semantic_contract"), dict)
                    and record["semantic_contract"].get("passed") is True
                )
                accepted_status = (
                    (action in mutating_actions and status == "applied")
                    or (action == "keep" and status == "skipped")
                    or (action in {"reject_merge", "uncertain"} and status == "rejected")
                )
                if not contract_passed or not accepted_status:
                    failure = {
                        "decision_index": index,
                        "status": status,
                        "violation_counts": (
                            record.get("semantic_contract", {}).get("violation_counts", {})
                            if isinstance(record.get("semantic_contract"), dict) else {}
                        ),
                    }
                    break
                if action == "flatten" and record.get("lineage_target"):
                    candidate_trace[record["source_id"]] = record["lineage_target"]
                    candidate_lineage = merge_lineage_maps(
                        candidate_lineage,
                        {record["source_id"]: record["lineage_target"]},
                    )
        except Exception as exc:
            failure = {
                "decision_index": len(records),
                "exception_type": type(exc).__name__,
            }

        if failure is not None:
            aborted_records = []
            for index, (executed_decision, _level, _error) in enumerate(prepared):
                violation_code = (
                    "BATCH_SEMANTIC_ABORTED"
                    if index == failure.get("decision_index")
                    else "BATCH_TRANSACTION_ABORTED"
                )
                aborted = rejected_parse_record(
                    f"structure_balancing_{stage}",
                    [{
                        "code": violation_code,
                        "message": "the whole decision batch was discarded before live-tree commit",
                        "context": failure,
                    }],
                )
                aborted["batch_status"] = "aborted"
                aborted["batch_index"] = index
                aborted["batch_size"] = len(prepared)
                _retain_executed_decision(aborted, executed_decision)
                attach_operation_audit(
                    aborted,
                    call_audit=call_audit,
                    decision_index=index,
                    executed_decision=executed_decision,
                    execution_transform=execution_transform(
                        index, executed_decision
                    ),
                )
                aborted_records.append(aborted)
            _atomic_append_jsonl_batch(self.ops_log, aborted_records)
            return False

        changed = candidate_tm.root != original_root
        for index, record in enumerate(records):
            record["batch_status"] = "committed"
            record["batch_index"] = index
            record["batch_size"] = len(records)

        original_trace = dict(getattr(self, "trace_map", {}))
        log_existed = self.ops_log.exists()
        original_log = self.ops_log.read_bytes() if log_existed else b""
        try:
            self.tm._restore(copy.deepcopy(candidate_tm.root))
            if not hasattr(self, "trace_map"):
                self.trace_map = {}
            self.trace_map.clear()
            self.trace_map.update(candidate_trace)
            live_before = canonical_payload_sha256(original_root)
            live_after = canonical_payload_sha256(self.tm.root)
            for index, record in enumerate(records):
                attach_operation_audit(
                    record,
                    call_audit=call_audit,
                    decision_index=index,
                    executed_decision=prepared[index][0],
                    live_tree_before_sha256=live_before,
                    live_tree_after_sha256=live_after,
                    execution_transform=execution_transform(
                        index, prepared[index][0]
                    ),
                )
            _atomic_append_jsonl_batch(self.ops_log, records)
        except Exception as exc:
            rollback_errors = []
            try:
                self.tm._restore(copy.deepcopy(original_root))
            except Exception as rollback_exc:
                rollback_errors.append(f"tree={type(rollback_exc).__name__}")
            try:
                if not hasattr(self, "trace_map"):
                    self.trace_map = {}
                self.trace_map.clear()
                self.trace_map.update(original_trace)
            except Exception as rollback_exc:
                rollback_errors.append(f"trace={type(rollback_exc).__name__}")
            try:
                _restore_file_snapshot(
                    self.ops_log, log_existed, original_log
                )
            except Exception as rollback_exc:
                rollback_errors.append(f"log={type(rollback_exc).__name__}")
            suffix = (
                f"; rollback_errors={','.join(rollback_errors)}"
                if rollback_errors else ""
            )
            raise RuntimeError(
                f"structure-balancing batch commit failed: {type(exc).__name__}{suffix}"
            ) from exc
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
