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
import copy
import time
from pathlib import Path
from typing import Dict, Tuple

from llm_runtime import call_llm_json
from common_utils import jaccard_overlap
from utils.step4_shared import (
    Step4Env, EmbeddingHelper, load_tree, dump_tree,
    append_jsonl, read_membership_map, read_title_map
)
from utils.tree_manager import TreeManager
from utils.tree_integrity import atomic_write_bytes, atomic_write_json, atomic_write_jsonl
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

# Scope messages that describe a legitimate restructure this stage cannot
# express, as opposed to a factual misread the model must repair.  Membership is
# per message, not per whole list: a decision carrying one of these plus an
# ordinary violation must still report the deferrable half instead of dropping
# it (C13RF16 fix ②).
PAIR_EXTERNAL_CHILD_MESSAGE = "merge child target is outside the exact pair role"
INEXPRESSIBLE_SCOPE_MESSAGES = frozenset({PAIR_EXTERNAL_CHILD_MESSAGE})
VOID_CHILD_PLAN_MESSAGE = "source has no children; child_plan must be empty"

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
        child_labels = [
            f"{c.get('node_id')}={c.get('label', '?')[:20]}"
            for c in children
        ]
        children_summary = f"\n直属子节点: {', '.join(child_labels)}"

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
        self.membership_counts = membership_counts_from_level_maps(self.membership)
        self.ops_log = env.outdir / "tree_refinement_operations.jsonl"
        self.llm_log = env.log_dir / "llm_collapse_redundant_hierarchy.jsonl"
        self.redirect_map = {}

    def run(self):
        print("[Step 4.1] Starting Skeleton Refinement (Fixed)...")
        atomic_write_jsonl(self.ops_log, [])
        candidate_parents = [nid for nid in self.tm.get_all_node_ids() if self.tm.get_children(nid)]

        for parent_id in candidate_parents:
            if not self.tm.exists(parent_id): continue
            self._process_parent(parent_id)

        dump_tree(Path(self.args.output), self.tm.root)
        trace_path = self.env.outdir / "vertical_collapse_trace.json"
        atomic_write_json(trace_path, self.redirect_map)
        print(f"[DONE] Skeleton Refined. Tree saved to {self.args.output}")

    def _process_parent(self, parent_id):
        children = list(self.tm.get_children(parent_id)) # Snapshot

        for child_snapshot in children:
            child_id = child_snapshot["node_id"]
            if not self.tm.exists(child_id): continue
            parent_node = self.tm.get_node(parent_id)
            child_node = self.tm.get_node(child_id)
            if not parent_node or not child_node: continue

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

            system_prompt = PROMPT_COLLAPSE.read_text(encoding="utf-8")
            preferred_refs = [("PARENT", parent_id), ("CHILD", child_id)]
            grandparent_id = str(self.tm.get_parent_id(parent_id) or "")
            if grandparent_id:
                preferred_refs.append(("GRANDPARENT", grandparent_id))
            local_context = build_local_reference_context(
                task="collapse_redundant_hierarchy",
                user_text=evidence,
                candidate_node_ids=self.tm.get_all_node_ids(),
                preferred_refs=preferred_refs,
                contract_text=system_prompt,
                expected_count=1,
            )
            resp = call_local_reference_json(
                transport=call_llm_json,
                profile=self.llm_profile,
                system=system_prompt,
                context=local_context,
                task="collapse_redundant_hierarchy",
                expected_count=1,
                temperature=0.0,
                protected_manager=self.tm,
                bound_validator=lambda payload: self._scope_issues(
                    payload,
                    parent_id,
                    child_id,
                ),
            )

            append_jsonl(self.llm_log, {
                "ts": int(time.time()), "parent": parent_id, "child": child_id,
                "is_single": is_single_child, "metrics": {"jac":jac, "cos":cos}, "resp": resp
            })

            payload = resp.get("json") if isinstance(resp, dict) else None
            decisions, errors = parse_semantic_decisions(payload, expected_count=1)
            if errors:
                if self._is_deferrable_response(resp):
                    append_jsonl(
                        self.ops_log,
                        self._deferred_record(resp, "vertical_collapse"),
                    )
                    continue
                append_jsonl(
                    self.ops_log,
                    rejected_parse_record(
                        "vertical_collapse",
                        errors,
                        logical_call_id=(
                            resp.get("logical_call_id")
                            if isinstance(resp, dict) else None
                        ),
                    ),
                )
                continue
            self._execute_decision(
                decisions[0], parent_id, child_id, call_audit=resp
            )

    def _build_evidence(self, pid, cid, jac, cos, single):
        note = "【单脉传场景】" if single else "【多子节点场景】"
        grandparent_id = str(self.tm.get_parent_id(pid) or "")
        grandparent = (
            "# 父节点的父节点（用于完整 child plan）\n"
            + describe_node(grandparent_id, self.tm, self.membership, self.title_map)
            + "\n"
            if grandparent_id else "# 父节点无父节点\n"
        )
        return (
            f"# 场景：父子语义重叠检测 {note}\n"
            f"{grandparent}"
            f"# 父节点\n{describe_node(pid, self.tm, self.membership, self.title_map)}\n"
            f"# 子节点\n{describe_node(cid, self.tm, self.membership, self.title_map)}\n"
            f"相似度: Jaccard={jac:.2f}, Cosine={cos:.2f}\n"
        )

    def _scope_issues(self, payload, parent_id, child_id):
        decisions = payload.get("decisions", []) if isinstance(payload, dict) else []
        pair_ids = {str(parent_id), str(child_id)}
        displayed_ids = set(pair_ids)
        for node_id in tuple(pair_ids):
            displayed_ids.update(
                str(item.get("node_id")) for item in self.tm.get_children(node_id)
            )
            displayed_parent = str(self.tm.get_parent_id(node_id) or "")
            if displayed_parent:
                displayed_ids.add(displayed_parent)
        allowed_actions = {
            "merge", "rename", "split_reparent",
            "keep", "reject_merge", "uncertain",
        }
        issues = []
        for index, decision in enumerate(decisions):
            source_id = str(decision.get("source_id") or "")
            target_id = str(decision.get("target_id") or "")
            action = decision.get("action")
            mutable = []
            messages = []
            repairable = True
            # C13RF16 fix ②: the deferrable component is tracked separately so a
            # mixed decision can report it instead of silently losing it, and so
            # its refs can be withheld from the repair round.
            inexpressible_children = []
            inexpressible_mutable = []
            void_child_plan = False
            if action not in allowed_actions:
                messages.append("action is not allowed in vertical collapse")
                repairable = False
            elif action in {"rename", "split_reparent"} and not (
                source_id == target_id and source_id in pair_ids
            ):
                messages.append(f"{action} must operate on one displayed node")
                mutable.extend([
                    f"decisions[{index}].source_ref",
                    f"decisions[{index}].target_ref",
                ])
            elif action in {"merge", "keep", "reject_merge", "uncertain"} and (
                {source_id, target_id} != pair_ids
            ):
                messages.append("decision must reference the displayed parent-child pair")
                mutable.extend([
                    f"decisions[{index}].source_ref",
                    f"decisions[{index}].target_ref",
                ])

            if action in {"merge", "split_reparent"}:
                source_children = {
                    str(item.get("node_id"))
                    for item in self.tm.get_children(source_id)
                }
                plan_items = decision.get("child_plan", [])
                plan_items = plan_items if isinstance(plan_items, list) else []
                # No children at all means no child_ref is nameable, so every
                # item below is reported and no per-item repair exists.  Flag
                # it: the empty plan is the correction (C13RF16 fix ①).
                void_child_plan = bool(plan_items) and not source_children
                for child_index, item in enumerate(plan_items):
                    if not isinstance(item, dict):
                        continue
                    if str(item.get("child_id") or "") not in source_children:
                        messages.append("child plan contains a non-source child")
                        mutable.append(
                            f"decisions[{index}].child_plan[{child_index}].child_ref"
                        )
                    plan_target = str(item.get("target_parent_id") or "")
                    if action == "merge":
                        expected_target = (
                            str(self.tm.get_parent_id(source_id) or "")
                            if self.tm.is_descendant(target_id, source_id)
                            else target_id
                        )
                        if plan_target != expected_target:
                            messages.append(PAIR_EXTERNAL_CHILD_MESSAGE)
                            inexpressible_children.append(child_index)
                            inexpressible_mutable.append(
                                f"decisions[{index}].child_plan[{child_index}].target_parent_ref"
                            )
                            mutable.append(
                                f"decisions[{index}].child_plan[{child_index}].target_parent_ref"
                            )
                    elif plan_target not in displayed_ids:
                        messages.append("split target is outside the displayed call context")
                        mutable.append(
                            f"decisions[{index}].child_plan[{child_index}].target_parent_ref"
                        )
                    if action == "split_reparent":
                        child_id = str(item.get("child_id") or "")
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
                            messages.append("split child target does not match its disposition role")
                            mutable.append(
                                f"decisions[{index}].child_plan[{child_index}].target_parent_ref"
                            )
            if messages:
                if void_child_plan:
                    messages.append(VOID_CHILD_PLAN_MESSAGE)
                unique_messages = list(dict.fromkeys(messages))
                inexpressible_messages = [
                    message for message in unique_messages
                    if message in INEXPRESSIBLE_SCOPE_MESSAGES
                ]
                other_messages = [
                    message for message in unique_messages
                    if message not in INEXPRESSIBLE_SCOPE_MESSAGES
                ]
                inexpressible = (
                    action == "merge"
                    and repairable
                    and bool(inexpressible_messages)
                    and not other_messages
                )
                # Mixed decision: withhold the deferrable half's refs so the
                # repair round cannot coerce a pair-external destination into a
                # legal-but-wrong one, and record what was withheld.
                withheld = (
                    list(dict.fromkeys(inexpressible_mutable))
                    if inexpressible_messages and other_messages else []
                )
                repair_paths = [
                    path for path in dict.fromkeys(mutable)
                    if path not in set(withheld)
                ]
                extra_context = {}
                if withheld:
                    extra_context["inexpressible_components"] = inexpressible_messages
                    extra_context["inexpressible_child_indexes"] = list(
                        dict.fromkeys(inexpressible_children)
                    )
                    extra_context["withheld_ref_paths"] = withheld
                if void_child_plan:
                    extra_context[VOID_CHILD_PLAN_FLAG] = True
                issues.append({
                    "code": (
                        "DECISION_SCOPE_INEXPRESSIBLE"
                        if inexpressible else "DECISION_SCOPE_VIOLATION"
                    ),
                    "message": (
                        "merge with a pair-external child target is not expressible in this stage"
                        if inexpressible else "; ".join(unique_messages)
                    ),
                    "context": {
                        "decision_index": index,
                        **({
                            "child_indexes": [
                                child_index
                                for child_index, item in enumerate(
                                    decision.get("child_plan", [])
                                )
                                if isinstance(item, dict)
                                and str(item.get("target_parent_id") or "")
                                != (
                                    str(self.tm.get_parent_id(source_id) or "")
                                    if self.tm.is_descendant(target_id, source_id)
                                    else target_id
                                )
                            ],
                            "mutable_ref_paths": [],
                            "repairable": False,
                        } if inexpressible else {
                            "mutable_ref_paths": repair_paths,
                            "repairable": repairable and bool(repair_paths),
                            **extra_context,
                        }),
                    },
                })
        return issues

    @staticmethod
    def _is_deferrable_response(resp):
        if not isinstance(resp, dict):
            return False
        scope_errors = resp.get("initial_scope_errors")
        return (
            resp.get("ok") is not True
            and resp.get("final_disposition") == "scope_rejected"
            and resp.get("error") == "BOUND_SCOPE_VALIDATION_FAILED"
            and resp.get("mutation_before_validation") is False
            and resp.get("final_bound") is None
            and resp.get("repair_count") == 0
            and isinstance(scope_errors, list)
            and bool(scope_errors)
            and all(
                isinstance(item, dict)
                and item.get("code") == "DECISION_SCOPE_INEXPRESSIBLE"
                for item in scope_errors
            )
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
        return deferred_restructure_record(
            stage,
            logical_call_id=resp["logical_call_id"],
            local_proposal=local_proposal,
            resolved_proposal=resolved_proposal,
        )

    def _execute_decision(
        self, decision, parent_id, child_id, call_audit=None
    ):
        if isinstance(call_audit, dict) and call_audit.get("ok") is True:
            assert_call_audit_payload(
                call_audit, {"decisions": [decision]}
            )
        scope_issues = self._scope_issues(
            {"decisions": [decision]},
            parent_id,
            child_id,
        )
        if scope_issues:
            append_jsonl(
                self.ops_log,
                rejected_parse_record(
                    "vertical_collapse",
                    scope_issues,
                ),
            )
            return
        source_id = str(decision.get("source_id") or "")
        target_id = str(decision.get("target_id") or "")
        if (
            decision.get("action") == "merge"
            and self.tm.exists(source_id)
            and self.tm.exists(target_id)
            and self.tm.is_descendant(target_id, source_id)
        ):
            ok, reason = can_promote_safely(self.tm, target_id, source_id)
            if not ok:
                decision = dict(decision)
                evidence = dict(decision.get("evidence") or {})
                warnings = list(evidence.get("warnings") or [])
                warnings.append(f"promote_safety:{reason}")
                evidence["warnings"] = warnings
                decision["evidence"] = evidence

        original_root = copy.deepcopy(self.tm.root)
        original_redirect = dict(self.redirect_map)
        log_existed = self.ops_log.exists()
        original_log = self.ops_log.read_bytes() if log_existed else b""
        try:
            live_before = canonical_payload_sha256(self.tm.root)
            record = execute_semantic_decision(
                self.tm,
                decision,
                direct_membership_counts=self.membership_counts,
                membership_known=True,
                stage="vertical_collapse",
                lineage=self.redirect_map,
                allow_cross_l1=False,
            )
            live_after = canonical_payload_sha256(self.tm.root)
            attach_operation_audit(
                record,
                call_audit=call_audit,
                decision_index=0,
                executed_decision=decision,
                live_tree_before_sha256=live_before,
                live_tree_after_sha256=live_after,
                execution_transform=(
                    "promote_safety_fail_closed_warning"
                    if decision
                    != call_audit.get("final_bound", {}).get("decisions", [decision])[0]
                    else None
                ) if isinstance(call_audit, dict) else None,
            )
            if record.get("status") == "applied" and record.get("lineage_target"):
                self.redirect_map[record["source_id"]] = record["lineage_target"]
            append_jsonl(self.ops_log, record)
        except Exception:
            self.tm._restore(original_root)
            self.redirect_map.clear()
            self.redirect_map.update(original_redirect)
            if log_existed:
                atomic_write_bytes(self.ops_log, original_log)
            elif self.ops_log.exists():
                self.ops_log.unlink()
            raise

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
