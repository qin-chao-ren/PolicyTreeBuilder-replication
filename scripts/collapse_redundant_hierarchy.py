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
    DEFAULT_DEFERRED_SCOPE_MESSAGE,
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

# C13RF29 inverts this stage's scope criterion.
#
# Until this card the rule was a blacklist: enumerate the shapes that are not
# allowed, and treat everything that trips one as repairable -- ask the model to
# fix it, and if the repair does not satisfy the same predicate, the private
# harness guard stops the whole run.  Two named messages had a defer channel
# (C13RF6/RF16/RF18); every other rejection kept the stop-the-run disposition.
# C13RF28 died exactly there: call 54 of 53 successful calls proposed merging the
# CHILD with its *sibling*, the repair round returned a semantically identical
# answer, and 53 completed calls were thrown away with the run.  Since the list of
# ways a decision can fall outside a stage's stage is open-ended, each new shape
# bought another full-tree rerun.
#
# The inverted rule: EXECUTION is still gated on the same single positive shape
# this stage always required -- ``action == "merge"`` acting on exactly the
# displayed pair, plus a child_plan whose destinations follow from that merge --
# and the *disposition* of everything else changes from "repair, then stop the
# run" to "record it and move to the next pair".  Nothing new becomes
# executable: a merge naming a node outside the pair is refused exactly as
# before, it simply no longer takes the run down with it.  The cost, accepted on
# the record by the user, is the case where the model merely mistyped a
# reference and a repair round would have fixed it: that repair is no longer
# requested, so the tree may change less than it could have -- never more.
#
# ``keep``/``reject_merge``/``uncertain`` are in scope for the same reason: they
# were 49 of C13RF28's 53 calls, and a non-mutating verb pointing off-stage asks
# the stage to execute nothing at all.  Stopping a run over one is pure loss.
PAIR_EXTERNAL_CHILD_MESSAGE = "merge child target is outside the exact pair role"
# The two messages that had a defer channel before C13RF29.  Kept as a named set
# because the tests pin them and because they are the only ones whose text is
# asserted elsewhere; after this card every scope message defers, so this set no
# longer decides the disposition on its own.
INEXPRESSIBLE_SCOPE_MESSAGES = frozenset({PAIR_EXTERNAL_CHILD_MESSAGE})
VOID_CHILD_PLAN_MESSAGE = "source has no children; child_plan must be empty"


# C13RF29 load-bearing gate, stated positively and independently of the scope
# predicates below.  This stage mutates the tree for exactly one shape: a merge
# whose source and target are the two nodes it displayed.  Executing a merge that
# names anything else would have this stage absorb a node it was never shown --
# no membership list, no similarity measurement -- which is strictly worse than
# the stop-the-run behaviour C13RF29 removes.  So the gate is checked again at
# execution time rather than trusted from the validator: if a future edit
# loosens a predicate in _scope_issues, this still refuses.
MERGE_EXECUTION_GATE_MESSAGE = (
    "merge may only execute on the exact displayed parent-child pair"
)


def merge_execution_gate_error(decision, pair_ids):
    """Return a message when `decision` must not be executed, else None."""
    if not isinstance(decision, dict):
        return "decision is not an object"
    if decision.get("action") != "merge":
        return None
    ids = {
        str(decision.get("source_id") or ""),
        str(decision.get("target_id") or ""),
    }
    return None if ids == set(pair_ids) else MERGE_EXECUTION_GATE_MESSAGE


def _deferred_scope_messages(resp):
    """Every scope message the binder recorded for the channel that refused.

    C13RF29.  Read from repair_scope_errors when a repair round ran, else from
    initial_scope_errors -- the same channel choice the defer predicate makes.
    Falls back to the shared generic rather than inventing a specific reason.
    """
    if not isinstance(resp, dict):
        return []
    channel = (
        "repair_scope_errors" if resp.get("repair_count")
        else "initial_scope_errors"
    )
    messages = []
    for item in resp.get(channel) or []:
        if not isinstance(item, dict):
            continue
        context = item.get("context")
        recorded = (
            context.get("scope_messages")
            if isinstance(context, dict) else None
        )
        if isinstance(recorded, list) and recorded:
            messages.extend(str(message) for message in recorded)
        elif item.get("message"):
            messages.append(str(item["message"]))
    return list(dict.fromkeys(messages))


def _all_scope_errors_inexpressible(scope_errors):
    """True when a scope-error channel is non-empty and wholly inexpressible.

    Read by both defer channels in _is_deferrable_response (C13RF18).  An empty
    channel is never deferrable: "no recorded error" must not be mistaken for
    "every recorded error was benign".
    """
    return (
        isinstance(scope_errors, list)
        and bool(scope_errors)
        and all(
            isinstance(item, dict)
            and item.get("code") == "DECISION_SCOPE_INEXPRESSIBLE"
            # C13RF29: the verdict fields are checked too, not just the code.
            # 14b's predicate always did this; these three checked the code alone,
            # so an entry marked inexpressible AND repairable would have deferred
            # here and stopped the run there.  No stage emits that combination
            # today -- after this card nothing is repairable at all -- but the
            # asymmetry is exactly the "parallel verdict channels" shape that put
            # D2b into production, and a floor is worth having before it is needed.
            and isinstance(item.get("context"), dict)
            and item["context"].get("repairable") is False
            and item["context"].get("mutable_ref_paths") == []
            for item in scope_errors
        )
    )

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
            messages = []
            # C13RF29: `mutable`/`repairable`/`inexpressible_mutable` are gone
            # with the repair round they existed to steer -- nothing that reaches
            # this loop is repairable any more, so keeping a per-path ref list
            # would only suggest to a later reader that a repair still happens.
            # `inexpressible_children` stays: it names the off-stage child_plan
            # items in the record, which is reporting, not repair.
            inexpressible_children = []
            void_child_plan = False
            if action not in allowed_actions:
                messages.append("action is not allowed in vertical collapse")
            elif action in {"rename", "split_reparent"} and not (
                source_id == target_id and source_id in pair_ids
            ):
                messages.append(f"{action} must operate on one displayed node")
            elif action in {"merge", "keep", "reject_merge", "uncertain"} and (
                {source_id, target_id} != pair_ids
            ):
                messages.append("decision must reference the displayed parent-child pair")

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
                    elif plan_target not in displayed_ids:
                        messages.append("split target is outside the displayed call context")
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
            if messages:
                if void_child_plan:
                    messages.append(VOID_CHILD_PLAN_MESSAGE)
                unique_messages = list(dict.fromkeys(messages))
                # C13RF29: reaching here means the decision failed the positive
                # check above, so it is not executable and no repair round is
                # requested for it.  Every such decision is recorded as
                # DECISION_SCOPE_INEXPRESSIBLE with repairable=False and no
                # mutable ref path, which is what makes the binder skip the
                # repair attempt, the harness guard let the call through, and the
                # loop continue with the next pair.
                #
                # `child_indexes` keeps naming the child_plan items whose
                # destination did not follow from the merge, unchanged from
                # C13RF16, so the deferred record still says which part was
                # off-stage.  The pre-RF29 fields that only existed to steer a
                # repair round -- withheld_ref_paths, inexpressible_components,
                # repair_paths -- are gone with the round itself; VOID_CHILD_PLAN
                # is still flagged because it describes the proposal, not a
                # repair instruction.
                extra_context = {}
                if inexpressible_children:
                    extra_context["child_indexes"] = list(
                        dict.fromkeys(inexpressible_children)
                    )
                if void_child_plan:
                    extra_context[VOID_CHILD_PLAN_FLAG] = True
                issues.append({
                    "code": "DECISION_SCOPE_INEXPRESSIBLE",
                    "message": "; ".join(unique_messages),
                    "context": {
                        "decision_index": index,
                        "mutable_ref_paths": [],
                        "repairable": False,
                        "scope_messages": unique_messages,
                        **extra_context,
                    },
                })
        return issues

    @staticmethod
    def _is_deferrable_response(resp):
        if not isinstance(resp, dict):
            return False
        # Conditions shared by both channels: the call was refused, nothing was
        # bound, and the tree did not move underneath the validator.
        if not (
            resp.get("ok") is not True
            and resp.get("mutation_before_validation") is False
            and resp.get("final_bound") is None
        ):
            return False
        # Channel 1 (C13RF6, unchanged): the initial answer was refused outright
        # and every scope error it left behind is inexpressible.
        if (
            resp.get("final_disposition") == "scope_rejected"
            and resp.get("error") == "BOUND_SCOPE_VALIDATION_FAILED"
            and resp.get("repair_count") == 0
            and _all_scope_errors_inexpressible(
                resp.get("initial_scope_errors")
            )
        ):
            return True
        # Channel 2 (C13RF18): a mixed answer whose repair round honestly fixed
        # the repairable half, leaving a remainder that is entirely
        # inexpressible.  The remainder is read from repair_scope_errors -- the
        # parallel field the binder has always produced from the same rule as
        # initial_scope_errors, and which no defer predicate used to read.  Any
        # non-inexpressible residue keeps the run stopping, by design.
        # repair_count is pinned to exactly 1 because the binder performs at
        # most one repair round; a future multi-round repair must fail closed
        # rather than silently inherit this allowance.
        return (
            resp.get("final_disposition") == "repair_scope_rejected"
            and resp.get("error") == "REPAIR_SCOPE_VALIDATION_FAILED"
            and resp.get("repair_count") == 1
            and _all_scope_errors_inexpressible(resp.get("repair_scope_errors"))
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
            # C13RF29: the record states the reasons this call actually carried
            # instead of the one hard-coded merge sentence it used to claim.
            scope_messages=_deferred_scope_messages(resp),
        )

    @staticmethod
    def _deferred_execution_record(stage, decision, messages):
        """Record an unexecutable decision at the execution channel and skip it.

        C13RF29.  Status stays ``rejected`` rather than ``deferred`` on purpose:
        a ``deferred`` record is contractually a parked *proposal* and the
        offline auditor (``scan_deferred_records.py``) requires it to carry the
        binder's ``logical_call_id`` and the local/resolved proposal pair.  This
        channel sees a single already-bound decision, so it has no proposal to
        park; what changes versus the pre-RF29 behaviour is the verdict it
        carries -- ``DECISION_SCOPE_INEXPRESSIBLE`` plus every reason -- instead
        of a bare parse rejection.  Either way the tree does not move, no repair
        is requested, and the loop continues: `rejected` is not in the stage
        verifier's applied set, so it is classified `presented_only` (the defence
        working) and not as a stage-failing incident.
        """
        reasons = [str(message) for message in messages if str(message).strip()]
        record = rejected_parse_record(
            stage,
            [
                {
                    "code": "DECISION_SCOPE_INEXPRESSIBLE",
                    "message": "; ".join(reasons) or DEFAULT_DEFERRED_SCOPE_MESSAGE,
                    "scope_messages": reasons,
                    "mutable_ref_paths": [],
                    "repairable": False,
                }
            ],
        )
        record["message"] = "semantic decision not expressible in this stage; skipped"
        return record

    def _execute_decision(
        self, decision, parent_id, child_id, call_audit=None
    ):
        if isinstance(call_audit, dict) and call_audit.get("ok") is True:
            assert_call_audit_payload(
                call_audit, {"decisions": [decision]}
            )
        # C13RF29: the second of this stage's two scope channels.  It re-runs the
        # same predicate at execution time (the "parallel verdict channels"
        # lesson: a criterion changed in one channel and not the other is how
        # C13RF11's D2b slipped through), and its disposition is inverted the
        # same way -- record the reason and skip this pair instead of rejecting
        # into the log.  The independent merge gate is checked first and on its
        # own, so it holds even if _scope_issues is later loosened.
        pair_ids = {str(parent_id), str(child_id)}
        gate_error = merge_execution_gate_error(decision, pair_ids)
        scope_issues = self._scope_issues(
            {"decisions": [decision]},
            parent_id,
            child_id,
        )
        if gate_error or scope_issues:
            messages = ([gate_error] if gate_error else []) + [
                str(issue.get("message") or "") for issue in scope_issues
            ]
            append_jsonl(
                self.ops_log,
                self._deferred_execution_record(
                    "vertical_collapse", decision, messages
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
