#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 4.3 · Polishing (Refactored)
功能：精细润色 (兄弟合并 / 跨父统一 / 标签精炼)
核心职责：
1. 结构收尾：使用 TreeManager 进行安全的 Merge/Move/Rename。
2. 链路闭环：加载 Step 4.1/4.2 的 Trace，合并本步骤产生的 Trace。
3. 最终产出：生成 policy_tree_final_membership.csv，确保所有样本能找到最终归属。
"""

import argparse
import copy
import time
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from common_utils import jaccard_overlap
from llm_runtime import call_llm_json
from utils.step4_shared import (
    Step4Env,
    EmbeddingHelper,
    load_tree,
    dump_tree,
    append_jsonl,
    read_membership_map,
    read_title_map
)
from utils.tree_manager import TreeManager
from utils.tree_integrity import (
    STAGE_LINEAGE_LEDGERS,
    LineageError,
    atomic_write_bytes,
    atomic_write_csv,
    atomic_write_json,
    cross_validate_ledger_against_operations,
    load_stage_lineage_ledger,
    merge_lineage_maps,
    read_jsonl,
    read_membership_csv,
    redirect_membership_rows,
    stages_present_in_operations,
)
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

# Scope messages that describe a legitimate restructure this stage cannot
# express, as opposed to a factual misread the model must repair.  Membership is
# per message, not per whole list: a decision carrying one of these plus an
# ordinary violation must still report the deferrable half instead of dropping
# it (C13RF16 fix ②).
# C13RF29 inverts this stage's scope criterion the same way it inverts 14a's
# (see collapse_redundant_hierarchy.py's header for the full account).  This
# stage keeps two mutating verbs besides merge -- `move` (reparent one displayed
# node to one displayed parent) and `rename` -- and both keep their positive
# scope requirement unchanged; only the disposition of a failed check changes,
# from "repair, then stop the run if unfixed" to "record it and skip this pair".
PAIR_EXTERNAL_CHILD_MESSAGE = "merge child target is outside the exact pair role"
# Named set kept for the tests that pin its text; after C13RF29 every scope
# message defers, so this set no longer decides the disposition on its own.
INEXPRESSIBLE_SCOPE_MESSAGES = frozenset({PAIR_EXTERNAL_CHILD_MESSAGE})
VOID_CHILD_PLAN_MESSAGE = "source has no children; child_plan must be empty"


# C13RF29 load-bearing gate, stated positively and independently of the scope
# predicates below (see collapse_redundant_hierarchy.py for the full rationale).
# This stage's merge absorbs one displayed node into the other; executing a merge
# that names anything else would act on a node it never measured.
MERGE_EXECUTION_GATE_MESSAGE = (
    "merge may only execute on the exact displayed pair"
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

    C13RF29; identical to 14a's reader.  Reads repair_scope_errors when a repair
    round ran, else initial_scope_errors -- the same channel choice the defer
    predicate makes.
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

# --- 1. 路径锚点 (Path Anchors) ---
# 无论在哪里运行命令，__file__ 都能定位到 scripts/ 目录
HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent

# --- 2. 关键资源路径 (基于锚点) ---
# 这样写绝对不会错
ENV_PATH = PROJECT_ROOT / "configs" / ".env"

# --- 3. Prompt 路径 (每个脚本不一样，请按需保留) ---
PROMPT_POLISH = PROJECT_ROOT / "prompts" / "polish_tree_labels.md"

class PolishingProcess:
    def __init__(self, env: Step4Env, tm: TreeManager, args):
        self.env = env
        self.llm_profile = env.primary_llm_profile()
        self.tm = tm
        self.args = args

        # 加载辅助数据
        self.emb_helper = EmbeddingHelper(Path(env.config["paths"]["embeddings"]))
        corpus_path = Path(env.config["paths"]["corpus"])
        self.title_map = read_title_map(corpus_path)
        self.membership = {
            lvl: read_membership_map(env.outdir, lvl)
            for lvl in ["L4", "L3", "L2", "L1"]
        }
        self.membership_counts = membership_counts_from_level_maps(self.membership)

        # 日志
        self.ops_log = env.outdir / "tree_refinement_operations.jsonl"
        self.llm_log = env.log_dir / "llm_polish_tree_labels.jsonl"

        # 本步骤产生的 ID 变更记录 (Old ID -> New ID)
        self.local_trace_map: Dict[str, str] = {}
        # C13RF22: 上游 ledger 必需性由 operations 账反推，缓存一次即可
        self._required_stages: Optional[Set[str]] = None
        self.ledger_audit: List[Dict[str, Any]] = []

    def run(self):
        # 1. 兄弟节点合并 (同父)
        print("[Step 4.3] Starting Sibling Merge...")
        self._run_sibling_merge()

        # 2. 跨父节点统一 (同层级)
        print("[Step 4.3] Starting Cross-Parent Unify...")
        self._run_cross_parent_unify()

        # 3. 追溯与导出 (The Core Traceability Logic)
        print("[Step 4.3] Consolidating Traces & Exporting Final Membership...")
        final_trace = self._consolidate_all_traces()
        self._export_final_membership(final_trace)
        atomic_write_json(self.env.outdir / "policy_tree_lineage.json", final_trace)

        # 4. 保存最终树
        dump_tree(Path(self.args.output), self.tm.root)
        print(f"[DONE] Polishing Completed. Tree saved to {self.args.output}")

    # --- Logic 1: Sibling Merge ---

    def _run_sibling_merge(self):
        # 遍历所有非叶子节点
        parents = [nid for nid in self.tm.get_all_node_ids() if self.tm.get_children(nid)]

        for parent_id in parents:
            if not self.tm.exists(parent_id): continue

            # 获取子节点快照
            children = list(self.tm.get_children(parent_id))
            if len(children) < 2: continue

            # 两两比较 (简单冒泡以覆盖所有对，或限制 Top-K)
            # 为了效率，我们这里只比较 jaccard > 0.6 的对
            handled_pairs = set()

            for i in range(len(children)):
                for j in range(i + 1, len(children)):
                    a = children[i]
                    b = children[j]

                    # 动态检查有效性 (因为前面的循环可能已经 merge 掉了某节点)
                    if not self.tm.exists(a["node_id"]) or not self.tm.exists(b["node_id"]):
                        continue

                    pair_key = tuple(sorted([a["node_id"], b["node_id"]]))
                    if pair_key in handled_pairs: continue
                    handled_pairs.add(pair_key)

                    # 预筛选
                    if not self._quick_check_similarity(a, b, thr_jac=0.75, thr_cos=0.85):
                        continue

                    # LLM 决策
                    self._process_pair(a, b, case="sibling_merge", parent_id=parent_id)

    # --- Logic 2: Cross-Parent Unify ---

    def _run_cross_parent_unify(self):
        # 按层级桶 (Bucket) 聚合所有节点
        level_buckets: Dict[str, List[Dict]] = {}
        for nid in self.tm.get_all_node_ids():
            node = self.tm.get_node(nid)
            lvl = node.get("level", "")
            if lvl in ["L3", "L2"]: # L4 太多通常不做跨父，L1 不动
                level_buckets.setdefault(lvl, []).append(node)

        for lvl, nodes in level_buckets.items():
            # 限制每层处理数量，避免 O(N^2) 爆炸，或者使用聚类加速
            # 这里简化逻辑：只比较相邻/高相似对，实际生产环境建议配合 Faiss 检索
            # 本代码演示：简单双重循环，加严格预筛选

            print(f"  > Processing {lvl} ({len(nodes)} nodes)...")
            handled_cross = set()

            for i in range(len(nodes)):
                for j in range(i + 1, min(i + 50, len(nodes))): # 滑动窗口减少计算量
                    a = nodes[i]
                    b = nodes[j]

                    if not self.tm.exists(a["node_id"]) or not self.tm.exists(b["node_id"]):
                        continue

                    # 必须是不同父
                    pa = self.tm.get_parent_id(a["node_id"])
                    pb = self.tm.get_parent_id(b["node_id"])
                    if pa == pb: continue

                    # 严格预筛选
                    if not self._quick_check_similarity(a, b, thr_jac=0.80, thr_cos=0.90):
                        continue

                    self._process_pair(a, b, case="cross_parent_unify", parent_id=None)

    # --- Helper: LLM Interaction & Execution ---

    def _quick_check_similarity(self, a, b, thr_jac, thr_cos) -> bool:
        jac = jaccard_overlap(a.get("label", ""), b.get("label", ""))
        if jac >= thr_jac: return True

        vec_a = self.emb_helper.get_centroid(self._get_members(a))
        vec_b = self.emb_helper.get_centroid(self._get_members(b))
        cos = self.emb_helper.cosine_sim(vec_a, vec_b)

        return cos >= thr_cos

    def _get_members(self, node) -> List[str]:
        return self.membership.get(node.get("level"), {}).get(node["node_id"], [])

    def _process_pair(self, a, b, case, parent_id):
        # Earlier pairs may have committed a merge and invalidated snapshots.
        if not self.tm.exists(a["node_id"]) or not self.tm.exists(b["node_id"]):
            return
        a = self.tm.get_node(a["node_id"])
        b = self.tm.get_node(b["node_id"])
        if not a or not b:
            return
        # 1. 构造 Context
        ctx_a = self._describe_node(a["node_id"])
        ctx_b = self._describe_node(b["node_id"])

        # 计算当前指标供 LLM 参考
        jac = jaccard_overlap(a.get("label", ""), b.get("label", ""))
        vec_a = self.emb_helper.get_centroid(self._get_members(a))
        vec_b = self.emb_helper.get_centroid(self._get_members(b))
        cos = self.emb_helper.cosine_sim(vec_a, vec_b)

        evidence = (
            f"# 场景: {case}\n"
            f"## 节点 A\n{ctx_a}\n\n"
            f"## 节点 B\n{ctx_b}\n\n"
            f"## 相似度指标\nJaccard={jac:.2f}, Cosine={cos:.2f}\n"
            "请根据 label 语义、父节点语境和样本内容，决定是否合并、移动或重命名。\n"
        )

        preferred_refs = [("LEFT", a["node_id"]), ("RIGHT", b["node_id"])]
        parent_a = str(self.tm.get_parent_id(a["node_id"]) or "")
        parent_b = str(self.tm.get_parent_id(b["node_id"]) or "")
        if parent_a and parent_a == parent_b:
            preferred_refs.append(("SHARED_PARENT", parent_a))
        else:
            if parent_a:
                preferred_refs.append(("LEFT_PARENT", parent_a))
            if parent_b:
                preferred_refs.append(("RIGHT_PARENT", parent_b))
        system_prompt = PROMPT_POLISH.read_text(encoding="utf-8")
        local_context = build_local_reference_context(
            task="polish_tree_labels",
            user_text=evidence,
            candidate_node_ids=self.tm.get_all_node_ids(),
            preferred_refs=preferred_refs,
            contract_text=system_prompt,
            expected_count=1,
        )

        # 2. Call LLM
        resp = call_local_reference_json(
            transport=call_llm_json,
            profile=self.llm_profile,
            system=system_prompt,
            context=local_context,
            task="polish_tree_labels",
            expected_count=1,
            protected_manager=self.tm,
            bound_validator=lambda payload: self._scope_issues(
                payload,
                a["node_id"],
                b["node_id"],
            ),
        )

        append_jsonl(self.llm_log, {
            "ts": int(time.time()), "case": case,
            "node_a": a["node_id"], "node_b": b["node_id"],
            "metrics": {"jac": jac, "cos": cos}, "resp": resp
        })

        payload = resp.get("json") if isinstance(resp, dict) else None
        decisions, errors = parse_semantic_decisions(payload, expected_count=1)
        if errors:
            if self._is_deferrable_response(resp):
                append_jsonl(
                    self.ops_log,
                    self._deferred_record(resp, "label_polishing"),
                )
                return
            append_jsonl(
                self.ops_log,
                rejected_parse_record(
                    "label_polishing",
                    errors,
                    logical_call_id=(
                        resp.get("logical_call_id")
                        if isinstance(resp, dict) else None
                    ),
                ),
            )
            return

        self._execute_decision(
            decisions[0], a["node_id"], b["node_id"], case,
            call_audit=resp,
        )

    def _scope_issues(self, payload, node_a_id, node_b_id):
        decisions = payload.get("decisions", []) if isinstance(payload, dict) else []
        pair_ids = {str(node_a_id), str(node_b_id)}
        parent_ids = {
            str(self.tm.get_parent_id(node_a_id) or ""),
            str(self.tm.get_parent_id(node_b_id) or ""),
        }
        parent_ids.discard("")
        displayed_ids = set(pair_ids) | set(parent_ids)
        for node_id in tuple(pair_ids):
            displayed_ids.update(
                str(item.get("node_id")) for item in self.tm.get_children(node_id)
            )
        allowed_actions = {
            "merge", "move", "rename", "split_reparent",
            "keep", "reject_merge", "uncertain",
        }
        issues = []
        for index, decision in enumerate(decisions):
            action = decision.get("action")
            source_id = str(decision.get("source_id") or "")
            target_id = str(decision.get("target_id") or "")
            messages = []
            # C13RF29: `mutable`/`repairable`/`inexpressible_mutable` are gone
            # with the repair round they existed to steer.  `inexpressible_children`
            # stays: it names off-stage child_plan items in the record, which is
            # reporting, not repair.
            inexpressible_children = []
            void_child_plan = False
            if action not in allowed_actions:
                messages.append("action is not allowed in label polishing")
            elif action == "merge" and {source_id, target_id} != pair_ids:
                messages.append("merge source and target must be the displayed pair")
            elif action == "move" and (
                source_id not in pair_ids or target_id not in parent_ids
            ):
                messages.append("move must reparent one displayed node to one displayed parent")
            elif action in {"rename", "split_reparent"} and not (
                source_id == target_id and source_id in pair_ids
            ):
                messages.append(f"{action} must operate on one displayed node")
            elif action in {"keep", "reject_merge", "uncertain"} and {source_id, target_id} != pair_ids:
                messages.append("non-mutating decision must reference the displayed pair")

            if action in {"merge", "move", "split_reparent"}:
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
                    elif action == "move" and plan_target != source_id:
                        messages.append("moved node children must remain under their source")
                    elif action == "split_reparent" and plan_target not in displayed_ids:
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
                # C13RF29: every failed check is DECISION_SCOPE_INEXPRESSIBLE,
                # repairable=False, no mutable ref path -- see
                # collapse_redundant_hierarchy.py's header for the full account.
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
            # C13RF29: state the reasons this call carried, not one hard-coded
            # merge sentence.
            scope_messages=_deferred_scope_messages(resp),
        )

    @staticmethod
    def _deferred_execution_record(stage, decision, messages):
        """Record an unexecutable decision at the execution channel and skip it.

        C13RF29; same shape and same reasoning as 14a's method of this name.
        Status stays ``rejected`` rather than ``deferred`` because this channel
        holds a single already-bound decision and so has no binder proposal to
        park, which is what the offline deferred-record auditor requires.
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
        self, decision, node_a_id, node_b_id, case, call_audit=None
    ):
        if isinstance(call_audit, dict) and call_audit.get("ok") is True:
            assert_call_audit_payload(
                call_audit, {"decisions": [decision]}
            )
        # C13RF29: this stage's second scope channel, inverted the same way as
        # the binder channel -- record the reason and skip this pair instead of
        # requesting a repair that can stop the run.  The independent merge gate
        # is checked first and on its own so it holds even if _scope_issues is
        # later loosened.
        pair_ids = {str(node_a_id), str(node_b_id)}
        gate_error = merge_execution_gate_error(decision, pair_ids)
        scope_issues = self._scope_issues(
            {"decisions": [decision]},
            node_a_id,
            node_b_id,
        )
        if gate_error or scope_issues:
            messages = ([gate_error] if gate_error else []) + [
                str(issue.get("message") or "") for issue in scope_issues
            ]
            append_jsonl(
                self.ops_log,
                self._deferred_execution_record(
                    "label_polishing", decision, messages
                ),
            )
            return

        original_root = copy.deepcopy(self.tm.root)
        original_trace = dict(self.local_trace_map)
        log_existed = self.ops_log.exists()
        original_log = self.ops_log.read_bytes() if log_existed else b""
        try:
            live_before = canonical_payload_sha256(self.tm.root)
            record = execute_semantic_decision(
                self.tm,
                decision,
                direct_membership_counts=self.membership_counts,
                membership_known=True,
                stage="label_polishing",
                lineage=self._current_lineage(),
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
            )
            record["case"] = case
            append_jsonl(self.ops_log, record)
            if record.get("status") == "applied" and record.get("lineage_target"):
                self.local_trace_map[record["source_id"]] = record["lineage_target"]
        except Exception:
            self.tm._restore(original_root)
            self.local_trace_map.clear()
            self.local_trace_map.update(original_trace)
            if log_existed:
                atomic_write_bytes(self.ops_log, original_log)
            elif self.ops_log.exists():
                self.ops_log.unlink()
            raise

    def _describe_node(self, node_id):
        node = self.tm.get_node(node_id)
        pid = self.tm.get_parent_id(node_id)
        p_label = self.tm.get_node(pid)["label"] if pid and self.tm.exists(pid) else "ROOT"
        children = self.tm.get_children(node_id)
        child_text = ", ".join(
            f"{child.get('node_id')}={child.get('label', '')}"
            for child in children
        ) or "none"

        titles = self._collect_titles(node_id)
        return (
            f"ID: {node_id} · Label: {node.get('label','')}\n"
            f"Level: {node.get('level')} · Parent: {pid or 'ROOT'}={p_label}\n"
            f"Direct children: {child_text}\n"
            f"Direct membership count: {len(self._get_members(node))}\n"
            f"Examples: {', '.join(titles)}"
        )

    def _required_upstream_stages(self) -> Set[str]:
        """Which upstream ledgers must exist, judged from the operations log.

        C13RF22: deciding this from `path.exists()` is the fail-open bug.  The
        operations log is the evidence that a stage ran -- 14a's records are in
        it whether or not anyone remembered to stage its ledger file.
        """
        if self._required_stages is None:
            records = read_jsonl(self.ops_log) if self.ops_log.exists() else []
            self._required_stages = stages_present_in_operations(records)
        return self._required_stages

    def _load_upstream_ledgers(self) -> List[Dict[str, str]]:
        required = self._required_upstream_stages()
        return [
            load_stage_lineage_ledger(
                self.env.outdir, stage, filename, required=stage in required
            )
            for stage, filename in STAGE_LINEAGE_LEDGERS
        ]

    def _current_lineage(self) -> Dict[str, str]:
        maps = self._load_upstream_ledgers()
        maps.append(self.local_trace_map)
        return merge_lineage_maps(*maps)

    def _collect_titles(self, node_id):
        node = self.tm.get_node(node_id)
        mems = self._get_members(node)
        return [self.title_map[sid] for sid in mems if sid in self.title_map][:5]

    # --- Logic 3: Trace Consolidation (Critical) ---

    def _consolidate_all_traces(self) -> Dict[str, str]:
        """
        合并 Trace 4.1 + 4.2 + 4.3 (Local)
        逻辑：链式更新 A->B, B->C  =>  A->C
        """
        maps = []

        # 1. 加载历史 Trace（C13RF22：缺 ledger 时 fail-closed，见 _load_upstream_ledgers）
        required = self._required_upstream_stages()
        records = read_jsonl(self.ops_log) if self.ops_log.exists() else []
        self.ledger_audit = []
        for stage, filename in STAGE_LINEAGE_LEDGERS:
            sub_map = load_stage_lineage_ledger(
                self.env.outdir, stage, filename, required=stage in required
            )
            maps.append(sub_map)
            state = (
                "required" if stage in required
                else "absent upstream (no operations records)"
            )
            print(f"  + Loaded {len(sub_map)} redirects from {stage} [{state}]")
            verdict = cross_validate_ledger_against_operations(sub_map, records, stage)
            self.ledger_audit.append(verdict)
            if verdict["comparable"] and not verdict["agrees"]:
                raise LineageError(
                    f"lineage ledger for {stage} disagrees with the operations log: "
                    f"{verdict['disagreements']}"
                )
            if not verdict["comparable"] and verdict["applied_records"]:
                print(
                    f"    (cross-check skipped for {stage}: {verdict['reason']})"
                )

        # 2. 合并当前步骤 Trace
        maps.append(self.local_trace_map)
        print(f"  + Loaded {len(self.local_trace_map)} redirects from 4.3 (Local)")
        try:
            return merge_lineage_maps(*maps)
        except LineageError as exc:
            raise RuntimeError(f"Lineage consolidation failed: {exc}") from exc

    def _export_final_membership(self, trace_map: Dict[str, str]):
        """
        读取原始 CSV，应用 trace_map，生成 policy_tree_final_membership.csv
        """
        source_rows = []
        fieldnames = ["sample_id", "final_node_id", "original_node_id", "original_level"]

        for level in ["L4", "L3", "L2"]:
            path = self.env.outdir / f"tree_node_membership_{level}.csv"
            if not path.exists(): continue

            _, rows = read_membership_csv(path)
            print(f"  > Processing {level} membership ({len(rows)} rows)...")
            for row in rows:
                original_nid = str(row.get("node_id", ""))
                source_rows.append({
                    "sample_id": str(row.get("member_id", "")),
                    "final_node_id": original_nid,
                    "original_node_id": original_nid,
                    "original_level": level,
                })

        output_rows = redirect_membership_rows(
            source_rows, trace_map, self.tm.get_all_node_ids()
        )
        out_path = self.env.outdir / "policy_tree_final_membership.csv"
        atomic_write_csv(out_path, fieldnames, output_rows)
        print(f"[SUCCESS] Final membership exported to {out_path} ({len(output_rows)} rows)")

def main():
    parser = argparse.ArgumentParser(description="PolicyTreeBuilder final replication · Step4.3 Polishing")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    # 1. 初始化
    env = Step4Env(args.config, str(ENV_PATH))

    # 2. 加载数据
    raw_tree = load_tree(Path(args.input))
    tm = TreeManager(raw_tree)

    # 3. 执行
    process = PolishingProcess(env, tm, args)
    process.run()

if __name__ == "__main__":
    main()
