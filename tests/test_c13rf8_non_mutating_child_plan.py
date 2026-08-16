"""C13RF8 (D1 option 2): a non-mutating action may carry an all-no-op child_plan.

Background: the C13RF7 live full-tree run was lost at 14a logical call #23. The
model correctly answered ``reject_merge`` but attached a single ``keep`` entry in
``child_plan``. The pre-C13RF8 contract failed that closed unconditionally
(``NON_MUTATING_CHILD_PLAN_NOT_EMPTY``), which the stage verifier later surfaced
as a critical violation and the whole run was written off. Offline replay of the
326-response corpus (C13M2/C13M3) measured the shape at 2/326 and proved that
tolerating it flips only that one record.

These tests pin the relaxation and, just as importantly, its limits:
an all-``keep`` plan is tolerated, anything proposing a real change still fails
closed, per-item field validation is untouched, and the executor still returns
for these three actions before any ``child_plan`` is consumed.
"""
from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from utils.local_reference_binding import canonical_payload_sha256  # noqa: E402
from utils.semantic_contract import (  # noqa: E402
    execute_semantic_decision,
    validate_semantic_decision,
    build_semantic_context,
)
from utils.tree_manager import TreeManager  # noqa: E402


NON_MUTATING_ACTIONS = ("keep", "reject_merge", "uncertain")
RELAXED_CODE = "NON_MUTATING_CHILD_PLAN_NOT_EMPTY"


def node(node_id, label, level, children=None):
    return {
        "node_id": node_id,
        "label": label,
        "level": level,
        "children": list(children or []),
    }


def evidence(**overrides):
    value = {
        "summary": "The child is a distinct policy instrument, not a duplicate.",
        "warnings": [],
        "target_represents_all_source_members": False,
        "membership_basis": "",
        "pure_structural_redundancy": False,
        "cross_l1_authorized": False,
    }
    value.update(overrides)
    return value


def plan_item(child_id, target_parent_id, disposition="keep", **overrides):
    value = {
        "child_id": child_id,
        "disposition": disposition,
        "target_parent_id": target_parent_id,
        "relation": "broader_narrower",
        "same_domain": True,
        "evidence": "The child keeps its current parent.",
    }
    value.update(overrides)
    return value


def decision(action, relation, source_id, target_id, *, children=None, **changes):
    value = {
        "relation": relation,
        "action": action,
        "source_id": source_id,
        "target_id": target_id,
        "new_label": None,
        "confidence": 0.9,
        "evidence": evidence(),
        "child_plan": list(children or []),
    }
    value.update(changes)
    return value


def relation_for(action):
    return "uncertain" if action == "uncertain" else "broader_narrower"


class NonMutatingChildPlanTests(unittest.TestCase):
    """The pair shape of 14a call #23: PARENT with one CHILD that has a child."""

    def setUp(self):
        self.tree = node(
            "ROOT", "ROOT", "ROOT",
            [node("L1", "domain", "L1", [
                node("PARENT", "parent scope", "L2", [
                    node("CHILD", "child scope", "L3", [
                        node("GRANDCHILD", "grandchild scope", "L4"),
                    ]),
                ]),
            ])],
        )
        self.manager = TreeManager(copy.deepcopy(self.tree))

    def validate(self, value, manager=None):
        manager = manager or self.manager
        context = build_semantic_context(
            manager,
            value,
            direct_membership_counts={},
            membership_known=True,
            allow_cross_l1=False,
            allowed_l1_id=None,
        )
        return validate_semantic_decision(value, context)

    def codes(self, report):
        return [item["code"] for item in report["violations"]]

    def execute(self, value, manager=None):
        manager = manager or self.manager
        return execute_semantic_decision(
            manager,
            value,
            direct_membership_counts={},
            membership_known=True,
            stage="c13rf8_synthetic",
            lineage=None,
            allow_cross_l1=False,
        )

    # ------------------------------------------------------------------
    # 1. the relaxation itself, on all three non-mutating actions
    # ------------------------------------------------------------------
    def test_all_keep_child_plan_is_tolerated_on_every_non_mutating_action(self):
        for action in NON_MUTATING_ACTIONS:
            with self.subTest(action=action):
                value = decision(
                    action, relation_for(action), "CHILD", "PARENT",
                    children=[plan_item("GRANDCHILD", "CHILD")],
                )
                report = self.validate(value)
                self.assertNotIn(RELAXED_CODE, self.codes(report))
                self.assertTrue(report["passed"], report["violations"])

    def test_multi_entry_all_keep_child_plan_is_tolerated(self):
        tree = node(
            "ROOT", "ROOT", "ROOT",
            [node("L1", "domain", "L1", [
                node("PARENT", "parent scope", "L2", [
                    node("CHILD", "child scope", "L3", [
                        node("G1", "first grandchild", "L4"),
                        node("G2", "second grandchild", "L4"),
                    ]),
                ]),
            ])],
        )
        manager = TreeManager(copy.deepcopy(tree))
        value = decision(
            "reject_merge", "broader_narrower", "CHILD", "PARENT",
            children=[plan_item("G1", "CHILD"), plan_item("G2", "CHILD")],
        )
        report = self.validate(value, manager)
        self.assertTrue(report["passed"], report["violations"])

    def test_empty_child_plan_remains_valid(self):
        for action in NON_MUTATING_ACTIONS:
            with self.subTest(action=action):
                report = self.validate(
                    decision(action, relation_for(action), "CHILD", "PARENT")
                )
                self.assertTrue(report["passed"], report["violations"])

    # ------------------------------------------------------------------
    # 2. anything proposing an actual change still fails closed
    # ------------------------------------------------------------------
    def test_mixed_keep_and_move_child_plan_still_fails_closed(self):
        tree = node(
            "ROOT", "ROOT", "ROOT",
            [node("L1", "domain", "L1", [
                node("PARENT", "parent scope", "L2", [
                    node("CHILD", "child scope", "L3", [
                        node("G1", "first grandchild", "L4"),
                        node("G2", "second grandchild", "L4"),
                    ]),
                ]),
            ])],
        )
        manager = TreeManager(copy.deepcopy(tree))
        for action in NON_MUTATING_ACTIONS:
            with self.subTest(action=action):
                value = decision(
                    action, relation_for(action), "CHILD", "PARENT",
                    children=[
                        plan_item("G1", "CHILD"),
                        plan_item("G2", "PARENT", disposition="move"),
                    ],
                )
                report = self.validate(value, manager)
                self.assertIn(RELAXED_CODE, self.codes(report))
                self.assertFalse(report["passed"])

    def test_pure_move_child_plan_still_fails_closed(self):
        for action in NON_MUTATING_ACTIONS:
            with self.subTest(action=action):
                value = decision(
                    action, relation_for(action), "CHILD", "PARENT",
                    children=[
                        plan_item("GRANDCHILD", "PARENT", disposition="move"),
                    ],
                )
                report = self.validate(value)
                self.assertIn(RELAXED_CODE, self.codes(report))
                self.assertFalse(report["passed"])

    def test_unknown_disposition_is_not_treated_as_a_no_op(self):
        """An unknown disposition must never be read as an implicit keep."""
        value = decision(
            "reject_merge", "broader_narrower", "CHILD", "PARENT",
            children=[plan_item("GRANDCHILD", "CHILD", disposition="delete")],
        )
        report = self.validate(value)
        codes = self.codes(report)
        self.assertIn("CHILD_DISPOSITION_UNKNOWN", codes)
        self.assertIn(RELAXED_CODE, codes)
        self.assertFalse(report["passed"])

    def test_missing_disposition_is_not_treated_as_a_no_op(self):
        item = plan_item("GRANDCHILD", "CHILD")
        item.pop("disposition")
        value = decision(
            "reject_merge", "broader_narrower", "CHILD", "PARENT",
            children=[item],
        )
        report = self.validate(value)
        codes = self.codes(report)
        self.assertIn("CHILD_PLAN_FIELDS_MISSING", codes)
        self.assertIn(RELAXED_CODE, codes)
        self.assertFalse(report["passed"])

    # ------------------------------------------------------------------
    # 3. per-item field validation is NOT relaxed for tolerated plans
    # ------------------------------------------------------------------
    def test_malformed_no_op_entries_still_report_their_own_violations(self):
        cases = {
            "CHILD_ID_MISSING": plan_item("", "CHILD"),
            "CHILD_TARGET_MISSING": plan_item("GRANDCHILD", ""),
            "CHILD_RELATION_UNKNOWN": plan_item(
                "GRANDCHILD", "CHILD", relation="not_a_relation"
            ),
            "CHILD_SAME_DOMAIN_INVALID": plan_item(
                "GRANDCHILD", "CHILD", same_domain="yes"
            ),
            "CHILD_EVIDENCE_MISSING": plan_item("GRANDCHILD", "CHILD", evidence="  "),
        }
        for expected_code, item in cases.items():
            with self.subTest(code=expected_code):
                value = decision(
                    "reject_merge", "broader_narrower", "CHILD", "PARENT",
                    children=[item],
                )
                report = self.validate(value)
                self.assertIn(expected_code, self.codes(report))
                self.assertFalse(report["passed"])

    def test_unknown_fields_on_a_no_op_entry_still_fail_closed(self):
        item = plan_item("GRANDCHILD", "CHILD")
        item["unexpected"] = True
        value = decision(
            "reject_merge", "broader_narrower", "CHILD", "PARENT",
            children=[item],
        )
        report = self.validate(value)
        self.assertIn("CHILD_PLAN_FIELDS_UNKNOWN", self.codes(report))
        self.assertFalse(report["passed"])

    def test_non_object_child_plan_entry_still_fails_closed(self):
        value = decision("reject_merge", "broader_narrower", "CHILD", "PARENT")
        value["child_plan"] = ["not an object"]
        report = self.validate(value)
        self.assertIn("CHILD_PLAN_ITEM_NOT_OBJECT", self.codes(report))
        self.assertFalse(report["passed"])

    def test_child_plan_must_still_be_a_list(self):
        value = decision("reject_merge", "broader_narrower", "CHILD", "PARENT")
        value["child_plan"] = {"child_id": "GRANDCHILD"}
        report = self.validate(value)
        self.assertIn("CHILD_PLAN_NOT_LIST", self.codes(report))
        self.assertFalse(report["passed"])

    # ------------------------------------------------------------------
    # 4. the executor never consumes a tolerated child_plan
    # ------------------------------------------------------------------
    def test_executor_returns_before_consuming_a_tolerated_child_plan(self):
        expected_status = {
            "keep": "skipped",
            "reject_merge": "rejected",
            "uncertain": "rejected",
        }
        for action in NON_MUTATING_ACTIONS:
            with self.subTest(action=action):
                manager = TreeManager(copy.deepcopy(self.tree))
                before = canonical_payload_sha256(manager.root)
                record = self.execute(
                    decision(
                        action, relation_for(action), "CHILD", "PARENT",
                        children=[plan_item("GRANDCHILD", "CHILD")],
                    ),
                    manager,
                )
                after = canonical_payload_sha256(manager.root)
                self.assertEqual(record["status"], expected_status[action])
                self.assertTrue(record["semantic_contract"]["passed"])
                self.assertEqual(record["semantic_contract"]["critical_count"], 0)
                self.assertEqual(before, after)
                # no mutation bookkeeping may appear for a non-mutating action
                self.assertNotIn("lineage_target", record)
                self.assertNotIn("after_path", record)

    def test_tolerated_plan_does_not_move_the_named_child(self):
        manager = TreeManager(copy.deepcopy(self.tree))
        self.execute(
            decision(
                "reject_merge", "broader_narrower", "CHILD", "PARENT",
                children=[plan_item("GRANDCHILD", "PARENT")],
            ),
            manager,
        )
        # even though the entry names PARENT as target_parent_id, a keep
        # disposition is inert and the grandchild must stay under CHILD
        self.assertEqual(str(manager.get_parent_id("GRANDCHILD")), "CHILD")
        self.assertEqual(
            [str(item.get("node_id")) for item in manager.get_children("PARENT")],
            ["CHILD"],
        )

    # ------------------------------------------------------------------
    # 5. mutating actions keep their own child_plan rules unchanged
    # ------------------------------------------------------------------
    def test_rename_still_forbids_any_child_plan_including_all_keep(self):
        value = decision(
            "rename", "synonym", "CHILD", "CHILD",
            new_label="renamed child",
            children=[plan_item("GRANDCHILD", "CHILD")],
        )
        report = self.validate(value)
        self.assertIn("RENAME_CHILD_PLAN_NOT_EMPTY", self.codes(report))
        self.assertFalse(report["passed"])

    def test_merge_still_requires_a_complete_move_plan(self):
        value = decision(
            "merge", "synonym", "CHILD", "PARENT",
            children=[plan_item("GRANDCHILD", "CHILD")],
        )
        report = self.validate(value)
        codes = self.codes(report)
        self.assertNotIn(RELAXED_CODE, codes)
        self.assertFalse(report["passed"])

    def test_split_reparent_still_requires_at_least_one_move(self):
        value = decision(
            "split_reparent", "broader_narrower", "CHILD", "CHILD",
            children=[plan_item("GRANDCHILD", "CHILD")],
        )
        report = self.validate(value)
        codes = self.codes(report)
        self.assertNotIn(RELAXED_CODE, codes)
        self.assertIn("SPLIT_HAS_NO_MOVES", codes)
        self.assertFalse(report["passed"])

    def test_flatten_and_create_bridge_child_plan_rules_are_unchanged(self):
        flatten = decision(
            "flatten", "broader_narrower", "CHILD", "PARENT",
            children=[plan_item("GRANDCHILD", "CHILD")],
            evidence=evidence(pure_structural_redundancy=True),
        )
        report = self.validate(flatten)
        self.assertNotIn(RELAXED_CODE, self.codes(report))
        self.assertFalse(report["passed"])

        bridge = decision(
            "create_bridge", "broader_narrower", "CHILD", "NEW_BRIDGE",
            new_label="bridge label",
            children=[plan_item("GRANDCHILD", "NEW_BRIDGE")],
        )
        report = self.validate(bridge)
        self.assertNotIn(RELAXED_CODE, self.codes(report))
        self.assertFalse(report["passed"])


if __name__ == "__main__":
    unittest.main()
