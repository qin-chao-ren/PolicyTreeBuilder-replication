from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from utils.semantic_contract import (  # noqa: E402
    ACTIONS,
    RELATIONS,
    execute_semantic_decision,
    parse_semantic_decisions,
    validate_semantic_history,
)
from utils.tree_manager import TreeManager  # noqa: E402


def node(node_id, label, level, children=None):
    return {
        "node_id": node_id,
        "label": label,
        "level": level,
        "children": list(children or []),
    }


def evidence(**overrides):
    value = {
        "summary": "Labels and represented records have the same meaning.",
        "warnings": [],
        "target_represents_all_source_members": False,
        "membership_basis": "",
        "pure_structural_redundancy": False,
        "cross_l1_authorized": False,
    }
    value.update(overrides)
    return value


def child_plan(child_id, target_id, disposition="move", relation="broader_narrower"):
    return {
        "child_id": child_id,
        "disposition": disposition,
        "target_parent_id": target_id,
        "relation": relation,
        "same_domain": True,
        "evidence": "The child remains inside the destination policy domain.",
    }


def decision(action, relation, source_id, target_id, *, children=None, **changes):
    value = {
        "relation": relation,
        "action": action,
        "source_id": source_id,
        "target_id": target_id,
        "new_label": None,
        "confidence": 0.95,
        "evidence": evidence(),
        "child_plan": list(children or []),
    }
    value.update(changes)
    return value


class SemanticContractTests(unittest.TestCase):
    def execute(
        self,
        manager,
        value,
        *,
        counts=None,
        membership_known=True,
        lineage=None,
        allow_cross_l1=False,
        allowed_l1_id=None,
        new_node_level=None,
    ):
        return execute_semantic_decision(
            manager,
            value,
            direct_membership_counts=counts or {},
            membership_known=membership_known,
            stage="synthetic_test",
            lineage=lineage,
            allow_cross_l1=allow_cross_l1,
            allowed_l1_id=allowed_l1_id,
            new_node_level=new_node_level,
        )

    def test_public_schema_enums_match_runtime_contract(self):
        schema = json.loads(
            (REPO_ROOT / "schemas" / "semantic_tree_decision.schema.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(set(schema["$defs"]["relation"]["enum"]), set(RELATIONS))
        self.assertEqual(set(schema["$defs"]["action"]["enum"]), set(ACTIONS))

    def test_shared_parser_never_invents_missing_or_invalid_decisions(self):
        values, errors = parse_semantic_decisions(None, expected_count=1)
        self.assertEqual(values, [])
        self.assertEqual(errors[0]["code"], "ENVELOPE_NOT_OBJECT")

        values, errors = parse_semantic_decisions({}, expected_count=1)
        self.assertEqual(values, [])
        self.assertEqual(errors[0]["code"], "DECISIONS_MISSING")

        values, errors = parse_semantic_decisions(
            {"decisions": ["bad"]}, expected_count=1
        )
        self.assertEqual(values, [])
        self.assertEqual(errors[0]["code"], "DECISION_NOT_OBJECT")

        values, errors = parse_semantic_decisions(
            {"decisions": [], "unexpected": True}, expected_count=0
        )
        self.assertEqual(values, [])
        self.assertEqual(errors[0]["code"], "ENVELOPE_FIELDS_UNKNOWN")

    def test_runtime_rejects_fields_forbidden_by_the_public_schema(self):
        tree = node(
            "ROOT", "ROOT", "ROOT",
            [node("L1", "domain", "L1", [node("W", "winner", "L2"), node("S", "source", "L2")])],
        )
        cases = []
        extra_decision = decision("merge", "synonym", "S", "W")
        extra_decision["unexpected"] = True
        cases.append((extra_decision, "DECISION_FIELDS_UNKNOWN"))
        extra_evidence = decision("merge", "synonym", "S", "W")
        extra_evidence["evidence"]["unexpected"] = True
        cases.append((extra_evidence, "EVIDENCE_FIELDS_UNKNOWN"))

        tree_with_child = node(
            "ROOT", "ROOT", "ROOT",
            [node("L1", "domain", "L1", [
                node("W", "winner", "L2"),
                node("S", "source", "L2", [node("C", "child", "L3")]),
            ])],
        )
        extra_child = decision(
            "merge", "synonym", "S", "W", children=[child_plan("C", "W")]
        )
        extra_child["child_plan"][0]["unexpected"] = True
        cases.append((extra_child, "CHILD_PLAN_FIELDS_UNKNOWN", tree_with_child))

        for item in cases:
            value, expected_code, *custom_tree = item
            with self.subTest(expected_code=expected_code):
                manager = TreeManager(copy.deepcopy(custom_tree[0] if custom_tree else tree))
                record = self.execute(manager, value)
                self.assertEqual(record["status"], "rejected")
                self.assertIn(expected_code, record["semantic_contract"]["violation_counts"])

    def test_direct_non_object_execution_fails_closed_with_a_record(self):
        tree = node("ROOT", "ROOT", "ROOT", [node("L1", "domain", "L1")])
        manager = TreeManager(tree)
        record = self.execute(manager, "not-an-object")
        self.assertEqual(record["status"], "rejected")
        self.assertIn(
            "DECISION_NOT_OBJECT",
            record["semantic_contract"]["violation_counts"],
        )

    def test_exact_duplicate_and_synonym_merges_remain_available(self):
        for relation in ("exact_duplicate", "synonym"):
            with self.subTest(relation=relation):
                tree = node(
                    "ROOT", "ROOT", "ROOT",
                    [node("L1", "domain", "L1", [
                        node("W", "same", "L2"),
                        node("S", "same wording", "L2"),
                    ])],
                )
                manager = TreeManager(tree)
                record = self.execute(manager, decision("merge", relation, "S", "W"))
                self.assertEqual(record["status"], "applied", record)
                self.assertFalse(manager.exists("S"))
                self.assertTrue(manager.exists("W"))

    def test_non_synonym_relation_cannot_authorize_destructive_merge(self):
        tree = node(
            "ROOT", "ROOT", "ROOT",
            [node("L1", "domain", "L1", [node("W", "goal", "L2"), node("S", "means", "L2")])],
        )
        manager = TreeManager(tree)
        before = copy.deepcopy(tree)
        record = self.execute(manager, decision("merge", "means_goal", "S", "W"))
        self.assertEqual(record["status"], "rejected")
        self.assertIn(
            "DESTRUCTIVE_MERGE_RELATION_FORBIDDEN",
            record["semantic_contract"]["violation_counts"],
        )
        self.assertEqual(tree, before)

    def test_missing_unknown_and_conflicting_fields_fail_closed(self):
        tree = node(
            "ROOT", "ROOT", "ROOT",
            [node("L1", "domain", "L1", [node("W", "winner", "L2"), node("S", "source", "L2")])],
        )
        cases = [
            {"relation": "synonym"},
            decision("merge", "unknown_relation", "S", "W"),
            decision("unknown_action", "synonym", "S", "W"),
            decision("uncertain", "synonym", "S", "W"),
        ]
        for value in cases:
            with self.subTest(value=value):
                manager = TreeManager(copy.deepcopy(tree))
                record = self.execute(manager, value)
                self.assertEqual(record["status"], "rejected")
                self.assertEqual(set(manager.index), {"ROOT", "L1", "W", "S"})

    def test_warning_flags_and_warning_text_block_merge(self):
        tree = node(
            "ROOT", "ROOT", "ROOT",
            [node("L1", "domain", "L1", [node("W", "winner", "L2"), node("S", "source", "L2")])],
        )
        warning_cases = [
            evidence(warnings=["heterogeneous_membership"]),
            evidence(summary="The labels are related but not equivalent."),
            evidence(summary="两个标签是上下位关系。"),
        ]
        for proof in warning_cases:
            with self.subTest(proof=proof):
                manager = TreeManager(copy.deepcopy(tree))
                value = decision("merge", "synonym", "S", "W", evidence=proof)
                record = self.execute(manager, value)
                self.assertEqual(record["status"], "rejected")
                self.assertTrue(manager.exists("S"))

    def test_direct_membership_requires_explicit_full_representation_proof(self):
        tree = node(
            "ROOT", "ROOT", "ROOT",
            [node("L1", "domain", "L1", [node("W", "winner", "L2"), node("S", "source", "L2")])],
        )
        manager = TreeManager(copy.deepcopy(tree))
        rejected = self.execute(
            manager,
            decision("merge", "synonym", "S", "W"),
            counts={"S": 2},
        )
        self.assertEqual(rejected["status"], "rejected")
        self.assertIn(
            "SOURCE_MEMBERSHIP_NOT_REPRESENTED",
            rejected["semantic_contract"]["violation_counts"],
        )

        manager = TreeManager(copy.deepcopy(tree))
        proof = evidence(
            target_represents_all_source_members=True,
            membership_basis="Both source records are exact title variants represented by W.",
        )
        accepted = self.execute(
            manager,
            decision("merge", "synonym", "S", "W", evidence=proof),
            counts={"S": 2},
        )
        self.assertEqual(accepted["status"], "applied", accepted)

    def test_merge_requires_a_complete_compatible_child_plan(self):
        tree = node(
            "ROOT", "ROOT", "ROOT",
            [node("L1", "domain", "L1", [
                node("W", "winner", "L2"),
                node("S", "source", "L2", [node("C1", "one", "L3"), node("C2", "two", "L3")]),
            ])],
        )
        manager = TreeManager(copy.deepcopy(tree))
        incomplete = decision(
            "merge", "synonym", "S", "W",
            children=[child_plan("C1", "W")],
        )
        record = self.execute(manager, incomplete)
        self.assertEqual(record["status"], "rejected")
        self.assertIn("CHILD_PLAN_INCOMPLETE", record["semantic_contract"]["violation_counts"])

        manager = TreeManager(copy.deepcopy(tree))
        complete = decision(
            "merge", "synonym", "S", "W",
            children=[child_plan("C1", "W"), child_plan("C2", "W")],
        )
        record = self.execute(manager, complete)
        self.assertEqual(record["status"], "applied", record)
        self.assertEqual(manager.get_parent_id("C1"), "W")
        self.assertEqual(manager.get_parent_id("C2"), "W")

    def test_parent_into_child_merge_requires_single_child_and_explicit_plan(self):
        tree = node(
            "ROOT", "ROOT", "ROOT",
            [node("L1", "domain", "L1", [
                node("P", "same", "L2", [node("C", "same", "L3")]),
            ])],
        )
        manager = TreeManager(tree)
        value = decision(
            "merge", "exact_duplicate", "P", "C",
            children=[child_plan("C", "L1")],
        )
        record = self.execute(manager, value)
        self.assertEqual(record["status"], "applied", record)
        self.assertEqual(record["type"], "promote_child")
        self.assertFalse(manager.exists("P"))
        self.assertEqual(manager.get_parent_id("C"), "L1")

    def test_flatten_requires_zero_membership_redundancy_and_all_children(self):
        base = node(
            "ROOT", "ROOT", "ROOT",
            [node("L1", "domain", "L1", [
                node("F", "wrapper", "L2", [node("C1", "one", "L3"), node("C2", "two", "L3")]),
            ])],
        )
        value = decision(
            "flatten", "broader_narrower", "F", "L1",
            children=[child_plan("C1", "L1"), child_plan("C2", "L1")],
            evidence=evidence(pure_structural_redundancy=True),
        )
        manager = TreeManager(copy.deepcopy(base))
        rejected = self.execute(manager, value, counts={"F": 1})
        self.assertEqual(rejected["status"], "rejected")
        self.assertIn("FLATTEN_SOURCE_HAS_MEMBERSHIP", rejected["semantic_contract"]["violation_counts"])

        manager = TreeManager(copy.deepcopy(base))
        accepted = self.execute(manager, value)
        self.assertEqual(accepted["status"], "applied", accepted)
        self.assertFalse(manager.exists("F"))
        self.assertEqual(manager.get_parent_id("C1"), "L1")

    def test_split_reparent_is_atomic_and_preserves_the_umbrella(self):
        tree = node(
            "ROOT", "ROOT", "ROOT",
            [node("L1", "domain", "L1", [
                node("P", "umbrella", "L2", [node("C1", "one", "L3"), node("C2", "two", "L3")]),
                node("T1", "target one", "L2"),
                node("T2", "target two", "L2"),
            ])],
        )
        value = decision(
            "split_reparent", "related", "P", "P",
            children=[child_plan("C1", "T1"), child_plan("C2", "T2")],
        )
        manager = TreeManager(copy.deepcopy(tree))
        record = self.execute(manager, value)
        self.assertEqual(record["status"], "applied", record)
        self.assertTrue(manager.exists("P"))
        self.assertEqual(manager.get_parent_id("C1"), "T1")
        self.assertEqual(manager.get_parent_id("C2"), "T2")

        manager = TreeManager(copy.deepcopy(tree))
        before = copy.deepcopy(manager.root)
        original_move = manager._move_unchecked
        calls = 0

        def fail_second(source_id, target_id):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("injected split failure")
            original_move(source_id, target_id)

        manager._move_unchecked = fail_second
        record = self.execute(manager, value)
        self.assertEqual(record["status"], "rejected")
        self.assertEqual(manager.root, before)
        self.assertEqual(manager.validate_consistency(), [])

    def test_split_plan_must_cover_kept_and_moved_children(self):
        tree = node(
            "ROOT", "ROOT", "ROOT",
            [node("L1", "domain", "L1", [
                node("P", "umbrella", "L2", [node("C1", "one", "L3"), node("C2", "two", "L3")]),
                node("T", "target", "L2"),
            ])],
        )
        manager = TreeManager(tree)
        value = decision(
            "split_reparent", "related", "P", "P",
            children=[child_plan("C1", "T")],
        )
        record = self.execute(manager, value)
        self.assertEqual(record["status"], "rejected")
        self.assertIn("CHILD_PLAN_INCOMPLETE", record["semantic_contract"]["violation_counts"])

    def test_split_target_must_preserve_the_source_umbrella(self):
        tree = node(
            "ROOT", "ROOT", "ROOT",
            [node("L1", "domain", "L1", [
                node("P", "umbrella", "L2", [node("C", "child", "L3")]),
                node("T", "target", "L2"),
            ])],
        )
        manager = TreeManager(tree)
        value = decision(
            "split_reparent", "related", "P", "T",
            children=[child_plan("C", "T")],
        )
        record = self.execute(manager, value)
        self.assertEqual(record["status"], "rejected")
        self.assertIn("SPLIT_TARGET_CONFLICT", record["semantic_contract"]["violation_counts"])
        self.assertEqual(manager.get_parent_id("C"), "P")

    def test_move_of_parent_requires_child_compatibility_plan(self):
        tree = node(
            "ROOT", "ROOT", "ROOT",
            [node("L1", "domain", "L1", [
                node("P1", "old parent", "L2", [node("S", "source", "L3", [node("C", "child", "L4")])]),
                node("P2", "new parent", "L2"),
            ])],
        )
        manager = TreeManager(copy.deepcopy(tree))
        rejected = self.execute(manager, decision("move", "misplaced", "S", "P2"))
        self.assertEqual(rejected["status"], "rejected")
        self.assertIn("CHILD_PLAN_INCOMPLETE", rejected["semantic_contract"]["violation_counts"])

        manager = TreeManager(copy.deepcopy(tree))
        value = decision(
            "move", "misplaced", "S", "P2",
            children=[child_plan("C", "S", disposition="retain_under_source")],
        )
        accepted = self.execute(manager, value)
        self.assertEqual(accepted["status"], "applied", accepted)
        self.assertEqual(manager.get_parent_id("S"), "P2")
        self.assertEqual(manager.get_parent_id("C"), "S")

    def test_move_requires_misplaced_relation(self):
        tree = node(
            "ROOT", "ROOT", "ROOT",
            [node("L1", "domain", "L1", [
                node("P1", "old parent", "L2", [node("S", "source", "L3")]),
                node("P2", "new parent", "L2"),
            ])],
        )
        manager = TreeManager(tree)
        record = self.execute(manager, decision("move", "related", "S", "P2"))
        self.assertEqual(record["status"], "rejected")
        self.assertIn(
            "MOVE_RELATION_CONFLICT",
            record["semantic_contract"]["violation_counts"],
        )
        self.assertEqual(manager.get_parent_id("S"), "P1")

    def test_cross_l1_move_requires_explicit_action_and_two_authorizations(self):
        tree = node(
            "ROOT", "ROOT", "ROOT",
            [
                node("L1A", "domain A", "L1", [node("S", "source", "L2")]),
                node("L1B", "domain B", "L1", [node("T", "target", "L2")]),
            ],
        )
        manager = TreeManager(copy.deepcopy(tree))
        record = self.execute(manager, decision("move", "misplaced", "S", "T"))
        self.assertEqual(record["status"], "rejected")

        manager = TreeManager(copy.deepcopy(tree))
        value = decision(
            "move_across_l1", "misplaced", "S", "T",
            evidence=evidence(cross_l1_authorized=True),
        )
        record = self.execute(manager, value, allow_cross_l1=False)
        self.assertEqual(record["status"], "rejected")

        manager = TreeManager(copy.deepcopy(tree))
        record = self.execute(manager, value, allow_cross_l1=True)
        self.assertEqual(record["status"], "applied", record)
        self.assertEqual(manager.get_parent_id("S"), "T")

    def test_create_bridge_requires_per_child_domain_evidence(self):
        tree = node(
            "ROOT", "ROOT", "ROOT",
            [node("L1", "domain", "L1", [node("C1", "one", "L2"), node("C2", "two", "L2")])],
        )
        value = decision(
            "create_bridge", "broader_narrower", "L1", "BR",
            children=[child_plan("C1", "BR"), child_plan("C2", "BR")],
            new_label="shared group",
        )
        manager = TreeManager(tree)
        record = self.execute(manager, value, new_node_level="L2")
        self.assertEqual(record["status"], "applied", record)
        self.assertEqual(manager.get_parent_id("BR"), "L1")
        self.assertEqual(manager.get_parent_id("C1"), "BR")

    def test_publication_gate_revalidates_applied_semantic_operations(self):
        tree = node(
            "ROOT", "ROOT", "ROOT",
            [node("L1", "domain", "L1", [node("W", "winner", "L2"), node("S", "source", "L2")])],
        )
        manager = TreeManager(tree)
        record = self.execute(manager, decision("merge", "synonym", "S", "W"))
        report = validate_semantic_history([record])
        self.assertTrue(report["passed"], report["violations"])

        tampered = copy.deepcopy(record)
        tampered["relation"] = "related"
        report = validate_semantic_history([tampered])
        self.assertFalse(report["passed"])
        self.assertIn("APPLIED_SEMANTIC_DECISION_INVALID", report["violation_counts"])

        legacy = {
            "type": "merge",
            "source_id": "S",
            "target_id": "W",
            "status": "applied",
        }
        report = validate_semantic_history([legacy])
        self.assertIn("APPLIED_SEMANTIC_CONTEXT_MISSING", report["violation_counts"])

    def test_all_four_prompts_use_the_shared_envelope(self):
        prompt_paths = [
            REPO_ROOT / "prompts" / "collapse_redundant_hierarchy.md",
            REPO_ROOT / "prompts" / "balance_tree_structure.md",
            REPO_ROOT / "prompts" / "polish_tree_labels.md",
            REPO_ROOT / "prompts" / "finalize_tree_structure.md",
        ]
        required_tokens = (
            '"decisions"', '"relation"', '"action"', '"source_id"',
            '"target_id"', '"new_label"', '"confidence"', '"evidence"',
            '"child_plan"', '"target_represents_all_source_members"',
            '"pure_structural_redundancy"', '"cross_l1_authorized"',
        )
        for path in prompt_paths:
            with self.subTest(prompt=path.name):
                text = path.read_text(encoding="utf-8")
                for token in required_tokens:
                    self.assertIn(token, text)

        collapse = prompt_paths[0].read_text(encoding="utf-8")
        balance = prompt_paths[1].read_text(encoding="utf-8")
        polish = prompt_paths[2].read_text(encoding="utf-8")
        finalizer = prompt_paths[3].read_text(encoding="utf-8")
        self.assertNotIn("promote_child", collapse)
        self.assertNotIn('"groups"', balance)
        self.assertNotIn('"operation"', polish)
        self.assertNotIn('"operations"', finalizer)


if __name__ == "__main__":
    unittest.main()
