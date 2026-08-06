from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from utils.tree_integrity import (  # noqa: E402
    E0ValidationError,
    LineageError,
    close_lineage,
    publish_tree_if_valid,
    redirect_membership_rows,
    validate_tree_e0,
)
from utils.tree_manager import TreeManager  # noqa: E402


FIXTURE_PATH = REPO_ROOT / "tests" / "fixtures" / "c13_duplicate_cases.json"


def node(node_id, label, level, children=None):
    return {
        "node_id": node_id,
        "label": label,
        "level": level,
        "children": list(children or []),
    }


def build_duplicate_fixture(manifest):
    root_children = []
    sibling_pairs = []
    parent_child_pairs = []
    for index, case in enumerate(manifest["sibling_cases"]):
        left_id = f"S{index}_LEFT"
        right_id = f"S{index}_RIGHT"
        parent_id = f"S{index}_PARENT"
        root_children.append(
            node(
                parent_id,
                f"Sibling fixture {index}",
                "L1",
                [
                    node(left_id, case["label"], "L2"),
                    node(right_id, case["label"], "L2"),
                ],
            )
        )
        sibling_pairs.append((left_id, right_id))
    for index, case in enumerate(manifest["parent_child_cases"]):
        parent_id = f"P{index}_PARENT"
        child_id = f"P{index}_CHILD"
        root_children.append(
            node(parent_id, case["label"], "L1", [node(child_id, case["label"], "L2")])
        )
        parent_child_pairs.append((parent_id, child_id))
    return node("ROOT", "ROOT", "ROOT", root_children), sibling_pairs, parent_child_pairs


class TreeIntegrityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))

    def test_lineage_closure_and_cycle_detection(self):
        self.assertEqual(close_lineage({"A": "B", "B": "C"}), {"A": "C", "B": "C"})
        with self.assertRaises(LineageError):
            close_lineage({"A": "B", "B": "A"})

    def test_membership_redirect_is_closed_and_conserved(self):
        source = [
            {"sample_id": "S1", "final_node_id": "A", "original_node_id": "A"},
            {"sample_id": "S2", "final_node_id": "C", "original_node_id": "C"},
        ]
        result = redirect_membership_rows(source, {"A": "B", "B": "C"}, {"C"})
        self.assertEqual([row["final_node_id"] for row in result], ["C", "C"])
        self.assertEqual([row["sample_id"] for row in result], ["S1", "S2"])
        with self.assertRaises(LineageError):
            redirect_membership_rows(source, {"A": "DELETED"}, {"C"})

    def test_all_15_plus_3_real_cases_trigger_and_can_be_repaired(self):
        self.assertEqual(len(self.manifest["sibling_cases"]), 15)
        self.assertEqual(len(self.manifest["parent_child_cases"]), 3)
        tree, sibling_pairs, parent_child_pairs = build_duplicate_fixture(self.manifest)
        report = validate_tree_e0(tree)
        self.assertFalse(report["passed"])
        self.assertEqual(report["stats"]["sibling_duplicate_groups"], 15)
        self.assertEqual(report["stats"]["parent_child_duplicates"], 3)
        self.assertEqual(report["stats"]["level_depth_mismatches"], 0)

        manager = TreeManager(tree)
        for winner, loser in sibling_pairs + parent_child_pairs:
            self.assertTrue(manager.absorb_node(winner, loser), manager.last_error)
        repaired = validate_tree_e0(tree, manager=manager)
        self.assertTrue(repaired["passed"], repaired["violations"])

    def test_applied_operation_membership_and_lineage_truth(self):
        tree = node("ROOT", "ROOT", "ROOT", [node("W", "winner", "L1")])
        manager = TreeManager(tree)
        source_membership = [
            {"sample_id": "S1", "final_node_id": "L", "original_node_id": "L"}
        ]
        candidate_membership = redirect_membership_rows(source_membership, {"L": "W"}, {"W"})
        operation = {
            "type": "merge",
            "source_id": "L",
            "target_id": "W",
            "status": "applied",
        }
        report = validate_tree_e0(
            tree,
            manager=manager,
            membership_rows=candidate_membership,
            expected_membership_rows=source_membership,
            lineage={"L": "W"},
            operations=[operation],
            require_membership=True,
        )
        self.assertTrue(report["passed"], report["violations"])

        false_operation = {**operation, "source_id": "W"}
        rejected = validate_tree_e0(tree, operations=[false_operation])
        self.assertIn("APPLIED_MERGE_UNTRUE", rejected["violation_counts"])

    def test_chained_merge_lineage_resolves_to_live_terminal_target(self):
        tree = node("ROOT", "ROOT", "ROOT", [node("C", "terminal", "L1")])
        operations = [
            {"type": "merge", "source_id": "A", "target_id": "B", "status": "applied"},
            {"type": "merge", "source_id": "B", "target_id": "C", "status": "applied"},
        ]
        report = validate_tree_e0(
            tree,
            lineage={"A": "B", "B": "C"},
            operations=operations,
        )
        self.assertTrue(report["passed"], report["violations"])

    def test_lineage_sources_and_unknown_applied_operations_fail_closed(self):
        tree = node(
            "ROOT", "ROOT", "ROOT",
            [node("A", "source", "L1"), node("C", "target", "L1")],
        )
        report = validate_tree_e0(
            tree,
            lineage={"A": "C"},
            operations=[{"type": "unvalidated_edit", "status": "applied"}],
        )
        self.assertIn("LINEAGE_SOURCE_STILL_LIVE", report["violation_counts"])
        self.assertIn("APPLIED_OPERATION_UNSUPPORTED", report["violation_counts"])

    def test_fail_closed_publisher_preserves_existing_output(self):
        bad_tree = node(
            "ROOT", "ROOT", "ROOT",
            [node("P", "parent", "L1", [node("A", "same", "L2"), node("B", "same", "L2")])],
        )
        with tempfile.TemporaryDirectory() as temporary_dir:
            output = Path(temporary_dir) / "policy_tree_final.json"
            audit = Path(temporary_dir) / "policy_tree_final_audit.json"
            output.write_text("sentinel", encoding="utf-8")
            with self.assertRaises(E0ValidationError):
                publish_tree_if_valid(bad_tree, output, audit)
            self.assertEqual(output.read_text(encoding="utf-8"), "sentinel")
            self.assertFalse(json.loads(audit.read_text(encoding="utf-8"))["passed"])

            good_tree = node("ROOT", "ROOT", "ROOT", [node("A", "alpha", "L1")])
            report = publish_tree_if_valid(good_tree, output, audit)
            self.assertTrue(report["passed"])
            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), good_tree)

    def test_frozen_353_tree_is_an_exact_negative_control(self):
        source = REPO_ROOT / "data" / "final_tree" / "policy_tree_final.json"
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        expected = self.manifest["sources"]["old_353"]
        self.assertEqual(digest, expected["tree_sha256"])
        tree = json.loads(source.read_text(encoding="utf-8"))
        report = validate_tree_e0(tree)
        self.assertFalse(report["passed"])
        self.assertEqual(report["stats"]["raw_node_count"], expected["expected_nodes"])
        self.assertEqual(
            report["stats"]["sibling_duplicate_groups"],
            expected["expected_sibling_groups"],
        )
        self.assertEqual(
            report["stats"]["parent_child_duplicates"],
            expected["expected_parent_child_pairs"],
        )
        self.assertEqual(
            report["stats"]["level_depth_mismatches"],
            expected["expected_level_mismatches"],
        )


if __name__ == "__main__":
    unittest.main()
