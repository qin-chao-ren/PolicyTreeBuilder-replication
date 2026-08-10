from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

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
        "summary": "The requested structure preserves each policy domain.",
        "warnings": [],
        "target_represents_all_source_members": False,
        "membership_basis": "",
        "pure_structural_redundancy": False,
        "cross_l1_authorized": False,
    }
    value.update(overrides)
    return value


def child_plan(child_id, target_id, disposition="move"):
    return {
        "child_id": child_id,
        "disposition": disposition,
        "target_parent_id": target_id,
        "relation": "broader_narrower",
        "same_domain": True,
        "evidence": "The child remains in the displayed destination domain.",
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


def append_jsonl(path, payload):
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


def load_stage_module(name):
    runtime = types.ModuleType("llm_runtime")
    runtime.call_llm_json = lambda **kwargs: {"json": {"decisions": []}}
    sys.modules["llm_runtime"] = runtime

    common = types.ModuleType("common_utils")
    common.jaccard_overlap = lambda left, right: 0.0
    sys.modules["common_utils"] = common

    shared = types.ModuleType("utils.step4_shared")
    shared.Step4Env = object
    shared.EmbeddingHelper = object
    shared.load_tree = lambda path: {}
    shared.dump_tree = lambda path, payload: None
    shared.append_jsonl = append_jsonl
    shared.read_membership_map = lambda outdir, level: {}
    shared.read_title_map = lambda path: {}
    sys.modules["utils.step4_shared"] = shared

    spec = importlib.util.spec_from_file_location(
        f"c13r_{name}", SCRIPTS / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


class SemanticStageDispatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.collapse = load_stage_module("collapse_redundant_hierarchy")
        cls.balance = load_stage_module("balance_tree_structure")
        cls.polish = load_stage_module("polish_tree_labels")

    def test_vertical_split_preserves_umbrella_and_reparents_one_child(self):
        tree = node("ROOT", "ROOT", "ROOT", [
            node("L1", "domain", "L1", [
                node("P", "parent", "L2", [
                    node("C", "umbrella", "L3", [
                        node("G1", "kept", "L4"),
                        node("G2", "moved", "L4"),
                    ]),
                ]),
            ]),
        ])
        with tempfile.TemporaryDirectory() as temporary_dir:
            process = self.collapse.SkeletonRefiner.__new__(
                self.collapse.SkeletonRefiner
            )
            process.tm = TreeManager(tree)
            process.membership_counts = {}
            process.redirect_map = {}
            process.ops_log = Path(temporary_dir) / "ops.jsonl"
            value = decision(
                "split_reparent", "related", "C", "C",
                children=[
                    child_plan("G1", "C", disposition="keep"),
                    child_plan("G2", "P"),
                ],
            )
            process._execute_decision(value, "P", "C")

            self.assertTrue(process.tm.exists("C"))
            self.assertEqual(process.tm.get_parent_id("G1"), "C")
            self.assertEqual(process.tm.get_parent_id("G2"), "P")
            record = json.loads(process.ops_log.read_text(encoding="utf-8"))
            self.assertEqual(record["status"], "applied", record)
            self.assertEqual(record["type"], "split_reparent")

    def test_balancer_requires_explicit_bridge_placeholder_then_materializes_id(self):
        base = node("ROOT", "ROOT", "ROOT", [
            node("L1", "domain", "L1", [
                node("P", "parent", "L2", [
                    node("C1", "one", "L3"),
                    node("C2", "two", "L3"),
                ]),
            ]),
        ])
        with tempfile.TemporaryDirectory() as temporary_dir:
            process = self.balance.ShapingProcess.__new__(self.balance.ShapingProcess)
            process.tm = TreeManager(json.loads(json.dumps(base)))
            process.membership_counts = {}
            process.ops_log = Path(temporary_dir) / "rejected.jsonl"
            process._current_lineage = lambda: {}
            unsafe = decision(
                "create_bridge", "broader_narrower", "P", "invented",
                children=[child_plan("C1", "invented")],
                new_label="group",
            )
            changed = process._apply("P", {"decisions": [unsafe]}, "fanout")
            self.assertFalse(changed)
            self.assertEqual(set(process.tm.get_all_node_ids()), {"ROOT", "L1", "P", "C1", "C2"})

            process.ops_log = Path(temporary_dir) / "accepted.jsonl"
            placeholder = self.balance.BRIDGE_TARGET_PLACEHOLDER
            safe = decision(
                "create_bridge", "broader_narrower", "P", placeholder,
                children=[child_plan("C1", placeholder), child_plan("C2", placeholder)],
                new_label="group",
            )
            changed = process._apply("P", {"decisions": [safe]}, "fanout")
            self.assertTrue(changed)
            record = json.loads(process.ops_log.read_text(encoding="utf-8"))
            bridge_id = record["bridge_id"]
            self.assertNotEqual(bridge_id, placeholder)
            self.assertEqual(process.tm.get_parent_id("C1"), bridge_id)
            self.assertEqual(process.tm.get_parent_id("C2"), bridge_id)

    def test_polisher_split_targets_are_limited_to_displayed_scope(self):
        base = node("ROOT", "ROOT", "ROOT", [
            node("L1", "domain", "L1", [
                node("P1", "parent one", "L2", [
                    node("A", "umbrella", "L3", [node("C", "child", "L4")]),
                ]),
                node("P2", "parent two", "L2", [node("B", "peer", "L3")]),
                node("H", "hidden", "L2"),
            ]),
        ])
        with tempfile.TemporaryDirectory() as temporary_dir:
            process = self.polish.PolishingProcess.__new__(self.polish.PolishingProcess)
            process.tm = TreeManager(base)
            process.membership_counts = {}
            process.local_trace_map = {}
            process._current_lineage = lambda: {}
            process.ops_log = Path(temporary_dir) / "ops.jsonl"
            unsafe = decision(
                "split_reparent", "related", "A", "A",
                children=[child_plan("C", "H")],
            )
            process._execute_decision(unsafe, "A", "B", "cross_parent_unify")

            self.assertEqual(process.tm.get_parent_id("C"), "A")
            record = json.loads(process.ops_log.read_text(encoding="utf-8"))
            self.assertEqual(record["status"], "rejected")
            self.assertIn(
                "DECISION_SCOPE_VIOLATION",
                record["semantic_contract"]["violation_counts"],
            )


if __name__ == "__main__":
    unittest.main()
