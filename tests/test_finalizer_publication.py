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

from utils.tree_integrity import E0ValidationError, atomic_write_csv  # noqa: E402
from utils.tree_manager import TreeManager  # noqa: E402


def load_finalizer_module():
    runtime = types.ModuleType("llm_runtime")
    runtime.call_llm_json = lambda **kwargs: {"json": {"operations": []}}
    sys.modules["llm_runtime"] = runtime

    shared = types.ModuleType("utils.step4_shared")
    shared.Step4Env = object
    shared.load_tree = lambda path: json.loads(Path(path).read_text(encoding="utf-8"))
    shared.append_jsonl = lambda path, data: Path(path).open("a", encoding="utf-8").write(
        json.dumps(data, ensure_ascii=False) + "\n"
    )
    sys.modules["utils.step4_shared"] = shared

    spec = importlib.util.spec_from_file_location(
        "c13_finalize_policy_tree", SCRIPTS / "finalize_policy_tree.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def node(node_id, label, level, children=None):
    return {
        "node_id": node_id,
        "label": label,
        "level": level,
        "children": list(children or []),
    }


class FakeEnv:
    def __init__(self, root: Path):
        self.outdir = root / "intermediate"
        self.log_dir = self.outdir / "logs"
        self.outdir.mkdir(parents=True)
        self.log_dir.mkdir(parents=True)

    def primary_llm_profile(self):
        return "offline-test"


class FinalizerPublicationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.finalizer = load_finalizer_module()

    def make_args(self, root: Path):
        formal = root / "formal"
        formal.mkdir()
        l1_def = root / "l1.json"
        l1_def.write_text('{"categories": []}', encoding="utf-8")
        return types.SimpleNamespace(
            output=str(formal / "policy_tree_final.json"),
            audit_out=str(formal / "policy_tree_final_audit.json"),
            flat_csv=str(formal / "policy_tree_final_flat.csv"),
            membership_input=None,
            membership_out=str(formal / "policy_tree_final_membership.csv"),
            lineage_in=None,
            lineage_out=str(formal / "policy_tree_final_lineage.json"),
            operations_input=None,
            operations_out=str(formal / "policy_tree_final_operations.jsonl"),
            l1_def=str(l1_def),
        )

    def prepare_membership(self, env: FakeEnv, node_id: str):
        atomic_write_csv(
            env.outdir / "policy_tree_final_membership.csv",
            ["sample_id", "final_node_id", "original_node_id", "original_level"],
            [{
                "sample_id": "S1",
                "final_node_id": node_id,
                "original_node_id": node_id,
                "original_level": "L2",
            }],
        )

    def test_e0_failure_leaves_all_formal_outputs_unchanged(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root_dir = Path(temporary_dir)
            env = FakeEnv(root_dir)
            args = self.make_args(root_dir)
            tree = node(
                "ROOT", "ROOT", "ROOT",
                [node("P", "parent", "L1", [node("A", "same", "L2"), node("B", "same", "L2")])],
            )
            self.prepare_membership(env, "A")
            formal_paths = [
                Path(args.output),
                Path(args.flat_csv),
                Path(args.membership_out),
                Path(args.lineage_out),
                Path(args.operations_out),
            ]
            for path in formal_paths:
                path.write_text("sentinel", encoding="utf-8")

            process = self.finalizer.OverallStructureAudit(env, TreeManager(tree), args)
            process._call_llm = lambda context, l1_id: []
            with self.assertRaises(E0ValidationError):
                process.run()

            for path in formal_paths:
                self.assertEqual(path.read_text(encoding="utf-8"), "sentinel")
            audit = json.loads(Path(args.audit_out).read_text(encoding="utf-8"))
            self.assertFalse(audit["e0"]["passed"])

    def test_candidate_preparation_failure_writes_audit_without_publication(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root_dir = Path(temporary_dir)
            env = FakeEnv(root_dir)
            args = self.make_args(root_dir)
            tree = node("ROOT", "ROOT", "ROOT", [node("A", "alpha", "L1")])
            self.prepare_membership(env, "A")
            (env.outdir / "policy_tree_lineage.json").write_text(
                '{"A": "B", "B": "A"}', encoding="utf-8"
            )
            formal_paths = [
                Path(args.output),
                Path(args.flat_csv),
                Path(args.membership_out),
                Path(args.lineage_out),
                Path(args.operations_out),
            ]
            for path in formal_paths:
                path.write_text("sentinel", encoding="utf-8")

            process = self.finalizer.OverallStructureAudit(env, TreeManager(tree), args)
            process._call_llm = lambda context, l1_id: []
            with self.assertRaises(E0ValidationError):
                process.run()

            for path in formal_paths:
                self.assertEqual(path.read_text(encoding="utf-8"), "sentinel")
            audit = json.loads(Path(args.audit_out).read_text(encoding="utf-8"))
            self.assertIn(
                "CANDIDATE_PREPARATION_FAILED",
                audit["e0"]["violation_counts"],
            )

    def test_explicit_missing_membership_fails_without_publication(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root_dir = Path(temporary_dir)
            env = FakeEnv(root_dir)
            args = self.make_args(root_dir)
            args.membership_input = str(root_dir / "missing-membership.csv")
            tree = node("ROOT", "ROOT", "ROOT", [node("A", "alpha", "L1")])
            Path(args.output).write_text("sentinel", encoding="utf-8")

            process = self.finalizer.OverallStructureAudit(env, TreeManager(tree), args)
            process._call_llm = lambda context, l1_id: []
            with self.assertRaises(E0ValidationError):
                process.run()

            self.assertEqual(Path(args.output).read_text(encoding="utf-8"), "sentinel")
            audit = json.loads(Path(args.audit_out).read_text(encoding="utf-8"))
            self.assertIn(
                "CANDIDATE_PREPARATION_FAILED",
                audit["e0"]["violation_counts"],
            )

    def test_false_pre_finalization_operation_blocks_publication(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root_dir = Path(temporary_dir)
            env = FakeEnv(root_dir)
            args = self.make_args(root_dir)
            tree = node("ROOT", "ROOT", "ROOT", [node("A", "alpha", "L1")])
            self.prepare_membership(env, "A")
            (env.outdir / "tree_refinement_operations.jsonl").write_text(
                json.dumps({
                    "type": "merge",
                    "source_id": "A",
                    "target_id": "A",
                    "status": "applied",
                }) + "\n",
                encoding="utf-8",
            )
            Path(args.output).write_text("sentinel", encoding="utf-8")

            process = self.finalizer.OverallStructureAudit(env, TreeManager(tree), args)
            process._call_llm = lambda context, l1_id: []
            with self.assertRaises(E0ValidationError):
                process.run()

            self.assertEqual(Path(args.output).read_text(encoding="utf-8"), "sentinel")
            audit = json.loads(Path(args.audit_out).read_text(encoding="utf-8"))
            self.assertIn("APPLIED_MERGE_UNTRUE", audit["e0"]["violation_counts"])

    def test_e0_pass_publishes_formal_tree_last(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root_dir = Path(temporary_dir)
            env = FakeEnv(root_dir)
            args = self.make_args(root_dir)
            tree = node("ROOT", "ROOT", "ROOT", [node("A", "alpha", "L1")])
            self.prepare_membership(env, "A")

            events = []
            original_json = self.finalizer.atomic_write_json
            original_csv = self.finalizer.atomic_write_csv

            def record_json(path, payload):
                events.append(Path(path))
                original_json(path, payload)

            def record_csv(path, fieldnames, rows):
                events.append(Path(path))
                original_csv(path, fieldnames, rows)

            self.finalizer.atomic_write_json = record_json
            self.finalizer.atomic_write_csv = record_csv
            try:
                process = self.finalizer.OverallStructureAudit(env, TreeManager(tree), args)
                process._call_llm = lambda context, l1_id: []
                process.run()
            finally:
                self.finalizer.atomic_write_json = original_json
                self.finalizer.atomic_write_csv = original_csv

            self.assertEqual(events[-1], Path(args.output))
            self.assertEqual(json.loads(Path(args.output).read_text(encoding="utf-8")), tree)
            audit = json.loads(Path(args.audit_out).read_text(encoding="utf-8"))
            self.assertTrue(audit["e0"]["passed"])

    def test_publication_write_failure_rolls_back_all_formal_outputs(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root_dir = Path(temporary_dir)
            env = FakeEnv(root_dir)
            args = self.make_args(root_dir)
            tree = node("ROOT", "ROOT", "ROOT", [node("A", "alpha", "L1")])
            self.prepare_membership(env, "A")
            formal_paths = [
                Path(args.output),
                Path(args.flat_csv),
                Path(args.membership_out),
                Path(args.lineage_out),
                Path(args.operations_out),
            ]
            for path in formal_paths:
                path.write_text("sentinel", encoding="utf-8")

            original_jsonl = self.finalizer.atomic_write_jsonl

            def fail_operations_write(path, records):
                raise OSError("injected publication failure")

            self.finalizer.atomic_write_jsonl = fail_operations_write
            try:
                process = self.finalizer.OverallStructureAudit(env, TreeManager(tree), args)
                process._call_llm = lambda context, l1_id: []
                with self.assertRaises(E0ValidationError):
                    process.run()
            finally:
                self.finalizer.atomic_write_jsonl = original_jsonl

            for path in formal_paths:
                self.assertEqual(path.read_text(encoding="utf-8"), "sentinel")
            audit = json.loads(Path(args.audit_out).read_text(encoding="utf-8"))
            self.assertFalse(audit["e0"]["passed"])
            self.assertIn("PUBLICATION_FAILED", audit["e0"]["violation_counts"])


if __name__ == "__main__":
    unittest.main()
