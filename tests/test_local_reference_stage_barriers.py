from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tests.test_semantic_stage_dispatch import (
    child_plan,
    decision,
    load_stage_module,
    node,
)

from utils.tree_manager import TreeManager


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


def payload_sha256(value):
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def successful_call_audit(payload, fill="A"):
    return {
        "ok": True,
        "logical_call_id": f"CALL_{fill * 64}_00000001",
        "json": copy.deepcopy(payload),
        "final_bound": copy.deepcopy(payload),
        "events": [{"seq": 4, "event": "scope_validation_pass"}],
    }


class LocalReferenceStageBarrierTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.collapse = load_stage_module("collapse_redundant_hierarchy")
        cls.balance = load_stage_module("balance_tree_structure")
        cls.polish = load_stage_module("polish_tree_labels")
        cls.finalize = load_stage_module("finalize_policy_tree")

    def assert_non_applied_audit_closes(self, records, call_audit):
        decisions = call_audit["final_bound"]["decisions"]
        self.assertEqual(
            [record.get("decision_index") for record in records],
            list(range(len(decisions))),
        )
        for record, bound in zip(records, decisions):
            executed = {
                field: copy.deepcopy(record.get(field))
                for field in AUDITED_DECISION_FIELDS
            }
            self.assertEqual(
                record.get("bound_decision_sha256"), payload_sha256(bound)
            )
            self.assertEqual(
                record.get("executed_decision_sha256"),
                payload_sha256(executed),
            )
            transform = record.get("execution_transform")
            if transform is None:
                self.assertEqual(executed, bound)
            else:
                self.assertEqual(transform, "materialize_new_bridge")
                expected = copy.deepcopy(bound)
                materialized_target = executed["target_id"]
                expected["target_id"] = materialized_target
                expected["child_plan"] = [
                    {
                        **copy.deepcopy(item),
                        "target_parent_id": materialized_target,
                    }
                    if item.get("target_parent_id") == "__NEW_BRIDGE__"
                    else copy.deepcopy(item)
                    for item in expected["child_plan"]
                ]
                self.assertEqual(executed, expected)
            self.assertEqual(record.get("status"), "rejected")
            self.assertEqual(record.get("batch_status"), "aborted")
            self.assertEqual(record.get("validated_event_seq"), 4)
            self.assertEqual(
                record.get("semantic_contract_passed"),
                record["semantic_contract"].get("passed"),
            )
            for field in (
                "mutation_seq",
                "live_tree_sha256_before",
                "live_tree_sha256_after",
            ):
                self.assertNotIn(field, record)

    def test_balancer_scope_preflights_the_whole_envelope_before_first_mutation(self):
        base = node("ROOT", "ROOT", "ROOT", [
            node("L1", "domain", "L1", [
                node("P", "parent", "L2", [
                    node("C1", "one", "L3", [node("GC", "grandchild", "L4")]),
                    node("C2", "two", "L3"),
                ]),
            ]),
        ])
        with tempfile.TemporaryDirectory() as temporary_dir:
            process = self.balance.ShapingProcess.__new__(self.balance.ShapingProcess)
            process.tm = TreeManager(copy.deepcopy(base))
            process.membership_counts = {}
            process.ops_log = Path(temporary_dir) / "ops.jsonl"
            process._current_lineage = lambda: {}
            process.trace_map = {}
            placeholder = self.balance.BRIDGE_TARGET_PLACEHOLDER
            valid_first = decision(
                "create_bridge",
                "broader_narrower",
                "P",
                placeholder,
                children=[child_plan("C1", placeholder)],
                new_label="group",
            )
            invalid_second = decision(
                "move",
                "misplaced",
                "HIDDEN",
                "L1",
            )
            payload = {"decisions": [valid_first, invalid_second]}
            call_audit = successful_call_audit(payload)
            changed = process._apply(
                "P",
                payload,
                "fanout",
                call_audit=call_audit,
            )
            self.assertFalse(changed)
            self.assertEqual(
                set(process.tm.get_all_node_ids()),
                {"ROOT", "L1", "P", "C1", "C2", "GC"},
            )
            self.assertEqual(process.tm.get_parent_id("C1"), "P")
            scope_records = [
                json.loads(line)
                for line in process.ops_log.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(scope_records), 2)
            self.assert_non_applied_audit_closes(scope_records, call_audit)

            wrong_keep = decision(
                "split_reparent", "related", "C1", "C1",
                children=[child_plan("GC", "P", disposition="keep")],
            )
            collapse = self.collapse.SkeletonRefiner.__new__(
                self.collapse.SkeletonRefiner
            )
            collapse.tm = TreeManager(copy.deepcopy(base))
            self.assertTrue(collapse._scope_issues(
                {"decisions": [wrong_keep]}, "P", "C1"
            ))

            wrong_balancer_keep = decision(
                "split_reparent", "related", "P", "P",
                children=[child_plan("C1", "L1", disposition="keep")],
            )
            self.assertTrue(process._scope_issues(
                "P", {"decisions": [wrong_balancer_keep]}
            ))

            polish = self.polish.PolishingProcess.__new__(
                self.polish.PolishingProcess
            )
            polish.tm = TreeManager(copy.deepcopy(base))
            self.assertTrue(polish._scope_issues(
                {"decisions": [wrong_keep]}, "C1", "C2"
            ))

    def test_balancer_semantic_failure_aborts_prior_staged_mutation(self):
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
            process.tm = TreeManager(copy.deepcopy(base))
            process.membership_counts = {}
            process.ops_log = Path(temporary_dir) / "ops.jsonl"
            process._current_lineage = lambda: {}
            process.trace_map = {}
            placeholder = self.balance.BRIDGE_TARGET_PLACEHOLDER
            valid_first = decision(
                "create_bridge",
                "broader_narrower",
                "P",
                placeholder,
                children=[child_plan("C1", placeholder)],
                new_label="group",
            )
            semantic_invalid_second = decision(
                "keep",
                "broader_narrower",
                "P",
                "P",
                children=[child_plan("C1", "P")],
            )
            semantic_payload = {
                "decisions": [valid_first, semantic_invalid_second]
            }
            semantic_call_audit = successful_call_audit(
                semantic_payload, fill="B"
            )
            changed = process._apply(
                "P",
                semantic_payload,
                "fanout",
                call_audit=semantic_call_audit,
            )
            self.assertFalse(changed)
            self.assertEqual(process.tm.root, base)
            self.assertEqual(process.trace_map, {})
            log_text = process.ops_log.read_text(encoding="utf-8")
            self.assertIn("BATCH_SEMANTIC_ABORTED", log_text)
            self.assertNotIn('"status": "applied"', log_text)
            semantic_records = [
                json.loads(line) for line in log_text.splitlines()
            ]
            self.assertEqual(len(semantic_records), 2)
            self.assert_non_applied_audit_closes(
                semantic_records, semantic_call_audit
            )

            transactional = self.balance.ShapingProcess.__new__(
                self.balance.ShapingProcess
            )
            transactional.tm = TreeManager(copy.deepcopy(base))
            transactional.membership_counts = {}
            transactional.ops_log = Path(temporary_dir) / "transactional.jsonl"
            transactional.ops_log.write_text("prior\n", encoding="utf-8")
            transactional._current_lineage = lambda: {}
            transactional.trace_map = {"OLD": "TARGET"}
            complete_bridge = decision(
                "create_bridge",
                "broader_narrower",
                "P",
                placeholder,
                children=[
                    child_plan("C1", placeholder),
                    child_plan("C2", placeholder),
                ],
                new_label="group",
            )

            def fail_after_partial_write(path, records):
                Path(path).write_text("partial\n", encoding="utf-8")
                raise OSError("injected log failure")

            with mock.patch.object(
                self.balance,
                "_atomic_append_jsonl_batch",
                side_effect=fail_after_partial_write,
            ), self.assertRaises(RuntimeError):
                transactional._apply(
                    "P", {"decisions": [complete_bridge]}, "fanout"
                )
            self.assertEqual(transactional.tm.root, base)
            self.assertEqual(transactional.trace_map, {"OLD": "TARGET"})
            self.assertEqual(
                transactional.ops_log.read_text(encoding="utf-8"), "prior\n"
            )

            def fail_single_after_partial_write(path, _record):
                Path(path).write_text("partial\n", encoding="utf-8")
                raise OSError("injected single-operation log failure")

            rename = decision(
                "rename", "synonym", "C1", "C1", new_label="renamed"
            )
            collapse = self.collapse.SkeletonRefiner.__new__(
                self.collapse.SkeletonRefiner
            )
            collapse.tm = TreeManager(copy.deepcopy(base))
            collapse.membership_counts = {}
            collapse.redirect_map = {"OLD": "TARGET"}
            collapse.ops_log = Path(temporary_dir) / "collapse.jsonl"
            collapse.ops_log.write_text("prior\n", encoding="utf-8")
            with mock.patch.object(
                self.collapse,
                "append_jsonl",
                side_effect=fail_single_after_partial_write,
            ), self.assertRaises(OSError):
                collapse._execute_decision(rename, "P", "C1")
            self.assertEqual(collapse.tm.root, base)
            self.assertEqual(collapse.redirect_map, {"OLD": "TARGET"})
            self.assertEqual(
                collapse.ops_log.read_text(encoding="utf-8"), "prior\n"
            )

            polish = self.polish.PolishingProcess.__new__(
                self.polish.PolishingProcess
            )
            polish.tm = TreeManager(copy.deepcopy(base))
            polish.membership_counts = {}
            polish.local_trace_map = {"OLD": "TARGET"}
            polish._current_lineage = lambda: {}
            polish.ops_log = Path(temporary_dir) / "polish.jsonl"
            polish.ops_log.write_text("prior\n", encoding="utf-8")
            with mock.patch.object(
                self.polish,
                "append_jsonl",
                side_effect=fail_single_after_partial_write,
            ), self.assertRaises(OSError):
                polish._execute_decision(rename, "C1", "C2", "audit")
            self.assertEqual(polish.tm.root, base)
            self.assertEqual(polish.local_trace_map, {"OLD": "TARGET"})
            self.assertEqual(
                polish.ops_log.read_text(encoding="utf-8"), "prior\n"
            )

            bound = decision(
                "rename", "synonym", "C1", "C1", new_label="bound rename"
            )
            altered = decision(
                "rename", "synonym", "C2", "C2", new_label="wrong rename"
            )
            call_audit = {
                "ok": True,
                "logical_call_id": "CALL_" + "A" * 64 + "_00000001",
                "json": {"decisions": [bound]},
                "final_bound": {"decisions": [bound]},
                "events": [{"seq": 4, "event": "scope_validation_pass"}],
            }
            collapse.tm = TreeManager(copy.deepcopy(base))
            with self.assertRaisesRegex(ValueError, "final_bound"):
                collapse._execute_decision(
                    altered, "P", "C1", call_audit=call_audit
                )
            self.assertEqual(collapse.tm.root, base)
            polish.tm = TreeManager(copy.deepcopy(base))
            with self.assertRaisesRegex(ValueError, "final_bound"):
                polish._execute_decision(
                    altered, "C1", "C2", "audit", call_audit=call_audit
                )
            self.assertEqual(polish.tm.root, base)

    def test_finalizer_scope_preflights_the_whole_batch_before_first_mutation(self):
        base = node("ROOT", "ROOT", "ROOT", [
            node("L1", "domain", "L1", [
                node("P", "parent", "L2", [node("C", "child", "L3")]),
            ]),
        ])
        process = self.finalize.OverallStructureAudit.__new__(
            self.finalize.OverallStructureAudit
        )
        process.tm = TreeManager(copy.deepcopy(base))
        process.stats = {"ops_applied": 0, "ops_skipped": 0}
        process.redirect_map = {}
        process._load_membership_source = lambda: (["final_node_id"], [])
        valid_first = decision(
            "rename",
            "synonym",
            "C",
            "C",
            new_label="renamed child",
        )
        invalid_second = decision(
            "rename",
            "broader_narrower",
            "P",
            "P",
            new_label="invalid parent",
        )
        payload = {"decisions": [valid_first, invalid_second]}
        call_audit = successful_call_audit(payload, fill="C")
        records = process._apply_operations(
            [valid_first, invalid_second], "L1", call_audit=call_audit
        )
        self.assertEqual(process.tm.get_node("C")["label"], "child")
        self.assertEqual(process.stats["ops_applied"], 0)
        self.assertEqual(process.stats["ops_skipped"], 2)
        self.assertEqual(len(records), 2)
        self.assertTrue(all(record.get("status") == "rejected" for record in records))
        self.assert_non_applied_audit_closes(records, call_audit)

        single = self.finalize.OverallStructureAudit.__new__(
            self.finalize.OverallStructureAudit
        )
        single.tm = TreeManager(copy.deepcopy(base))
        single.stats = {"ops_applied": 0, "ops_skipped": 0}
        single.redirect_map = {"OLD": "TARGET"}
        single.operation_records = []
        single.audit_entries = []
        single_temporary_dir = tempfile.TemporaryDirectory()
        self.addCleanup(single_temporary_dir.cleanup)
        single.ops_log = Path(single_temporary_dir.name) / "operations.jsonl"
        single._load_membership_source = lambda: (["final_node_id"], [])
        invalid_single = decision(
            "rename",
            "broader_narrower",
            "C",
            "C",
            new_label="invalid child",
        )
        single_payload = {"decisions": [invalid_single]}
        single_call_audit = successful_call_audit(single_payload, fill="D")
        single_records = single._apply_operations(
            [invalid_single], "L1", call_audit=single_call_audit
        )
        self.assertEqual(single.tm.root, base)
        self.assertEqual(single.redirect_map, {"OLD": "TARGET"})
        self.assertEqual(single.stats, {"ops_applied": 0, "ops_skipped": 1})
        self.assertEqual(single.operation_records, single_records)
        self.assertEqual(len(single.audit_entries), 1)
        exported_single_records = [
            json.loads(line)
            for line in single.ops_log.read_text(encoding="utf-8").splitlines()
        ]
        self.assertEqual(exported_single_records, single_records)
        self.assertNotIn(
            '"status": "applied"',
            single.ops_log.read_text(encoding="utf-8"),
        )
        self.assert_non_applied_audit_closes(
            exported_single_records, single_call_audit
        )

        with tempfile.TemporaryDirectory() as temporary_dir:
            transactional = self.finalize.OverallStructureAudit.__new__(
                self.finalize.OverallStructureAudit
            )
            transactional.tm = TreeManager(copy.deepcopy(base))
            transactional.stats = {
                "llm_calls": 3,
                "llm_failures": 1,
                "ops_applied": 0,
                "ops_skipped": 0,
            }
            transactional.redirect_map = {"OLD": "TARGET"}
            transactional.operation_records = [{"prior": True}]
            transactional.audit_entries = [{"prior": True}]
            transactional.ops_log = Path(temporary_dir) / "final.jsonl"
            transactional.ops_log.write_text("prior\n", encoding="utf-8")
            transactional._load_membership_source = lambda: (["final_node_id"], [])

            def fail_after_partial_write(path, batch):
                Path(path).write_text("partial\n", encoding="utf-8")
                raise OSError("injected log failure")

            with mock.patch.object(
                self.finalize,
                "_atomic_append_jsonl_batch",
                side_effect=fail_after_partial_write,
            ), self.assertRaises(RuntimeError):
                transactional._apply_operations([valid_first], "L1")
            self.assertEqual(transactional.tm.root, base)
            self.assertEqual(transactional.redirect_map, {"OLD": "TARGET"})
            self.assertEqual(
                transactional.stats,
                {
                    "llm_calls": 3,
                    "llm_failures": 1,
                    "ops_applied": 0,
                    "ops_skipped": 0,
                },
            )
            self.assertEqual(transactional.operation_records, [{"prior": True}])
            self.assertEqual(transactional.audit_entries, [{"prior": True}])
            self.assertEqual(
                transactional.ops_log.read_text(encoding="utf-8"), "prior\n"
            )


if __name__ == "__main__":
    unittest.main()
