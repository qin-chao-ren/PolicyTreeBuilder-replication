from __future__ import annotations

import copy
import inspect
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tests.test_semantic_stage_dispatch import (
    append_jsonl,
    decision,
    load_stage_module,
    node,
)

from utils.local_reference_binding import (
    build_local_reference_context,
    call_local_reference_json,
)
from utils.semantic_contract import (
    execute_semantic_decision,
    validate_semantic_history,
)
from utils.tree_integrity import validate_tree_e0
from utils.tree_manager import TreeManager


REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_PATH = REPO_ROOT / "tests/fixtures/c13rf6_inexpressible_cases.json"


class FakeTransport:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(copy.deepcopy(kwargs))
        return {
            "ok": True,
            "json": copy.deepcopy(self.payload),
            "raw": json.dumps(self.payload),
            "error": None,
            "status": 200,
            "latency_ms": 1,
            "profile": "fake",
            "provider": "fake",
            "model": "fake",
            "attempts": 1,
            "attempt_history": [{
                "attempt": 1,
                "status": 200,
                "latency_ms": 1,
                "raw": json.dumps(self.payload),
                "parse_ok": True,
                "error": None,
            }],
        }


class C13RF6DeferredRestructureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
        cls.collapse = load_stage_module("collapse_redundant_hierarchy")
        cls.polish = load_stage_module("polish_tree_labels")
        cls.balance = load_stage_module("balance_tree_structure")
        cls.finalize = load_stage_module("finalize_policy_tree")

    def process(self, module, class_name):
        value = getattr(module, class_name).__new__(getattr(module, class_name))
        value.tm = TreeManager(copy.deepcopy(self.fixture["tree"]))
        return value

    def assert_inexpressible(self, issues):
        self.assertEqual(len(issues), 1, issues)
        issue = issues[0]
        self.assertEqual(issue["code"], "DECISION_SCOPE_INEXPRESSIBLE")
        self.assertEqual(issue["context"]["child_indexes"], [1])
        self.assertIs(issue["context"]["repairable"], False)
        self.assertEqual(issue["context"]["mutable_ref_paths"], [])
        encoded = json.dumps(issue["context"], ensure_ascii=False)
        for real_id in self.process(
            self.collapse, "SkeletonRefiner"
        ).tm.get_all_node_ids():
            self.assertNotIn(real_id, encoded)

    def test_call_85_classifies_identically_in_14a_and_14c(self):
        pair = self.fixture["pair"]
        payload = {"decisions": [copy.deepcopy(self.fixture["bound_decision"])]}
        collapse = self.process(self.collapse, "SkeletonRefiner")
        self.assert_inexpressible(collapse._scope_issues(
            payload, pair["parent_id"], pair["child_id"]
        ))

        polish = self.process(self.polish, "PolishingProcess")
        self.assert_inexpressible(polish._scope_issues(
            payload, pair["parent_id"], pair["child_id"]
        ))

    def local_context_and_payload(self):
        pair = self.fixture["pair"]
        bound = self.fixture["bound_decision"]
        preferred = [
            ("PARENT", pair["parent_id"]),
            ("CHILD", pair["child_id"]),
            ("GRANDPARENT", "ROOT"),
            ("N3", pair["external_target_id"]),
            ("N26", bound["child_plan"][0]["child_id"]),
            ("N27", bound["child_plan"][1]["child_id"]),
            ("N28", bound["child_plan"][2]["child_id"]),
        ]
        user_text = " ".join(
            f"{ref}={node_id}" for ref, node_id in preferred
        )
        context = build_local_reference_context(
            task="collapse_redundant_hierarchy",
            user_text=user_text,
            candidate_node_ids=TreeManager(
                copy.deepcopy(self.fixture["tree"])
            ).get_all_node_ids(),
            preferred_refs=preferred,
            contract_text="c13rf6 fixture contract",
            expected_count=1,
        )
        local = {
            "context_token": context.context_token,
            "decisions": [{
                "relation": bound["relation"],
                "action": bound["action"],
                "source_ref": "CHILD",
                "target_ref": "PARENT",
                "new_label": None,
                "confidence": bound["confidence"],
                "evidence": copy.deepcopy(bound["evidence"]),
                "child_plan": [
                    {
                        "child_ref": ref,
                        "disposition": item["disposition"],
                        "target_parent_ref": target_ref,
                        "relation": item["relation"],
                        "same_domain": item["same_domain"],
                        "evidence": item["evidence"],
                    }
                    for item, ref, target_ref in zip(
                        bound["child_plan"],
                        ("N26", "N27", "N28"),
                        ("PARENT", "N3", "PARENT"),
                    )
                ],
            }],
        }
        return context, local

    def test_inexpressible_call_is_terminal_without_repair(self):
        pair = self.fixture["pair"]
        process = self.process(self.collapse, "SkeletonRefiner")
        context, local = self.local_context_and_payload()
        transport = FakeTransport(local)
        result = call_local_reference_json(
            transport=transport,
            profile="fake",
            system="c13rf6 fixture contract",
            context=context,
            task="collapse_redundant_hierarchy",
            expected_count=1,
            protected_manager=process.tm,
            bound_validator=lambda payload: process._scope_issues(
                payload, pair["parent_id"], pair["child_id"]
            ),
        )
        self.assertEqual(len(transport.calls), 1)
        self.assertIs(result["ok"], False)
        self.assertEqual(result["repair_count"], 0)
        self.assertIsNone(result["final_bound"])
        self.assertEqual(result["final_disposition"], "scope_rejected")
        self.assertEqual(result["error"], "BOUND_SCOPE_VALIDATION_FAILED")
        self.assertEqual(result["mutable_ref_paths"], [])
        self.assertTrue(result["stable_valid_refs"])

    def deferred_response(self, module, class_name):
        pair = self.fixture["pair"]
        process = self.process(module, class_name)
        context, local = self.local_context_and_payload()
        transport = FakeTransport(local)
        response = call_local_reference_json(
            transport=transport,
            profile="fake",
            system="c13rf6 fixture contract",
            context=context,
            task="collapse_redundant_hierarchy",
            expected_count=1,
            protected_manager=process.tm,
            bound_validator=lambda payload: process._scope_issues(
                payload, pair["parent_id"], pair["child_id"]
            ),
        )
        return process, response

    def test_deferred_record_resolves_ids_without_mutation(self):
        for module, class_name in (
            (self.collapse, "SkeletonRefiner"),
            (self.polish, "PolishingProcess"),
        ):
            with self.subTest(stage=module.__name__):
                process, response = self.deferred_response(module, class_name)
                before = copy.deepcopy(process.tm.root)
                record = process._deferred_record(response, module.__name__)
                self.assertEqual(record["status"], "deferred")
                self.assertEqual(record["type"], "semantic_decision")
                self.assertEqual(
                    record["logical_call_id"], response["logical_call_id"]
                )
                self.assertNotIn("decision_index", record)
                self.assertNotIn("mutation_seq", record)
                self.assertNotIn("live_tree_sha256_before", record)
                self.assertNotIn("live_tree_sha256_after", record)
                resolved = record["resolved_proposal"]["decisions"][0]
                self.assertEqual(
                    resolved["child_plan"][1]["target_parent_id"],
                    self.fixture["pair"]["external_target_id"],
                )
                self.assertEqual(process.tm.root, before)

    def test_14a_stage_continues_after_deferred_candidate(self):
        base = node("ROOT", "ROOT", "ROOT", [
            node("L1_NEXT0001", "external", "L1"),
            node("L1_NPARENT1", "parent", "L1", [
                node("L2_NCHILD01", "child one", "L2", [
                    node("L3_NGRAND01", "grandchild", "L3"),
                ]),
                node("L2_NCHILD02", "child two", "L2"),
            ]),
        ])
        with tempfile.TemporaryDirectory() as temporary_dir:
            process = self.collapse.SkeletonRefiner.__new__(
                self.collapse.SkeletonRefiner
            )
            process.tm = TreeManager(copy.deepcopy(base))
            process.membership = {}
            process.title_map = {}
            process.llm_profile = "fake"
            process.emb_helper = type("Embedding", (), {
                "get_centroid": staticmethod(lambda members: []),
                "cosine_sim": staticmethod(lambda left, right: 1.0),
            })()
            process.redirect_map = {}
            process.ops_log = Path(temporary_dir) / "ops.jsonl"
            process.llm_log = Path(temporary_dir) / "llm.jsonl"
            local = {
                "context_token": "CTX_STAGE",
                "decisions": [{
                    "relation": "exact_duplicate",
                    "action": "merge",
                    "source_ref": "CHILD",
                    "target_ref": "PARENT",
                    "new_label": None,
                    "confidence": 0.95,
                    "evidence": copy.deepcopy(
                        self.fixture["bound_decision"]["evidence"]
                    ),
                    "child_plan": [{
                        "child_ref": "GRANDCHILD",
                        "disposition": "move",
                        "target_parent_ref": "EXTERNAL",
                        "relation": "broader_narrower",
                        "same_domain": True,
                        "evidence": "Move to the external displayed target.",
                    }],
                }],
            }
            deferred = {
                "ok": False,
                "json": None,
                "error": "BOUND_SCOPE_VALIDATION_FAILED",
                "final_disposition": "scope_rejected",
                "mutation_before_validation": False,
                "final_bound": None,
                "repair_count": 0,
                "logical_call_id": "CALL_STAGE_00000001",
                "initial_scope_errors": [{
                    "code": "DECISION_SCOPE_INEXPRESSIBLE",
                    "context": {
                        "decision_index": 0,
                        "child_indexes": [0],
                        "repairable": False,
                        "mutable_ref_paths": [],
                    },
                }],
                "final_local": local,
                "context": {"ref_mapping": [
                    {"ref": "PARENT", "node_id": "L1_NPARENT1"},
                    {"ref": "CHILD", "node_id": "L2_NCHILD01"},
                    {"ref": "GRANDCHILD", "node_id": "L3_NGRAND01"},
                    {"ref": "EXTERNAL", "node_id": "L1_NEXT0001"},
                ]},
            }
            accepted = {
                "ok": True,
                "json": {"decisions": [decision(
                    "keep", "related", "L1_NPARENT1", "L2_NCHILD02"
                )]},
            }
            with mock.patch.object(
                self.collapse,
                "call_local_reference_json",
                side_effect=[deferred, accepted],
            ) as call_mock, mock.patch.object(
                process, "_execute_decision"
            ) as execute_mock:
                process._process_parent("L1_NPARENT1")
            self.assertEqual(call_mock.call_count, 2)
            execute_mock.assert_called_once()
            record = json.loads(process.ops_log.read_text(encoding="utf-8"))
            self.assertEqual(record["status"], "deferred")
            self.assertEqual(process.tm.root, base)

    def test_14c_stage_records_deferred_without_execution(self):
        pair = self.fixture["pair"]
        process, response = self.deferred_response(
            self.polish, "PolishingProcess"
        )
        with tempfile.TemporaryDirectory() as temporary_dir:
            process.membership = {}
            process.llm_profile = "fake"
            process.emb_helper = type("Embedding", (), {
                "get_centroid": staticmethod(lambda members: []),
                "cosine_sim": staticmethod(lambda left, right: 1.0),
            })()
            process.ops_log = Path(temporary_dir) / "ops.jsonl"
            process.llm_log = Path(temporary_dir) / "llm.jsonl"
            process._describe_node = lambda node_id: (
                f"node={node_id} parent={process.tm.get_parent_id(node_id)}"
            )
            left = process.tm.get_node(pair["parent_id"])
            right = process.tm.get_node(pair["child_id"])
            before = copy.deepcopy(process.tm.root)
            with mock.patch.object(
                self.polish,
                "call_local_reference_json",
                return_value=response,
            ), mock.patch.object(
                self.polish,
                "build_local_reference_context",
                return_value=object(),
            ), mock.patch.object(process, "_execute_decision") as execute_mock:
                process._process_pair(
                    left, right, "cross_parent_unify", parent_id=None
                )
            execute_mock.assert_not_called()
            record = json.loads(process.ops_log.read_text(encoding="utf-8"))
            self.assertEqual(record["status"], "deferred")
            self.assertEqual(process.tm.root, before)

    def test_deferrable_whitelist_rejects_partial_and_mixed_failures(self):
        process = self.process(self.collapse, "SkeletonRefiner")
        _, response = self.deferred_response(
            self.collapse, "SkeletonRefiner"
        )
        self.assertTrue(process._is_deferrable_response(response))
        changes = {
            "transport": {"final_disposition": "transport_rejected"},
            "binding": {"error": "BINDING_FAILED"},
            "semantic_drift": {"repair_count": 1},
            "real_id": {"final_disposition": "raw_response_rejected"},
            "mutation": {"mutation_before_validation": True},
            "mixed": {"initial_scope_errors": [
                response["initial_scope_errors"][0],
                {"code": "DECISION_SCOPE_VIOLATION", "context": {}},
            ]},
        }
        for name, values in changes.items():
            with self.subTest(name=name):
                candidate = copy.deepcopy(response)
                candidate.update(values)
                self.assertFalse(process._is_deferrable_response(candidate))

    def test_scope_misclassification_defenses_hold_for_14a_and_14c(self):
        pair = self.fixture["pair"]
        base = self.fixture["bound_decision"]
        cases = []
        invalid_action = copy.deepcopy(base)
        invalid_action["action"] = "invented"
        cases.append(invalid_action)
        wrong_pair = copy.deepcopy(base)
        wrong_pair["target_id"] = pair["external_target_id"]
        cases.append(wrong_pair)
        non_source = copy.deepcopy(base)
        non_source["child_plan"][1]["child_id"] = pair["external_target_id"]
        cases.append(non_source)
        for module, class_name in (
            (self.collapse, "SkeletonRefiner"),
            (self.polish, "PolishingProcess"),
        ):
            process = self.process(module, class_name)
            for candidate in cases:
                with self.subTest(stage=module.__name__, action=candidate["action"]):
                    issues = process._scope_issues(
                        {"decisions": [candidate]},
                        pair["parent_id"],
                        pair["child_id"],
                    )
                    self.assertTrue(issues)
                    self.assertTrue(any(
                        item["code"] == "DECISION_SCOPE_VIOLATION"
                        for item in issues
                    ))

    def test_binding_failures_never_reach_deferred_classification(self):
        context, valid = self.local_context_and_payload()
        process = self.process(self.collapse, "SkeletonRefiner")
        pair = self.fixture["pair"]
        cases = {}
        invalid_ref = copy.deepcopy(valid)
        invalid_ref["decisions"][0]["target_ref"] = "UNKNOWN"
        cases["invalid_ref"] = invalid_ref
        multiple = copy.deepcopy(valid)
        multiple["decisions"].append(copy.deepcopy(multiple["decisions"][0]))
        cases["multiple_decisions"] = multiple
        invalid_relation = copy.deepcopy(valid)
        invalid_relation["decisions"][0]["relation"] = "invented"
        cases["invalid_relation"] = invalid_relation
        invalid_action = copy.deepcopy(valid)
        invalid_action["decisions"][0]["action"] = "invented"
        cases["invalid_action"] = invalid_action
        for name, payload in cases.items():
            with self.subTest(name=name):
                transport = FakeTransport(payload)
                result = call_local_reference_json(
                    transport=transport,
                    profile="fake",
                    system="c13rf6 fixture contract",
                    context=context,
                    task="collapse_redundant_hierarchy",
                    expected_count=1,
                    protected_manager=process.tm,
                    bound_validator=lambda value: process._scope_issues(
                        value, pair["parent_id"], pair["child_id"]
                    ),
                )
                self.assertFalse(result["ok"])
                self.assertFalse(process._is_deferrable_response(result))

    def test_execute_never_returns_deferred_and_batch_gates_reject_it(self):
        source = inspect.getsource(execute_semantic_decision)
        self.assertNotIn('"deferred"', source)
        deferred = {
            "action": "merge",
            "status": "deferred",
            "semantic_contract": {"passed": True, "violation_counts": {}},
        }
        base = node("ROOT", "ROOT", "ROOT", [
            node("L1", "domain", "L1", [
                node("P", "parent", "L2", [
                    node("C1", "one", "L3"),
                    node("C2", "two", "L3"),
                ]),
            ]),
        ])
        with tempfile.TemporaryDirectory() as temporary_dir:
            process = self.balance.ShapingProcess.__new__(
                self.balance.ShapingProcess
            )
            process.tm = TreeManager(copy.deepcopy(base))
            process.membership_counts = {}
            process.ops_log = Path(temporary_dir) / "ops.jsonl"
            process._current_lineage = lambda: {}
            process.trace_map = {}
            payload = {"decisions": [decision(
                "keep", "related", "P", "P"
            )]}
            with mock.patch.object(
                self.balance, "execute_semantic_decision", return_value=deferred
            ):
                changed = process._apply("P", payload, "fanout")
            self.assertFalse(changed)
            self.assertIn(
                "BATCH_SEMANTIC_ABORTED",
                process.ops_log.read_text(encoding="utf-8"),
            )

        finalizer = self.finalize.OverallStructureAudit.__new__(
            self.finalize.OverallStructureAudit
        )
        finalizer.tm = TreeManager(copy.deepcopy(base))
        finalizer.redirect_map = {}
        finalizer._load_membership_source = lambda: ([], [])
        final_op = decision("keep", "related", "C1", "P")
        with mock.patch.object(
            self.finalize, "execute_semantic_decision", return_value=deferred
        ), mock.patch.object(
            finalizer,
            "_commit_operation_batch",
            side_effect=lambda **kwargs: kwargs["records"],
        ):
            records = finalizer._apply_operations([final_op], "L1")
        self.assertEqual(records[0]["status"], "rejected")
        self.assertIn(
            "BATCH_SEMANTIC_ABORTED",
            records[0]["semantic_contract"]["violation_counts"],
        )

    def test_deferred_is_inert_to_public_semantic_and_e0_gates(self):
        process, response = self.deferred_response(
            self.collapse, "SkeletonRefiner"
        )
        record = process._deferred_record(response, "vertical_collapse")
        semantic = validate_semantic_history([record])
        baseline_e0 = validate_tree_e0(process.tm.root)
        e0 = validate_tree_e0(process.tm.root, operations=[record])
        self.assertTrue(semantic["passed"], semantic)
        self.assertEqual(
            semantic["stats"]["applied_semantic_operations_checked"], 0
        )
        self.assertEqual(
            semantic["stats"]["non_applied_semantic_operations_seen"], 0
        )
        self.assertEqual(e0["critical_count"], baseline_e0["critical_count"])
        self.assertEqual(
            e0["violation_counts"], baseline_e0["violation_counts"]
        )
        self.assertEqual(e0["stats"]["applied_operations_checked"], 0)


if __name__ == "__main__":
    unittest.main()
