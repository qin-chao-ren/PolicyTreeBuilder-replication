from __future__ import annotations

import copy
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
from utils.local_reference_binding import (
    build_local_reference_context,
    call_local_reference_json,
)
from utils.tree_manager import TreeManager


class FakeTransport:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(copy.deepcopy(kwargs))
        raw = json.dumps(self.payload, ensure_ascii=False)
        return {
            "ok": True,
            "json": copy.deepcopy(self.payload),
            "raw": raw,
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
                "raw": raw,
                "parse_ok": True,
                "error": None,
            }],
        }


class C13RF10BalanceDeferTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.balance = load_stage_module("balance_tree_structure")

    @staticmethod
    def deep_tree():
        children = [
            node(f"L4_NCHILD{i:02d}", f"child {i}", "L4")
            for i in range(8)
        ]
        return node("ROOT", "ROOT", "ROOT", [
            node("L1_NDOMAIN1", "domain", "L1", [
                node("L2_NPARENT1", "parent", "L2", [
                    node("L3_NCAND001", "candidate", "L3", children),
                ]),
            ]),
        ])

    def process(self, temporary_dir):
        process = self.balance.ShapingProcess.__new__(
            self.balance.ShapingProcess
        )
        process.tm = TreeManager(copy.deepcopy(self.deep_tree()))
        process.membership_counts = {}
        process.ops_log = Path(temporary_dir) / "ops.jsonl"
        process.llm_log = Path(temporary_dir) / "llm.jsonl"
        process.trace_map = {}
        process.prior_lineage = {}
        process.deferred_candidates = set()
        return process

    def bound_decisions(self):
        placeholder = self.balance.BRIDGE_TARGET_PLACEHOLDER
        child_ids = [f"L4_NCHILD{i:02d}" for i in range(8)]
        groups = (child_ids[0:2], child_ids[2:5], child_ids[5:8])
        return [
            decision(
                "create_bridge",
                "broader_narrower",
                "L3_NCAND001",
                placeholder,
                children=[child_plan(child_id, placeholder) for child_id in group],
                new_label=f"group {index}",
            )
            for index, group in enumerate(groups, 1)
        ]

    def local_context_and_payload(self, process):
        placeholder = self.balance.BRIDGE_TARGET_PLACEHOLDER
        refs = [
            ("CANDIDATE", "L3_NCAND001"),
            ("PARENT", "L2_NPARENT1"),
            *[
                (f"N{i}", f"L4_NCHILD{i:02d}")
                for i in range(8)
            ],
        ]
        user_text = "\n".join(
            [*(f"{ref}={node_id}" for ref, node_id in refs), placeholder]
        )
        system = "c13rf10 structure-balancing fixture contract"
        context = build_local_reference_context(
            task="balance_tree_fanout",
            user_text=user_text,
            candidate_node_ids=process.tm.get_all_node_ids(),
            preferred_refs=refs,
            virtual_refs=(("NEW_BRIDGE", placeholder),),
            contract_text=system,
            expected_count=None,
        )
        local_decisions = []
        for bound in self.bound_decisions():
            local = copy.deepcopy(bound)
            local["source_ref"] = "CANDIDATE"
            local["target_ref"] = "NEW_BRIDGE"
            local.pop("source_id")
            local.pop("target_id")
            for item in local["child_plan"]:
                index = int(item["child_id"].rsplit("D", 1)[1])
                item["child_ref"] = f"N{index}"
                item["target_parent_ref"] = "NEW_BRIDGE"
                item.pop("child_id")
                item.pop("target_parent_id")
            local_decisions.append(local)
        payload = {
            "context_token": context.context_token,
            "decisions": local_decisions,
        }
        return system, context, payload

    def replayed_response(self, process):
        system, context, payload = self.local_context_and_payload(process)
        transport = FakeTransport(payload)
        response = call_local_reference_json(
            transport=transport,
            profile="fake",
            system=system,
            context=context,
            task="balance_tree_fanout",
            expected_count=None,
            protected_manager=process.tm,
            bound_validator=lambda value: process._scope_issues(
                "L3_NCAND001", value
            ),
        )
        return transport, response

    def test_depth_ceiling_batch_is_classified_as_inexpressible(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            process = self.process(temporary_dir)
            issues = process._scope_issues(
                "L3_NCAND001", {"decisions": self.bound_decisions()}
            )
        self.assertEqual(len(issues), 3, issues)
        self.assertEqual(
            {item["code"] for item in issues},
            {"DECISION_SCOPE_INEXPRESSIBLE"},
        )
        for index, issue in enumerate(issues):
            self.assertEqual(issue["context"]["decision_index"], index)
            self.assertEqual(issue["context"]["mutable_ref_paths"], [])
            self.assertIs(issue["context"]["repairable"], False)
            # C13RF29: the issue reports the scope predicate's own message
            # (BRIDGE_TOO_DEEP_MESSAGE) rather than the deferral sentence that
            # used to be substituted for it.  The deferral sentence was written for
            # a channel that only ever carried depth-ceiling bridges; now that any
            # off-stage decision can defer, substituting one stage-specific
            # sentence for every reason would misreport most of them.
            self.assertEqual(
                issue["message"], self.balance.BRIDGE_TOO_DEEP_MESSAGE
            )

    def test_mixed_scope_failures_are_refused_and_recorded_not_repaired(self):
        """C13RF29 renamed what these two shapes get, not whether they are refused.

        Both were DECISION_SCOPE_VIOLATION (repairable -> a repair round -> the run
        stops if the repair fails the same predicate).  Both are now
        DECISION_SCOPE_INEXPRESSIBLE with no repair round.  What the test defends is
        unchanged and asserted below: exactly one issue each, non-repairable, no
        mutable ref path, and neither shape becomes executable.
        """
        with tempfile.TemporaryDirectory() as temporary_dir:
            process = self.process(temporary_dir)
            wrong_target = copy.deepcopy(self.bound_decisions()[0])
            wrong_target["target_id"] = "L2_NPARENT1"
            issues = process._scope_issues(
                "L3_NCAND001", {"decisions": [wrong_target]}
            )
            self.assertEqual(len(issues), 1, issues)
            self.assertEqual(issues[0]["code"], "DECISION_SCOPE_INEXPRESSIBLE")
            self.assertIs(issues[0]["context"]["repairable"], False)
            self.assertEqual(issues[0]["context"]["mutable_ref_paths"], [])

            non_direct_child = copy.deepcopy(self.bound_decisions()[0])
            non_direct_child["child_plan"][0]["child_id"] = "L2_NPARENT1"
            issues = process._scope_issues(
                "L3_NCAND001", {"decisions": [non_direct_child]}
            )
            self.assertEqual(len(issues), 1, issues)
            self.assertEqual(issues[0]["code"], "DECISION_SCOPE_INEXPRESSIBLE")
            self.assertIs(issues[0]["context"]["repairable"], False)
            self.assertEqual(issues[0]["context"]["mutable_ref_paths"], [])

    def test_archived_shape_uses_one_attempt_and_no_repair(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            process = self.process(temporary_dir)
            transport, response = self.replayed_response(process)
        self.assertEqual(len(transport.calls), 1)
        self.assertIs(response["ok"], False)
        self.assertEqual(response["final_disposition"], "scope_rejected")
        self.assertEqual(response["error"], "BOUND_SCOPE_VALIDATION_FAILED")
        self.assertEqual(response["repair_count"], 0)
        self.assertIsNone(response["final_bound"])
        self.assertEqual(len(response["initial_scope_errors"]), 3)
        self.assertTrue(process._is_deferrable_response(response))

    def test_deferred_batch_is_logged_once_without_mutation(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            process = self.process(temporary_dir)
            before = copy.deepcopy(process.tm.root)
            _transport, response = self.replayed_response(process)
            changed = process._apply(
                "L3_NCAND001", response["json"], "fanout", call_audit=response
            )
            records = [
                json.loads(line)
                for line in process.ops_log.read_text(encoding="utf-8").splitlines()
            ]
        self.assertFalse(changed)
        self.assertEqual(process.tm.root, before)
        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(record["status"], "deferred")
        self.assertEqual(record["type"], "semantic_decision")
        self.assertEqual(record["logical_call_id"], response["logical_call_id"])
        self.assertEqual(len(record["local_proposal"]["decisions"]), 3)
        self.assertEqual(len(record["resolved_proposal"]["decisions"]), 3)
        self.assertNotIn("decision_index", record)
        self.assertNotIn("mutation_seq", record)
        for resolved in record["resolved_proposal"]["decisions"]:
            self.assertEqual(resolved["source_id"], "L3_NCAND001")
            self.assertEqual(
                resolved["target_id"],
                self.balance.BRIDGE_TARGET_PLACEHOLDER,
            )
            self.assertNotIn("source_ref", resolved)
            self.assertNotIn("target_ref", resolved)
        # C13RF29: the shared record takes its messages as a parameter now, so the
        # violation states the reason the binder actually recorded instead of the
        # sentence 14b used to overwrite it with.
        violation = record["semantic_contract"]["violations"][0]
        self.assertEqual(
            violation["message"], self.balance.BRIDGE_TOO_DEEP_MESSAGE
        )
        self.assertEqual(violation["code"], "DECISION_SCOPE_INEXPRESSIBLE")

    def test_deferred_candidate_is_not_queried_again_in_the_same_stage(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            process = self.process(temporary_dir)
            _transport, response = self.replayed_response(process)
            with mock.patch.object(
                process, "_call_local", return_value=response
            ) as call_mock:
                self.assertFalse(process._fix_fanout("L3_NCAND001"))
                self.assertFalse(process._fix_fanout("L3_NCAND001"))
            self.assertEqual(call_mock.call_count, 1)
            records = process.ops_log.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(records), 1)

    def test_deferrable_whitelist_rejects_partial_or_mixed_results(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            process = self.process(temporary_dir)
            _transport, response = self.replayed_response(process)
        self.assertTrue(process._is_deferrable_response(response))
        for field, value in (
            ("error", "OTHER_ERROR"),
            ("mutation_before_validation", True),
            ("repair_count", 1),
            ("final_bound", {}),
        ):
            with self.subTest(field=field):
                candidate = copy.deepcopy(response)
                candidate[field] = value
                self.assertFalse(process._is_deferrable_response(candidate))
        mixed = copy.deepcopy(response)
        mixed["initial_scope_errors"].append({
            "code": "DECISION_SCOPE_VIOLATION",
            "context": {"repairable": False, "mutable_ref_paths": []},
        })
        self.assertFalse(process._is_deferrable_response(mixed))

        # C13RF29: the message text no longer decides deferrability -- the code,
        # the repairable flag and the empty ref-path list do.  A different message
        # carrying the same inexpressible verdict now defers, which is the point of
        # the card (the open-ended list of message texts was what made every new
        # off-stage shape cost another full-tree rerun).  What still must NOT defer
        # is a channel entry that is not inexpressible, which `mixed` above pins.
        other_message = copy.deepcopy(response)
        other_message["initial_scope_errors"][0]["message"] = "other scope issue"
        self.assertTrue(process._is_deferrable_response(other_message))

        # Still refused: a repairable entry anywhere in the channel.  Deferring one
        # would park a decision the binder is simultaneously asking the model to
        # fix.
        repairable_entry = copy.deepcopy(response)
        repairable_entry["initial_scope_errors"][0]["context"]["repairable"] = True
        repairable_entry["initial_scope_errors"][0]["context"][
            "mutable_ref_paths"] = ["decisions[0].target_ref"]
        self.assertFalse(process._is_deferrable_response(repairable_entry))

        # Still refused: an empty channel.  "No recorded error" must never be read
        # as "every recorded error was benign" (C13RF18's floor, kept by RF29).
        empty_channel = copy.deepcopy(response)
        empty_channel["initial_scope_errors"] = []
        self.assertFalse(process._is_deferrable_response(empty_channel))

    def test_deferrable_whitelist_rejects_legal_decision_mixed_with_depth_error(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            process = self.process(temporary_dir)
            _transport, response = self.replayed_response(process)
        mixed = copy.deepcopy(response)
        # The first proposal is legal in this hypothetical batch; only the
        # remaining two proposals are represented by depth-ceiling errors.
        mixed["final_local"]["decisions"][0]["action"] = "keep"
        mixed["initial_scope_errors"] = [
            {
                "code": "DECISION_SCOPE_INEXPRESSIBLE",
                "message": self.balance.BRIDGE_DEPTH_DEFERRED_MESSAGE,
                "context": {
                    "decision_index": index,
                    "mutable_ref_paths": [],
                    "repairable": False,
                },
            }
            for index in (1, 2)
        ]
        self.assertFalse(process._is_deferrable_response(mixed))


if __name__ == "__main__":
    unittest.main()
