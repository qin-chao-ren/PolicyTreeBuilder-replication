"""C13RF18: defer the inexpressible remainder that survives an honest repair.

A single model answer can carry two different kinds of error at once: a child
the stage genuinely cannot express (C13RF6 built ``defer`` for exactly this) and
a factual misread the model can and should repair.  C13RF16 made the classifier
report both halves per message instead of collapsing them, and withheld the
inexpressible half's ref paths so the repair round cannot coerce it.  But the
defer predicate still demanded ``repair_count == 0``, so the moment a repair
happened the deferrable half lost its defer eligibility for good and the whole
call stopped the run with ``REPAIR_SCOPE_VALIDATION_FAILED``.

The card assumed the repair round produced no classification to read.  The
mandatory channel scan found the opposite, and this file pins it: the binder has
always re-run each stage's ``_scope_issues`` on the repaired payload and stored
the result in ``repair_scope_errors`` -- a field parallel to
``initial_scope_errors`` that no defer predicate read.  This is the third
appearance of the "two parallel ledgers" defect (C13RF12's critical vs
``bad_responses``, C13RF16's four independent copies).  The fix is therefore not
to produce a classification but to *read the ledger that was already there*.

Strictness floor, pinned by the user on 2026-08-18: the remainder left after a
repair must be **entirely** inexpressible.  One non-inexpressible residue and
the run still stops.  A pure subject misread still never defers, and multi-round
repair -- which the binder does not perform today -- fails closed rather than
inheriting the allowance.
"""
from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from tests.test_semantic_stage_dispatch import load_stage_module
from utils.local_reference_binding import (
    build_local_reference_context,
    call_local_reference_json,
)
from utils.tree_manager import TreeManager

INEXPRESSIBLE = "DECISION_SCOPE_INEXPRESSIBLE"
VIOLATION = "DECISION_SCOPE_VIOLATION"


class StageLoadingTestCase(unittest.TestCase):
    """Loads the stage modules in setUpClass, never at import time.

    load_stage_module installs a fake ``llm_runtime`` into sys.modules; doing
    that while this file is merely being imported leaks the stub into
    test_llm_runtime_attempts when the whole suite runs together.  Every other
    stage test file loads inside setUpClass for the same reason.
    """

    @classmethod
    def setUpClass(cls):
        cls.collapse = load_stage_module("collapse_redundant_hierarchy")
        cls.balance = load_stage_module("balance_tree_structure")
        cls.polish = load_stage_module("polish_tree_labels")
        # The three stage predicates.  A fourth copy lives in the private
        # harness (run_stage_fail_closed_v2.is_deferrable_result), exercised by
        # the private-side test in the C13RF18 experiment directory; it cannot
        # be imported from the public tree, and verify_stage_v2 imports that
        # function rather than keeping a fifth copy.
        cls.plain_predicates = (
            ("collapse", cls.collapse.SkeletonRefiner._is_deferrable_response),
            ("polish", cls.polish.PolishingProcess._is_deferrable_response),
        )
        # staticmethod, or attribute access would re-bind it as an instance
        # method and pass self as the response.
        cls.balance_predicate = staticmethod(
            cls.balance.ShapingProcess._is_deferrable_response
        )


def scope_error(code, *, message="merge child target is outside the exact pair role"):
    return {
        "code": code,
        "message": message,
        "context": {"mutable_ref_paths": [], "repairable": False},
    }


def repaired_response(*, remainder, **overrides):
    """The C13RF18 shape: initial answer was mixed, repair left a remainder."""
    resp = {
        "ok": False,
        "final_disposition": "repair_scope_rejected",
        "error": "REPAIR_SCOPE_VALIDATION_FAILED",
        "mutation_before_validation": False,
        "final_bound": None,
        "repair_count": 1,
        "initial_scope_errors": [scope_error(VIOLATION)],
        "repair_scope_errors": (
            list(remainder) if isinstance(remainder, list) else remainder
        ),
    }
    resp.update(overrides)
    return resp


def pure_response(*, errors, **overrides):
    """The C13RF6 shape: initial answer refused outright, no repair."""
    resp = {
        "ok": False,
        "final_disposition": "scope_rejected",
        "error": "BOUND_SCOPE_VALIDATION_FAILED",
        "mutation_before_validation": False,
        "final_bound": None,
        "repair_count": 0,
        "initial_scope_errors": list(errors),
        "repair_scope_errors": [],
    }
    resp.update(overrides)
    return resp


class MixedDeferPredicateTests(StageLoadingTestCase):
    """Channel 2 opens for a wholly inexpressible remainder, and only that."""

    def test_repaired_inexpressible_remainder_is_deferrable(self):
        resp = repaired_response(remainder=[scope_error(INEXPRESSIBLE)])
        for name, predicate in self.plain_predicates:
            with self.subTest(stage=name):
                self.assertTrue(predicate(resp))

    def test_pure_initial_case_is_unchanged(self):
        """Channel 1 must keep behaving exactly as C13RF6 left it."""
        resp = pure_response(errors=[scope_error(INEXPRESSIBLE)])
        for name, predicate in self.plain_predicates:
            with self.subTest(stage=name):
                self.assertTrue(predicate(resp))

    def test_impure_remainder_still_stops_the_run(self):
        """The pinned strictness floor: one non-inexpressible residue stops."""
        remainders = {
            "all violations": [scope_error(VIOLATION)],
            "inexpressible plus violation": [
                scope_error(INEXPRESSIBLE),
                scope_error(VIOLATION),
            ],
            "violation plus inexpressible": [
                scope_error(VIOLATION),
                scope_error(INEXPRESSIBLE),
            ],
            "unknown code": [scope_error("DECISION_SCOPE_SOMETHING_NEW")],
            "non-dict member": [scope_error(INEXPRESSIBLE), "not a dict"],
        }
        for label, remainder in remainders.items():
            resp = repaired_response(remainder=remainder)
            for name, predicate in self.plain_predicates:
                with self.subTest(stage=name, remainder=label):
                    self.assertFalse(predicate(resp))

    def test_empty_remainder_is_not_deferrable(self):
        """No recorded error is not the same as every error being benign."""
        for name, predicate in self.plain_predicates:
            with self.subTest(stage=name):
                self.assertFalse(predicate(repaired_response(remainder=[])))
                self.assertFalse(
                    predicate(repaired_response(remainder=None))
                )

    def test_multi_round_repair_fails_closed(self):
        """The binder performs at most one repair round; more must not inherit."""
        resp = repaired_response(
            remainder=[scope_error(INEXPRESSIBLE)], repair_count=2
        )
        for name, predicate in self.plain_predicates:
            with self.subTest(stage=name):
                self.assertFalse(predicate(resp))

    def test_shared_conditions_still_gate_channel_two(self):
        """Everything C13RF6 required of channel 1 also guards channel 2."""
        negatives = {
            "accepted call": {"ok": True},
            "something was bound": {"final_bound": {"decisions": []}},
            "tree moved under the validator": {
                "mutation_before_validation": True
            },
            "wrong disposition": {"final_disposition": "repair_binding_rejected"},
            "wrong error code": {"error": "REPAIR_BINDING_FAILED"},
            "transport-level failure": {
                "final_disposition": "transport_failed",
                "error": "TRANSPORT_EXHAUSTED",
            },
        }
        for label, override in negatives.items():
            resp = repaired_response(
                remainder=[scope_error(INEXPRESSIBLE)], **override
            )
            for name, predicate in self.plain_predicates:
                with self.subTest(stage=name, negative=label):
                    self.assertFalse(predicate(resp))

    def test_channels_do_not_cross_contaminate(self):
        """A clean initial ledger cannot rescue a dirty remainder, or vice versa."""
        dirty_remainder_clean_initial = repaired_response(
            remainder=[scope_error(VIOLATION)],
            initial_scope_errors=[scope_error(INEXPRESSIBLE)],
        )
        clean_remainder_but_pure_disposition = pure_response(
            errors=[scope_error(VIOLATION)],
            repair_scope_errors=[scope_error(INEXPRESSIBLE)],
        )
        for name, predicate in self.plain_predicates:
            with self.subTest(stage=name, case="dirty remainder"):
                self.assertFalse(predicate(dirty_remainder_clean_initial))
            with self.subTest(stage=name, case="remainder read on channel 1"):
                self.assertFalse(predicate(clean_remainder_but_pure_disposition))

    def test_non_mapping_response(self):
        for name, predicate in self.plain_predicates:
            with self.subTest(stage=name):
                self.assertFalse(predicate(None))
                self.assertFalse(predicate([]))


class BalanceNarrownessTests(StageLoadingTestCase):
    """C13RF18 widens *when* a channel is read, never what counts as deferrable.

    Balance deliberately defers only homogeneous ``create_bridge`` batches whose
    sole issue is the depth ceiling (C13RF10).  That narrowness must survive on
    channel 2 unchanged, so the benefit surface in 14b is smaller than in
    14a/14c -- recorded here rather than left to be rediscovered.
    """

    def depth_error(self, index):
        return {
            "code": INEXPRESSIBLE,
            "message": self.balance.BRIDGE_DEPTH_DEFERRED_MESSAGE,
            "context": {
                "decision_index": index,
                "mutable_ref_paths": [],
                "repairable": False,
            },
        }

    def bridges(self, count):
        return {
            "decisions": [
                {"action": "create_bridge"} for _ in range(count)
            ]
        }

    def test_repaired_depth_only_batch_is_deferrable(self):
        resp = repaired_response(
            remainder=[self.depth_error(0)],
            final_local=self.bridges(1),
        )
        self.assertTrue(self.balance_predicate(resp))

    def test_repaired_inexpressible_but_not_depth_is_refused(self):
        """Right code, wrong reason -- balance's message literal still gates."""
        remainder = [self.depth_error(0)]
        remainder[0]["message"] = "merge child target is outside the exact pair role"
        resp = repaired_response(
            remainder=remainder, final_local=self.bridges(1)
        )
        self.assertFalse(self.balance_predicate(resp))

    def test_repaired_batch_with_a_non_bridge_decision_is_refused(self):
        resp = repaired_response(
            remainder=[self.depth_error(0)],
            final_local={
                "decisions": [{"action": "create_bridge"}, {"action": "merge"}]
            },
        )
        self.assertFalse(self.balance_predicate(resp))

    def test_repaired_batch_not_fully_covered_by_errors_is_refused(self):
        """A surviving legal bridge alongside a deferred one still stops."""
        resp = repaired_response(
            remainder=[self.depth_error(0)],
            final_local=self.bridges(2),
        )
        self.assertFalse(self.balance_predicate(resp))

    def test_pure_depth_batch_unchanged(self):
        resp = pure_response(
            errors=[self.depth_error(0), self.depth_error(1)],
            final_local=self.bridges(2),
        )
        self.assertTrue(self.balance_predicate(resp))


# ---------------------------------------------------------------------------
# End-to-end: the same four scenarios the reachability probe ran, but driven
# through the real binder and the real _scope_issues classifier.  HTTP is
# structurally impossible -- the transport raises when its queue is exhausted.
# ---------------------------------------------------------------------------

SOURCE_FULL = "L2_NFULL001"
TARGET = "L2_NTGT001"
REFS = (
    ("FULL", SOURCE_FULL),
    ("TGT", TARGET),
    ("SCHILD1", "L3_NSC0001"),
    ("SCHILD2", "L3_NSC0002"),
    ("TCHILD", "L3_NTC0001"),
    ("OTHER", "L2_NOTHER1"),
)


class ScriptedTransport:
    def __init__(self, *payloads):
        self.payloads = list(payloads)
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(copy.deepcopy(kwargs))
        if not self.payloads:
            raise AssertionError("scripted transport exhausted -- no network here")
        payload = self.payloads.pop(0)
        raw = json.dumps(payload, ensure_ascii=False)
        return {
            "ok": True, "json": copy.deepcopy(payload), "raw": raw, "error": None,
            "status": 200, "latency_ms": 1, "profile": "fake", "provider": "fake",
            "model": "fake", "attempts": 1,
            "attempt_history": [{
                "attempt": 1, "status": 200, "latency_ms": 1, "raw": raw,
                "parse_ok": True, "error": None,
            }],
        }


def fixture_tree():
    def node(node_id, level, children=()):
        return {
            "node_id": node_id,
            "label": node_id.lower(),
            "level": level,
            "children": list(children),
        }

    return node("ROOT", "ROOT", [
        node("L1_NDOMAIN1", "L1", [
            node(TARGET, "L2", [node("L3_NTC0001", "L3")]),
            node(SOURCE_FULL, "L2", [
                node("L3_NSC0001", "L3"),
                node("L3_NSC0002", "L3"),
            ]),
        ]),
        node("L1_NDOMAIN2", "L1", [node("L2_NOTHER1", "L2")]),
    ])


def merge_decision(children):
    return {
        "relation": "exact_duplicate",
        "action": "merge",
        "source_ref": "FULL",
        "target_ref": "TGT",
        "new_label": None,
        "confidence": 0.95,
        "evidence": {
            "summary": "The two labels name the same policy instrument.",
            "warnings": [],
            "target_represents_all_source_members": False,
            "membership_basis": "",
            "pure_structural_redundancy": False,
            "cross_l1_authorized": False,
        },
        "child_plan": list(children),
    }


def child(child_ref, target_parent_ref):
    return {
        "child_ref": child_ref,
        "disposition": "move",
        "target_parent_ref": target_parent_ref,
        "relation": "broader_narrower",
        "same_domain": True,
        "evidence": "The child stays inside the displayed destination domain.",
    }


class EndToEndMixedDeferTests(StageLoadingTestCase):
    def run_call(self, initial_children, repair_children):
        system = "c13rf18 fixture contract"
        manager = TreeManager(copy.deepcopy(fixture_tree()))
        context = build_local_reference_context(
            task="polish_tree_labels",
            user_text="\n".join(f"{ref}={node_id}" for ref, node_id in REFS),
            candidate_node_ids=manager.get_all_node_ids(),
            preferred_refs=list(REFS),
            contract_text=system,
            expected_count=None,
        )
        payloads = [
            {
                "context_token": context.context_token,
                "decisions": [merge_decision(copy.deepcopy(children))],
            }
            for children in (initial_children, repair_children)
        ]
        transport = ScriptedTransport(*payloads)
        process = self.polish.PolishingProcess.__new__(self.polish.PolishingProcess)
        process.tm = manager
        before = copy.deepcopy(manager.root)
        with tempfile.TemporaryDirectory() as temporary_dir:
            process.ops_log = Path(temporary_dir) / "ops.jsonl"
            process.llm_log = Path(temporary_dir) / "llm.jsonl"
            process.membership_counts = {}
            process.trace_map = {}
            process.prior_lineage = {}
            process.deferred_candidates = set()
            resp = call_local_reference_json(
                transport=transport,
                profile="fake",
                system=system,
                context=context,
                task="polish_tree_labels",
                expected_count=None,
                protected_manager=process.tm,
                bound_validator=lambda value: process._scope_issues(
                    value, SOURCE_FULL, TARGET
                ),
            )
        self.assertEqual(
            before, manager.root, "a refused call must not touch the tree"
        )
        return resp

    def codes(self, items):
        return [item.get("code") for item in (items or []) if isinstance(item, dict)]

    def test_mixed_answer_repaired_to_an_inexpressible_remainder_defers(self):
        """The card's target case, end to end."""
        resp = self.run_call(
            [child("SCHILD1", "OTHER"), child("TCHILD", "TGT")],
            [child("SCHILD1", "OTHER"), child("SCHILD2", "TGT")],
        )
        self.assertFalse(resp["ok"])
        self.assertEqual(resp["error"], "REPAIR_SCOPE_VALIDATION_FAILED")
        self.assertEqual(resp["repair_count"], 1)
        # The ledger the predicate now reads, produced by the real classifier.
        self.assertEqual(self.codes(resp["initial_scope_errors"]), [VIOLATION])
        self.assertEqual(self.codes(resp["repair_scope_errors"]), [INEXPRESSIBLE])
        self.assertTrue(
            self.polish.PolishingProcess._is_deferrable_response(resp)
        )

    def test_pure_inexpressible_still_defers_on_channel_one(self):
        resp = self.run_call(
            [child("SCHILD1", "OTHER")], [child("SCHILD1", "OTHER")]
        )
        self.assertEqual(resp["error"], "BOUND_SCOPE_VALIDATION_FAILED")
        self.assertEqual(resp["repair_count"], 0)
        self.assertEqual(self.codes(resp["initial_scope_errors"]), [INEXPRESSIBLE])
        self.assertTrue(self.polish.PolishingProcess._is_deferrable_response(resp))

    def test_repaired_subject_error_is_accepted_not_deferred(self):
        resp = self.run_call(
            [child("TCHILD", "TGT")], [child("SCHILD2", "TGT")]
        )
        self.assertTrue(resp["ok"])
        self.assertEqual(resp["final_disposition"], "bound_after_repair")
        self.assertEqual(resp["repair_scope_errors"], [])
        self.assertFalse(self.polish.PolishingProcess._is_deferrable_response(resp))

    def test_repair_leaving_an_impurity_still_stops(self):
        """Control for the strictness floor, end to end."""
        resp = self.run_call(
            [child("SCHILD1", "OTHER"), child("TCHILD", "TGT")],
            [child("SCHILD1", "OTHER"), child("TCHILD", "TGT")],
        )
        self.assertEqual(resp["error"], "REPAIR_SCOPE_VALIDATION_FAILED")
        self.assertEqual(self.codes(resp["repair_scope_errors"]), [VIOLATION])
        self.assertFalse(self.polish.PolishingProcess._is_deferrable_response(resp))


if __name__ == "__main__":
    unittest.main()
