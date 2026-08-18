"""C13RF19: give 14d the defer channel it is the only stage never to have had.

C13RF15 measured the gap: ``DECISION_SCOPE_INEXPRESSIBLE`` counts are 2 / 2 / 2
for collapse / balance / polish and **0** for finalize, which did not even carry
the word ``deferred``.  It also classified all 14 of this stage's scope
predicates, and that classification is the whole scope of this card: **3** of
them describe a restructure the contract cannot express (a target in another L1,
a child sent somewhere its role does not allow, a split child rehomed across L1),
**3** are the model misreading its own call context, and **8** are structural
invariants where parking a proposal would be meaningless.  Only the first three
get a channel here.

The predicate itself is C13RF18's landed two-channel form, installed verbatim
rather than reinvented -- that is precisely why RF18 was sequenced before RF19.
``DeferPredicateShapeTests`` pins the equivalence to the two plain stages so a
future edit to one cannot silently fork the other.

Two boundaries the card names explicitly and this file locks:

* ``ACTION_NOT_ALLOWED_IN_STAGE`` is 14d's only terminal verdict and must stay
  terminal -- a verb the stage does not implement is not a verb it merely
  cannot spell.
* ``flatten`` never defers, even when it raises the very same
  "child target does not match the operation role" message that makes a merge
  deferrable.  C13RF16 withheld flatten from its empty-plan authorisation on the
  same reasoning, and ``FlattenExclusionTests`` keeps both halves pinned.
"""
from __future__ import annotations

import copy
import json
import tempfile
import types
import unittest
from pathlib import Path

from tests.test_semantic_stage_dispatch import load_stage_module
from utils.local_reference_binding import (
    VOID_CHILD_PLAN_FLAG,
    build_local_reference_context,
    call_local_reference_json,
)
from utils.semantic_contract import (
    validate_semantic_decision,
    validate_semantic_history,
)
from utils.tree_manager import TreeManager

INEXPRESSIBLE = "DECISION_SCOPE_INEXPRESSIBLE"
VIOLATION = "DECISION_SCOPE_VIOLATION"
NOT_ALLOWED = "ACTION_NOT_ALLOWED_IN_STAGE"
L1_ID = "L1_A"

REFS = (
    ("SRC", "L2_SRC"),
    ("TGT", "L2_TGT"),
    ("S1", "L3_S1"),
    ("S2", "L3_S2"),
    ("S1A", "L4_S1A"),
    ("T1", "L3_T1"),
    ("FOREIGN", "L2_FOREIGN"),
    ("L1", L1_ID),
)


def audited_tree() -> dict:
    """One audited L1 (source with two children, target with one) plus a second
    L1 holding a node that is out of scope for the audited batch."""
    return {
        "node_id": "ROOT", "label": "root", "level": "ROOT", "children": [
            {"node_id": "L1_A", "label": "l1a", "level": "L1", "children": [
                {"node_id": "L2_SRC", "label": "src", "level": "L2", "children": [
                    {"node_id": "L3_S1", "label": "s1", "level": "L3", "children": [
                        {"node_id": "L4_S1A", "label": "s1a", "level": "L4",
                         "children": []},
                    ]},
                    {"node_id": "L3_S2", "label": "s2", "level": "L3",
                     "children": []},
                ]},
                {"node_id": "L2_TGT", "label": "tgt", "level": "L2", "children": [
                    {"node_id": "L3_T1", "label": "t1", "level": "L3",
                     "children": []},
                ]},
            ]},
            {"node_id": "L1_B", "label": "l1b", "level": "L1", "children": [
                {"node_id": "L2_FOREIGN", "label": "foreign", "level": "L2",
                 "children": []},
            ]},
        ],
    }


def childless_source_tree() -> dict:
    """C13RF13 call 25's topology, transposed into 14d: the source owns nothing,
    so the only legal child_plan is the empty one (C13RF16 ①)."""
    return {
        "node_id": "ROOT", "label": "root", "level": "ROOT", "children": [
            {"node_id": "L1_A", "label": "l1a", "level": "L1", "children": [
                {"node_id": "L2_SRC", "label": "src", "level": "L2",
                 "children": []},
                {"node_id": "L2_TGT", "label": "tgt", "level": "L2", "children": [
                    {"node_id": f"L3_C{i}", "label": f"c{i}", "level": "L3",
                     "children": []} for i in (1, 2, 3)
                ]},
            ]},
        ],
    }


def op(action: str, source_id: str, target_id: str, **extra) -> dict:
    decision = {"action": action, "source_id": source_id, "target_id": target_id}
    decision.update(extra)
    return decision


def child(child_id: str, target_parent_id: str, disposition: str = "move") -> dict:
    return {
        "child_id": child_id,
        "target_parent_id": target_parent_id,
        "disposition": disposition,
    }


class StageLoadingTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.finalize = load_stage_module("finalize_policy_tree")
        cls.collapse = load_stage_module("collapse_redundant_hierarchy")
        cls.polish = load_stage_module("polish_tree_labels")
        cls.balance = load_stage_module("balance_tree_structure")

    def issues(self, decisions, tree=None, l1_id=L1_ID):
        manager = TreeManager(copy.deepcopy(tree or audited_tree()))
        probe = types.SimpleNamespace(tm=manager)
        return self.finalize.OverallStructureAudit._scope_issues(
            probe, {"decisions": list(decisions)}, l1_id
        )

    def only(self, decision, tree=None):
        found = self.issues([decision], tree=tree)
        self.assertEqual(len(found), 1, found)
        return found[0]


# --------------------------------------------------------------------------
# The three predicates that gain a channel.
# --------------------------------------------------------------------------

class DeferredPredicateTests(StageLoadingTestCase):
    def assert_deferred(self, issue):
        self.assertEqual(issue["code"], INEXPRESSIBLE)
        # Non-repairable with no mutable path is what makes the binder skip the
        # repair round entirely, so the stage stops burning an attempt on a
        # request the model cannot satisfy.
        self.assertIs(issue["context"]["repairable"], False)
        self.assertEqual(issue["context"]["mutable_ref_paths"], [])

    def test_d4_target_in_another_l1_defers(self):
        """RF5 call-85's shape at 14d: relocating across L1 can be right, but
        finalization is scoped per L1 and cannot express it."""
        issue = self.only(op("merge", "L2_SRC", "L2_FOREIGN"))
        self.assert_deferred(issue)
        self.assertEqual(
            issue["message"], self.finalize.TARGET_OUTSIDE_L1_MESSAGE
        )

    def test_d12_child_target_outside_its_role_defers(self):
        """"Merge these two, but send one child elsewhere" -- coherent intent,
        no way to encode it.  14c has had a branch for this family since RF6."""
        issue = self.only(
            op("merge", "L2_SRC", "L2_TGT",
               child_plan=[child("L3_S1", "L2_FOREIGN")])
        )
        self.assert_deferred(issue)
        self.assertEqual(
            issue["message"], self.finalize.CHILD_ROLE_MISMATCH_MESSAGE
        )
        self.assertEqual(issue["context"]["child_indexes"], [0])

    def test_d14_split_child_rehomed_across_l1_defers(self):
        issue = self.only(
            op("split_reparent", "L2_SRC", "L2_SRC",
               child_plan=[child("L3_S1", "L2_FOREIGN")])
        )
        self.assert_deferred(issue)
        self.assertEqual(
            issue["message"], self.finalize.SPLIT_CHILD_OUTSIDE_L1_MESSAGE
        )

    def test_move_child_left_behind_defers(self):
        """The same expressiveness gap reached through `move` rather than
        `merge`: a move must carry every child, so "move it but leave one child
        here" has no encoding."""
        issue = self.only(
            op("move", "L3_S1", "L2_TGT",
               child_plan=[child("L4_S1A", "L2_FOREIGN")])
        )
        self.assert_deferred(issue)

    def test_whitelist_is_exactly_the_three_measured_messages(self):
        self.assertEqual(
            set(self.finalize.INEXPRESSIBLE_SCOPE_MESSAGES),
            {
                self.finalize.TARGET_OUTSIDE_L1_MESSAGE,
                self.finalize.CHILD_ROLE_MISMATCH_MESSAGE,
                self.finalize.SPLIT_CHILD_OUTSIDE_L1_MESSAGE,
            },
        )


# --------------------------------------------------------------------------
# Strictness floor: nothing else may reach the channel.
# --------------------------------------------------------------------------

class SubjectErrorNeverDefersTests(StageLoadingTestCase):
    """The 3 predicates C13RF15 classified as the model misreading its own call
    context.  These are factual mistakes, and a parked factual mistake is just a
    lost correction."""

    def test_d2_source_in_another_l1(self):
        issue = self.only(op("merge", "L2_FOREIGN", "L2_TGT"))
        self.assertEqual(issue["code"], VIOLATION)
        self.assertIs(issue["context"]["repairable"], True)

    def test_d3_source_is_the_audited_l1_itself(self):
        issue = self.only(op("merge", "L1_A", "L2_TGT"))
        self.assertEqual(issue["code"], VIOLATION)
        self.assertIs(issue["context"]["repairable"], True)

    def test_d11_child_plan_names_a_non_source_child(self):
        issue = self.only(
            op("merge", "L2_SRC", "L2_TGT",
               child_plan=[child(f"L3_C{i}", "L2_TGT") for i in (1, 2, 3)]),
            tree=childless_source_tree(),
        )
        self.assertEqual(issue["code"], VIOLATION)
        self.assertIs(issue["context"]["repairable"], True)
        # C13RF16 ① must survive untouched: emptying the plan is the correction.
        self.assertIs(issue["context"][VOID_CHILD_PLAN_FLAG], True)


class HardConstraintNeverDefersTests(StageLoadingTestCase):
    """The 8 structural invariants.  Deferring one would park a proposal no
    later stage could execute either -- the same reasoning RF9 recorded when
    14b's depth ceiling was found to have no defer channel *correctly*."""

    def test_all_eight_hard_constraints_stay_violations(self):
        cases = {
            "D1_action_not_allowed": (
                op("create_bridge", "L2_SRC", "L2_TGT"), None, NOT_ALLOWED),
            "D5_rename_source_target_differ": (
                op("rename", "L2_SRC", "L2_TGT"), None, VIOLATION),
            "D6_split_source_target_differ": (
                op("split_reparent", "L2_SRC", "L2_TGT"), None, VIOLATION),
            "D7_flatten_target_not_parent": (
                op("flatten", "L3_S1", "L2_TGT"), None, VIOLATION),
            "D8_merge_source_equals_target": (
                op("merge", "L2_SRC", "L2_SRC"), None, VIOLATION),
            "D9_move_target_is_current_parent": (
                op("move", "L3_S1", "L2_SRC"), None, VIOLATION),
            "D10_move_into_own_subtree": (
                op("move", "L3_S1", "L4_S1A"), None, VIOLATION),
            "D13_split_target_is_child_itself": (
                op("split_reparent", "L2_SRC", "L2_SRC",
                   child_plan=[child("L3_S1", "L3_S1")]), None, VIOLATION),
        }
        for name, (decision, tree, expected) in cases.items():
            with self.subTest(case=name):
                issue = self.only(decision, tree=tree)
                self.assertEqual(issue["code"], expected)
                self.assertNotEqual(issue["code"], INEXPRESSIBLE)

    def test_action_not_allowed_stays_terminal(self):
        issue = self.only(op("create_bridge", "L2_SRC", "L2_TGT"))
        self.assertEqual(issue["code"], NOT_ALLOWED)
        self.assertIs(issue["context"]["repairable"], False)

    def test_terminal_inventory_did_not_grow(self):
        """Besides the new defer code -- which is non-repairable by design, not
        by terminality -- ACTION_NOT_ALLOWED_IN_STAGE remains 14d's only
        non-repairable verdict (C13RF15 case D16)."""
        seen = set()
        for decision in (
            op("create_bridge", "L2_SRC", "L2_TGT"),
            op("rename", "L2_SRC", "L2_TGT"),
            op("merge", "L2_FOREIGN", "L2_TGT"),
            op("move", "L3_S1", "L4_S1A"),
            op("merge", "L2_SRC", "L2_SRC"),
        ):
            issue = self.only(decision)
            if issue["context"]["repairable"] is False:
                seen.add(issue["code"])
        self.assertEqual(seen, {NOT_ALLOWED})


class FlattenExclusionTests(StageLoadingTestCase):
    """`flatten` is in 14d's whitelist and can raise the deferrable message, yet
    must never defer."""

    def test_flatten_with_wrong_child_role_is_a_violation_not_a_defer(self):
        issue = self.only(
            op("flatten", "L3_S1", "L2_SRC",
               child_plan=[child("L4_S1A", "L2_FOREIGN")])
        )
        self.assertEqual(issue["code"], VIOLATION)
        self.assertIn(
            self.finalize.CHILD_ROLE_MISMATCH_MESSAGE, issue["message"]
        )
        self.assertIs(issue["context"]["repairable"], True)
        # An excluded verb keeps its whole pre-C13RF19 repair surface: nothing
        # was withheld, so the model can still be asked to fix it.
        self.assertNotIn("withheld_ref_paths", issue["context"])
        self.assertEqual(
            issue["context"]["mutable_ref_paths"],
            ["decisions[0].child_plan[0].target_parent_ref"],
        )

    def test_flatten_is_not_in_the_deferrable_action_set(self):
        self.assertEqual(
            set(self.finalize.DEFERRABLE_ACTIONS),
            {"merge", "move", "split_reparent"},
        )

    def test_flatten_on_a_childless_source_is_still_rejected(self):
        """C13RF15's F2 kept as a control: the shared contract refuses a flatten
        with no children, and C13RF19 must not open a side door for it."""
        report = validate_semantic_decision(
            {
                "relation": "broader_narrower",
                "action": "flatten",
                "source_id": "L2_SRC",
                "target_id": "L1_A",
                "new_label": None,
                "confidence": 0.9,
                "evidence": {
                    "summary": "dissolve the empty umbrella",
                    "warnings": [],
                    "pure_structural_redundancy": True,
                    "cross_l1_authorized": False,
                },
                "child_plan": [],
            },
            {
                "source_children": [],
                "target_child_ids": [],
                "source_direct_membership_count": 0,
                "source_parent_id": "L1_A",
                "source_l1_id": L1_ID,
                "target_l1_id": L1_ID,
            },
        )
        self.assertIs(report["passed"], False)
        self.assertIn(
            "FLATTEN_SOURCE_HAS_NO_CHILDREN",
            [item["code"] for item in report["violations"]],
        )

    def test_flatten_never_gets_the_void_child_plan_authorisation(self):
        """C13RF16 restricted the empty-plan authorisation to merge and move; an
        empty plan is permanently illegal for flatten, so authorising it would
        only relocate the rejection."""
        issue = self.only(
            op("flatten", "L2_SRC", "L1_A",
               child_plan=[child("L3_C1", "L1_A")]),
            tree=childless_source_tree(),
        )
        self.assertEqual(issue["code"], VIOLATION)
        self.assertNotIn(VOID_CHILD_PLAN_FLAG, issue["context"])


class MixedDecisionTests(StageLoadingTestCase):
    """A decision carrying both kinds of error must report both and withhold the
    deferrable half's refs, so the repair round cannot coerce an out-of-scope
    destination into a legal-but-wrong one (the harm C13RF16 measured)."""

    def test_mixed_decision_is_a_violation_that_records_its_deferrable_half(self):
        issue = self.only(
            op("merge", "L2_SRC", "L2_TGT", child_plan=[
                child("L3_S1", "L2_FOREIGN"),   # inexpressible half
                child("L3_T1", "L2_TGT"),       # subject error, repairable
            ])
        )
        self.assertEqual(issue["code"], VIOLATION)
        context = issue["context"]
        self.assertEqual(
            context["inexpressible_components"],
            [self.finalize.CHILD_ROLE_MISMATCH_MESSAGE],
        )
        self.assertEqual(context["inexpressible_child_indexes"], [0])
        self.assertEqual(
            context["withheld_ref_paths"],
            ["decisions[0].child_plan[0].target_parent_ref"],
        )
        # The withheld path is gone from the repair surface; the subject error's
        # own path is still there, so the repair round has real work to do.
        self.assertNotIn(
            "decisions[0].child_plan[0].target_parent_ref",
            context["mutable_ref_paths"],
        )
        self.assertIn(
            "decisions[0].child_plan[1].child_ref", context["mutable_ref_paths"]
        )
        self.assertIs(context["repairable"], True)

    def test_void_child_plan_message_blocks_the_defer_channel(self):
        """A childless source additionally raises the C13RF16 ① message, which
        is not on the whitelist, so the decision stays a violation."""
        issue = self.only(
            op("merge", "L2_SRC", "L2_TGT",
               child_plan=[child("L3_C1", "L2_FOREIGN")]),
            tree=childless_source_tree(),
        )
        self.assertEqual(issue["code"], VIOLATION)
        self.assertIs(issue["context"][VOID_CHILD_PLAN_FLAG], True)


class LegalCallControlTests(StageLoadingTestCase):
    def test_a_well_formed_batch_still_passes_cleanly(self):
        self.assertEqual(
            self.issues([
                op("merge", "L2_SRC", "L2_TGT",
                   child_plan=[child("L3_S1", "L2_TGT"),
                               child("L3_S2", "L2_TGT")]),
                op("keep", "L3_T1", "L3_T1"),
            ]),
            [],
        )


# --------------------------------------------------------------------------
# The predicate is C13RF18's, not a second shape.
# --------------------------------------------------------------------------

def response(**overrides) -> dict:
    base = {
        "ok": False,
        "mutation_before_validation": False,
        "final_bound": None,
        "final_disposition": "scope_rejected",
        "error": "BOUND_SCOPE_VALIDATION_FAILED",
        "repair_count": 0,
        "initial_scope_errors": [{"code": INEXPRESSIBLE}],
        "repair_scope_errors": [],
    }
    base.update(overrides)
    return base


class DeferPredicateShapeTests(StageLoadingTestCase):
    """Card requirement ③: the new predicate must agree field for field with the
    one C13RF18 landed, and the agreement must be *checked*, not asserted in
    prose.  `balance` is excluded on purpose -- RF10 gave it an extra narrowness
    (homogeneous depth-ceiling create_bridge batches only) that RF18 preserved on
    both channels, so it is deliberately stricter than the two plain stages."""

    BATTERY = {
        # channel 1
        "c1_pure_initial": (response(), True),
        # channel 2
        "c2_repaired_remainder": (response(
            final_disposition="repair_scope_rejected",
            error="REPAIR_SCOPE_VALIDATION_FAILED",
            repair_count=1,
            initial_scope_errors=[{"code": VIOLATION}],
            repair_scope_errors=[{"code": INEXPRESSIBLE}],
        ), True),
        # shared preconditions
        "neg_ok_true": (response(ok=True), False),
        "neg_mutation_seen": (response(mutation_before_validation=True), False),
        "neg_bound_not_none": (response(final_bound={"decisions": []}), False),
        "neg_not_a_dict": ("not a response", False),
        # channel 1 negatives
        "neg_c1_wrong_disposition": (response(final_disposition="binding_rejected"), False),
        "neg_c1_wrong_error": (response(error="REPAIR_BINDING_FAILED"), False),
        "neg_c1_empty_channel": (response(initial_scope_errors=[]), False),
        "neg_c1_impure_channel": (response(
            initial_scope_errors=[{"code": INEXPRESSIBLE}, {"code": VIOLATION}],
        ), False),
        "neg_c1_repair_happened": (response(repair_count=1), False),
        # channel 2 negatives
        "neg_c2_empty_remainder": (response(
            final_disposition="repair_scope_rejected",
            error="REPAIR_SCOPE_VALIDATION_FAILED",
            repair_count=1,
            repair_scope_errors=[],
        ), False),
        "neg_c2_impure_remainder": (response(
            final_disposition="repair_scope_rejected",
            error="REPAIR_SCOPE_VALIDATION_FAILED",
            repair_count=1,
            repair_scope_errors=[{"code": INEXPRESSIBLE}, {"code": VIOLATION}],
        ), False),
        "neg_c2_multi_round_repair": (response(
            final_disposition="repair_scope_rejected",
            error="REPAIR_SCOPE_VALIDATION_FAILED",
            repair_count=2,
            repair_scope_errors=[{"code": INEXPRESSIBLE}],
        ), False),
        # cross-channel smearing, both directions
        "neg_initial_ledger_read_on_repair_path": (response(
            final_disposition="repair_scope_rejected",
            error="REPAIR_SCOPE_VALIDATION_FAILED",
            repair_count=1,
            initial_scope_errors=[{"code": INEXPRESSIBLE}],
            repair_scope_errors=[{"code": VIOLATION}],
        ), False),
        "neg_repair_ledger_read_on_initial_path": (response(
            repair_scope_errors=[{"code": INEXPRESSIBLE}],
            initial_scope_errors=[{"code": VIOLATION}],
        ), False),
    }

    def test_finalize_predicate_matches_the_expected_verdicts(self):
        predicate = self.finalize.OverallStructureAudit._is_deferrable_response
        for name, (payload, expected) in self.BATTERY.items():
            with self.subTest(case=name):
                self.assertIs(predicate(payload), expected)

    def test_finalize_agrees_with_both_plain_stages_on_every_case(self):
        finalize = self.finalize.OverallStructureAudit._is_deferrable_response
        others = {
            "collapse": self.collapse.SkeletonRefiner._is_deferrable_response,
            "polish": self.polish.PolishingProcess._is_deferrable_response,
        }
        for name, (payload, _expected) in self.BATTERY.items():
            for stage, other in others.items():
                with self.subTest(case=name, stage=stage):
                    self.assertIs(finalize(payload), other(payload))


# --------------------------------------------------------------------------
# End to end, through the real binder.
# --------------------------------------------------------------------------

class ScriptedTransport:
    """Returns pre-scripted model answers and raises once exhausted, so a test
    that accidentally asks for another round fails loudly instead of reaching a
    network."""

    def __init__(self, *payloads):
        self.payloads = list(payloads)
        self.calls = 0

    def __call__(self, **_kwargs):
        self.calls += 1
        if not self.payloads:
            raise AssertionError("scripted transport exhausted")
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


def local_decision(action, source_ref, target_ref, children) -> dict:
    return {
        "relation": "exact_duplicate",
        "action": action,
        "source_ref": source_ref,
        "target_ref": target_ref,
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


def local_child(child_ref, target_parent_ref) -> dict:
    return {
        "child_ref": child_ref,
        "disposition": "move",
        "target_parent_ref": target_parent_ref,
        "relation": "broader_narrower",
        "same_domain": True,
        "evidence": "The child stays inside the displayed destination domain.",
    }


class EndToEndDeferTests(StageLoadingTestCase):
    """Whole logical calls through the real binder, with 14d's own _scope_issues
    as the bound validator and a scripted transport in place of a network."""

    def bind(self, *payloads, decisions=None, repeat=1):
        system = "c13rf19 contract"
        manager = TreeManager(copy.deepcopy(audited_tree()))
        context = build_local_reference_context(
            task="finalize_policy_tree",
            user_text="\n".join(f"{ref}={node_id}" for ref, node_id in REFS),
            candidate_node_ids=manager.get_all_node_ids(),
            preferred_refs=list(REFS),
            contract_text=system,
            expected_count=None,
        )
        scripted = [
            {
                "context_token": context.context_token,
                "decisions": copy.deepcopy(decisions) if decisions else [
                    local_decision("merge", "SRC", "TGT", children)
                ],
            }
            for children in (payloads or [None] * repeat)
        ]
        transport = ScriptedTransport(*scripted)
        audit = self.finalize.OverallStructureAudit.__new__(
            self.finalize.OverallStructureAudit
        )
        audit.tm = manager
        response_obj = call_local_reference_json(
            transport=transport,
            profile="fake",
            system=system,
            context=context,
            task="finalize_policy_tree",
            expected_count=None,
            protected_manager=audit.tm,
            bound_validator=lambda value: audit._scope_issues(value, L1_ID),
        )
        return response_obj, audit, transport

    def record_for(self, response_obj):
        audit = self.finalize.OverallStructureAudit.__new__(
            self.finalize.OverallStructureAudit
        )
        audit.operation_records = []
        audit.stats = {
            "llm_calls": 0, "llm_failures": 0,
            "ops_applied": 0, "ops_skipped": 0, "ops_deferred": 0,
        }
        with tempfile.TemporaryDirectory() as directory:
            audit.ops_log = Path(directory) / "ops.jsonl"
            returned = audit._record_deferred_restructure(response_obj)
            written = [
                json.loads(line) for line
                in audit.ops_log.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        return returned, written, audit

    def test_pure_inexpressible_call_defers_without_a_repair_round(self):
        response_obj, _audit, transport = self.bind(
            [local_child("S1", "FOREIGN")]
        )
        self.assertIs(
            self.finalize.OverallStructureAudit._is_deferrable_response(
                response_obj
            ),
            True,
        )
        self.assertEqual(response_obj["final_disposition"], "scope_rejected")
        self.assertEqual(response_obj["repair_count"], 0)
        # The deferrable verdict is non-repairable, so the binder never asked
        # the model to try again: the fix also stops burning that attempt.
        self.assertEqual(transport.calls, 1)

        returned, written, audit = self.record_for(response_obj)
        self.assertEqual(returned, [])  # run() skips the batch and continues
        self.assertEqual(len(written), 1)
        record = written[0]
        self.assertEqual(record["status"], "deferred")
        self.assertEqual(record["step"], "finalization")
        self.assertEqual(record["type"], "semantic_decision")
        self.assertEqual(audit.stats["ops_deferred"], 1)
        # Real node ids are resolved back from the call-local refs.
        resolved = record["resolved_proposal"]["decisions"][0]
        self.assertEqual(resolved["source_id"], "L2_SRC")
        self.assertEqual(
            resolved["child_plan"][0]["target_parent_id"], "L2_FOREIGN"
        )

    def test_deferred_record_names_the_scope_error_it_actually_deferred(self):
        """deferred_restructure_record() hard-codes 14a/14c's single deferrable
        message, which is wrong for this stage's three.  The stage attaches what
        it really deferred rather than reword a shared record."""
        response_obj, _audit, _transport = self.bind(
            [local_child("S1", "FOREIGN")]
        )
        _returned, written, _audit2 = self.record_for(response_obj)
        record = written[0]
        self.assertEqual(record["deferred_scope_channel"], "initial_scope_errors")
        self.assertEqual(
            [item["code"] for item in record["deferred_scope_errors"]],
            [INEXPRESSIBLE],
        )
        self.assertEqual(
            record["deferred_scope_errors"][0]["message"],
            self.finalize.CHILD_ROLE_MISMATCH_MESSAGE,
        )

    def test_mixed_call_defers_only_after_an_honest_repair(self):
        """Channel 2: the repair round fixes the subject error and cannot touch
        the withheld half, so the remainder is wholly inexpressible."""
        response_obj, _audit, transport = self.bind(
            [local_child("S1", "FOREIGN"), local_child("T1", "TGT")],
            [local_child("S1", "FOREIGN"), local_child("S2", "TGT")],
        )
        self.assertEqual(transport.calls, 2)
        self.assertEqual(
            response_obj["final_disposition"], "repair_scope_rejected"
        )
        self.assertEqual(response_obj["repair_count"], 1)
        self.assertIs(
            self.finalize.OverallStructureAudit._is_deferrable_response(
                response_obj
            ),
            True,
        )
        _returned, written, _audit2 = self.record_for(response_obj)
        self.assertEqual(
            written[0]["deferred_scope_channel"], "repair_scope_errors"
        )

    def test_mixed_call_with_an_impure_remainder_still_fails(self):
        """The pinned severity floor: one non-inexpressible residue and the call
        stays a plain failure, which is what stops the run."""
        response_obj, _audit, _transport = self.bind(
            [local_child("S1", "FOREIGN"), local_child("T1", "TGT")],
            [local_child("S1", "FOREIGN"), local_child("T1", "TGT")],
        )
        self.assertEqual(
            response_obj["error"], "REPAIR_SCOPE_VALIDATION_FAILED"
        )
        self.assertIs(
            self.finalize.OverallStructureAudit._is_deferrable_response(
                response_obj
            ),
            False,
        )

    def test_repaired_subject_error_is_accepted_and_never_deferred(self):
        response_obj, _audit, _transport = self.bind(
            [local_child("T1", "TGT")],
            [local_child("S1", "TGT")],
        )
        self.assertIs(response_obj["ok"], True)
        self.assertIs(
            self.finalize.OverallStructureAudit._is_deferrable_response(
                response_obj
            ),
            False,
        )

    def test_flatten_call_never_reaches_the_defer_channel(self):
        response_obj, _audit, _transport = self.bind(
            decisions=[local_decision(
                "flatten", "S1", "SRC", [local_child("S1A", "FOREIGN")]
            )],
            repeat=2,
        )
        self.assertIs(
            self.finalize.OverallStructureAudit._is_deferrable_response(
                response_obj
            ),
            False,
        )

    def test_action_not_allowed_call_never_reaches_the_defer_channel(self):
        response_obj, _audit, _transport = self.bind(
            decisions=[local_decision(
                "create_bridge", "SRC", "TGT", [local_child("S1", "TGT")]
            )],
        )
        self.assertEqual(
            [item["code"] for item in response_obj["initial_scope_errors"]],
            [NOT_ALLOWED],
        )
        self.assertIs(
            self.finalize.OverallStructureAudit._is_deferrable_response(
                response_obj
            ),
            False,
        )

    def test_a_deferred_batch_parks_its_legal_decisions_too(self):
        """Registered boundary, not a defect introduced here: 14d binds and
        commits a whole L1 batch at once, so deferring one inexpressible
        decision parks the batch around it.  The baseline discarded the same
        batch *and* stopped the run, so this is strictly the smaller loss --
        but a 14d defer must never be read as "only that one decision was
        dropped"."""
        response_obj, _audit, _transport = self.bind(
            decisions=[
                local_decision("merge", "SRC", "TGT",
                               [local_child("S1", "FOREIGN")]),
                local_decision("keep", "T1", "T1", []),
            ],
        )
        self.assertIs(
            self.finalize.OverallStructureAudit._is_deferrable_response(
                response_obj
            ),
            True,
        )
        # Exactly one decision was faulted, yet both are parked together.
        self.assertEqual(len(response_obj["initial_scope_errors"]), 1)
        self.assertEqual(len(response_obj["final_local"]["decisions"]), 2)

    def test_deferred_record_is_inert_for_the_history_validator(self):
        """C13RF6's requirement, re-checked for this stage: a deferred record
        must not look like an applied operation missing its approval."""
        response_obj, _audit, _transport = self.bind(
            [local_child("S1", "FOREIGN")]
        )
        _returned, written, _audit2 = self.record_for(response_obj)
        report = validate_semantic_history(written)
        self.assertIs(report["passed"], True)
        self.assertEqual(report["violations"], [])


class CallLlmHookTests(StageLoadingTestCase):
    """_call_llm must route a deferrable response to the deferred record and
    everything else to the existing rejection path."""

    def drive(self, canned):
        audit = self.finalize.OverallStructureAudit.__new__(
            self.finalize.OverallStructureAudit
        )
        audit.tm = TreeManager(copy.deepcopy(audited_tree()))
        audit.llm_profile = "fake"
        audit.operation_records = []
        audit.stats = {
            "llm_calls": 0, "llm_failures": 0,
            "ops_applied": 0, "ops_skipped": 0, "ops_deferred": 0,
        }
        original = self.finalize.call_local_reference_json
        self.finalize.call_local_reference_json = lambda **_kwargs: canned
        with tempfile.TemporaryDirectory() as directory:
            audit.ops_log = Path(directory) / "ops.jsonl"
            audit.llm_log = Path(directory) / "llm.jsonl"
            try:
                ops = audit._call_llm(
                    "batch listing node L1_A", L1_ID
                )
            finally:
                self.finalize.call_local_reference_json = original
            written = [
                json.loads(line) for line
                in audit.ops_log.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        return ops, written, audit

    def deferrable_response(self):
        response_obj, _audit, _transport = EndToEndDeferTests.bind(
            self, [local_child("S1", "FOREIGN")]
        )
        return response_obj

    def test_deferrable_response_is_parked_and_the_stage_continues(self):
        ops, written, audit = self.drive(self.deferrable_response())
        self.assertEqual(ops, [])          # run() moves to the next batch
        self.assertEqual(len(written), 1)
        self.assertEqual(written[0]["status"], "deferred")
        self.assertEqual(audit.stats["ops_deferred"], 1)
        self.assertEqual(audit.stats["llm_failures"], 0)

    def test_a_plain_failed_response_is_still_a_rejection(self):
        canned = {
            "ok": False, "json": None, "error": "BOUND_SCOPE_VALIDATION_FAILED",
            "final_disposition": "scope_rejected", "repair_count": 0,
            "mutation_before_validation": False, "final_bound": None,
            "initial_scope_errors": [{"code": VIOLATION}],
            "repair_scope_errors": [],
        }
        ops, written, audit = self.drive(canned)
        self.assertEqual(ops, [])
        self.assertEqual(audit.stats["ops_deferred"], 0)
        self.assertEqual(audit.stats["llm_failures"], 1)
        self.assertNotEqual(written[0].get("status"), "deferred")


class BatchPreflightChannelTests(StageLoadingTestCase):
    """The channel registry's family one: _scope_issues is also called a second
    time, as a batch preflight inside _apply_operations, where there is no
    concept of defer.  That path is unreachable for an inexpressible verdict --
    it only runs once the binder has already returned ok -- and this pins it so a
    later edit cannot quietly create a second, contradictory ledger."""

    def test_batch_preflight_cannot_see_an_inexpressible_verdict(self):
        manager = TreeManager(copy.deepcopy(audited_tree()))
        audit = self.finalize.OverallStructureAudit.__new__(
            self.finalize.OverallStructureAudit
        )
        audit.tm = manager
        # Anything _apply_operations receives has already passed the same
        # validator as bound_validator, so by construction it has no issues.
        accepted = [
            op("merge", "L2_SRC", "L2_TGT",
               child_plan=[child("L3_S1", "L2_TGT"), child("L3_S2", "L2_TGT")]),
        ]
        self.assertEqual(audit._scope_issues({"decisions": accepted}, L1_ID), [])
        # And a batch that would defer never reaches it: _call_llm returns [] and
        # run() skips to the next batch on an empty list.
        deferring = [
            op("merge", "L2_SRC", "L2_TGT",
               child_plan=[child("L3_S1", "L2_FOREIGN")]),
        ]
        issues = audit._scope_issues({"decisions": deferring}, L1_ID)
        self.assertEqual([item["code"] for item in issues], [INEXPRESSIBLE])


if __name__ == "__main__":
    unittest.main()
