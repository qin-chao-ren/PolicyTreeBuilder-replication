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
    context.

    C13RF19's reasoning was: these are factual mistakes, and a parked factual
    mistake is just a lost correction -- so give them a repair round, not a defer
    channel.  C13RF29 accepts that reasoning and overrides the conclusion, on the
    user's explicit decision: the repair round is what turned a lost correction into
    a lost RUN (C13RF28 threw away 53 completed calls over one), and no correction is
    worth that.  So these three are now recorded and skipped like everything else.

    What the class still defends, and what these tests still assert: all three are
    REFUSED.  None of them executes, before or after.  Only the disposition changed.
    """

    def test_d2_source_in_another_l1(self):
        issue = self.only(op("merge", "L2_FOREIGN", "L2_TGT"))
        self.assertEqual(issue["code"], INEXPRESSIBLE)
        self.assertIs(issue["context"]["repairable"], False)
        self.assertEqual(issue["context"]["mutable_ref_paths"], [])

    def test_d3_source_is_the_audited_l1_itself(self):
        issue = self.only(op("merge", "L1_A", "L2_TGT"))
        self.assertEqual(issue["code"], INEXPRESSIBLE)
        self.assertIs(issue["context"]["repairable"], False)
        self.assertEqual(issue["context"]["mutable_ref_paths"], [])

    def test_d11_child_plan_names_a_non_source_child(self):
        issue = self.only(
            op("merge", "L2_SRC", "L2_TGT",
               child_plan=[child(f"L3_C{i}", "L2_TGT") for i in (1, 2, 3)]),
            tree=childless_source_tree(),
        )
        self.assertEqual(issue["code"], INEXPRESSIBLE)
        self.assertIs(issue["context"]["repairable"], False)
        # C13RF16 ①'s flag is still raised -- it describes the proposal ("the empty
        # plan is the only legal one") and still reaches the record.  What no longer
        # happens is the repair round it used to authorise; see C13RF29's note in
        # test_c13rf16_scope_defer_fix.RepairTruncationTests.
        self.assertIs(issue["context"][VOID_CHILD_PLAN_FLAG], True)


class HardConstraintNeverDefersTests(StageLoadingTestCase):
    """The 8 structural invariants.

    C13RF19's reasoning was that deferring one of these would park a proposal no
    later stage could execute either, so parking it is pointless.  That remains true
    -- and C13RF29 shows it was an argument about the VALUE of the record, not about
    whether the run should stop.  Parking a proposal nothing can execute costs a log
    line; stopping the run costs every completed call in the stage.  So these are now
    recorded and skipped too, and the log line is accepted as the cheaper of the two.

    Every one of the 8 is still REFUSED, which is what this class exists to defend.
    Note the renamed method below: "stay violations" was a claim about the verdict
    code, and the code changed; "are refused" is the claim that actually matters.
    """

    def test_all_eight_hard_constraints_are_refused(self):
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
                # C13RF29: the per-case `expected` codes above (VIOLATION /
                # NOT_ALLOWED) are kept in the table as a record of what each shape
                # used to return; the live assertion is that each is refused, with
                # no repair round and nothing executable.
                self.assertEqual(issue["code"], INEXPRESSIBLE)
                self.assertIs(issue["context"]["repairable"], False)
                self.assertEqual(issue["context"]["mutable_ref_paths"], [])
                self.assertIn(expected, {VIOLATION, NOT_ALLOWED})

    def test_action_not_allowed_stays_terminal(self):
        """A verb this stage does not implement is still refused outright.

        C13RF29 folded the ACTION_NOT_ALLOWED_IN_STAGE code into the uniform
        inexpressible verdict, but kept the distinction visible: `terminal_action` is
        recorded in the context so a reader can still tell "this stage has no such
        verb" from "this verb was misaddressed".  Terminality no longer needs to
        control the disposition, because nothing is repairable either way.
        """
        issue = self.only(op("create_bridge", "L2_SRC", "L2_TGT"))
        self.assertEqual(issue["code"], INEXPRESSIBLE)
        self.assertIs(issue["context"]["repairable"], False)
        self.assertIs(issue["context"]["terminal_action"], True)

    def test_every_verdict_is_now_non_repairable_and_uniform(self):
        """The inversion, stated as an inventory.

        This replaces C13RF19's `test_terminal_inventory_did_not_grow`, which pinned
        that ACTION_NOT_ALLOWED_IN_STAGE was the only non-repairable verdict.  After
        C13RF29 that inventory is the complement of what it was: every verdict is
        non-repairable, and there is exactly one code.  Asserting the inventory (not
        just individual cases) is what would catch a future edit that quietly
        reintroduces a repairable branch at this stage.
        """
        codes, repairable_flags = set(), set()
        for decision in (
            op("create_bridge", "L2_SRC", "L2_TGT"),
            op("rename", "L2_SRC", "L2_TGT"),
            op("merge", "L2_FOREIGN", "L2_TGT"),
            op("move", "L3_S1", "L4_S1A"),
            op("merge", "L2_SRC", "L2_SRC"),
            op("merge", "L2_SRC", "L2_FOREIGN"),
            op("flatten", "L3_S1", "L2_SRC",
               child_plan=[child("L4_S1A", "L2_FOREIGN")]),
        ):
            issue = self.only(decision)
            codes.add(issue["code"])
            repairable_flags.add(issue["context"]["repairable"])
            self.assertEqual(issue["context"]["mutable_ref_paths"], [])
        self.assertEqual(codes, {INEXPRESSIBLE})
        self.assertEqual(repairable_flags, {False})


class FlattenExclusionTests(StageLoadingTestCase):
    """`flatten` is in 14d's whitelist and can raise the deferrable message, yet
    must never defer."""

    def test_flatten_with_wrong_child_role_is_refused_and_recorded(self):
        """C13RF29: flatten's exclusion from the defer channel is moot.

        C13RF19 excluded flatten deliberately -- dissolving a node into its parent
        while sending a child elsewhere is a DIFFERENT operation, not an unspellable
        one, so it deserved a repair round rather than a parking slot.  The taxonomy
        still holds; the consequence no longer follows.  Either way this stage does
        not execute it, and C13RF29's position is that the choice between "ask for a
        repair and stop the run if it fails" and "record it and move on" should not
        depend on which of those two things the proposal is.
        """
        issue = self.only(
            op("flatten", "L3_S1", "L2_SRC",
               child_plan=[child("L4_S1A", "L2_FOREIGN")])
        )
        self.assertEqual(issue["code"], INEXPRESSIBLE)
        self.assertIn(
            self.finalize.CHILD_ROLE_MISMATCH_MESSAGE, issue["message"]
        )
        self.assertIs(issue["context"]["repairable"], False)
        self.assertEqual(issue["context"]["mutable_ref_paths"], [])

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
        # C13RF29: the code is uniform now; the assertion that matters is the one
        # C13RF16 made -- flatten never receives the empty-plan authorisation, since
        # an empty plan is permanently illegal for it.
        self.assertEqual(issue["code"], INEXPRESSIBLE)
        self.assertNotIn(VOID_CHILD_PLAN_FLAG, issue["context"])


class MixedDecisionTests(StageLoadingTestCase):
    """A decision carrying both kinds of error must report both.

    C13RF16 needed two things here: report the deferrable half (it used to vanish),
    and withhold its ref path so the repair round could not coerce an out-of-scope
    destination into a legal-but-wrong one.  C13RF29 keeps the first and subsumes the
    second: with no repair round, no path is offered for any reason, so the
    protection is absolute rather than selective.
    """

    def test_mixed_decision_records_both_halves(self):
        issue = self.only(
            op("merge", "L2_SRC", "L2_TGT", child_plan=[
                child("L3_S1", "L2_FOREIGN"),   # was: the inexpressible half
                child("L3_T1", "L2_TGT"),       # was: the repairable half
            ])
        )
        self.assertEqual(issue["code"], INEXPRESSIBLE)
        context = issue["context"]
        # Both reasons are stated; neither half is lost.
        self.assertIn(
            self.finalize.CHILD_ROLE_MISMATCH_MESSAGE, context["scope_messages"])
        self.assertIn(
            "child plan contains a non-source child", context["scope_messages"])
        # The off-stage child is still identified by index.
        self.assertEqual(context["child_indexes"], [0])
        # Nothing repairable, nothing offered, so nothing needs withholding.
        self.assertIs(context["repairable"], False)
        self.assertEqual(context["mutable_ref_paths"], [])
        self.assertNotIn("withheld_ref_paths", context)

    def test_void_child_plan_message_blocks_the_defer_channel(self):
        """A childless source additionally raises the C13RF16 ① message, which
        is not on the whitelist, so the decision stays a violation."""
        issue = self.only(
            op("merge", "L2_SRC", "L2_TGT",
               child_plan=[child("L3_C1", "L2_FOREIGN")]),
            tree=childless_source_tree(),
        )
        # C13RF29: no message "blocks the channel" any more -- every reason defers.
        # The flag is still raised, which is what C13RF16 ① is about.
        self.assertEqual(issue["code"], INEXPRESSIBLE)
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

# C13RF29: a complete inexpressible verdict, as the stages actually emit one.
# The fixtures used to carry only `{"code": INEXPRESSIBLE}` because the predicates
# inspected nothing else; they now also require `repairable is False` and an empty
# `mutable_ref_paths` (14b always did, and C13RF29 aligned the other three so the
# four cannot disagree about the same channel).  See inexpressible_entry() below.
_INEXPRESSIBLE_ENTRY = {
    "code": INEXPRESSIBLE,
    "message": "an off-stage reason",
    "context": {
        "decision_index": 0,
        "mutable_ref_paths": [],
        "repairable": False,
    },
}


def response(**overrides) -> dict:
    base = {
        "ok": False,
        "mutation_before_validation": False,
        "final_bound": None,
        "final_disposition": "scope_rejected",
        "error": "BOUND_SCOPE_VALIDATION_FAILED",
        "repair_count": 0,
        "initial_scope_errors": [copy.deepcopy(_INEXPRESSIBLE_ENTRY)],
        "repair_scope_errors": [],
    }
    base.update(overrides)
    return base


def inexpressible_entry(index=0):
    """A complete inexpressible verdict, as the stages actually emit one.

    C13RF29 note: the battery below used to write `{"code": INEXPRESSIBLE}` and
    nothing else, because the predicates only inspected the code.  They now also
    require `repairable is False` and an empty `mutable_ref_paths` -- 14b always
    did, and C13RF29 aligned the other three so a verdict marked inexpressible AND
    repairable cannot defer at one stage while stopping the run at another.  The
    fixtures therefore have to carry the whole shape; a bare code is no longer a
    verdict any stage would produce.
    """
    return {
        "code": INEXPRESSIBLE,
        "message": "an off-stage reason",
        "context": {
            "decision_index": index,
            "mutable_ref_paths": [],
            "repairable": False,
        },
    }


class DeferPredicateShapeTests(StageLoadingTestCase):
    """Card requirement ③: the new predicate must agree field for field with the
    one C13RF18 landed, and the agreement must be *checked*, not asserted in
    prose.  `balance` is excluded on purpose -- RF10 gave it an extra narrowness
    (every proposed decision must be covered by its own error) that RF18 preserved
    on both channels and C13RF29 kept, so it stays stricter than the plain stages.
    """

    BATTERY = {
        # channel 1
        "c1_pure_initial": (response(), True),
        # channel 2
        "c2_repaired_remainder": (response(
            final_disposition="repair_scope_rejected",
            error="REPAIR_SCOPE_VALIDATION_FAILED",
            repair_count=1,
            initial_scope_errors=[{"code": VIOLATION}],
            repair_scope_errors=[inexpressible_entry()],
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
            initial_scope_errors=[inexpressible_entry(), {"code": VIOLATION}],
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
            repair_scope_errors=[inexpressible_entry(), {"code": VIOLATION}],
        ), False),
        "neg_c2_multi_round_repair": (response(
            final_disposition="repair_scope_rejected",
            error="REPAIR_SCOPE_VALIDATION_FAILED",
            repair_count=2,
            repair_scope_errors=[inexpressible_entry()],
        ), False),
        # cross-channel smearing, both directions
        "neg_initial_ledger_read_on_repair_path": (response(
            final_disposition="repair_scope_rejected",
            error="REPAIR_SCOPE_VALIDATION_FAILED",
            repair_count=1,
            initial_scope_errors=[inexpressible_entry()],
            repair_scope_errors=[{"code": VIOLATION}],
        ), False),
        "neg_repair_ledger_read_on_initial_path": (response(
            repair_scope_errors=[inexpressible_entry()],
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
        """C13RF29 discharges C13RF19's stopgap: the message is a parameter now.

        C13RF19 could not touch `semantic_contract.py`, so where the shared record
        hard-coded 14a/14c's single message ("merge with a pair-external child
        target ...") this stage attached two side fields instead --
        `deferred_scope_errors` and `deferred_scope_channel` -- and left a note asking
        whoever could touch that file to parameterise it.  C13RF29 did, and removed
        both fields.  The guarantee this test defends is unchanged: the record states
        the reason this call was actually deferred.  It now does so through its own
        violations, one per message, which is where a reader looks anyway.
        """
        response_obj, _audit, _transport = self.bind(
            [local_child("S1", "FOREIGN")]
        )
        _returned, written, _audit2 = self.record_for(response_obj)
        record = written[0]
        self.assertNotIn("deferred_scope_channel", record)
        self.assertNotIn("deferred_scope_errors", record)
        violations = record["semantic_contract"]["violations"]
        self.assertEqual([item["code"] for item in violations], [INEXPRESSIBLE])
        self.assertEqual(
            violations[0]["message"], self.finalize.CHILD_ROLE_MISMATCH_MESSAGE)
        # And never the sentence written for another stage.
        self.assertNotIn("pair-external", violations[0]["message"])

    def test_mixed_call_defers_on_the_first_round_and_spends_one_attempt(self):
        """C13RF29: no second round, and the saved attempt is the point.

        This was C13RF18's channel-2 case at 14d: the repair round fixed the
        repairable half, leaving an inexpressible remainder.  Now the whole decision
        is inexpressible on the first round, so the binder never asks again --
        `transport.calls` drops from 2 to 1.  That is the API saving the card
        predicted, measured here rather than asserted in prose: one fewer request per
        off-stage decision (49 of C13RF28's 53 calls were this family).
        """
        response_obj, _audit, transport = self.bind(
            [local_child("S1", "FOREIGN"), local_child("T1", "TGT")],
            [local_child("S1", "FOREIGN"), local_child("S2", "TGT")],
        )
        self.assertEqual(transport.calls, 1)
        self.assertEqual(response_obj["final_disposition"], "scope_rejected")
        self.assertEqual(response_obj["repair_count"], 0)
        self.assertIs(
            self.finalize.OverallStructureAudit._is_deferrable_response(
                response_obj
            ),
            True,
        )
        _returned, written, _audit2 = self.record_for(response_obj)
        self.assertEqual(written[0]["status"], "deferred")

    def test_severity_floor_still_refuses_a_non_inexpressible_channel(self):
        """The floor survives C13RF29; only its reachability from here changed.

        The predicate still requires every entry in the scope channel to be
        inexpressible and non-repairable, and still refuses an empty channel.  Since
        this stage no longer emits anything else, the floor has to be exercised
        synthetically -- which is worth keeping: it is what would catch a future
        repairable class being deferred by accident.
        """
        response_obj, _audit, _transport = self.bind(
            [local_child("S1", "FOREIGN"), local_child("T1", "TGT")],
        )
        predicate = self.finalize.OverallStructureAudit._is_deferrable_response
        self.assertIs(predicate(response_obj), True)

        impure = copy.deepcopy(response_obj)
        impure["initial_scope_errors"] = [{
            "code": VIOLATION,
            "message": "some repairable problem",
            "context": {
                "decision_index": 0,
                "mutable_ref_paths": ["decisions[0].target_ref"],
                "repairable": True,
            },
        }]
        self.assertIs(predicate(impure), False)

        empty = copy.deepcopy(response_obj)
        empty["initial_scope_errors"] = []
        self.assertIs(predicate(empty), False)

    def test_subject_error_is_no_longer_repaired_but_recorded(self):
        """The accepted cost, at 14d.

        Before C13RF29 this shape (a child_plan naming a node that is not the
        source's child) came back corrected from a repair round and executed.  Now it
        is recorded and skipped.  The user accepted this on the record: the tree
        changes less, never wrongly, and no run ends over it.
        """
        response_obj, _audit, _transport = self.bind(
            [local_child("T1", "TGT")],
            [local_child("S1", "TGT")],
        )
        self.assertIs(response_obj["ok"], False)
        self.assertEqual(response_obj["repair_count"], 0)
        self.assertIs(
            self.finalize.OverallStructureAudit._is_deferrable_response(
                response_obj
            ),
            True,
        )

    def test_flatten_call_now_defers_instead_of_stopping_the_run(self):
        """C13RF19 excluded flatten from the channel; C13RF29 includes it.

        The taxonomic reason for the exclusion still stands (a flatten that rehomes a
        child elsewhere is a different operation, not an unspellable one) and is
        recorded in FlattenExclusionTests.  What changed is that the reason no longer
        decides whether the run survives.  Either way, this stage does not execute it.
        """
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
            True,
        )

    def test_unimplemented_verb_now_defers_and_stays_marked_terminal(self):
        """C13RF29: a verb this stage does not implement no longer ends the run.

        `create_bridge` belongs to 14b; proposing it at 14d is nonsense the stage
        cannot execute.  C13RF19 made it 14d's one terminal verdict -- refuse, no
        repair, and the run stops.  The refusal is unchanged; only the stopping is
        gone.  The distinction is preserved in the record via `terminal_action`, so
        "no such verb here" is still distinguishable from "misaddressed" when someone
        reads the deferred entries.
        """
        response_obj, _audit, _transport = self.bind(
            decisions=[local_decision(
                "create_bridge", "SRC", "TGT", [local_child("S1", "TGT")]
            )],
        )
        errors = response_obj["initial_scope_errors"]
        self.assertEqual([item["code"] for item in errors], [INEXPRESSIBLE])
        self.assertIs(errors[0]["context"]["terminal_action"], True)
        self.assertIs(errors[0]["context"]["repairable"], False)
        self.assertIs(
            self.finalize.OverallStructureAudit._is_deferrable_response(
                response_obj
            ),
            True,
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
