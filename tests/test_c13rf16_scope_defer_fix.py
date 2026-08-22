"""C13RF16: the zero-child child_plan deadlock and the lost defer component.

Two defects, both in the scope/defer family, both reproduced here against the
real stage code rather than a paraphrase of it.

Fix ① -- C13RF13 call 25 died because a decision whose source node has no
children at all has no legal ``child_ref`` to name, while the only correct plan
(the empty one) was read as ``REPAIR_SEMANTIC_DRIFT``.  Both doors were shut, so
the run stopped.  The stages now flag that topology and the repair round may
truncate exactly that decision's ``child_plan``.

Fix ② -- the deferrable classification required the whole deduplicated message
list to equal one sentence, so a decision carrying a genuinely inexpressible
component *plus* an ordinary violation lost the deferrable half silently, and
worse, went on to the repair round where the model could coerce the
pair-external destination into a legal-but-wrong one.  Classification is now per
message, the deferrable component is recorded, and its refs are withheld.

Strictness must not move: a source that *does* have children still gets its
subject error rejected, an unauthorised truncation still trips the drift guard,
and a factual misread still gets no defer path.
"""
from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from tests.test_semantic_stage_dispatch import (
    child_plan,
    decision,
    evidence,
    load_stage_module,
    node,
)
from utils.local_reference_binding import (
    VOID_CHILD_PLAN_FLAG,
    build_local_reference_context,
    call_local_reference_json,
)
from utils.semantic_contract import validate_semantic_decision
from utils.tree_manager import TreeManager


class ScriptedTransport:
    """Returns a queued payload per call and fails loudly when exhausted.

    HTTP is structurally impossible here: nothing in this file can reach a
    network transport, and an unexpected extra call raises rather than silently
    returning a stub.
    """

    def __init__(self, *payloads):
        self.payloads = list(payloads)
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(copy.deepcopy(kwargs))
        if not self.payloads:
            raise AssertionError("scripted transport exhausted")
        payload = self.payloads.pop(0)
        raw = json.dumps(payload, ensure_ascii=False)
        return {
            "ok": True,
            "json": copy.deepcopy(payload),
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


def pair_tree():
    """L2_NSRC001 is childless; its sibling L2_NTGT001 has one child.

    The childless source is fix ①'s topology.  ``L2_NOTHER1`` sits under a
    different L1, so naming it as a child destination is the pair-external
    (deferrable) message, while ``L3_NTC0001`` belongs to the target rather than
    the source, giving the subject-error message.  One tree therefore supports
    the pure, the mixed and the has-children cases.
    """
    return node("ROOT", "ROOT", "ROOT", [
        node("L1_NDOMAIN1", "domain one", "L1", [
            node("L2_NSRC001", "childless source", "L2"),
            node("L2_NTGT001", "target", "L2", [
                node("L3_NTC0001", "target child", "L3"),
            ]),
            node("L2_NFULL001", "source with children", "L2", [
                node("L3_NSC0001", "source child", "L3"),
            ]),
        ]),
        node("L1_NDOMAIN2", "domain two", "L1", [
            node("L2_NOTHER1", "pair external", "L2"),
        ]),
    ])


def stage_process(module, class_name, tree, temporary_dir):
    process = getattr(module, class_name).__new__(getattr(module, class_name))
    process.tm = TreeManager(copy.deepcopy(tree))
    process.membership_counts = {}
    process.ops_log = Path(temporary_dir) / "ops.jsonl"
    process.llm_log = Path(temporary_dir) / "llm.jsonl"
    process.trace_map = {}
    process.prior_lineage = {}
    process.deferred_candidates = set()
    process.redirect_map = {}
    return process


class VoidChildPlanScopeTests(unittest.TestCase):
    """Fix ①, at the scope-validator layer, in all four stages."""

    @classmethod
    def setUpClass(cls):
        cls.collapse = load_stage_module("collapse_redundant_hierarchy")
        cls.polish = load_stage_module("polish_tree_labels")
        cls.balance = load_stage_module("balance_tree_structure")
        cls.finalize = load_stage_module("finalize_policy_tree")

    def collapse_issues(self, decisions, parent="L2_NSRC001", child="L2_NTGT001"):
        with tempfile.TemporaryDirectory() as temporary_dir:
            process = stage_process(
                self.collapse, "SkeletonRefiner", pair_tree(), temporary_dir
            )
            return process._scope_issues({"decisions": decisions}, parent, child)

    def polish_issues(self, decisions, node_a="L2_NSRC001", node_b="L2_NTGT001"):
        with tempfile.TemporaryDirectory() as temporary_dir:
            process = stage_process(
                self.polish, "PolishingProcess", pair_tree(), temporary_dir
            )
            return process._scope_issues({"decisions": decisions}, node_a, node_b)

    def finalize_issues(self, decisions, l1_id="L1_NDOMAIN1"):
        with tempfile.TemporaryDirectory() as temporary_dir:
            process = stage_process(
                self.finalize, "OverallStructureAudit", pair_tree(), temporary_dir
            )
            return process._scope_issues({"decisions": decisions}, l1_id)

    def balance_issues(self, decisions, pid="L1_NDOMAIN1"):
        with tempfile.TemporaryDirectory() as temporary_dir:
            process = stage_process(
                self.balance, "ShapingProcess", pair_tree(), temporary_dir
            )
            return process._scope_issues(pid, {"decisions": decisions})

    # --- the flagged topology, stage by stage ---------------------------------

    def assert_void_flagged(self, issues, *, stage):
        """C13RF16 fix ① after C13RF29: the flag is still raised, but nothing
        repairs on it any more.

        The flag says "this decision's only legal child_plan is the empty one".
        C13RF16 raised it so the binder could authorise the model to truncate the
        plan in a repair round.  C13RF29 makes every scope failure non-repairable,
        so the binder returns before it ever reads the flag
        (`_void_child_plan_indexes` is called after the `_repairable` early return
        in `call_local_reference_json`).  The flag is kept because it describes the
        proposal and lands in the record, but the truncation-authorisation path it
        fed is now unreachable from these four stages -- recorded as a consequence
        of this card, not a silent loss.
        """
        self.assertEqual(len(issues), 1, f"{stage}: {issues}")
        details = issues[0]["context"]
        self.assertIs(details.get(VOID_CHILD_PLAN_FLAG), True, f"{stage}: {details}")
        self.assertIs(details["repairable"], False, f"{stage}: {details}")
        self.assertEqual(details["mutable_ref_paths"], [], f"{stage}: {details}")
        self.assertIn(
            "source has no children", issues[0]["message"], f"{stage}: {issues}"
        )
        return issues[0]

    def test_collapse_flags_childless_merge_source(self):
        issue = self.assert_void_flagged(
            self.collapse_issues([decision(
                "merge", "exact_duplicate", "L2_NSRC001", "L2_NTGT001",
                children=[child_plan("L3_NTC0001", "L2_NTGT001")],
            )]),
            stage="14a",
        )
        self.assertEqual(issue["code"], "DECISION_SCOPE_INEXPRESSIBLE")

    def test_polish_flags_childless_merge_source(self):
        issue = self.assert_void_flagged(
            self.polish_issues([decision(
                "merge", "exact_duplicate", "L2_NSRC001", "L2_NTGT001",
                children=[child_plan("L3_NTC0001", "L2_NTGT001")],
            )]),
            stage="14c",
        )
        self.assertEqual(issue["code"], "DECISION_SCOPE_INEXPRESSIBLE")

    def test_balance_flags_childless_move_source(self):
        issue = self.assert_void_flagged(
            self.balance_issues([decision(
                "move", "misplaced", "L2_NSRC001", "ROOT",
                children=[child_plan("L3_NTC0001", "L2_NSRC001")],
            )]),
            stage="14b",
        )
        self.assertEqual(issue["code"], "DECISION_SCOPE_INEXPRESSIBLE")

    def test_finalize_flags_childless_merge_source(self):
        issue = self.assert_void_flagged(
            self.finalize_issues([decision(
                "merge", "exact_duplicate", "L2_NSRC001", "L2_NTGT001",
                children=[child_plan("L3_NTC0001", "L2_NTGT001")],
            )]),
            stage="14d",
        )
        self.assertEqual(issue["code"], "DECISION_SCOPE_INEXPRESSIBLE")

    # --- strictness: a source that has children keeps its subject error ------

    def assert_subject_error_unchanged(self, issues, *, stage):
        """A source that HAS children keeps its subject error and gets no void flag.

        That distinction is C13RF16 fix ①'s whole point and C13RF29 does not touch
        it: the flag means "the empty plan is the only legal one", which is false
        when the source owns children.  What C13RF29 changes is only the
        disposition -- the subject error is no longer repairable, so it is recorded
        and skipped instead of driving a repair round that could stop the run.
        """
        self.assertEqual(len(issues), 1, f"{stage}: {issues}")
        details = issues[0]["context"]
        self.assertEqual(issues[0]["code"], "DECISION_SCOPE_INEXPRESSIBLE")
        self.assertNotIn(VOID_CHILD_PLAN_FLAG, details, f"{stage}: {details}")
        self.assertIs(details["repairable"], False, f"{stage}: {details}")
        self.assertEqual(details["mutable_ref_paths"], [], f"{stage}: {details}")
        self.assertIn("non-source child", issues[0]["message"], f"{stage}: {issues}")

    def test_collapse_keeps_rejecting_subject_error_with_children(self):
        self.assert_subject_error_unchanged(
            self.collapse_issues(
                [decision(
                    "merge", "exact_duplicate", "L2_NFULL001", "L2_NTGT001",
                    children=[child_plan("L3_NTC0001", "L2_NTGT001")],
                )],
                parent="L2_NFULL001",
            ),
            stage="14a",
        )

    def test_polish_keeps_rejecting_subject_error_with_children(self):
        self.assert_subject_error_unchanged(
            self.polish_issues(
                [decision(
                    "merge", "exact_duplicate", "L2_NFULL001", "L2_NTGT001",
                    children=[child_plan("L3_NTC0001", "L2_NTGT001")],
                )],
                node_a="L2_NFULL001",
            ),
            stage="14c",
        )

    def test_finalize_keeps_rejecting_subject_error_with_children(self):
        self.assert_subject_error_unchanged(
            self.finalize_issues([decision(
                "merge", "exact_duplicate", "L2_NFULL001", "L2_NTGT001",
                children=[child_plan("L3_NTC0001", "L2_NTGT001")],
            )]),
            stage="14d",
        )

    def test_balance_keeps_rejecting_subject_error_with_children(self):
        issues = self.balance_issues([decision(
            "move", "misplaced", "L2_NFULL001", "ROOT",
            children=[child_plan("L3_NTC0001", "L2_NFULL001")],
        )])
        self.assertEqual(len(issues), 1, issues)
        self.assertEqual(issues[0]["code"], "DECISION_SCOPE_INEXPRESSIBLE")
        self.assertNotIn(VOID_CHILD_PLAN_FLAG, issues[0]["context"])
        self.assertIn("non-source child", issues[0]["message"])

    # --- the flag is withheld where an empty plan is never legal -------------

    def test_finalize_does_not_flag_childless_flatten(self):
        """F2 control: flattening a leaf is void as a whole, not repairable.

        ``FLATTEN_SOURCE_HAS_NO_CHILDREN`` (and ``SPLIT_HAS_NO_MOVES``) make an
        empty plan illegal for these verbs, so authorising truncation would only
        move the rejection, never resolve it.  The flag must stay off.
        """
        issues = self.finalize_issues([decision(
            "flatten", "exact_duplicate", "L2_NSRC001", "L1_NDOMAIN1",
            children=[child_plan("L3_NTC0001", "L1_NDOMAIN1")],
        )])
        self.assertEqual(len(issues), 1, issues)
        self.assertNotIn(VOID_CHILD_PLAN_FLAG, issues[0]["context"])

    def test_balance_does_not_flag_childless_split_reparent(self):
        issues = self.balance_issues(
            [decision(
                "split_reparent", "broader_narrower", "L2_NSRC001", "L2_NSRC001",
                children=[child_plan("L3_NTC0001", "L2_NSRC001")],
            )],
            pid="L2_NSRC001",
        )
        self.assertEqual(len(issues), 1, issues)
        self.assertNotIn(VOID_CHILD_PLAN_FLAG, issues[0]["context"])


class InexpressibleComponentTests(unittest.TestCase):
    """Fix ②, at the scope-validator layer."""

    @classmethod
    def setUpClass(cls):
        cls.collapse = load_stage_module("collapse_redundant_hierarchy")
        cls.polish = load_stage_module("polish_tree_labels")

    def nested_tree(self):
        """The parent/child pair shape 14a's vertical collapse inspects."""
        return node("ROOT", "ROOT", "ROOT", [
            node("L1_NDOMAIN1", "domain one", "L1", [
                node("L2_NSRC001", "source", "L2", [
                    node("L3_NTGT001", "target", "L3", [
                        node("L4_NTC0001", "target child", "L4"),
                    ]),
                    node("L3_NSC0001", "source child", "L3"),
                ]),
            ]),
            node("L1_NDOMAIN2", "domain two", "L1", [
                node("L2_NOTHER1", "pair external", "L2"),
            ]),
        ])

    def collapse_issues(self, decisions):
        with tempfile.TemporaryDirectory() as temporary_dir:
            process = stage_process(
                self.collapse, "SkeletonRefiner", self.nested_tree(), temporary_dir
            )
            return process._scope_issues(
                {"decisions": decisions}, "L2_NSRC001", "L3_NTGT001"
            )

    def polish_issues(self, decisions):
        with tempfile.TemporaryDirectory() as temporary_dir:
            process = stage_process(
                self.polish, "PolishingProcess", pair_tree(), temporary_dir
            )
            return process._scope_issues(
                {"decisions": decisions}, "L2_NFULL001", "L2_NTGT001"
            )

    def test_pure_pair_external_child_still_defers_in_collapse(self):
        issues = self.collapse_issues([decision(
            "merge", "exact_duplicate", "L2_NSRC001", "L3_NTGT001",
            children=[child_plan("L3_NSC0001", "L2_NOTHER1")],
        )])
        self.assertEqual(len(issues), 1, issues)
        self.assertEqual(issues[0]["code"], "DECISION_SCOPE_INEXPRESSIBLE")
        self.assertEqual(issues[0]["context"]["mutable_ref_paths"], [])
        self.assertIs(issues[0]["context"]["repairable"], False)
        self.assertNotIn("inexpressible_components", issues[0]["context"])

    def test_pure_pair_external_child_still_defers_in_polish(self):
        issues = self.polish_issues([decision(
            "merge", "exact_duplicate", "L2_NFULL001", "L2_NTGT001",
            children=[child_plan("L3_NSC0001", "L2_NOTHER1")],
        )])
        self.assertEqual(len(issues), 1, issues)
        self.assertEqual(issues[0]["code"], "DECISION_SCOPE_INEXPRESSIBLE")

    def test_mixed_decision_reports_both_halves_and_withholds_nothing(self):
        """C13RF16 fix ② after C13RF29: reporting kept, withholding obsolete.

        Fix ② solved two problems with a mixed decision (one half deferrable, one
        half repairable): the deferrable half used to vanish from the record, and
        its ref path had to be withheld from the repair round so the model could not
        re-aim an off-stage destination into a legal-but-wrong one.

        The reporting half is preserved -- both reasons are in the record, and the
        off-stage child is still named by index.  The withholding half is obsolete:
        with no repair round there is nothing to withhold a path FROM, and the
        protection it provided is now absolute rather than selective (no ref path is
        offered for any reason).  `inexpressible_components` and `withheld_ref_paths`
        are gone with the mechanism; `child_indexes` carries the surviving
        information.
        """
        issues = self.polish_issues([decision(
            "merge", "exact_duplicate", "L2_NFULL001", "L2_NTGT001",
            children=[
                child_plan("L3_NSC0001", "L2_NOTHER1"),
                child_plan("L3_NTC0001", "L2_NTGT001"),
            ],
        )])
        self.assertEqual(len(issues), 1, issues)
        details = issues[0]["context"]
        self.assertEqual(issues[0]["code"], "DECISION_SCOPE_INEXPRESSIBLE")
        # Both reasons survive in the record -- neither half is silently lost.
        self.assertIn(
            "merge child target is outside the exact pair role",
            details["scope_messages"],
        )
        self.assertIn("child plan contains a non-source child",
                      details["scope_messages"])
        # The off-stage child is still identified.
        self.assertEqual(details["child_indexes"], [0])
        # Nothing is repairable, so no ref path is offered at all -- the selective
        # withholding fix ② needed is subsumed.
        self.assertIs(details["repairable"], False)
        self.assertEqual(details["mutable_ref_paths"], [])
        self.assertNotIn("withheld_ref_paths", details)
        self.assertNotIn("inexpressible_components", details)

    def test_pure_subject_error_gets_no_defer_path(self):
        """The §4 red line: a factual misread must never become deferrable."""
        issues = self.polish_issues([decision(
            "merge", "exact_duplicate", "L2_NFULL001", "L2_NTGT001",
            children=[child_plan("L3_NTC0001", "L2_NTGT001")],
        )])
        self.assertEqual(len(issues), 1, issues)
        self.assertEqual(issues[0]["code"], "DECISION_SCOPE_INEXPRESSIBLE")
        self.assertNotIn("inexpressible_components", issues[0]["context"])


class BalanceDeferMessageTests(unittest.TestCase):
    """Fix ② must not disturb the depth-ceiling defer RF10/RF11 rely on."""

    @classmethod
    def setUpClass(cls):
        cls.balance = load_stage_module("balance_tree_structure")

    def deep_tree(self):
        children = [node(f"L4_NCHILD{i:02d}", f"child {i}", "L4") for i in range(8)]
        return node("ROOT", "ROOT", "ROOT", [
            node("L1_NDOMAIN1", "domain", "L1", [
                node("L2_NPARENT1", "parent", "L2", [
                    node("L3_NCAND001", "candidate", "L3", children),
                ]),
            ]),
        ])

    def bridge_decision(self, **changes):
        placeholder = self.balance.BRIDGE_TARGET_PLACEHOLDER
        return decision(
            "create_bridge", "broader_narrower", "L3_NCAND001", placeholder,
            children=[child_plan("L4_NCHILD00", placeholder)],
            new_label="group 1",
            **changes,
        )

    def issues(self, decisions):
        with tempfile.TemporaryDirectory() as temporary_dir:
            process = stage_process(
                self.balance, "ShapingProcess", self.deep_tree(), temporary_dir
            )
            return process._scope_issues("L3_NCAND001", {"decisions": decisions})

    def test_depth_ceiling_reports_the_predicate_message(self):
        """C13RF29: the issue carries the predicate's own reason.

        14b used to substitute BRIDGE_DEPTH_DEFERRED_MESSAGE ("create_bridge is not
        expressible at the structure-balancing depth ceiling") for the predicate's
        BRIDGE_TOO_DEEP_MESSAGE ("bridge parent is too deep").  That substitution
        only made sense while the depth ceiling was the single deferrable reason at
        this stage; now that any off-stage decision defers, one stage-specific
        sentence would misreport most of them.  The constant is kept and still names
        the deferral in the stage's own vocabulary.
        """
        issues = self.issues([self.bridge_decision()])
        self.assertEqual(len(issues), 1, issues)
        self.assertEqual(issues[0]["code"], "DECISION_SCOPE_INEXPRESSIBLE")
        self.assertEqual(
            issues[0]["message"], self.balance.BRIDGE_TOO_DEEP_MESSAGE
        )
        self.assertNotIn("inexpressible_components", issues[0]["context"])

    def test_mixed_bridge_records_every_reason(self):
        """C13RF16 fix ② after C13RF29: same guarantee, simpler carrier.

        Fix ② was that the deferrable component of a mixed decision must not vanish
        from the record.  It still does not: both reasons appear, now in
        ``scope_messages`` rather than in a separate ``inexpressible_components``
        field that existed to distinguish the deferrable half from the repairable
        one.  With nothing repairable, that distinction has no consumer.
        """
        broken = self.bridge_decision()
        broken["child_plan"][0]["child_id"] = "L2_NPARENT1"
        issues = self.issues([broken])
        self.assertEqual(len(issues), 1, issues)
        self.assertEqual(issues[0]["code"], "DECISION_SCOPE_INEXPRESSIBLE")
        messages = issues[0]["context"]["scope_messages"]
        self.assertIn("bridge parent is too deep", messages)
        self.assertIn("bridge plan contains a non-direct child", messages)
        self.assertIs(issues[0]["context"]["repairable"], False)
        self.assertEqual(issues[0]["context"]["mutable_ref_paths"], [])


class RepairTruncationTests(unittest.TestCase):
    """Fix ① end to end through the repair round, with its strictness intact.

    C13RF29 note on how these tests are wired.  They exercise machinery that lives
    in the BINDER (`local_reference_binding`): when a scope error authorises it via
    ``context[VOID_CHILD_PLAN_FLAG]``, the repair round may empty a child_plan, and
    a battery of drift checks constrains what else that repair may change.  None of
    that machinery is touched by C13RF29 and all of it still works -- but after this
    card no stage validator returns a repairable error, so it can no longer be
    reached by driving a real stage.  Rather than delete coverage of a live
    mechanism, these tests now supply the authorising verdict directly through a
    synthetic validator (`authorising_validator` below), which is exactly the
    contract the binder documents for `bound_validator`.

    Two things this preserves: the drift checks stay pinned for whatever repairable
    class is introduced next, and the fact that these paths are currently
    unreachable from 14a/14b/14c/14d is stated here rather than discovered later by
    someone wondering why the code looks dead.
    """

    @classmethod
    def setUpClass(cls):
        cls.polish = load_stage_module("polish_tree_labels")

    @staticmethod
    def authorising_validator(payload, *, void_indexes=(0,)):
        """The pre-C13RF29 void-child-plan verdict, supplied directly.

        Shape copied from what the stage validators emitted before this card: a
        repairable scope error whose mutable path is the plan's target ref, plus the
        VOID_CHILD_PLAN_FLAG authorisation that lets the repair empty the plan.
        """
        issues = []
        decisions = payload.get("decisions") or []
        for index, item in enumerate(decisions):
            if index not in void_indexes:
                continue
            plan = item.get("child_plan") or []
            if not plan:
                continue
            issues.append({
                "code": "DECISION_SCOPE_VIOLATION",
                "message": "source has no children; child_plan must be empty",
                "context": {
                    "decision_index": index,
                    "mutable_ref_paths": [
                        f"decisions[{index}].child_plan[{child_index}]"
                        ".target_parent_ref"
                        for child_index in range(len(plan))
                    ],
                    "repairable": True,
                    VOID_CHILD_PLAN_FLAG: True,
                },
            })
        return issues

    def setUp(self):
        self.system = "c13rf16 label-polishing fixture contract"
        self.refs = [
            ("SRC", "L2_NSRC001"),
            ("TGT", "L2_NTGT001"),
            ("TCHILD", "L3_NTC0001"),
            ("OTHER", "L2_NOTHER1"),
        ]
        self.tm = TreeManager(copy.deepcopy(pair_tree()))
        self.context = build_local_reference_context(
            task="polish_tree_labels",
            user_text="\n".join(f"{ref}={node_id}" for ref, node_id in self.refs),
            candidate_node_ids=self.tm.get_all_node_ids(),
            preferred_refs=self.refs,
            contract_text=self.system,
            expected_count=None,
        )

    def local_decision(self, children):
        value = decision("merge", "exact_duplicate", "L2_NSRC001", "L2_NTGT001")
        value.pop("source_id")
        value.pop("target_id")
        value["source_ref"] = "SRC"
        value["target_ref"] = "TGT"
        value["child_plan"] = children
        return value

    def local_child(self, child_ref, target_ref):
        item = child_plan("placeholder", "placeholder")
        item.pop("child_id")
        item.pop("target_parent_id")
        item["child_ref"] = child_ref
        item["target_parent_ref"] = target_ref
        return item

    def payload(self, children):
        return {
            "context_token": self.context.context_token,
            "decisions": [self.local_decision(children)],
        }

    def run_call(self, *payloads, validator=None):
        transport = ScriptedTransport(*payloads)
        with tempfile.TemporaryDirectory() as temporary_dir:
            process = stage_process(
                self.polish, "PolishingProcess", pair_tree(), temporary_dir
            )
            response = call_local_reference_json(
                transport=transport,
                profile="fake",
                system=self.system,
                context=self.context,
                task="polish_tree_labels",
                expected_count=None,
                protected_manager=process.tm,
                # C13RF29: the authorising verdict is supplied directly -- see the
                # class docstring.  No stage validator produces a repairable error
                # any more, so driving one would never reach the repair round.
                bound_validator=validator or self.authorising_validator,
            )
        return transport, response

    def test_authorised_truncation_is_accepted(self):
        initial = self.payload([self.local_child("TCHILD", "TGT")])
        repaired = self.payload([])
        transport, response = self.run_call(initial, repaired)

        self.assertEqual(len(transport.calls), 2, transport.calls)
        self.assertIs(response["ok"], True, response.get("error"))
        self.assertEqual(response["final_disposition"], "bound_after_repair")
        self.assertEqual(response["repair_count"], 1)
        self.assertIs(response["mutation_before_validation"], False)
        self.assertEqual(response["json"]["decisions"][0]["child_plan"], [])
        self.assertEqual(
            response["initial_scope_errors"][0]["context"][VOID_CHILD_PLAN_FLAG],
            True,
        )

    def test_repair_prompt_names_the_authorised_index_without_node_ids(self):
        initial = self.payload([self.local_child("TCHILD", "TGT")])
        transport, _ = self.run_call(initial, self.payload([]))
        repair_user = transport.calls[1]["user"]
        self.assertIn("child_plan_must_be_empty=[0]", repair_user)
        for _ref, node_id in self.refs:
            self.assertNotIn(node_id, repair_user)

    def test_truncation_without_authorisation_still_drifts(self):
        """A source that has children may not have its plan emptied."""
        source_children = [self.local_child("SCHILD", "TGT")]
        refs = [*self.refs, ("SCHILD", "L3_NSC0001"), ("FULL", "L2_NFULL001")]
        context = build_local_reference_context(
            task="polish_tree_labels",
            user_text="\n".join(f"{ref}={node_id}" for ref, node_id in refs),
            candidate_node_ids=self.tm.get_all_node_ids(),
            preferred_refs=refs,
            contract_text=self.system,
            expected_count=None,
        )
        local = self.local_decision(source_children)
        local["source_ref"] = "FULL"
        local["child_plan"][0]["child_ref"] = "TCHILD"
        initial = {"context_token": context.context_token, "decisions": [local]}
        repaired = copy.deepcopy(initial)
        repaired["decisions"][0]["child_plan"] = []

        def unauthorising_validator(payload):
            """Repairable, but WITHOUT the void-plan authorisation.

            This is the control: the source has children, so emptying the plan is
            not a correction the binder may accept.  C13RF29 changed which verdicts
            stage validators emit, not this rule, so the verdict is supplied directly
            here for the same reason as the rest of this class.
            """
            return [{
                "code": "DECISION_SCOPE_VIOLATION",
                "message": "child plan contains a non-source child",
                "context": {
                    "decision_index": 0,
                    "mutable_ref_paths": [
                        "decisions[0].child_plan[0].child_ref",
                    ],
                    "repairable": True,
                },
            }]

        transport = ScriptedTransport(initial, repaired)
        with tempfile.TemporaryDirectory() as temporary_dir:
            process = stage_process(
                self.polish, "PolishingProcess", pair_tree(), temporary_dir
            )
            response = call_local_reference_json(
                transport=transport,
                profile="fake",
                system=self.system,
                context=context,
                task="polish_tree_labels",
                expected_count=None,
                protected_manager=process.tm,
                bound_validator=unauthorising_validator,
            )
        self.assertIs(response["ok"], False)
        self.assertEqual(response["error"], "REPAIR_SEMANTIC_DRIFT")

    def test_authorised_truncation_may_not_change_frozen_fields(self):
        initial = self.payload([self.local_child("TCHILD", "TGT")])
        repaired = self.payload([])
        repaired["decisions"][0]["relation"] = "synonym"
        _transport, response = self.run_call(initial, repaired)
        self.assertIs(response["ok"], False)
        self.assertEqual(response["error"], "REPAIR_SEMANTIC_DRIFT")

    def test_authorised_decision_may_not_keep_a_non_empty_plan(self):
        initial = self.payload([self.local_child("TCHILD", "TGT")])
        repaired = self.payload([self.local_child("TCHILD", "OTHER")])
        _transport, response = self.run_call(initial, repaired)
        self.assertIs(response["ok"], False)
        self.assertIn(
            response["error"],
            {"REPAIR_SCOPE_VALIDATION_FAILED", "REPAIR_VALID_REF_DRIFT"},
        )


class FlattenControlTests(unittest.TestCase):
    """F2 control: fix ① must not loosen the childless-flatten rejection."""

    def context_for(self, **overrides):
        value = {
            "source_id": "L2_NSRC001",
            "target_id": "L1_NDOMAIN1",
            "source_exists": True,
            "target_exists": True,
            "source_child_ids": [],
            "source_parent_id": "L1_NDOMAIN1",
            "source_l1_id": "L1_NDOMAIN1",
            "target_l1_id": "L1_NDOMAIN1",
            "allowed_l1_id": "L1_NDOMAIN1",
            "membership_known": True,
            "source_direct_membership_count": 0,
            "plan_targets": {},
        }
        value.update(overrides)
        return value

    @staticmethod
    def flatten_decision(children):
        return decision(
            "flatten", "exact_duplicate", "L2_NSRC001", "L1_NDOMAIN1",
            children=children,
            evidence=evidence(pure_structural_redundancy=True),
        )

    def test_childless_flatten_is_rejected_even_with_an_empty_plan(self):
        report = validate_semantic_decision(
            self.flatten_decision([]), context=self.context_for(),
        )
        self.assertEqual(
            [item["code"] for item in report["violations"]],
            ["FLATTEN_SOURCE_HAS_NO_CHILDREN"],
        )
        self.assertIs(report["passed"], False)

    def test_flatten_with_children_and_a_complete_plan_passes(self):
        """Guards against a fixture that can never pass."""
        report = validate_semantic_decision(
            self.flatten_decision([child_plan("L3_NSC0001", "L1_NDOMAIN1")]),
            context=self.context_for(source_child_ids=["L3_NSC0001"]),
        )
        self.assertEqual(report["violations"], [])
        self.assertIs(report["passed"], True)


if __name__ == "__main__":
    unittest.main()
