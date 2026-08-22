"""C13RF29: invert the scope criterion in all four stages, both channels.

The rule before this card was a blacklist: enumerate the shapes a stage may not
execute, mark them repairable, ask the model to fix them, and -- when the repair
fails the same predicate -- let the private harness guard stop the whole run.
C13RF28 died there: call 54 of 53 successful calls proposed merging the displayed
CHILD with its SIBLING, the repair came back semantically identical, and 53
completed calls were discarded with the run.  Because the list of ways a decision
can fall outside a stage's stage is open-ended, each newly encountered shape cost
another full-tree rerun.

After this card each stage still executes exactly the positive shape it always
required, and every failed check is recorded and skipped instead.  For 14a and 14c
that positive shape is literally "a merge on the two displayed nodes"; 14b and 14d
have no `merge` verb on a displayed pair at all (14b balances one candidate with
create_bridge/flatten/split_reparent/move, 14d audits an L1 batch), so there the
same rule reads: every verb keeps the scope it already required, and only the
disposition of a failure changes.

What this file pins, in order of importance:

1. **Nothing new executes.**  A merge naming a node outside the displayed stage is
   still refused, and the tree does not move.  This is the load-bearing negative:
   executing one would have a stage absorb a node it never measured -- no membership
   list, no similarity -- which is worse than the stop-the-run behaviour removed
   here.  `MergeExecutionGateTests` and `ExecutionChannelTests` cover it, including
   the case where `_scope_issues` is deliberately stubbed out to prove the gate
   stands on its own.
2. **Both channels agree.**  Every stage judges scope twice (once as the binder's
   bound_validator, once at execution time).  C13RF11's D2b slipped through because
   a criterion was changed in one channel and not the other, so `BothChannelsTests`
   compares them directly.
3. **The run continues.**  The verdict shape (`DECISION_SCOPE_INEXPRESSIBLE`,
   `repairable=False`, empty `mutable_ref_paths`) is what makes the binder skip the
   repair round and the harness guard let the call through.
4. **The strictness floor survives.**  An empty scope channel is still never
   deferrable, and a channel carrying any repairable entry still stops the run.
5. **The accepted cost is real and bounded.**  A mistyped reference is no longer
   repaired; the tree changes less, never wrongly.
"""
from __future__ import annotations

import copy
import json
import tempfile
import types
import unittest
from pathlib import Path

from tests.test_semantic_stage_dispatch import load_stage_module
from utils.semantic_contract import (
    DEFAULT_DEFERRED_SCOPE_MESSAGE,
    deferred_restructure_record,
)
from utils.tree_manager import TreeManager

INEXPRESSIBLE = "DECISION_SCOPE_INEXPRESSIBLE"
VIOLATION = "DECISION_SCOPE_VIOLATION"


def pair_tree():
    """PARENT + CHILD (the displayed pair) plus the sibling a model may name."""
    return {"node_id": "ROOT", "label": "root", "level": "ROOT", "children": [
        {"node_id": "L1_P", "label": "parent", "level": "L1", "children": [
            {"node_id": "L2_C", "label": "child", "level": "L2", "children": [
                {"node_id": "L3_G", "label": "grandchild", "level": "L3",
                 "children": []}]},
            {"node_id": "L2_SIB", "label": "sibling", "level": "L2",
             "children": []}]}]}


def evidence():
    return {
        "summary": "probe",
        "warnings": [],
        "target_represents_all_source_members": True,
        "membership_basis": "probe",
        "pure_structural_redundancy": True,
        "cross_l1_authorized": False,
    }


def decision(action, source_id, target_id, *, children=None, **over):
    value = {
        "relation": "exact_duplicate",
        "action": action,
        "source_id": source_id,
        "target_id": target_id,
        "new_label": None,
        "confidence": 0.95,
        "evidence": evidence(),
        "child_plan": list(children or []),
    }
    value.update(over)
    return value


def child(child_id, target_parent_id, disposition="move"):
    return {
        "child_id": child_id,
        "disposition": disposition,
        "target_parent_id": target_parent_id,
        "relation": "broader_narrower",
        "same_domain": True,
        "evidence": "probe child",
    }


class StageLoadingTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.collapse = load_stage_module("collapse_redundant_hierarchy")
        cls.polish = load_stage_module("polish_tree_labels")
        cls.balance = load_stage_module("balance_tree_structure")
        cls.finalize = load_stage_module("finalize_policy_tree")

    def collapse_issues(self, value, parent="L1_P", child_id="L2_C", tree=None):
        probe = types.SimpleNamespace(tm=TreeManager(copy.deepcopy(tree or pair_tree())))
        return self.collapse.SkeletonRefiner._scope_issues(
            probe, {"decisions": [copy.deepcopy(value)]}, parent, child_id)

    def polish_issues(self, value, a="L2_C", b="L2_SIB", tree=None):
        probe = types.SimpleNamespace(tm=TreeManager(copy.deepcopy(tree or pair_tree())))
        return self.polish.PolishingProcess._scope_issues(
            probe, {"decisions": [copy.deepcopy(value)]}, a, b)

    def balance_issues(self, value, pid="L2_C", tree=None):
        probe = self.balance.ShapingProcess.__new__(self.balance.ShapingProcess)
        probe.tm = TreeManager(copy.deepcopy(tree or pair_tree()))
        return self.balance.ShapingProcess._scope_issues(
            probe, pid, {"decisions": [copy.deepcopy(value)]})

    def finalize_issues(self, value, l1_id="L1_P", tree=None):
        probe = types.SimpleNamespace(tm=TreeManager(copy.deepcopy(tree or pair_tree())))
        return self.finalize.OverallStructureAudit._scope_issues(
            probe, {"decisions": [copy.deepcopy(value)]}, l1_id)


# --------------------------------------------------------------------------
# 1. The load-bearing negative: off-stage merges still never execute.
# --------------------------------------------------------------------------

class MergeExecutionGateTests(StageLoadingTestCase):
    """The positive gate 14a and 14c check at execution time, on its own.

    The gate is deliberately independent of `_scope_issues`.  If a future edit
    loosens a scope predicate, the gate must still refuse to absorb a node this
    stage was never shown -- so it is written as a standalone function and tested
    as one.
    """

    def test_gate_permits_a_merge_on_exactly_the_displayed_pair(self):
        for module in (self.collapse, self.polish):
            with self.subTest(stage=module.__name__):
                self.assertIsNone(module.merge_execution_gate_error(
                    decision("merge", "L2_C", "L1_P"), {"L1_P", "L2_C"}))
                # Order must not matter: the pair is a set of two ids.
                self.assertIsNone(module.merge_execution_gate_error(
                    decision("merge", "L1_P", "L2_C"), {"L1_P", "L2_C"}))

    def test_gate_refuses_a_merge_naming_anything_else(self):
        for module in (self.collapse, self.polish):
            for source, target in (
                ("L2_C", "L2_SIB"),      # C13RF28's actual death: the sibling
                ("L2_SIB", "L2_C"),      # and the reverse
                ("L2_C", "L3_G"),        # a grandchild
                ("L2_SIB", "L3_G"),      # neither id displayed at all
            ):
                with self.subTest(stage=module.__name__, pair=(source, target)):
                    self.assertEqual(
                        module.merge_execution_gate_error(
                            decision("merge", source, target), {"L1_P", "L2_C"}),
                        module.MERGE_EXECUTION_GATE_MESSAGE)

    def test_gate_only_judges_merges(self):
        """Non-mutating verbs are not the gate's business -- they mutate nothing,
        and their own scope check still records and skips them."""
        for module in (self.collapse, self.polish):
            for action in ("keep", "reject_merge", "uncertain"):
                with self.subTest(stage=module.__name__, action=action):
                    self.assertIsNone(module.merge_execution_gate_error(
                        decision(action, "L2_C", "L2_SIB"), {"L1_P", "L2_C"}))

    def test_gate_refuses_a_non_object(self):
        for module in (self.collapse, self.polish):
            with self.subTest(stage=module.__name__):
                self.assertIsNotNone(
                    module.merge_execution_gate_error(None, {"L1_P", "L2_C"}))


class ExecutionChannelTests(StageLoadingTestCase):
    """The tree must not move for anything off-stage -- checked by hashing it."""

    def run_execution(self, module, class_name, value, args, *, stub_scope=False):
        process = getattr(module, class_name).__new__(getattr(module, class_name))
        process.tm = TreeManager(copy.deepcopy(pair_tree()))
        process.membership_counts = {}
        process.redirect_map = {}
        process.local_trace_map = {}
        process._current_lineage = lambda: {}
        if stub_scope:
            # Prove the merge gate stands alone: with the scope validator stubbed
            # to approve everything, an off-stage merge must STILL not execute.
            process._scope_issues = lambda *a, **k: []
        with tempfile.TemporaryDirectory() as directory:
            process.ops_log = Path(directory) / "ops.jsonl"
            before = json.dumps(process.tm.root, sort_keys=True, ensure_ascii=False)
            process._execute_decision(copy.deepcopy(value), *args)
            after = json.dumps(process.tm.root, sort_keys=True, ensure_ascii=False)
            records = [json.loads(line) for line
                       in process.ops_log.read_text(encoding="utf-8").splitlines()
                       if line.strip()] if process.ops_log.exists() else []
        return before == after, records

    def test_offstage_merge_does_not_move_the_tree_at_14a(self):
        unmoved, records = self.run_execution(
            self.collapse, "SkeletonRefiner",
            decision("merge", "L2_C", "L2_SIB"), ("L1_P", "L2_C"))
        self.assertTrue(unmoved)
        self.assertEqual([r["status"] for r in records], ["rejected"])
        self.assertEqual(
            {v["code"] for r in records
             for v in r["semantic_contract"]["violations"]},
            {INEXPRESSIBLE})

    def test_offstage_merge_does_not_move_the_tree_at_14c(self):
        unmoved, records = self.run_execution(
            self.polish, "PolishingProcess",
            decision("merge", "L2_C", "L1_P"), ("L2_C", "L2_SIB", "sibling_merge"))
        self.assertTrue(unmoved)
        self.assertEqual([r["status"] for r in records], ["rejected"])

    def test_the_gate_holds_even_with_the_scope_validator_stubbed_out(self):
        """The reason the gate is a separate check and not a corollary.

        If someone later loosens `_scope_issues` -- by accident or by a well-meant
        widening -- an off-stage merge must still be refused, because executing one
        is worse than the failure mode C13RF29 removed.
        """
        for module, class_name, args in (
            (self.collapse, "SkeletonRefiner", ("L1_P", "L2_C")),
            (self.polish, "PolishingProcess", ("L2_C", "L2_SIB", "case")),
        ):
            with self.subTest(stage=module.__name__):
                unmoved, records = self.run_execution(
                    module, class_name,
                    decision("merge", "L2_C", "L2_SIB"), args, stub_scope=True)
                self.assertTrue(unmoved, "off-stage merge executed with scope stubbed")
                self.assertEqual([r["status"] for r in records], ["rejected"])

    def test_control_an_onstage_merge_still_moves_the_tree(self):
        """Without this control, "the tree did not move" proves nothing -- the
        harness might simply be incapable of mutating anything."""
        unmoved, records = self.run_execution(
            self.collapse, "SkeletonRefiner",
            decision("merge", "L2_C", "L1_P",
                     children=[child("L3_G", "L1_P")]),
            ("L1_P", "L2_C"))
        self.assertFalse(unmoved, "the on-stage control did not execute")
        self.assertEqual([r["status"] for r in records], ["applied"])


# --------------------------------------------------------------------------
# 2. The inversion itself, all four stages.
# --------------------------------------------------------------------------

class InvertedDispositionTests(StageLoadingTestCase):
    """Every failed scope check yields the shape that keeps the run alive."""

    def assert_recorded_and_skipped(self, issues, *, stage):
        self.assertEqual(len(issues), 1, f"{stage}: {issues}")
        issue = issues[0]
        self.assertEqual(issue["code"], INEXPRESSIBLE, f"{stage}: {issue}")
        # These three fields together are what make the binder skip the repair
        # round and the harness guard (is_deferrable_result) let the call through.
        self.assertIs(issue["context"]["repairable"], False, f"{stage}: {issue}")
        self.assertEqual(issue["context"]["mutable_ref_paths"], [],
                         f"{stage}: {issue}")
        self.assertTrue(issue["context"]["scope_messages"], f"{stage}: {issue}")
        return issue

    def test_14a_offstage_shapes_are_recorded_and_skipped(self):
        for name, value in (
            ("rf28_death", decision("merge", "L2_C", "L2_SIB")),
            ("keep", decision("keep", "L2_C", "L2_SIB")),
            ("reject_merge", decision("reject_merge", "L2_C", "L2_SIB")),
            ("uncertain", decision("uncertain", "L2_C", "L2_SIB")),
            ("unknown_action", decision("teleport", "L2_C", "L1_P")),
            ("child_plan_offstage",
             decision("merge", "L2_C", "L1_P",
                      children=[child("L3_G", "L2_SIB")])),
        ):
            with self.subTest(case=name):
                self.assert_recorded_and_skipped(
                    self.collapse_issues(value), stage="14a")

    def test_14c_offstage_shapes_are_recorded_and_skipped(self):
        for name, value in (
            ("merge_offstage", decision("merge", "L2_C", "L1_P")),
            ("keep", decision("keep", "L2_C", "L1_P")),
            ("reject_merge", decision("reject_merge", "L2_C", "L1_P")),
            ("uncertain", decision("uncertain", "L2_C", "L1_P")),
            ("unknown_action", decision("teleport", "L2_C", "L2_SIB")),
        ):
            with self.subTest(case=name):
                self.assert_recorded_and_skipped(
                    self.polish_issues(value), stage="14c")

    def test_14b_offstage_shapes_are_recorded_and_skipped(self):
        for name, value in (
            # 14b has no `merge` verb at all, so proposing one is an unknown action.
            ("merge_is_not_a_balance_verb", decision("merge", "L2_C", "L2_SIB")),
            ("keep_offstage", decision("keep", "L2_SIB", "L2_SIB")),
            ("uncertain_offstage", decision("uncertain", "L2_SIB", "L2_SIB")),
            ("flatten_wrong_target",
             decision("flatten", "L2_C", "L2_SIB",
                      children=[child("L3_G", "L2_SIB")])),
            ("move_source_not_a_child", decision("move", "L2_SIB", "L1_P")),
        ):
            with self.subTest(case=name):
                self.assert_recorded_and_skipped(
                    self.balance_issues(value), stage="14b")

    def test_14d_offstage_shapes_are_recorded_and_skipped(self):
        foreign_tree = {
            "node_id": "ROOT", "label": "root", "level": "ROOT", "children": [
                pair_tree()["children"][0],
                {"node_id": "L1_OTHER", "label": "other", "level": "L1",
                 "children": [{"node_id": "L2_FOREIGN", "label": "foreign",
                               "level": "L2", "children": []}]}]}
        for name, value in (
            ("target_in_another_l1", decision("merge", "L2_C", "L2_FOREIGN")),
            ("source_in_another_l1", decision("merge", "L2_FOREIGN", "L2_C")),
            ("source_is_the_l1", decision("merge", "L1_P", "L2_C")),
            ("unknown_action", decision("teleport", "L2_C", "L2_SIB")),
            ("merge_with_itself", decision("merge", "L2_C", "L2_C")),
        ):
            with self.subTest(case=name):
                self.assert_recorded_and_skipped(
                    self.finalize_issues(value, tree=foreign_tree), stage="14d")


class LegalShapeControlTests(StageLoadingTestCase):
    """The controls: what executed before must still execute, unchanged.

    Without these the inversion could be "achieved" by refusing everything, which
    would pass every test above and produce an untouched tree.
    """

    def test_14a_legal_shapes_produce_no_issue(self):
        for name, value in (
            ("merge_on_pair", decision("merge", "L2_C", "L1_P",
                                       children=[child("L3_G", "L1_P")])),
            ("keep_on_pair", decision("keep", "L1_P", "L2_C")),
            ("reject_merge_on_pair", decision("reject_merge", "L1_P", "L2_C")),
            ("uncertain_on_pair", decision("uncertain", "L1_P", "L2_C")),
            ("rename_displayed", decision("rename", "L2_C", "L2_C",
                                          new_label="x")),
        ):
            with self.subTest(case=name):
                self.assertEqual(self.collapse_issues(value), [], name)

    def test_14c_legal_shapes_produce_no_issue(self):
        for name, value in (
            ("merge_on_sibling_pair",
             decision("merge", "L2_C", "L2_SIB",
                      children=[child("L3_G", "L2_SIB")])),
            ("keep_on_pair", decision("keep", "L2_C", "L2_SIB")),
            ("rename_displayed", decision("rename", "L2_C", "L2_C",
                                          new_label="x")),
            ("move_to_displayed_parent",
             decision("move", "L2_C", "L1_P",
                      children=[child("L3_G", "L2_C", "keep")])),
        ):
            with self.subTest(case=name):
                self.assertEqual(self.polish_issues(value), [], name)

    def test_14b_legal_shapes_produce_no_issue(self):
        for name, value in (
            ("keep_on_candidate", decision("keep", "L2_C", "L2_C")),
            ("uncertain_on_candidate", decision("uncertain", "L2_C", "L2_C")),
            ("flatten_to_parent",
             decision("flatten", "L2_C", "L1_P",
                      children=[child("L3_G", "L1_P")])),
        ):
            with self.subTest(case=name):
                self.assertEqual(self.balance_issues(value), [], name)

    def test_14d_legal_shapes_produce_no_issue(self):
        for name, value in (
            ("merge_within_l1",
             decision("merge", "L2_C", "L2_SIB",
                      children=[child("L3_G", "L2_SIB")])),
            ("keep_within_l1", decision("keep", "L2_C", "L2_C")),
            ("rename_within_l1", decision("rename", "L2_C", "L2_C",
                                          new_label="x")),
        ):
            with self.subTest(case=name):
                self.assertEqual(self.finalize_issues(value), [], name)


# --------------------------------------------------------------------------
# 3. Both channels must agree (the "parallel verdict channels" lesson).
# --------------------------------------------------------------------------

class BothChannelsTests(StageLoadingTestCase):
    """Each stage judges scope twice; a criterion changed in one only is a bug.

    C13RF11's D2b reached production exactly that way, and the C13RF29 card counts
    eight sites (four stages x two channels) for the same reason.  14a and 14c call
    the identical `_scope_issues` from both channels, so the check here is that the
    execution channel really does re-run it and act on the result -- not that two
    copies happen to agree.
    """

    def test_14a_execution_channel_reruns_the_same_predicate(self):
        value = decision("merge", "L2_C", "L2_SIB")
        binder_issues = self.collapse_issues(value)
        process = self.collapse.SkeletonRefiner.__new__(
            self.collapse.SkeletonRefiner)
        process.tm = TreeManager(copy.deepcopy(pair_tree()))
        process.membership_counts = {}
        process.redirect_map = {}
        seen = {}
        original = self.collapse.SkeletonRefiner._scope_issues

        def spy(self_, payload, parent_id, child_id):
            result = original(self_, payload, parent_id, child_id)
            seen["called"] = True
            seen["codes"] = [item["code"] for item in result]
            return result

        with tempfile.TemporaryDirectory() as directory:
            process.ops_log = Path(directory) / "ops.jsonl"
            process._scope_issues = types.MethodType(spy, process)
            process._execute_decision(copy.deepcopy(value), "L1_P", "L2_C")
        self.assertTrue(seen.get("called"), "execution channel skipped the predicate")
        self.assertEqual(seen["codes"], [item["code"] for item in binder_issues])

    def test_14b_and_14d_batch_channels_abort_the_whole_batch(self):
        """Whole-batch abort is kept on purpose.

        14b and 14d propose a batch as one plan (a bridge plus the child moves that
        populate it; one L1's decisions together).  Executing the sound half of a
        plan whose sibling was refused would WIDEN what the stage executes, and this
        card widens nothing -- its whole subject is the disposition of refusals.
        """
        # 14b: one legal flatten batched with an off-stage move.
        probe = self.balance.ShapingProcess.__new__(self.balance.ShapingProcess)
        probe.tm = TreeManager(copy.deepcopy(pair_tree()))
        issues = self.balance.ShapingProcess._scope_issues(
            probe, "L2_C", {"decisions": [
                decision("flatten", "L2_C", "L1_P",
                         children=[child("L3_G", "L1_P")]),
                decision("move", "L2_SIB", "L1_P"),
            ]})
        # Only the offending decision carries an issue; _apply turns the rest into
        # BATCH_SCOPE_ABORTED records and applies none of them.
        self.assertEqual([item["context"]["decision_index"] for item in issues], [1])
        self.assertEqual(issues[0]["code"], INEXPRESSIBLE)


# --------------------------------------------------------------------------
# 4. The strictness floor that survives the inversion.
# --------------------------------------------------------------------------

class StrictnessFloorTests(StageLoadingTestCase):
    """What still stops a run.  The inversion is not "defer everything".

    The defer predicate reads the binder's scope-error channel.  Two rules are kept
    from C13RF10/RF18 and are load-bearing in the other direction: an EMPTY channel
    is never deferrable ("no recorded error" must not be read as "every error was
    benign"), and any entry that is not an inexpressible verdict still stops the run.
    Nothing the four stages now emit looks like the second case, which is exactly why
    it has to be checked synthetically -- a future repairable class must fail closed
    rather than inherit this allowance.
    """

    def predicates(self):
        return (
            ("14a", self.collapse.SkeletonRefiner._is_deferrable_response),
            ("14b", self.balance.ShapingProcess._is_deferrable_response),
            ("14c", self.polish.PolishingProcess._is_deferrable_response),
            ("14d", self.finalize.OverallStructureAudit._is_deferrable_response),
        )

    def base_response(self, errors, *, decisions=None):
        return {
            "ok": False,
            "mutation_before_validation": False,
            "final_bound": None,
            "final_disposition": "scope_rejected",
            "error": "BOUND_SCOPE_VALIDATION_FAILED",
            "repair_count": 0,
            "initial_scope_errors": errors,
            "repair_scope_errors": [],
            "final_local": {"decisions": decisions or [{"action": "create_bridge"}]},
        }

    def inexpressible_entry(self, index=0):
        return {
            "code": INEXPRESSIBLE,
            "message": "some off-stage reason",
            "context": {
                "decision_index": index,
                "mutable_ref_paths": [],
                "repairable": False,
            },
        }

    def test_an_empty_channel_is_never_deferrable(self):
        for stage, predicate in self.predicates():
            with self.subTest(stage=stage):
                self.assertFalse(predicate(self.base_response([])))

    def test_a_repairable_entry_still_stops_the_run(self):
        entry = self.inexpressible_entry()
        entry["context"]["repairable"] = True
        entry["context"]["mutable_ref_paths"] = ["decisions[0].target_ref"]
        for stage, predicate in self.predicates():
            with self.subTest(stage=stage):
                self.assertFalse(predicate(self.base_response([entry])))

    def test_a_non_inexpressible_code_still_stops_the_run(self):
        entry = self.inexpressible_entry()
        entry["code"] = VIOLATION
        for stage, predicate in self.predicates():
            with self.subTest(stage=stage):
                self.assertFalse(predicate(self.base_response([entry])))

    def test_a_mutated_tree_still_stops_the_run(self):
        """`mutation_before_validation` is a hard floor: if the tree moved while the
        untrusted response was in flight, nothing about the response is trustworthy."""
        for stage, predicate in self.predicates():
            with self.subTest(stage=stage):
                response = self.base_response([self.inexpressible_entry()])
                response["mutation_before_validation"] = True
                self.assertFalse(predicate(response))

    def test_a_bound_payload_still_stops_the_run(self):
        for stage, predicate in self.predicates():
            with self.subTest(stage=stage):
                response = self.base_response([self.inexpressible_entry()])
                response["final_bound"] = {"decisions": []}
                self.assertFalse(predicate(response))

    def test_a_clean_inexpressible_channel_defers(self):
        for stage, predicate in self.predicates():
            with self.subTest(stage=stage):
                self.assertTrue(
                    predicate(self.base_response([self.inexpressible_entry()])))
