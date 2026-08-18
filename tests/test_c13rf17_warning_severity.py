"""C13RF17: merge warnings are graded by content, on both parallel channels.

Before this card a merge was rejected whenever ``evidence.warnings`` was
non-empty, whatever the warning said.  Measured over the C13M2 corpus: 13 of 20
merge proposals carried a warning and all 13 were rejected purely by that rule,
although every one had already cleared the destructive-merge whitelist.  The
only merge that was ever applied was the only one that stayed silent.  So merge
counts measured whether a model volunteered a caveat, not whether the merge was
sound.

Two channels had to move together, or the same concern keeps a second outlet:
  * the LIST channel  -- MERGE_WARNING_CONFLICT / MERGE_WARNING_ADVISORY
  * the PROSE channel -- MERGE_WARNING_TEXT_CONFLICT, keyed on summary regexes

The four-quadrant block below is the joint coverage: warnings non-empty /
summary matching / both / neither.
"""
from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from utils.semantic_contract import (  # noqa: E402
    SEVERITY_ADVISORY,
    SEVERITY_CRITICAL,
    classify_merge_warning,
    execute_semantic_decision,
    validate_semantic_decision,
)
from utils.tree_manager import TreeManager  # noqa: E402


def node(node_id, label, level, children=None):
    return {
        "node_id": node_id,
        "label": label,
        "level": level,
        "children": list(children or []),
    }


BASE_TREE = node(
    "ROOT", "ROOT", "ROOT",
    [node("L1", "domain", "L1", [
        node("W", "winner", "L2"),
        node("S", "source", "L2"),
    ])],
)


def merge_decision(*, warnings=None, summary=None, action="merge"):
    return {
        "relation": "synonym",
        "action": action,
        "source_id": "S",
        "target_id": "W",
        "new_label": None,
        "confidence": 0.95,
        "evidence": {
            "summary": summary or "Labels and represented records mean the same.",
            "warnings": list(warnings or []),
            "target_represents_all_source_members": True,
            "membership_basis": "direct members",
            "pure_structural_redundancy": True,
            "cross_l1_authorized": False,
        },
        "child_plan": [],
    }


def context_for(manager):
    return {
        "source_id": "S",
        "target_id": "W",
        "source_exists": manager.exists("S"),
        "target_exists": manager.exists("W"),
        "source_l1_id": "L1",
        "target_l1_id": "L1",
        "allowed_l1_id": "L1",
        "allow_cross_l1": False,
        "membership_known": True,
        "source_direct_membership_count": 0,
        "source_child_ids": [],
        "target_child_ids": [],
    }


def report_for(decision):
    manager = TreeManager(copy.deepcopy(BASE_TREE))
    return validate_semantic_decision(decision, context_for(manager))


def codes(report, prefix="MERGE_WARNING"):
    return sorted(
        str(item["code"]) for item in report["violations"]
        if str(item["code"]).startswith(prefix)
    )


# Real strings, taken verbatim from the C13M2 corpus and the RF13 14c archive.
BENIGN_LEXICAL = "minor_verb_nuance"
BENIGN_STRUCTURAL = "PARENT has 6 other children that remain unaffected"
CONFLICT_SHIPPED = "heterogeneous_membership"
CONFLICT_HYPERNYM = (
    "PARENT label is slightly broader ('服务效能' vs '通关便利化'), "
    "but example title alignment confirms synonymy in practice"
)
CONFLICT_OVERLAP = "labels_not_identical_but_semantically_overlapping"
BENIGN_SUBSUMED = "RIGHT's '关键' qualifier is dropped but subsumed by broader LEFT label"
MACHINE_SAFETY = "promote_safety:父节点有 2 个子节点，非单脉传，禁止 Promote"
CONFLICT_SUMMARY = "The labels are related but not equivalent."


class FourQuadrantTests(unittest.TestCase):
    """Joint coverage of the two channels: list x prose."""

    def test_q1_benign_list_clean_summary_passes(self):
        report = report_for(merge_decision(warnings=[BENIGN_LEXICAL]))
        self.assertTrue(report["passed"])
        self.assertEqual(codes(report), ["MERGE_WARNING_ADVISORY"])
        self.assertEqual(report["critical_count"], 0)
        self.assertEqual(report["advisory_count"], 1)

    def test_q2_clean_list_conflicting_summary_rejects(self):
        report = report_for(merge_decision(summary=CONFLICT_SUMMARY))
        self.assertFalse(report["passed"])
        self.assertEqual(codes(report), ["MERGE_WARNING_TEXT_CONFLICT"])

    def test_q3_benign_list_and_conflicting_summary_rejects(self):
        report = report_for(
            merge_decision(warnings=[BENIGN_LEXICAL], summary=CONFLICT_SUMMARY)
        )
        self.assertFalse(report["passed"])
        # the prose channel rejects; the benign caveat is still recorded
        self.assertEqual(
            codes(report),
            ["MERGE_WARNING_ADVISORY", "MERGE_WARNING_TEXT_CONFLICT"],
        )
        self.assertEqual(report["advisory_count"], 1)

    def test_q4_clean_list_clean_summary_passes(self):
        report = report_for(merge_decision())
        self.assertTrue(report["passed"])
        self.assertEqual(codes(report), [])
        self.assertEqual(report["advisory_count"], 0)


class ConflictStaysCriticalTests(unittest.TestCase):
    """Strictness is not relaxed for anything that asserts a conflict."""

    def test_shipped_conflict_vocabulary_still_blocks(self):
        report = report_for(merge_decision(warnings=[CONFLICT_SHIPPED]))
        self.assertFalse(report["passed"])
        self.assertEqual(codes(report), ["MERGE_WARNING_CONFLICT"])

    def test_hypernym_warning_blocks(self):
        # The corpus case the old rule caught by accident and that a naive
        # relaxation would have let through.
        report = report_for(merge_decision(warnings=[CONFLICT_HYPERNYM]))
        self.assertFalse(report["passed"])
        self.assertEqual(codes(report), ["MERGE_WARNING_CONFLICT"])

    def test_partial_overlap_warning_blocks(self):
        # Token form: '_' is a word character, so this only classifies
        # correctly because the matcher also tries an underscore-normalised
        # copy.  Regression guard for that.
        report = report_for(merge_decision(warnings=[CONFLICT_OVERLAP]))
        self.assertFalse(report["passed"])
        self.assertEqual(codes(report), ["MERGE_WARNING_CONFLICT"])

    def test_unrecognised_warning_fails_closed(self):
        report = report_for(merge_decision(warnings=["完全没见过的新措辞 xyz"]))
        self.assertFalse(report["passed"])
        self.assertEqual(codes(report), ["MERGE_WARNING_CONFLICT"])

    def test_mixed_list_reports_only_the_conflicting_half_as_critical(self):
        report = report_for(
            merge_decision(warnings=[BENIGN_LEXICAL, CONFLICT_SHIPPED])
        )
        self.assertFalse(report["passed"])
        entry = next(item for item in report["violations"]
                     if item["code"] == "MERGE_WARNING_CONFLICT")
        self.assertEqual(entry["context"]["warnings"], [CONFLICT_SHIPPED])
        self.assertEqual(entry["context"]["advisory_warnings"], [BENIGN_LEXICAL])
        self.assertEqual(len(entry["context"]["all_warnings"]), 2)


class MachineSafetyTests(unittest.TestCase):
    """CONTROL: the one code-side warning producer must never be graded down.

    collapse_redundant_hierarchy.py appends "promote_safety:<reason>" to this
    same list when can_promote_safely() refuses an unsafe promotion.  That is a
    safety verdict riding the self-disclosure channel; if grading downgraded it
    the fix would have silently disabled a structural guard.
    """

    def test_promote_safety_warning_is_always_critical(self):
        report = report_for(merge_decision(warnings=[MACHINE_SAFETY]))
        self.assertFalse(report["passed"])
        self.assertEqual(codes(report), ["MERGE_WARNING_CONFLICT"])
        severity, reason = classify_merge_warning(MACHINE_SAFETY)
        self.assertEqual(severity, SEVERITY_CRITICAL)
        self.assertEqual(reason, "machine_safety_refusal")

    def test_promote_safety_beats_an_otherwise_benign_wording(self):
        # Even if a future reason string happens to read like a benign note.
        warning = "promote_safety:minor_verb_nuance"
        self.assertEqual(classify_merge_warning(warning)[0], SEVERITY_CRITICAL)


class SubsumptionTests(unittest.TestCase):
    """Containment of a modifier is benign; partial overlap is not."""

    def test_subsumed_qualifier_is_advisory(self):
        report = report_for(merge_decision(warnings=[BENIGN_SUBSUMED]))
        self.assertTrue(report["passed"])
        self.assertEqual(codes(report), ["MERGE_WARNING_ADVISORY"])

    def test_broader_without_subsumption_is_critical(self):
        self.assertEqual(
            classify_merge_warning("PARENT label is slightly broader")[0],
            SEVERITY_CRITICAL,
        )

    def test_structural_disclosure_is_advisory(self):
        report = report_for(merge_decision(warnings=[BENIGN_STRUCTURAL]))
        self.assertTrue(report["passed"])
        self.assertEqual(codes(report), ["MERGE_WARNING_ADVISORY"])


class ScopeBoundaryTests(unittest.TestCase):
    """RF14 case C3: both channels remain scoped to action == merge."""

    def test_non_merge_actions_report_no_warning_violation(self):
        for action in ("move", "split_reparent", "keep",
                       "reject_merge", "uncertain", "rename"):
            with self.subTest(action=action):
                report = report_for(merge_decision(
                    action=action,
                    warnings=[CONFLICT_SHIPPED],
                    summary=CONFLICT_SUMMARY,
                ))
                self.assertEqual(codes(report), [])

    def test_many_warnings_still_yield_a_single_record(self):
        report = report_for(merge_decision(
            warnings=[CONFLICT_SHIPPED, "另一个 " + CONFLICT_SHIPPED, "第三个 异质"]
        ))
        self.assertEqual(codes(report), ["MERGE_WARNING_CONFLICT"])


class AccountingTests(unittest.TestCase):
    """Advisories are recorded in full but do not decide the verdict."""

    def test_advisory_is_recorded_with_its_severity_and_reason(self):
        report = report_for(merge_decision(warnings=[BENIGN_LEXICAL]))
        entry = next(item for item in report["violations"]
                     if item["code"] == "MERGE_WARNING_ADVISORY")
        self.assertEqual(entry["severity"], SEVERITY_ADVISORY)
        self.assertEqual(
            entry["context"]["warning_reasons"][BENIGN_LEXICAL],
            "benign_lexical_variance",
        )
        self.assertEqual(report["advisories"], [entry])

    def test_verifier_severity_filter_sees_only_criticals(self):
        # Mirrors harness/verify_stage_v2.py:147, which filters on
        # severity == "critical".  That is why the harness needs no change.
        report = report_for(merge_decision(warnings=[BENIGN_LEXICAL]))
        as_verifier_sees = [item for item in report["violations"]
                            if item.get("severity") == "critical"]
        self.assertEqual(as_verifier_sees, [])

    def test_critical_violations_still_carry_critical_severity(self):
        report = report_for(merge_decision(warnings=[CONFLICT_SHIPPED]))
        for item in report["violations"]:
            self.assertEqual(item["severity"], SEVERITY_CRITICAL)


class ExecutionTests(unittest.TestCase):
    """Grading decides admission, so check the tree actually moves."""

    def execute(self, decision):
        manager = TreeManager(copy.deepcopy(BASE_TREE))
        record = execute_semantic_decision(
            manager, decision,
            direct_membership_counts={},
            membership_known=True,
            stage="vertical_collapse",
        )
        return manager, record

    def test_benign_warning_merge_is_applied(self):
        manager, record = self.execute(merge_decision(warnings=[BENIGN_LEXICAL]))
        self.assertEqual(record["status"], "applied")
        self.assertFalse(manager.exists("S"))

    def test_conflicting_warning_merge_is_rejected_and_tree_untouched(self):
        manager, record = self.execute(merge_decision(warnings=[CONFLICT_SHIPPED]))
        self.assertEqual(record["status"], "rejected")
        self.assertTrue(manager.exists("S"))

    def test_machine_safety_warning_merge_is_rejected(self):
        manager, record = self.execute(merge_decision(warnings=[MACHINE_SAFETY]))
        self.assertEqual(record["status"], "rejected")
        self.assertTrue(manager.exists("S"))


class RealCorpusStringsTests(unittest.TestCase):
    """Every distinct warning the corpus and the RF13 archive actually contain."""

    CORPUS_ADVISORY = (
        "minor_lexical_variation",
        "minor_verb_nuance",
        "labels_not_identical_strings",
        "labels_not_identical_but_semantically_equivalent",
        "parent_has_other_children",
        "parent_has_other_children_not_affected",
        "child_has_children",
        "child_has_membership",
        "child_has_one_child_N13_requiring_plan",
        "single_child_chain_merge",
        "PARENT has 6 other children that remain unaffected",
        "PARENT has 1 direct member and 2 other children (N13, N14) that must "
        "be accounted for",
        "CHILD has 2 children (N15, N16) that must be reparented",
        "PARENT label includes '加快' (accelerate) which adds urgency nuance "
        "not present in CHILD",
        "PARENT uses '航线网络' while CHILD uses '航线' - minor scope difference "
        "but functionally equivalent in policy context",
        # RF13 14c
        "minor_modifier_difference",
        "slight_nuance_difference: 丰富 emphasizes enrichment while 完善 "
        "emphasizes perfection, but in policy context they are interchangeable goals",
        "LEFT has 2 direct members whose exact label text includes '航空'; "
        "merged under RIGHT they will appear under the shorter label, but "
        "semantic meaning is preserved by parent context.",
        BENIGN_SUBSUMED,
        # RF9 archive. Both of these initially hit the fail-closed default even
        # though an equivalent phrasing was already allowlisted, i.e. the same
        # statement was graded differently depending on wording. Locked here so
        # the paraphrase coverage cannot regress.
        "child_has_membership_and_children",
        "labels are not lexically identical but example titles and context "
        "confirm equivalence",
    )

    CORPUS_CRITICAL = (
        CONFLICT_HYPERNYM,
        CONFLICT_OVERLAP,
        # RF9 archive. These two must stay critical: a third node is also a
        # merge candidate (so which pair to merge is undecided), and the model
        # explicitly denies synonymy.
        "PARENT already has direct child N13 '完善国际航空货运航线' which is also "
        "very similar to CHILD label",
        "not_synonymous",
        "broader_narrower_relationship",
    )

    def test_corpus_benign_strings_grade_to_advisory(self):
        for warning in self.CORPUS_ADVISORY:
            with self.subTest(warning=warning[:40]):
                self.assertEqual(
                    classify_merge_warning(warning)[0], SEVERITY_ADVISORY
                )

    def test_corpus_conflicting_strings_stay_critical(self):
        for warning in self.CORPUS_CRITICAL:
            with self.subTest(warning=warning[:40]):
                self.assertEqual(
                    classify_merge_warning(warning)[0], SEVERITY_CRITICAL
                )


if __name__ == "__main__":
    unittest.main()
