"""C13RF25 (D7-1): `source_has_direct_membership` downgrades only in context.

W2 (the unlogged 2026-08-20 run) produced a tree that was one finding away from
publishable: the semantic gate passed, membership stayed conserved at 279 rows,
and built-in E0 rejected exactly one code -- `PARENT_CHILD_EXACT_DUPLICATE` for
the L2/L3 pair both labelled '争取政策倾斜'.  The merge that would have removed
that duplicate was itself blocked upstream: the model disclosed
`source_has_direct_membership` in ``evidence.warnings``, no word list matched it,
so the fail-closed default graded it critical -> MERGE_WARNING_CONFLICT ->
BATCH_SEMANTIC_ABORTED.

The plural form `source_has_direct_members` was already advisory via
BENIGN_STRUCTURAL_WARNING_PATTERNS, so the singular failing closed was a
word-form inconsistency rather than a policy.  This card downgrades it, but only
where the decision context makes the disclosure harmless -- BOTH of:
    relation == "exact_duplicate"
    evidence.target_represents_all_source_members is True

The strictness that must NOT move:
  * `source_has_direct_membership_not_represented_by_target` asserts a real
    conflict (it occurs once in the W1 data) and stays critical.  The match is
    string equality, so that longer token can never reach the downgrade path --
    a substring test or a `\\bdirect\\s+members(?:hip)?\\b` regex would have
    swallowed it.
  * SOURCE_MEMBERSHIP_NOT_REPRESENTED, the real membership guard, is untouched.
  * Any single condition missing keeps the warning critical.
"""
from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from utils.semantic_contract import (  # noqa: E402
    CONDITIONAL_MEMBERSHIP_WARNING,
    SEVERITY_ADVISORY,
    SEVERITY_CRITICAL,
    classify_merge_warning,
    downgrade_conditional_membership_warning,
    grade_merge_warnings,
    validate_semantic_decision,
)

TRAP_WARNING = "source_has_direct_membership_not_represented_by_target"

# The real rejected record, W2 archive line 3 (L2/L3 '争取政策倾斜').
W2_DECISION = {
    "relation": "exact_duplicate",
    "action": "merge",
    "source_id": "L3_N71ce64b1",
    "target_id": "L2_N71ce64b1",
    "new_label": None,
    "confidence": 0.99,
    "evidence": {
        "summary": (
            "N23 is a direct child of N22 with identical label '争取政策倾斜' (100% "
            "repetitive). N22 has direct_membership=1 and N23 has direct_membership=1. "
            "Merging N23 into N22 consolidates the duplicate label. N22's direct "
            "membership plus N23's membership are both represented by the identical "
            "label '争取政策倾斜'."
        ),
        "warnings": [CONDITIONAL_MEMBERSHIP_WARNING],
        "target_represents_all_source_members": True,
        "membership_basis": (
            "Both N22 and N23 share the identical label '争取政策倾斜'; all records "
            "under N23 are semantically covered by N22's identical label."
        ),
        "pure_structural_redundancy": False,
        "cross_l1_authorized": False,
    },
    "child_plan": [],
}

W2_CONTEXT = {
    "source_id": "L3_N71ce64b1",
    "target_id": "L2_N71ce64b1",
    "source_exists": True,
    "target_exists": True,
    "source_child_ids": [],
    "source_parent_id": "L2_N71ce64b1",
    "source_l1_id": "L1_NX",
    "target_l1_id": "L1_NX",
    "allowed_l1_id": "L1_NX",
    "allow_cross_l1": False,
    "membership_known": True,
    "source_direct_membership_count": 1,
    "target_is_descendant_of_source": False,
}


def w2_variant(**mutations):
    decision = copy.deepcopy(W2_DECISION)
    for key, value in mutations.items():
        if key.startswith("evidence_"):
            field = key[len("evidence_"):]
            if value is _ABSENT:
                decision["evidence"].pop(field, None)
            else:
                decision["evidence"][field] = value
        else:
            decision[key] = value
    return decision


class _Absent:
    pass


_ABSENT = _Absent()


class ConditionalDowngradeTargetCase(unittest.TestCase):
    """The one merge W2 needed: it must now pass, as advisory not silence."""

    def test_w2_pair_is_accepted_as_advisory(self):
        report = validate_semantic_decision(w2_variant(), W2_CONTEXT)
        self.assertTrue(report["passed"], report.get("violations"))
        self.assertEqual(report["critical_count"], 0)
        self.assertEqual(report["advisory_count"], 1)
        self.assertEqual(report["violation_counts"], {"MERGE_WARNING_ADVISORY": 1})

    def test_downgrade_is_recorded_with_its_own_reason(self):
        report = validate_semantic_decision(w2_variant(), W2_CONTEXT)
        advisory = [v for v in report["violations"] if v["code"] == "MERGE_WARNING_ADVISORY"]
        self.assertEqual(len(advisory), 1)
        self.assertEqual(advisory[0]["severity"], SEVERITY_ADVISORY)
        reasons = advisory[0]["context"]["warning_reasons"]
        self.assertEqual(
            reasons[CONDITIONAL_MEMBERSHIP_WARNING],
            "conditional_membership_disclosure",
        )

    def test_warning_is_not_silently_dropped(self):
        """Downgraded, still disclosed: the token is reported on the advisory
        channel rather than vanishing from the report."""
        report = validate_semantic_decision(w2_variant(), W2_CONTEXT)
        advisory = [v for v in report["violations"] if v["code"] == "MERGE_WARNING_ADVISORY"]
        self.assertIn(CONDITIONAL_MEMBERSHIP_WARNING, advisory[0]["context"]["warnings"])


class ConditionalDowngradeControlCases(unittest.TestCase):
    """Four controls.  Every one of these was REJECT on the baseline and must
    stay REJECT: if any flips, strictness regressed and the card fails."""

    def assert_still_critical(self, decision, expected_code="MERGE_WARNING_CONFLICT"):
        report = validate_semantic_decision(decision, W2_CONTEXT)
        self.assertFalse(report["passed"])
        self.assertGreaterEqual(report["critical_count"], 1)
        self.assertIn(expected_code, report["violation_counts"])
        return report

    def test_ctl_trapword_real_conflict_still_rejected(self):
        report = self.assert_still_critical(w2_variant(evidence_warnings=[TRAP_WARNING]))
        conflict = [v for v in report["violations"] if v["code"] == "MERGE_WARNING_CONFLICT"]
        self.assertIn(TRAP_WARNING, conflict[0]["context"]["warnings"])
        self.assertEqual(
            conflict[0]["context"]["warning_reasons"][TRAP_WARNING],
            "unrecognised_fail_closed",
        )

    def test_ctl_relation_non_exact_duplicate_still_rejected(self):
        self.assert_still_critical(w2_variant(relation="synonym"))

    def test_ctl_representation_false_still_rejected(self):
        report = self.assert_still_critical(
            w2_variant(evidence_target_represents_all_source_members=False)
        )
        self.assertIn("SOURCE_MEMBERSHIP_NOT_REPRESENTED", report["violation_counts"])

    def test_ctl_representation_absent_still_rejected(self):
        report = self.assert_still_critical(
            w2_variant(evidence_target_represents_all_source_members=_ABSENT)
        )
        self.assertIn("EVIDENCE_FIELDS_MISSING", report["violation_counts"])

    def test_every_other_relation_keeps_the_warning_critical(self):
        for relation in ("synonym", "broader_narrower", "related", "uncertain"):
            with self.subTest(relation=relation):
                self.assert_still_critical(w2_variant(relation=relation))


class ClassifierUntouchedCases(unittest.TestCase):
    """The downgrade sits outside the classifier; the classifier's own verdicts
    for both token forms are unchanged."""

    def test_classifier_still_fails_closed_on_the_bare_token(self):
        severity, reason = classify_merge_warning(CONDITIONAL_MEMBERSHIP_WARNING)
        self.assertEqual(severity, SEVERITY_CRITICAL)
        self.assertEqual(reason, "unrecognised_fail_closed")

    def test_classifier_still_fails_closed_on_the_trap_token(self):
        severity, _ = classify_merge_warning(TRAP_WARNING)
        self.assertEqual(severity, SEVERITY_CRITICAL)

    def test_plural_form_remains_advisory(self):
        severity, reason = classify_merge_warning("source_has_direct_members")
        self.assertEqual(severity, SEVERITY_ADVISORY)
        self.assertEqual(reason, "benign_structural_disclosure")


class DowngradeHelperUnitCases(unittest.TestCase):
    """Direct unit coverage of the helper, including the no-op paths."""

    def helper(self, critical, relation, evidence):
        reasons = {w: "unrecognised_fail_closed" for w in critical}
        return downgrade_conditional_membership_warning(
            critical, [], reasons, relation, evidence
        )

    OK_EVIDENCE = {"target_represents_all_source_members": True}

    def test_moves_only_the_exact_token(self):
        critical, advisory, reasons = self.helper(
            [CONDITIONAL_MEMBERSHIP_WARNING, "some_other_warning"],
            "exact_duplicate",
            self.OK_EVIDENCE,
        )
        self.assertEqual(critical, ["some_other_warning"])
        self.assertEqual(advisory, [CONDITIONAL_MEMBERSHIP_WARNING])
        self.assertEqual(
            reasons[CONDITIONAL_MEMBERSHIP_WARNING],
            "conditional_membership_disclosure",
        )
        self.assertEqual(reasons["some_other_warning"], "unrecognised_fail_closed")

    def test_string_equality_never_matches_the_longer_token(self):
        critical, advisory, _ = self.helper(
            [TRAP_WARNING], "exact_duplicate", self.OK_EVIDENCE
        )
        self.assertEqual(critical, [TRAP_WARNING])
        self.assertEqual(advisory, [])

    def test_truthy_but_non_true_representation_is_not_enough(self):
        for value in (1, "true", "True", [1], {"a": 1}):
            with self.subTest(value=value):
                critical, advisory, _ = self.helper(
                    [CONDITIONAL_MEMBERSHIP_WARNING],
                    "exact_duplicate",
                    {"target_represents_all_source_members": value},
                )
                self.assertEqual(critical, [CONDITIONAL_MEMBERSHIP_WARNING])
                self.assertEqual(advisory, [])

    def test_no_op_returns_reasons_object_unchanged(self):
        reasons = {CONDITIONAL_MEMBERSHIP_WARNING: "unrecognised_fail_closed"}
        _, _, out = downgrade_conditional_membership_warning(
            [CONDITIONAL_MEMBERSHIP_WARNING], [], reasons, "synonym", self.OK_EVIDENCE
        )
        self.assertIs(out, reasons)

    def test_absent_token_is_a_no_op_even_in_context(self):
        critical, advisory, _ = self.helper(
            ["unrelated_warning"], "exact_duplicate", self.OK_EVIDENCE
        )
        self.assertEqual(critical, ["unrelated_warning"])
        self.assertEqual(advisory, [])

    def test_grading_then_downgrading_composes(self):
        critical, advisory, reasons = grade_merge_warnings(
            [CONDITIONAL_MEMBERSHIP_WARNING, "source_has_direct_members"]
        )
        self.assertEqual(critical, [CONDITIONAL_MEMBERSHIP_WARNING])
        self.assertEqual(advisory, ["source_has_direct_members"])
        critical, advisory, _ = downgrade_conditional_membership_warning(
            critical, advisory, reasons, "exact_duplicate", self.OK_EVIDENCE
        )
        self.assertEqual(critical, [])
        self.assertCountEqual(
            advisory, ["source_has_direct_members", CONDITIONAL_MEMBERSHIP_WARNING]
        )


if __name__ == "__main__":
    unittest.main()
