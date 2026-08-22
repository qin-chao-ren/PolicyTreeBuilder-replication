from __future__ import annotations

import copy
import re
import time
from collections import Counter
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .tree_integrity import LineageError, close_lineage


DECISION_SCHEMA_VERSION = "semantic-tree-decision-v1"
PUBLICATION_SCHEMA_VERSION = "semantic-contract-publication-v1"

RELATIONS = frozenset({
    "exact_duplicate",
    "synonym",
    "broader_narrower",
    "related",
    "complementary",
    "means_goal",
    "carrier_outcome",
    "misplaced",
    "uncertain",
})

ACTIONS = frozenset({
    "merge",
    "keep",
    "reject_merge",
    "move",
    "move_across_l1",
    "split_reparent",
    "uncertain",
    "rename",
    "flatten",
    "create_bridge",
})

MERGE_RELATIONS = frozenset({"exact_duplicate", "synonym"})
CHILD_COMPATIBLE_RELATIONS = frozenset({
    "exact_duplicate",
    "synonym",
    "broader_narrower",
})
CHILD_DISPOSITIONS = frozenset({"move", "keep", "retain_under_source"})

SEMANTIC_OPERATION_TYPES = frozenset({
    "merge",
    "absorb",
    "promote_child",
    "move",
    "lift_sibling",
    "split_reparent",
    "rename",
    "flatten",
    "create_bridge",
    "insert_bridge",
})

ACTION_OPERATION_TYPES = {
    "merge": frozenset({"merge", "absorb", "promote_child"}),
    "move": frozenset({"move", "lift_sibling"}),
    "move_across_l1": frozenset({"move", "lift_sibling"}),
    "split_reparent": frozenset({"split_reparent"}),
    "rename": frozenset({"rename"}),
    "flatten": frozenset({"flatten"}),
    "create_bridge": frozenset({"create_bridge", "insert_bridge"}),
}

REQUIRED_DECISION_FIELDS = frozenset({
    "relation",
    "action",
    "source_id",
    "target_id",
    "new_label",
    "confidence",
    "evidence",
    "child_plan",
})

ALLOWED_DECISION_FIELDS = REQUIRED_DECISION_FIELDS | frozenset({"operation_id"})

REQUIRED_EVIDENCE_FIELDS = frozenset({
    "summary",
    "warnings",
    "target_represents_all_source_members",
    "membership_basis",
    "pure_structural_redundancy",
    "cross_l1_authorized",
})

REQUIRED_CHILD_PLAN_FIELDS = frozenset({
    "child_id",
    "disposition",
    "target_parent_id",
    "relation",
    "same_domain",
    "evidence",
})

ALLOWED_ENVELOPE_FIELDS = frozenset({"decisions"})
ALLOWED_EVIDENCE_FIELDS = REQUIRED_EVIDENCE_FIELDS
ALLOWED_CHILD_PLAN_FIELDS = REQUIRED_CHILD_PLAN_FIELDS

WARNING_PATTERNS = tuple(re.compile(pattern, re.IGNORECASE) for pattern in (
    r"\bnot\s+(?:an?\s+)?(?:exact\s+duplicate|synonym|equivalent)\b",
    r"\b(?:broader|narrower)\s+than\b",
    r"\brelated\s+but\b",
    r"\bdifferent\s+(?:meaning|scope|domain|axis)\b",
    r"\bheterogeneous\b",
    r"\buncertain\b",
    r"\binsufficient\s+evidence\b",
    r"\bmeans[- ]goal\b",
    r"\bcarrier[- ]outcome\b",
    r"\bmisplaced\b",
    r"不是(?:同义|重复|等价)",
    r"并非(?:同义|重复|等价)",
    r"上下位",
    r"相关但",
    r"语义不同",
    r"范围不同",
    r"异质",
    r"不确定",
    r"证据不足",
    r"手段.?目标",
    r"载体.?结果",
    r"错位",
))

SEVERITY_CRITICAL = "critical"
SEVERITY_ADVISORY = "advisory"

# --- merge warning grading (C13RF17) ---------------------------------------
# Before RF17 a merge was rejected whenever ``evidence.warnings`` was non-empty,
# regardless of what the warning said.  Measured consequence over the C13M2
# corpus: 13 of 20 merge proposals carried a warning and all 13 were rejected
# solely by that rule, while every one of them had already passed the
# destructive-merge whitelist.  Admission therefore tracked whether the model
# volunteered a caveat, not whether the merge was sound -- a model that says
# nothing scores better than one that discloses.  Grading replaces the
# presence test with a content test.
#
# Three invariants shape the tables below:
#   1. ALLOWLIST, NOT BLOCKLIST.  Anything unrecognised stays critical, so a
#      future model's new phrasing fails closed rather than slipping through.
#   2. MACHINE WARNINGS ARE NEVER GRADED.  collapse_redundant_hierarchy.py
#      delivers an unsafe-promotion refusal by appending "promote_safety:..."
#      to this same list; it is a safety verdict, not a self-disclosure.
#   3. THE CONFLICT SCREEN RUNS FIRST, so a warning that looks benign but
#      also admits a conflict stays critical.

# Hypernym / partial-overlap vocabulary.  The shipped WARNING_PATTERNS table
# matches "broader than" but not "slightly broader (...)", and hypernym merging
# is exactly the failure mode the C13F audit froze 143 nodes over.
#
# Applied to the warnings LIST ONLY, deliberately.  Measured: adding these to
# the summary prose channel newly condemns 5 of 20 real merges, because a
# summary legitimately uses "broader" to explain why a merge is safe, whereas a
# warning uses it as a flag.  The two channels see different genres of text.
HYPERNYM_WARNING_PATTERNS = tuple(re.compile(p, re.IGNORECASE) for p in (
    r"\b(?:slightly\s+)?(?:broader|narrower)\b",
    r"\boverlapping\b",
    r"\bpartial(?:ly)?\s+overlap\b",
    r"\bsubset\b",
    r"\bsuperset\b",
    r"更(?:宽|窄|上位|下位)",
    r"包含关系",
    r"部分重叠",
))

# The one exemption inside the hypernym screen.  "the dropped qualifier is
# subsumed by the surviving label" asserts containment of a MODIFIER, not that
# two different instruments are the same: merging 培育关键市场主体 into
# 培育市场主体 loses the word 关键, not a policy instrument.  Containment is
# recoverable; partial overlap is not, which is why "overlapping" is absent
# here and stays critical.
SUBSUMPTION_EXEMPT_PATTERNS = tuple(re.compile(p, re.IGNORECASE) for p in (
    r"\bsubsumed\b",
    r"\bsubsumes\b",
    r"\bencompassed\s+by\b",
    r"被.{0,6}涵盖",
    r"已涵盖",
    r"上位标签涵盖",
))

# Benign family 1: the label strings differ, the referent does not.
BENIGN_LEXICAL_WARNING_PATTERNS = tuple(re.compile(p, re.IGNORECASE) for p in (
    r"^minor_[a-z_]+$",
    r"^slight_[a-z_]+",
    r"^labels_not_identical[a-z_]*$",
    # Prose form of the same disclosure as labels_not_identical_strings.
    # Found by replaying RF9's eight criticals: the token form was already
    # allowlisted while this wording fell through to the fail-closed default,
    # so the same statement was graded differently depending on phrasing.
    # "not lexically identical" is a statement about SPELLING, unlike
    # "not equivalent"/"not synonymous", which stay critical.
    r"\bnot\s+lexically\s+identical\b",
    r"\bminor\s+\w+\s+(?:difference|nuance|variation)\b",
    r"\bminor\s+(?:verb|lexical|wording|scope)\b",
    r"\bfunctionally\s+equivalent\b",
    r"\binterchangeable\b",
    r"\burgency\s+nuance\b",
    r"\bnuance\s+not\s+present\b",
    r"\bsubsumed\b",
    r"\bqualifier\b.*\bdropped\b",
    r"措辞(?:微差|差异)",
    r"用词不同",
    r"可互换",
    r"近义",
))

# Benign family 2: factual topology notes.  Their real enforcement lives in
# dedicated codes (CHILD_PLAN_INCOMPLETE, SOURCE_MEMBERSHIP_NOT_REPRESENTED,
# membership conservation), so repeating them here as a rejection reason
# punishes the disclosure while the actual guard is elsewhere.
BENIGN_STRUCTURAL_WARNING_PATTERNS = tuple(re.compile(p, re.IGNORECASE) for p in (
    r"^parent_has_other_children(_not_affected)?$",
    # Both the single and the combined token form: RF9's record 4 wrote
    # "child_has_membership_and_children" while the two halves were separately
    # allowlisted, so the conjunction was graded stricter than either part.
    r"^child_has_(children|membership)(_and_(children|membership))?$",
    r"^child_has_one_child_[A-Za-z0-9_]+$",
    r"^single_child_chain_merge$",
    r"\bhas\s+\d+\s+(?:other\s+)?(?:children|direct\s+member)",
    r"\bchildren\b.*\bmust\s+be\s+reparented\b",
    r"\bremain\s+unaffected\b",
    r"\bmust\s+be\s+accounted\s+for\b",
    r"\bdirect\s+members\b",
))

# C13RF25 (D7-1): context-conditional downgrade, deliberately narrow.
# `source_has_direct_membership` is a factual topology note; its real guard is
# SOURCE_MEMBERSHIP_NOT_REPRESENTED further down, which is untouched.  The plural
# form `source_has_direct_members` is already benign above, so the singular being
# critical is a word-form inconsistency (same defect class as RF9 record 4).
# Exact equality only: `source_has_direct_membership_not_represented_by_target`
# asserts a REAL conflict and must keep failing closed.
CONDITIONAL_MEMBERSHIP_WARNING = "source_has_direct_membership"

MACHINE_WARNING_PREFIXES = ("promote_safety:",)


def _pattern_hit(patterns: Sequence[Any], text: str) -> Optional[str]:
    """Search ``text`` and an underscore-normalised copy of it.

    Warnings arrive both as prose and as tokens such as
    ``labels_not_identical_but_semantically_overlapping``.  ``_`` is a word
    character, so ``\\boverlapping\\b`` never matches inside a token: without
    normalisation the conflict screen would miss every token-form conflict and
    the benign allowlist would then claim it.
    """
    normalised = text.replace("_", " ")
    for pattern in patterns:
        if pattern.search(text) or pattern.search(normalised):
            return pattern.pattern
    return None


def classify_merge_warning(warning: str) -> Tuple[str, str]:
    """Grade one merge warning string.  Returns ``(severity, reason)``."""
    text = str(warning)
    if text.startswith(MACHINE_WARNING_PREFIXES):
        return SEVERITY_CRITICAL, "machine_safety_refusal"

    shipped = _pattern_hit(WARNING_PATTERNS, text)
    if shipped:
        return SEVERITY_CRITICAL, "semantic_conflict"

    hypernym = _pattern_hit(HYPERNYM_WARNING_PATTERNS, text)
    if hypernym and not _pattern_hit(SUBSUMPTION_EXEMPT_PATTERNS, text):
        return SEVERITY_CRITICAL, "hypernym_or_partial_overlap"

    if _pattern_hit(BENIGN_LEXICAL_WARNING_PATTERNS, text):
        return SEVERITY_ADVISORY, "benign_lexical_variance"
    if _pattern_hit(BENIGN_STRUCTURAL_WARNING_PATTERNS, text):
        return SEVERITY_ADVISORY, "benign_structural_disclosure"

    return SEVERITY_CRITICAL, "unrecognised_fail_closed"


def grade_merge_warnings(
    warnings: Sequence[str],
) -> Tuple[List[str], List[str], Dict[str, str]]:
    """Split a warnings list into (critical, advisory, reason-by-warning)."""
    critical: List[str] = []
    advisory: List[str] = []
    reasons: Dict[str, str] = {}
    for warning in warnings:
        severity, reason = classify_merge_warning(warning)
        reasons[str(warning)] = reason
        if severity == SEVERITY_CRITICAL:
            critical.append(str(warning))
        else:
            advisory.append(str(warning))
    return critical, advisory, reasons


def downgrade_conditional_membership_warning(
    critical: Sequence[str],
    advisory: Sequence[str],
    reasons: Dict[str, str],
    relation: Any,
    evidence: Dict[str, Any],
) -> Tuple[List[str], List[str], Dict[str, str]]:
    """C13RF25: move CONDITIONAL_MEMBERSHIP_WARNING critical->advisory, but only
    when the decision context makes the disclosure harmless.  Both conditions are
    required by the card face.  Exact equality means the
    `..._not_represented_by_target` variant can never reach this path."""
    critical_list = [str(w) for w in critical]
    advisory_list = [str(w) for w in advisory]
    if relation != "exact_duplicate":
        return critical_list, advisory_list, reasons
    if evidence.get("target_represents_all_source_members") is not True:
        return critical_list, advisory_list, reasons
    remaining = [w for w in critical_list if w != CONDITIONAL_MEMBERSHIP_WARNING]
    if len(remaining) == len(critical_list):
        return critical_list, advisory_list, reasons
    new_reasons = dict(reasons)
    new_reasons[CONDITIONAL_MEMBERSHIP_WARNING] = "conditional_membership_disclosure"
    return remaining, advisory_list + [CONDITIONAL_MEMBERSHIP_WARNING], new_reasons


class SemanticContractError(RuntimeError):
    """Raised when a candidate fails the semantic publication gate."""


def _clean_id(value: Any) -> str:
    return str(value or "").strip()


def _is_bool(value: Any) -> bool:
    return isinstance(value, bool)


def _path(manager: Any, node_id: str) -> List[Dict[str, str]]:
    if not node_id or not manager.exists(node_id):
        return []
    result: List[Dict[str, str]] = []
    current: Optional[str] = node_id
    seen = set()
    while current:
        if current in seen:
            return []
        seen.add(current)
        node = manager.get_node(current)
        if not node:
            return []
        result.append({
            "node_id": current,
            "label": str(node.get("label", "")),
        })
        current = manager.get_parent_id(current)
    result.reverse()
    return result


def _l1_ancestor(manager: Any, node_id: str) -> Optional[str]:
    current: Optional[str] = node_id
    seen = set()
    while current:
        if current in seen:
            return None
        seen.add(current)
        node = manager.get_node(current)
        if not node:
            return None
        if str(node.get("level", "")).upper() == "L1":
            return current
        current = manager.get_parent_id(current)
    return None


def parse_semantic_decisions(
    payload: Any,
    *,
    expected_count: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Parse the shared response envelope without inventing defaults."""
    errors: List[Dict[str, Any]] = []
    if not isinstance(payload, Mapping):
        return [], [{
            "code": "ENVELOPE_NOT_OBJECT",
            "message": "response JSON must be an object",
        }]
    extra_fields = sorted(set(payload) - ALLOWED_ENVELOPE_FIELDS)
    if extra_fields:
        errors.append({
            "code": "ENVELOPE_FIELDS_UNKNOWN",
            "message": "response object contains unknown fields",
            "fields": extra_fields,
        })
    if "decisions" not in payload:
        return [], [{
            "code": "DECISIONS_MISSING",
            "message": "response object must contain decisions",
        }]
    raw_decisions = payload.get("decisions")
    if not isinstance(raw_decisions, list):
        return [], [{
            "code": "DECISIONS_NOT_LIST",
            "message": "decisions must be a list",
        }]

    decisions: List[Dict[str, Any]] = []
    for index, item in enumerate(raw_decisions):
        if not isinstance(item, Mapping):
            errors.append({
                "code": "DECISION_NOT_OBJECT",
                "message": "every decision must be an object",
                "decision_index": index,
            })
            continue
        decisions.append(dict(item))

    if expected_count is not None and len(raw_decisions) != expected_count:
        errors.append({
            "code": "DECISION_COUNT_MISMATCH",
            "message": f"expected {expected_count} decision(s), got {len(raw_decisions)}",
        })
    return decisions, errors


def membership_counts_from_level_maps(
    level_maps: Mapping[str, Mapping[str, Sequence[Any]]],
) -> Dict[str, int]:
    counts: Counter[str] = Counter()
    for node_map in level_maps.values():
        if not isinstance(node_map, Mapping):
            continue
        for raw_node_id, members in node_map.items():
            node_id = _clean_id(raw_node_id)
            if not node_id:
                continue
            if isinstance(members, Sequence) and not isinstance(members, (str, bytes)):
                counts[node_id] += len(members)
    return dict(counts)


def membership_counts_from_rows(
    rows: Sequence[Mapping[str, Any]],
    node_field: str = "final_node_id",
) -> Dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        node_id = _clean_id(row.get(node_field))
        if node_id:
            counts[node_id] += 1
    return dict(counts)


def resolve_membership_counts(
    counts: Mapping[str, int],
    lineage: Optional[Mapping[str, str]],
    live_node_ids: Iterable[str],
) -> Dict[str, int]:
    """Aggregate direct memberships through the current closed lineage.

    C13RF22: a count whose node resolves to nothing live used to be dropped
    silently, which quietly *loosened* two gates rather than tightening them --
    `SOURCE_MEMBERSHIP_NOT_REPRESENTED` only fires when the count is > 0, and
    `FLATTEN_SOURCE_HAS_MEMBERSHIP` requires it to be 0, so a count lost to a
    lineage gap made both look satisfied.  An unresolvable count means the
    lineage we were handed is incomplete, which is the caller's bug; refuse it
    here instead of deciding merges against numbers we know are wrong.
    """
    closed = close_lineage(lineage or {})
    live = {_clean_id(node_id) for node_id in live_node_ids}
    resolved: Counter[str] = Counter()
    unresolved: Dict[str, str] = {}
    for raw_node_id, raw_count in counts.items():
        node_id = _clean_id(raw_node_id)
        target = closed.get(node_id, node_id)
        if target in live:
            resolved[target] += int(raw_count)
        else:
            unresolved[node_id] = target
    if unresolved:
        detail = ", ".join(
            f"{source} -> {target}" if source != target else source
            for source, target in sorted(unresolved.items())[:5]
        )
        raise LineageError(
            f"{len(unresolved)} membership count(s) resolve to nodes that are not live: "
            f"{detail}. The lineage passed in does not cover every deleted node, so "
            f"membership-dependent gates would be evaluated against undercounted "
            f"totals. Supply the complete upstream lineage instead of dropping rows."
        )
    return dict(resolved)


def build_semantic_context(
    manager: Any,
    decision: Mapping[str, Any],
    *,
    direct_membership_counts: Optional[Mapping[str, int]],
    membership_known: bool,
    allow_cross_l1: bool,
    allowed_l1_id: Optional[str] = None,
) -> Dict[str, Any]:
    source_id = _clean_id(decision.get("source_id"))
    target_id = _clean_id(decision.get("target_id"))
    source_exists = bool(source_id and manager.exists(source_id))
    target_exists = bool(target_id and manager.exists(target_id))
    source_children = (
        [_clean_id(child.get("node_id")) for child in manager.get_children(source_id)]
        if source_exists else []
    )
    target_children = (
        [_clean_id(child.get("node_id")) for child in manager.get_children(target_id)]
        if target_exists else []
    )

    plan_targets: Dict[str, Dict[str, Any]] = {}
    raw_plan = decision.get("child_plan")
    if isinstance(raw_plan, list):
        for item in raw_plan:
            if not isinstance(item, Mapping):
                continue
            child_id = _clean_id(item.get("child_id"))
            plan_target_id = _clean_id(item.get("target_parent_id"))
            if not plan_target_id or plan_target_id in plan_targets:
                continue
            exists = manager.exists(plan_target_id)
            plan_targets[plan_target_id] = {
                "exists": exists,
                "l1_id": _l1_ancestor(manager, plan_target_id) if exists else None,
                "path": _path(manager, plan_target_id) if exists else [],
                "is_descendant_of_child": bool(
                    exists
                    and child_id
                    and manager.exists(child_id)
                    and manager.is_descendant(plan_target_id, child_id)
                ),
            }

    source_parent_id = manager.get_parent_id(source_id) if source_exists else None
    source_l1_id = _l1_ancestor(manager, source_id) if source_exists else None
    target_l1_id = _l1_ancestor(manager, target_id) if target_exists else None
    if decision.get("action") == "create_bridge" and not target_exists:
        target_l1_id = source_l1_id

    source_membership_count: Optional[int]
    if membership_known and direct_membership_counts is not None:
        source_membership_count = int(direct_membership_counts.get(source_id, 0))
    else:
        source_membership_count = None

    return {
        "source_id": source_id,
        "target_id": target_id,
        "source_exists": source_exists,
        "target_exists": target_exists,
        "source_parent_id": source_parent_id,
        "source_child_ids": source_children,
        "target_child_ids": target_children,
        "source_direct_membership_count": source_membership_count,
        "membership_known": bool(membership_known),
        "source_l1_id": source_l1_id,
        "target_l1_id": target_l1_id,
        "allowed_l1_id": _clean_id(allowed_l1_id) or None,
        "allow_cross_l1": bool(allow_cross_l1),
        "target_is_descendant_of_source": bool(
            source_exists
            and target_exists
            and manager.is_descendant(target_id, source_id)
        ),
        "source_path": _path(manager, source_id),
        "target_path": _path(manager, target_id),
        "plan_targets": plan_targets,
    }


def _warning_in_summary(summary: str) -> Optional[str]:
    for pattern in WARNING_PATTERNS:
        match = pattern.search(summary)
        if match:
            return match.group(0)
    return None


def validate_semantic_decision(
    decision: Any,
    context: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    violations: List[Dict[str, Any]] = []

    def add(
        code: str,
        message: str,
        _severity: str = SEVERITY_CRITICAL,
        **details: Any,
    ) -> None:
        # Severity defaults to critical so every existing call site keeps its
        # meaning; only the graded merge-warning path passes anything else.
        item: Dict[str, Any] = {
            "severity": _severity,
            "code": code,
            "message": message,
        }
        if details:
            item["context"] = details
        violations.append(item)

    if not isinstance(decision, Mapping):
        add("DECISION_NOT_OBJECT", "semantic decision must be an object")
        return _validation_report(violations)

    missing_fields = sorted(REQUIRED_DECISION_FIELDS - set(decision))
    if missing_fields:
        add(
            "DECISION_FIELDS_MISSING",
            "semantic decision is missing required fields",
            fields=missing_fields,
        )
    extra_fields = sorted(set(decision) - ALLOWED_DECISION_FIELDS)
    if extra_fields:
        add(
            "DECISION_FIELDS_UNKNOWN",
            "semantic decision contains unknown fields",
            fields=extra_fields,
        )

    relation = decision.get("relation")
    action = decision.get("action")
    source_id = _clean_id(decision.get("source_id"))
    target_id = _clean_id(decision.get("target_id"))
    new_label = decision.get("new_label")
    confidence = decision.get("confidence")

    if relation not in RELATIONS:
        add("RELATION_UNKNOWN", "relation is missing or unknown", value=relation)
    if action not in ACTIONS:
        add("ACTION_UNKNOWN", "action is missing or unknown", value=action)
    if not source_id:
        add("SOURCE_ID_MISSING", "source_id must be non-empty")
    if not isinstance(target_id, str) or not target_id:
        add("TARGET_ID_MISSING", "target_id must be non-empty")
    if new_label is not None and not isinstance(new_label, str):
        add("NEW_LABEL_INVALID", "new_label must be a string or null")
    if not isinstance(confidence, (int, float)) or isinstance(confidence, bool):
        add("CONFIDENCE_INVALID", "confidence must be a number from 0 to 1")
    elif not 0.0 <= float(confidence) <= 1.0:
        add("CONFIDENCE_INVALID", "confidence must be a number from 0 to 1")

    evidence = decision.get("evidence")
    if not isinstance(evidence, Mapping):
        add("EVIDENCE_NOT_OBJECT", "evidence must be an object")
        evidence = {}
    else:
        missing_evidence = sorted(REQUIRED_EVIDENCE_FIELDS - set(evidence))
        if missing_evidence:
            add(
                "EVIDENCE_FIELDS_MISSING",
                "evidence is missing required fields",
                fields=missing_evidence,
            )
        extra_evidence = sorted(set(evidence) - ALLOWED_EVIDENCE_FIELDS)
        if extra_evidence:
            add(
                "EVIDENCE_FIELDS_UNKNOWN",
                "evidence contains unknown fields",
                fields=extra_evidence,
            )

    summary = evidence.get("summary")
    warnings = evidence.get("warnings")
    membership_basis = evidence.get("membership_basis")
    if not isinstance(summary, str) or not summary.strip():
        add("EVIDENCE_SUMMARY_MISSING", "evidence.summary must be non-empty")
        summary = ""
    if not isinstance(warnings, list) or any(
        not isinstance(item, str) or not item.strip() for item in warnings
    ):
        add("EVIDENCE_WARNINGS_INVALID", "evidence.warnings must be a list of strings")
        warnings = []
    if not isinstance(membership_basis, str):
        add("MEMBERSHIP_BASIS_INVALID", "evidence.membership_basis must be a string")
        membership_basis = ""
    for field in (
        "target_represents_all_source_members",
        "pure_structural_redundancy",
        "cross_l1_authorized",
    ):
        if not _is_bool(evidence.get(field)):
            add("EVIDENCE_FLAG_INVALID", f"evidence.{field} must be boolean", field=field)

    raw_child_plan = decision.get("child_plan")
    child_plan: List[Mapping[str, Any]] = []
    if not isinstance(raw_child_plan, list):
        add("CHILD_PLAN_NOT_LIST", "child_plan must be a list")
    else:
        for index, item in enumerate(raw_child_plan):
            if not isinstance(item, Mapping):
                add(
                    "CHILD_PLAN_ITEM_NOT_OBJECT",
                    "every child plan item must be an object",
                    child_index=index,
                )
                continue
            child_plan.append(item)
            missing_child = sorted(REQUIRED_CHILD_PLAN_FIELDS - set(item))
            if missing_child:
                add(
                    "CHILD_PLAN_FIELDS_MISSING",
                    "child plan item is missing required fields",
                    child_index=index,
                    fields=missing_child,
                )
            extra_child = sorted(set(item) - ALLOWED_CHILD_PLAN_FIELDS)
            if extra_child:
                add(
                    "CHILD_PLAN_FIELDS_UNKNOWN",
                    "child plan item contains unknown fields",
                    child_index=index,
                    fields=extra_child,
                )
            if not _clean_id(item.get("child_id")):
                add("CHILD_ID_MISSING", "child_id must be non-empty", child_index=index)
            if item.get("disposition") not in CHILD_DISPOSITIONS:
                add(
                    "CHILD_DISPOSITION_UNKNOWN",
                    "child disposition is unknown",
                    child_index=index,
                    value=item.get("disposition"),
                )
            if not _clean_id(item.get("target_parent_id")):
                add(
                    "CHILD_TARGET_MISSING",
                    "target_parent_id must be non-empty",
                    child_index=index,
                )
            if item.get("relation") not in RELATIONS:
                add(
                    "CHILD_RELATION_UNKNOWN",
                    "child relation is missing or unknown",
                    child_index=index,
                    value=item.get("relation"),
                )
            if not _is_bool(item.get("same_domain")):
                add(
                    "CHILD_SAME_DOMAIN_INVALID",
                    "child same_domain must be boolean",
                    child_index=index,
                )
            if not isinstance(item.get("evidence"), str) or not item.get("evidence", "").strip():
                add(
                    "CHILD_EVIDENCE_MISSING",
                    "child evidence must be non-empty",
                    child_index=index,
                )

    if action == "merge" and relation not in MERGE_RELATIONS:
        add(
            "DESTRUCTIVE_MERGE_RELATION_FORBIDDEN",
            "only exact_duplicate or synonym may authorize merge",
            relation=relation,
        )
    if action == "uncertain" and relation != "uncertain":
        add("UNCERTAIN_ACTION_CONFLICT", "uncertain action requires uncertain relation")
    if relation == "uncertain" and action not in {"uncertain", "keep", "reject_merge"}:
        add("UNCERTAIN_RELATION_CONFLICT", "uncertain relation cannot authorize mutation")

    if context is None or not isinstance(context, Mapping):
        add("SEMANTIC_CONTEXT_MISSING", "semantic context is required")
        return _validation_report(violations)

    if _clean_id(context.get("source_id")) != source_id:
        add("CONTEXT_SOURCE_MISMATCH", "context source_id does not match decision")
    if _clean_id(context.get("target_id")) != target_id:
        add("CONTEXT_TARGET_MISMATCH", "context target_id does not match decision")

    source_exists = context.get("source_exists") is True
    target_exists = context.get("target_exists") is True
    source_children = [str(item) for item in context.get("source_child_ids", [])]
    source_parent_id = _clean_id(context.get("source_parent_id"))
    source_l1 = _clean_id(context.get("source_l1_id"))
    target_l1 = _clean_id(context.get("target_l1_id"))
    allowed_l1 = _clean_id(context.get("allowed_l1_id"))
    allow_cross_l1 = context.get("allow_cross_l1") is True

    if not source_exists:
        add("SOURCE_NOT_FOUND", "source node does not exist", source_id=source_id)
    if action != "create_bridge" and not target_exists:
        add("TARGET_NOT_FOUND", "target node does not exist", target_id=target_id)

    if allowed_l1 and source_l1 != allowed_l1:
        add("SOURCE_OUTSIDE_ALLOWED_L1", "source is outside the audited L1")
    if allowed_l1 and action != "create_bridge" and target_l1 != allowed_l1:
        add("TARGET_OUTSIDE_ALLOWED_L1", "target is outside the audited L1")

    if action == "merge":
        if source_id == target_id:
            add("MERGE_INTO_SELF", "merge source and target must differ")
        if source_l1 != target_l1:
            add("CROSS_L1_MERGE_FORBIDDEN", "destructive merge may not cross L1")
        if warnings:
            # C13RF17: graded by content instead of by presence.  The code name
            # MERGE_WARNING_CONFLICT is deliberately unchanged so historical
            # reports stay directly comparable; what changed is that it now
            # fires only for warnings that actually assert a conflict.
            critical_warnings, advisory_warnings, warning_reasons = (
                grade_merge_warnings(warnings)
            )
            critical_warnings, advisory_warnings, warning_reasons = (
                downgrade_conditional_membership_warning(
                    critical_warnings, advisory_warnings, warning_reasons,
                    relation, evidence,
                )
            )
            if critical_warnings:
                add(
                    "MERGE_WARNING_CONFLICT",
                    "merge evidence contains conflicting warning flags",
                    warnings=list(critical_warnings),
                    advisory_warnings=list(advisory_warnings),
                    all_warnings=list(warnings),
                    warning_reasons=dict(warning_reasons),
                )
            else:
                add(
                    "MERGE_WARNING_ADVISORY",
                    "merge evidence discloses non-conflicting caveats",
                    _severity=SEVERITY_ADVISORY,
                    warnings=list(advisory_warnings),
                    warning_reasons=dict(warning_reasons),
                )
        # Parallel prose channel.  Its trigger vocabulary is deliberately left
        # as shipped: RF17 measured that extending the hypernym patterns to
        # summaries newly condemns 5 of 20 real merges, because a summary uses
        # "broader" to explain why a merge is safe while a warning uses it as a
        # flag.  Both channels now route through classify_merge_warning so they
        # agree on what the shipped vocabulary means, and both are exercised by
        # the same four-quadrant tests.
        warning_text = _warning_in_summary(summary)
        if warning_text:
            add(
                "MERGE_WARNING_TEXT_CONFLICT",
                "merge evidence text contains a semantic warning",
                matched=warning_text,
                severity_reason=classify_merge_warning(warning_text)[1],
            )
        member_count = context.get("source_direct_membership_count")
        if not context.get("membership_known") or not isinstance(member_count, int):
            add(
                "SOURCE_MEMBERSHIP_UNKNOWN",
                "merge requires known direct source membership",
            )
        elif member_count > 0:
            if evidence.get("target_represents_all_source_members") is not True:
                add(
                    "SOURCE_MEMBERSHIP_NOT_REPRESENTED",
                    "merge target must explicitly represent every direct source member",
                    source_direct_membership_count=member_count,
                )
            if not membership_basis.strip():
                add(
                    "SOURCE_MEMBERSHIP_BASIS_MISSING",
                    "merge with direct membership requires a non-empty membership basis",
                )
        if context.get("target_is_descendant_of_source"):
            if source_children != [target_id]:
                add(
                    "ANCESTOR_MERGE_NOT_SINGLE_CHILD",
                    "a parent may merge into its child only on an exact single-child chain",
                )
            _require_complete_child_plan(
                child_plan,
                source_children,
                target_parent_id=source_parent_id,
                dispositions={"move"},
                relation_whitelist=CHILD_COMPATIBLE_RELATIONS,
                add=add,
            )
        else:
            _require_complete_child_plan(
                child_plan,
                source_children,
                target_parent_id=target_id,
                dispositions={"move"},
                relation_whitelist=CHILD_COMPATIBLE_RELATIONS,
                add=add,
            )

    elif action in {"move", "move_across_l1"}:
        if relation != "misplaced":
            add("MOVE_RELATION_CONFLICT", "move requires misplaced relation")
        if source_id == target_id:
            add("MOVE_INTO_SELF", "move source and target must differ")
        if target_id == context.get("source_parent_id"):
            add("MOVE_ALREADY_UNDER_TARGET", "move target is already the source parent")
        if context.get("target_is_descendant_of_source"):
            add("MOVE_TARGET_DESCENDANT", "move target cannot be inside the source subtree")
        is_cross_l1 = bool(source_l1 and target_l1 and source_l1 != target_l1)
        if is_cross_l1:
            if action != "move_across_l1":
                add("CROSS_L1_MOVE_NOT_EXPLICIT", "cross-L1 move requires move_across_l1")
            if not allow_cross_l1 or evidence.get("cross_l1_authorized") is not True:
                add("CROSS_L1_MOVE_FORBIDDEN", "cross-L1 move lacks explicit authorization")
        elif action == "move_across_l1":
            add("CROSS_L1_ACTION_CONFLICT", "move_across_l1 requires different L1 branches")
        _require_complete_child_plan(
            child_plan,
            source_children,
            target_parent_id=source_id,
            dispositions={"retain_under_source"},
            relation_whitelist=CHILD_COMPATIBLE_RELATIONS,
            add=add,
        )

    elif action == "split_reparent":
        if target_id != source_id:
            add(
                "SPLIT_TARGET_CONFLICT",
                "split_reparent target_id must equal source_id so the umbrella is preserved",
            )
        if not source_children:
            add("SPLIT_SOURCE_HAS_NO_CHILDREN", "split_reparent requires direct children")
        _require_complete_child_plan(
            child_plan,
            source_children,
            target_parent_id=None,
            dispositions={"move", "keep"},
            relation_whitelist=CHILD_COMPATIBLE_RELATIONS,
            add=add,
        )
        moved = 0
        plan_targets = context.get("plan_targets", {})
        for index, item in enumerate(child_plan):
            disposition = item.get("disposition")
            plan_target = _clean_id(item.get("target_parent_id"))
            if disposition == "keep":
                if plan_target != source_id:
                    add(
                        "SPLIT_KEEP_TARGET_CONFLICT",
                        "kept child must remain under the split source",
                        child_index=index,
                    )
                continue
            if disposition != "move":
                continue
            moved += 1
            child_id = _clean_id(item.get("child_id"))
            if plan_target == source_id:
                add(
                    "SPLIT_MOVE_TARGET_SOURCE",
                    "moved child target must differ from the split source",
                    child_index=index,
                )
            if plan_target == child_id:
                add(
                    "SPLIT_MOVE_TARGET_SELF",
                    "moved child cannot target itself",
                    child_index=index,
                )
            target_context = plan_targets.get(plan_target, {}) if isinstance(plan_targets, Mapping) else {}
            if target_context.get("exists") is not True:
                add(
                    "SPLIT_TARGET_NOT_FOUND",
                    "split child target does not exist",
                    child_index=index,
                    target_parent_id=plan_target,
                )
                continue
            if target_context.get("is_descendant_of_child") is True:
                add(
                    "SPLIT_TARGET_DESCENDANT",
                    "split child target would create a cycle",
                    child_index=index,
                )
            plan_l1 = _clean_id(target_context.get("l1_id"))
            if source_l1 and plan_l1 and source_l1 != plan_l1:
                if not allow_cross_l1 or evidence.get("cross_l1_authorized") is not True:
                    add(
                        "SPLIT_CROSS_L1_FORBIDDEN",
                        "cross-L1 split target lacks explicit authorization",
                        child_index=index,
                    )
        if moved == 0:
            add("SPLIT_HAS_NO_MOVES", "split_reparent must move at least one child")

    elif action == "flatten":
        if target_id != source_parent_id:
            add("FLATTEN_TARGET_CONFLICT", "flatten target must be the source parent")
        if relation not in CHILD_COMPATIBLE_RELATIONS:
            add(
                "FLATTEN_RELATION_FORBIDDEN",
                "flatten requires duplicate, synonym, or broader/narrower relation",
            )
        member_count = context.get("source_direct_membership_count")
        if not context.get("membership_known") or not isinstance(member_count, int):
            add("FLATTEN_MEMBERSHIP_UNKNOWN", "flatten requires known direct membership")
        elif member_count != 0:
            add(
                "FLATTEN_SOURCE_HAS_MEMBERSHIP",
                "a flattened source must have no direct membership",
                source_direct_membership_count=member_count,
            )
        if evidence.get("pure_structural_redundancy") is not True:
            add(
                "FLATTEN_REDUNDANCY_UNPROVEN",
                "flatten requires explicit proof of pure structural redundancy",
            )
        if not source_children:
            add("FLATTEN_SOURCE_HAS_NO_CHILDREN", "flatten requires at least one child")
        _require_complete_child_plan(
            child_plan,
            source_children,
            target_parent_id=target_id,
            dispositions={"move"},
            relation_whitelist=CHILD_COMPATIBLE_RELATIONS,
            add=add,
        )

    elif action == "create_bridge":
        if relation != "broader_narrower":
            add("BRIDGE_RELATION_CONFLICT", "create_bridge requires broader_narrower relation")
        if not isinstance(new_label, str) or not new_label.strip():
            add("BRIDGE_LABEL_MISSING", "create_bridge requires a non-empty new_label")
        if not child_plan:
            add("BRIDGE_CHILD_PLAN_EMPTY", "create_bridge requires at least one child")
        allowed_children = set(source_children) | {
            str(item) for item in context.get("target_child_ids", [])
        }
        _require_selected_child_plan(
            child_plan,
            allowed_children,
            target_parent_id=target_id,
            dispositions={"move"},
            relation_whitelist=CHILD_COMPATIBLE_RELATIONS,
            add=add,
        )

    elif action == "rename":
        if source_id != target_id:
            add("RENAME_TARGET_CONFLICT", "rename target_id must equal source_id")
        if relation not in MERGE_RELATIONS:
            add("RENAME_RELATION_FORBIDDEN", "rename requires duplicate or synonym relation")
        if not isinstance(new_label, str) or not new_label.strip():
            add("RENAME_LABEL_MISSING", "rename requires a non-empty new_label")
        if child_plan:
            add("RENAME_CHILD_PLAN_NOT_EMPTY", "rename must not include a child plan")

    elif action in {"keep", "reject_merge", "uncertain"}:
        # C13RF8 (D1 option 2): a non-mutating action may carry a child_plan
        # only when every entry is itself a no-op. Any entry proposing an
        # actual change still fails closed. execute_semantic_decision()
        # returns for these three actions before any child_plan is consumed,
        # so a tolerated all-keep plan can never be executed.
        if child_plan and any(
            item.get("disposition") != "keep" for item in child_plan
        ):
            add("NON_MUTATING_CHILD_PLAN_NOT_EMPTY", "non-mutating action must not move children")

    return _validation_report(violations)


def _require_complete_child_plan(
    child_plan: Sequence[Mapping[str, Any]],
    expected_child_ids: Sequence[str],
    *,
    target_parent_id: Optional[str],
    dispositions: set[str],
    relation_whitelist: frozenset[str],
    add: Any,
) -> None:
    actual_ids = [_clean_id(item.get("child_id")) for item in child_plan]
    expected = [str(item) for item in expected_child_ids]
    if len(actual_ids) != len(set(actual_ids)):
        add("CHILD_PLAN_DUPLICATE", "child_plan contains duplicate child IDs")
    if set(actual_ids) != set(expected) or len(actual_ids) != len(expected):
        add(
            "CHILD_PLAN_INCOMPLETE",
            "child_plan must cover every direct child exactly once",
            expected=expected,
            actual=actual_ids,
        )
    _validate_child_items(
        child_plan,
        target_parent_id=target_parent_id,
        dispositions=dispositions,
        relation_whitelist=relation_whitelist,
        add=add,
    )


def _require_selected_child_plan(
    child_plan: Sequence[Mapping[str, Any]],
    allowed_child_ids: set[str],
    *,
    target_parent_id: str,
    dispositions: set[str],
    relation_whitelist: frozenset[str],
    add: Any,
) -> None:
    actual_ids = [_clean_id(item.get("child_id")) for item in child_plan]
    if len(actual_ids) != len(set(actual_ids)):
        add("CHILD_PLAN_DUPLICATE", "child_plan contains duplicate child IDs")
    invalid = sorted(set(actual_ids) - allowed_child_ids)
    if invalid:
        add(
            "CHILD_PLAN_NOT_DIRECT",
            "bridge child must currently belong to the source or existing bridge",
            child_ids=invalid,
        )
    _validate_child_items(
        child_plan,
        target_parent_id=target_parent_id,
        dispositions=dispositions,
        relation_whitelist=relation_whitelist,
        add=add,
    )


def _validate_child_items(
    child_plan: Sequence[Mapping[str, Any]],
    *,
    target_parent_id: Optional[str],
    dispositions: set[str],
    relation_whitelist: frozenset[str],
    add: Any,
) -> None:
    for index, item in enumerate(child_plan):
        if item.get("disposition") not in dispositions:
            add(
                "CHILD_DISPOSITION_CONFLICT",
                "child disposition conflicts with the requested action",
                child_index=index,
                disposition=item.get("disposition"),
            )
        if target_parent_id is not None and _clean_id(item.get("target_parent_id")) != target_parent_id:
            add(
                "CHILD_TARGET_CONFLICT",
                "child target conflicts with the requested action",
                child_index=index,
                expected_target=target_parent_id,
                actual_target=_clean_id(item.get("target_parent_id")),
            )
        if item.get("relation") not in relation_whitelist:
            add(
                "CHILD_RELATION_FORBIDDEN",
                "child is not proven compatible with its destination",
                child_index=index,
                relation=item.get("relation"),
            )
        if item.get("same_domain") is not True:
            add(
                "CHILD_DOMAIN_UNPROVEN",
                "every affected child must be explicitly proven in-domain",
                child_index=index,
            )


def _validation_report(violations: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    counts = Counter(str(item.get("code", "UNKNOWN")) for item in violations)
    # C13RF17: a decision fails on CRITICAL violations only.  Advisories are
    # recorded in full (so the model's caveat stays auditable and countable)
    # but do not reject the decision -- structurally the same move D2b made for
    # deferred records.  `violations` keeps carrying both so that consumers
    # filtering on `severity == "critical"` (harness/verify_stage_v2.py:147)
    # need no change at all.
    criticals = [
        item for item in violations
        if str(item.get("severity")) == SEVERITY_CRITICAL
    ]
    advisories = [
        item for item in violations
        if str(item.get("severity")) == SEVERITY_ADVISORY
    ]
    return {
        "schema_version": DECISION_SCHEMA_VERSION,
        "passed": not criticals,
        "critical_count": len(criticals),
        "advisory_count": len(advisories),
        "violation_counts": dict(sorted(counts.items())),
        "violations": [dict(item) for item in violations],
        "advisories": [dict(item) for item in advisories],
    }


def rejected_parse_record(
    stage: str,
    errors: Sequence[Mapping[str, Any]],
    *,
    logical_call_id: Optional[str] = None,
) -> Dict[str, Any]:
    violations = [
        {
            "severity": "critical",
            "code": str(item.get("code", "PARSE_FAILED")),
            "message": str(item.get("message", "semantic response parse failed")),
            **({"context": dict(item)} if item else {}),
        }
        for item in errors
    ]
    record = {
        "ts": int(time.time()),
        "step": stage,
        "type": "semantic_decision",
        "relation": None,
        "action": None,
        "status": "rejected",
        "message": "semantic response rejected before execution",
        "semantic_contract": _validation_report(violations),
    }
    if logical_call_id is not None:
        record["logical_call_id"] = logical_call_id
    return record


DEFAULT_DEFERRED_SCOPE_MESSAGE = (
    "the proposed restructure is not expressible in this stage"
)


def deferred_restructure_record(
    stage: str,
    *,
    logical_call_id: str,
    local_proposal: Mapping[str, Any],
    resolved_proposal: Mapping[str, Any],
    scope_messages: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Park an inexpressible restructure as a `deferred` operations record.

    C13RF29: ``scope_messages`` replaces a hard-coded sentence.  Until this card
    the record always said "merge with a pair-external child target ...", which
    was 14a/14c's single deferrable message and simply false everywhere else --
    14b defers a depth-ceiling bridge, 14d has three messages of its own, and
    after C13RF29 every stage can also defer `keep`/`reject_merge`/`uncertain`
    and unrecognised actions.  14b and 14d each worked around the wrong text
    differently (14b overwrote ``violations[0]["message"]`` after the fact, 14d
    attached ``deferred_scope_errors``/``deferred_scope_channel`` side fields and
    left a note asking whoever could touch this file to parameterise it).  Both
    workarounds are removed with this parameter.

    One violation is emitted per message so the record states every reason the
    proposal was parked, rather than collapsing them into one sentence.  Callers
    that pass nothing keep a single violation carrying
    ``DEFAULT_DEFERRED_SCOPE_MESSAGE``: an honest generic, not a specific claim
    about a shape the caller may not have seen.
    """
    messages = [
        str(message) for message in (scope_messages or ())
        if str(message).strip()
    ]
    violations = [
        {
            "severity": "critical",
            "code": "DECISION_SCOPE_INEXPRESSIBLE",
            "message": message,
        }
        for message in (messages or [DEFAULT_DEFERRED_SCOPE_MESSAGE])
    ]
    return {
        "ts": int(time.time()),
        "step": stage,
        "type": "semantic_decision",
        "relation": None,
        "action": None,
        "status": "deferred",
        "message": "semantic restructure deferred before execution",
        "logical_call_id": logical_call_id,
        "local_proposal": copy.deepcopy(dict(local_proposal)),
        "resolved_proposal": copy.deepcopy(dict(resolved_proposal)),
        "semantic_contract": _validation_report(violations),
    }


def execute_semantic_decision(
    manager: Any,
    decision: Any,
    *,
    direct_membership_counts: Optional[Mapping[str, int]],
    membership_known: bool,
    stage: str,
    lineage: Optional[Mapping[str, str]] = None,
    allow_cross_l1: bool = False,
    allowed_l1_id: Optional[str] = None,
    new_node_level: Optional[str] = None,
) -> Dict[str, Any]:
    if not isinstance(decision, Mapping):
        validation = validate_semantic_decision(decision, None)
        record = _base_operation_record(stage, {}, {}, validation)
        record.update({
            "status": "rejected",
            "message": ", ".join(validation["violation_counts"]),
        })
        return record

    live_ids = manager.get_all_node_ids()
    resolved_counts: Optional[Dict[str, int]] = None
    if membership_known and direct_membership_counts is not None:
        resolved_counts = resolve_membership_counts(
            direct_membership_counts,
            lineage,
            live_ids,
        )
    context = build_semantic_context(
        manager,
        decision,
        direct_membership_counts=resolved_counts,
        membership_known=membership_known,
        allow_cross_l1=allow_cross_l1,
        allowed_l1_id=allowed_l1_id,
    )
    validation = validate_semantic_decision(decision, context)
    record = _base_operation_record(stage, decision, context, validation)
    if not validation["passed"]:
        record["status"] = "rejected"
        record["message"] = ", ".join(validation["violation_counts"])
        return record

    action = str(decision.get("action"))
    source_id = _clean_id(decision.get("source_id"))
    target_id = _clean_id(decision.get("target_id"))
    new_label = decision.get("new_label")

    if action == "keep":
        record.update({"status": "skipped", "message": "decision keeps current structure"})
        return record
    if action == "reject_merge":
        record.update({"status": "rejected", "message": "destructive merge explicitly rejected"})
        return record
    if action == "uncertain":
        record.update({"status": "rejected", "message": "uncertain decision fails closed"})
        return record

    success = False
    if action == "merge":
        if context.get("target_is_descendant_of_source"):
            success = manager.promote_child_and_remove_parent(target_id, new_label=new_label)
            record["type"] = "promote_child"
        else:
            success = manager.absorb_node(target_id, source_id, new_label=new_label)
        record.update({
            "node_id": source_id,
            "merge_into": target_id,
            "lineage_target": target_id,
        })
    elif action in {"move", "move_across_l1"}:
        success = manager.move_node(source_id, target_id)
        record.update({
            "node_id": source_id,
            "target_parent_id": target_id,
        })
    elif action == "split_reparent":
        moves = [
            {
                "node_id": _clean_id(item.get("child_id")),
                "new_parent_id": _clean_id(item.get("target_parent_id")),
            }
            for item in decision.get("child_plan", [])
            if item.get("disposition") == "move"
            and _clean_id(item.get("target_parent_id")) != source_id
        ]
        success = manager.move_nodes_atomically(moves)
        record["node_id"] = source_id
    elif action == "flatten":
        success = manager.flatten_node(source_id)
        record.update({
            "node_id": source_id,
            "lineage_target": target_id,
        })
    elif action == "create_bridge":
        payload = {
            "node_id": target_id,
            "label": str(new_label).strip(),
            "level": str(new_node_level or ""),
            "children": [],
        }
        child_ids = [
            _clean_id(item.get("child_id"))
            for item in decision.get("child_plan", [])
            if item.get("disposition") == "move"
        ]
        bridge_id = manager.create_or_reuse_bridge(source_id, payload, child_ids)
        success = bridge_id == target_id
        record.update({
            "parent": source_id,
            "parent_id": source_id,
            "bridge": str(new_label).strip(),
            "bridge_id": target_id,
        })
    elif action == "rename":
        success = manager.rename_node(source_id, str(new_label))
        record.update({
            "node_id": source_id,
            "new_label": str(new_label),
        })

    if success:
        record.update({"status": "applied", "message": ""})
        after_id = target_id if action in {"merge", "flatten", "create_bridge"} else source_id
        record["after_path"] = _path(manager, after_id)
    else:
        record.update({
            "status": "rejected",
            "message": manager.last_error or "semantic operation failed",
        })
    return record


def _base_operation_record(
    stage: str,
    decision: Mapping[str, Any],
    context: Mapping[str, Any],
    validation: Mapping[str, Any],
) -> Dict[str, Any]:
    action = decision.get("action")
    operation_type = {
        "merge": "merge",
        "move": "move",
        "move_across_l1": "move",
        "split_reparent": "split_reparent",
        "rename": "rename",
        "flatten": "flatten",
        "create_bridge": "create_bridge",
    }.get(action, "semantic_decision")
    record = {
        "ts": int(time.time()),
        "step": stage,
        "type": operation_type,
        "op": action,
        "relation": decision.get("relation"),
        "action": action,
        "source_id": _clean_id(decision.get("source_id")),
        "target_id": _clean_id(decision.get("target_id")),
        "new_label": decision.get("new_label"),
        "confidence": decision.get("confidence"),
        "evidence": decision.get("evidence"),
        "child_plan": decision.get("child_plan"),
        "semantic_context": dict(context),
        "semantic_contract": dict(validation),
        "before_path": context.get("source_path", []),
    }
    if decision.get("operation_id") is not None:
        record["operation_id"] = decision.get("operation_id")
    return record


def validate_semantic_history(
    operations: Optional[Sequence[Mapping[str, Any]]],
) -> Dict[str, Any]:
    violations: List[Dict[str, Any]] = []
    applied_semantic = 0
    rejected_semantic = 0

    def add(code: str, message: str, **details: Any) -> None:
        item: Dict[str, Any] = {
            "severity": "critical",
            "code": code,
            "message": message,
        }
        if details:
            item["context"] = details
        violations.append(item)

    for index, operation in enumerate(operations or []):
        if not isinstance(operation, Mapping):
            continue
        operation_type = str(operation.get("type") or operation.get("op") or "").lower()
        status = str(operation.get("status", "")).lower()
        if operation_type not in SEMANTIC_OPERATION_TYPES:
            continue
        if status not in {"applied", "success"}:
            rejected_semantic += 1
            continue

        applied_semantic += 1
        context = operation.get("semantic_context")
        recorded = operation.get("semantic_contract")
        if not isinstance(context, Mapping):
            add(
                "APPLIED_SEMANTIC_CONTEXT_MISSING",
                "applied semantic operation lacks its pre-operation context",
                operation_index=index,
                operation_type=operation_type,
            )
            continue
        if not isinstance(recorded, Mapping) or recorded.get("passed") is not True:
            add(
                "APPLIED_SEMANTIC_APPROVAL_MISSING",
                "applied semantic operation lacks a passing decision report",
                operation_index=index,
                operation_type=operation_type,
            )

        decision = {
            field: operation.get(field)
            for field in REQUIRED_DECISION_FIELDS
        }
        report = validate_semantic_decision(decision, context)
        if not report["passed"]:
            add(
                "APPLIED_SEMANTIC_DECISION_INVALID",
                "applied operation fails deterministic semantic revalidation",
                operation_index=index,
                operation_type=operation_type,
                violation_counts=report["violation_counts"],
            )
        action = str(operation.get("action") or "")
        allowed_types = ACTION_OPERATION_TYPES.get(action, frozenset())
        if operation_type not in allowed_types:
            add(
                "SEMANTIC_ACTION_TYPE_CONFLICT",
                "operation type does not match its semantic action",
                operation_index=index,
                operation_type=operation_type,
                action=action,
            )

    counts = Counter(str(item["code"]) for item in violations)
    return {
        "schema_version": PUBLICATION_SCHEMA_VERSION,
        "generated_at_unix": int(time.time()),
        "passed": not violations,
        "critical_count": len(violations),
        "violation_counts": dict(sorted(counts.items())),
        "stats": {
            "applied_semantic_operations_checked": applied_semantic,
            "non_applied_semantic_operations_seen": rejected_semantic,
        },
        "violations": violations,
    }


__all__ = [
    "ACTIONS",
    "DECISION_SCHEMA_VERSION",
    "MERGE_RELATIONS",
    "PUBLICATION_SCHEMA_VERSION",
    "RELATIONS",
    "SEVERITY_ADVISORY",
    "SEVERITY_CRITICAL",
    "SemanticContractError",
    "DEFAULT_DEFERRED_SCOPE_MESSAGE",
    "classify_merge_warning",
    "grade_merge_warnings",
    "build_semantic_context",
    "deferred_restructure_record",
    "execute_semantic_decision",
    "membership_counts_from_level_maps",
    "membership_counts_from_rows",
    "parse_semantic_decisions",
    "rejected_parse_record",
    "resolve_membership_counts",
    "validate_semantic_decision",
    "validate_semantic_history",
]
