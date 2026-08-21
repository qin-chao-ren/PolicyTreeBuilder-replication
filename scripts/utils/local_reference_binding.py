from __future__ import annotations

import copy
import hashlib
import itertools
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


PROTOCOL_VERSION = "semantic-local-ref-v1"
LOCAL_SCHEMA_VERSION = "semantic-local-ref-envelope-v1"
LOCAL_REF_PATTERN = re.compile(r"^[A-Z][A-Z0-9_]{0,63}$")
RELATIONS = {
    "exact_duplicate", "synonym", "broader_narrower", "related",
    "complementary", "means_goal", "carrier_outcome", "misplaced",
    "uncertain",
}
ACTIONS = {
    "merge", "keep", "reject_merge", "move", "move_across_l1",
    "split_reparent", "uncertain", "rename", "flatten", "create_bridge",
}
CHILD_DISPOSITIONS = {"move", "keep", "retain_under_source"}
# --- Real node-ID grammar (C13RF21) -----------------------------------------
# The generators are the authority on this shape, not intuition:
#   * base IDs      ``scripts/common_id.py:make_node_id`` -> ``L{1-4}_N{md5[:8]}``
#   * bridge IDs    ``balance_tree_structure.py:generate_bridge_id``
#                   -> ``{parent}_BR_{md5[:6]}``, appendable more than once
# A byte-level census of every reachable tree (353 frozen, 268, 250-node
# scaffold; 445 distinct identifiers) found exactly one further shape not
# produced by today's generators but present in the still-live frozen 353
# baseline: a 4-hex group appended bare after a bridge segment, e.g.
# ``L1_N6be998d8_BR_8df89b_f5a4``.  It predates this repository and must stay
# covered -- narrowing past it would convert a false-positive fix into a
# missed detection.  See ``experiments/2026-08-19_c13rf21_raw_id_shape``.
_REAL_NODE_ID_BODY = (
    r"(?:ROOT|L[1-4]_N[0-9a-f]{8})"          # base identity
    r"(?:(?:_BR_[0-9a-f]{6})+(?:_[0-9a-f]{4})?)?"
    # Bridge segments chain, and the census found the bare 4-hex tail only ever
    # after at least one ``_BR_`` segment -- never directly on a base ID.  The
    # tail is therefore bound to the bridge chain rather than left dangling, so
    # the grammar stays exactly as wide as the observed identifier population.
)
REAL_NODE_ID_PATTERN = re.compile(rf"^{_REAL_NODE_ID_BODY}$")

# Tokenization-only grammar: intentionally the wide pre-RF21 shape.  Used to
# split prompt text into complete identifier tokens, never to decide whether a
# value leaked.  See ``_token_matches`` for why widening is the safe direction
# here and the narrow direction is safe for detection.
_REAL_NODE_ID_WIDE_BODY = (
    r"(?:ROOT|L[1-4]_[A-Za-z0-9][A-Za-z0-9_.:-]{1,126}[A-Za-z0-9])"
)
REAL_NODE_ID_TOKENIZATION_PATTERN = re.compile(rf"^{_REAL_NODE_ID_WIDE_BODY}$")
REAL_NODE_ID_TOKEN_PATTERN = re.compile(
    rf"(?<![A-Za-z0-9_]){_REAL_NODE_ID_WIDE_BODY}(?![A-Za-z0-9_])"
)
# Model-response and prompt safety is deliberately stricter than ref
# tokenization.  A model must not hide an otherwise valid node-ID shape by
# gluing ordinary characters around it (for example ``xL1_N762e6689y``).
# Exact context discovery continues to use REAL_NODE_ID_TOKEN_PATTERN above.
REAL_NODE_ID_EMBEDDED_PATTERN = re.compile(_REAL_NODE_ID_BODY)

# The pre-RF21 grammar treated ``L{1-4}_`` followed by any ordinary characters
# as an identifier, so natural-language phrases such as
# ``different_parent_L1_domains`` were hard-rejected as leaked IDs -- that
# false positive stopped the RF20 online run at 14c call 97.  The legacy shape
# is retained purely as a non-blocking telemetry signal so that narrowing
# remains observable in production rather than silent.
REAL_NODE_ID_LEGACY_WIDE_PATTERN = re.compile(_REAL_NODE_ID_WIDE_BODY)
_LOGICAL_CALL_SEQUENCE = itertools.count(1)

LOCAL_DECISION_FIELDS = {
    "relation",
    "action",
    "source_ref",
    "target_ref",
    "new_label",
    "confidence",
    "evidence",
    "child_plan",
}
LOCAL_CHILD_FIELDS = {
    "child_ref",
    "disposition",
    "target_parent_ref",
    "relation",
    "same_domain",
    "evidence",
}
BOUND_DECISION_FIELDS = {
    "relation",
    "action",
    "source_id",
    "target_id",
    "new_label",
    "confidence",
    "evidence",
    "child_plan",
}
BOUND_CHILD_FIELDS = {
    "child_id",
    "disposition",
    "target_parent_id",
    "relation",
    "same_domain",
    "evidence",
}
FORBIDDEN_ID_FIELDS = {
    "source_id",
    "target_id",
    "child_id",
    "target_parent_id",
}
DECISION_SEMANTIC_FIELDS = (
    "relation",
    "action",
    "new_label",
    "confidence",
    "evidence",
)
CHILD_SEMANTIC_FIELDS = (
    "disposition",
    "relation",
    "same_domain",
    "evidence",
)
# A stage sets this context flag when the decision's source node has no direct
# children at all.  Every ``child_ref`` the model could name is then a
# non-source child, so no per-item repair exists and the only correct plan is
# the empty one -- which the ordinary drift guard reads as a semantic edit.
# C13RF13 call 25 died in exactly that deadlock; the flag authorises the repair
# round to truncate that decision's ``child_plan`` and nothing else.
VOID_CHILD_PLAN_FLAG = "child_plan_must_be_empty"

LOCAL_SCHEMA_PATH = (
    Path(__file__).resolve().parents[2]
    / "schemas"
    / "local_reference_semantic_tree_decision.schema.json"
)
EVIDENCE_REQUIRED_FIELDS = {
    "summary",
    "warnings",
    "target_represents_all_source_members",
    "membership_basis",
    "pure_structural_redundancy",
    "cross_l1_authorized",
}


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def canonical_payload_sha256(value: Any) -> str:
    """Public canonical digest used by stage and independent audit records."""

    return _sha256(value)


def _token_pattern(value: str) -> re.Pattern[str]:
    return re.compile(
        rf"(?<![A-Za-z0-9_]){re.escape(value)}(?![A-Za-z0-9_])"
    )


def _token_matches(text: str, value: str) -> List[re.Match[str]]:
    """Return exact complete-token matches for one candidate identifier.

    Node IDs allow ``.``, ``:``, and ``-`` internally.  Ordinary word
    boundaries would therefore treat ``L2_ABC`` as displayed inside the
    longer token ``L2_ABC.DEF``.  Tokenize with the full node-ID grammar first
    and then compare the complete match; non-node virtual literals retain the
    ordinary exact-token rule.
    """

    # Tokenization and leak detection pull in OPPOSITE directions, so they use
    # different grammars on purpose (C13RF21).  Leak detection must be narrow:
    # a grammar wider than the real identifier population hard-rejects ordinary
    # prose (``different_parent_L1_domains`` stopped the RF20 run).  Context
    # discovery must stay wide: tokenizing greedily is what stops a short
    # candidate from being inferred as "displayed" out of the middle of a longer
    # identifier that uses the contract's ``.:-`` continuation characters.
    # Narrowing this call site too would silently reopen that prefix-inference
    # hole, so it keeps the pre-RF21 wide shape.
    if REAL_NODE_ID_TOKENIZATION_PATTERN.fullmatch(value):
        return [
            match
            for match in REAL_NODE_ID_TOKEN_PATTERN.finditer(text)
            if match.group(0) == value
        ]
    return list(_token_pattern(value).finditer(text))


def _token_position(text: str, value: str) -> Optional[int]:
    matches = _token_matches(text, value)
    return matches[0].start() if matches else None


def _replace_exact_token(text: str, value: str, replacement: str) -> str:
    matches = _token_matches(text, value)
    if not matches:
        return text
    pieces: List[str] = []
    cursor = 0
    for match in matches:
        pieces.append(text[cursor:match.start()])
        pieces.append(replacement)
        cursor = match.end()
    pieces.append(text[cursor:])
    return "".join(pieces)


def _looks_like_real_node_id(value: str) -> bool:
    return bool(REAL_NODE_ID_PATTERN.fullmatch(str(value)))


def _error(code: str, message: str, **context: Any) -> Dict[str, Any]:
    return {"code": code, "message": message, "context": context}


@dataclass(frozen=True)
class LocalReferenceEntry:
    ref: str
    node_id: str
    kind: str = "node"

    def audit_dict(self) -> Dict[str, str]:
        return {"ref": self.ref, "node_id": self.node_id, "kind": self.kind}


@dataclass(frozen=True)
class LocalReferenceContext:
    task: str
    context_token: str
    context_sha256: str
    contract_sha256: str
    schema_version: str
    schema_sha256: str
    runtime_contract_sha256: str
    system_sha256: str
    expected_count: Optional[int]
    known_node_ids: Tuple[str, ...]
    known_node_ids_sha256: str
    entries: Tuple[LocalReferenceEntry, ...]
    localized_user: str
    model_user: str

    @property
    def mapping(self) -> Dict[str, str]:
        return {entry.ref: entry.node_id for entry in self.entries}

    @property
    def kinds(self) -> Dict[str, str]:
        return {entry.ref: entry.kind for entry in self.entries}

    def audit_dict(self) -> Dict[str, Any]:
        return {
            "protocol_version": PROTOCOL_VERSION,
            "task": self.task,
            "context_token": self.context_token,
            "context_sha256": self.context_sha256,
            "contract_sha256": self.contract_sha256,
            "schema_version": self.schema_version,
            "schema_sha256": self.schema_sha256,
            "runtime_contract_sha256": self.runtime_contract_sha256,
            "system_sha256": self.system_sha256,
            "contract_descriptor": {
                "protocol_version": PROTOCOL_VERSION,
                "schema_version": self.schema_version,
                "schema_sha256": self.schema_sha256,
                "runtime_contract_sha256": self.runtime_contract_sha256,
                "system_sha256": self.system_sha256,
                "expected_count": self.expected_count,
            },
            "expected_count": self.expected_count,
            "known_node_id_count": len(self.known_node_ids),
            "known_node_ids_sha256": self.known_node_ids_sha256,
            "ref_mapping": [entry.audit_dict() for entry in self.entries],
            "localized_user": self.localized_user,
        }


@dataclass
class BindingResult:
    ok: bool
    bound_payload: Optional[Dict[str, Any]]
    errors: List[Dict[str, Any]]
    local_payload: Optional[Dict[str, Any]] = None


def _schema_sha256() -> str:
    try:
        payload = LOCAL_SCHEMA_PATH.read_bytes()
    except OSError as exc:
        raise RuntimeError(
            f"local-reference schema is unavailable: {LOCAL_SCHEMA_PATH}"
        ) from exc
    return hashlib.sha256(payload).hexdigest()


def _runtime_contract_descriptor() -> Dict[str, Any]:
    """Return every runtime-only constraint that changes binding safety."""

    return {
        "local_ref_pattern": LOCAL_REF_PATTERN.pattern,
        "real_node_id_pattern": REAL_NODE_ID_PATTERN.pattern,
        "real_node_id_token_pattern": REAL_NODE_ID_TOKEN_PATTERN.pattern,
        "real_node_id_embedded_pattern": REAL_NODE_ID_EMBEDDED_PATTERN.pattern,
        "decision_fields": sorted(LOCAL_DECISION_FIELDS),
        "child_fields": sorted(LOCAL_CHILD_FIELDS),
        "bound_decision_fields": sorted(BOUND_DECISION_FIELDS),
        "bound_child_fields": sorted(BOUND_CHILD_FIELDS),
        "forbidden_id_fields": sorted(FORBIDDEN_ID_FIELDS),
        "decision_semantic_fields": list(DECISION_SEMANTIC_FIELDS),
        "child_semantic_fields": list(CHILD_SEMANTIC_FIELDS),
        "evidence_required_fields": sorted(EVIDENCE_REQUIRED_FIELDS),
        "relation_enum": sorted(RELATIONS),
        "action_enum": sorted(ACTIONS),
        "child_disposition_enum": sorted(CHILD_DISPOSITIONS),
        "confidence_range": [0, 1],
        "evidence_nonempty_summary": True,
        "evidence_unique_nonempty_warnings": True,
        "child_evidence_nonempty": True,
        "exact_ref_matching": True,
        "case_sensitive_refs": True,
        "known_node_id_matching": "literal_substring",
        "unknown_node_id_matching": "embedded_pattern",
        "model_response_real_id_policy": "reject_all_fields",
        "raw_response_real_id_policy": "reject_all_attempts",
        "fuzzy_corrections": 0,
    }


def _runtime_contract_sha256() -> str:
    return _sha256(_runtime_contract_descriptor())


def _contract_descriptor(
    system: str,
    expected_count: Optional[int],
) -> Dict[str, Any]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "schema_version": LOCAL_SCHEMA_VERSION,
        "schema_sha256": _schema_sha256(),
        "runtime_contract_sha256": _runtime_contract_sha256(),
        "system_sha256": hashlib.sha256(str(system).encode("utf-8")).hexdigest(),
        "expected_count": expected_count,
    }


def _contract_sha256(system: str, expected_count: Optional[int]) -> str:
    return _sha256(_contract_descriptor(system, expected_count))


def _known_id_hits(value: Any, known_node_ids: Iterable[str]) -> List[str]:
    strings: List[str] = []

    def visit(item: Any) -> None:
        if isinstance(item, str):
            strings.append(item)
        elif isinstance(item, Mapping):
            for key, nested in item.items():
                visit(key)
                visit(nested)
        elif isinstance(item, (list, tuple)):
            for nested in item:
                visit(nested)

    visit(value)
    return sorted({
        node_id
        for node_id in known_node_ids
        # Known node identifiers are secrets of the binding context, not
        # natural-language tokens.  Reject a literal occurrence even when a
        # model or prompt author glues ASCII letters around it; token-boundary
        # matching would let ``x<node_id>y`` bypass the prompt/new-label gate.
        if any(node_id in text for text in strings)
    })


def _real_id_shape_hits(value: Any) -> List[str]:
    """Blocking tier: substrings matching the authoritative node-ID grammar."""

    hits: set[str] = set()

    def visit(item: Any) -> None:
        if isinstance(item, str):
            hits.update(
                match.group(0)
                for match in REAL_NODE_ID_EMBEDDED_PATTERN.finditer(item)
            )
        elif isinstance(item, Mapping):
            for key, nested in item.items():
                visit(key)
                visit(nested)
        elif isinstance(item, (list, tuple)):
            for nested in item:
                visit(nested)

    visit(value)
    return sorted(hits)


def _legacy_wide_shape_paths(value: Any) -> List[str]:
    """Non-blocking tier: field paths that only the pre-RF21 wide shape matched.

    Returned entries are *paths and counts only* -- never the matched text.
    A production response that trips this tier is allowed through; the signal
    exists so that narrowing the grammar stays observable instead of silent.
    """

    paths: Dict[str, int] = {}

    def visit(item: Any, path: str) -> None:
        if isinstance(item, str):
            wide = {m.group(0) for m in REAL_NODE_ID_LEGACY_WIDE_PATTERN.finditer(item)}
            if not wide:
                return
            narrow = {m.group(0) for m in REAL_NODE_ID_EMBEDDED_PATTERN.finditer(item)}
            residual = {
                token for token in wide
                if not any(token in exact or exact in token for exact in narrow)
            }
            if residual:
                paths[path] = paths.get(path, 0) + len(residual)
        elif isinstance(item, Mapping):
            for key, nested in item.items():
                visit(key, f"{path}.<key>" if path else "<key>")
                visit(nested, f"{path}.{key}" if path else str(key))
        elif isinstance(item, (list, tuple)):
            for index, nested in enumerate(item):
                visit(nested, f"{path}[{index}]")

    visit(value, "")
    return sorted(f"{key}:{count}" for key, count in paths.items())


def build_local_reference_context(
    *,
    task: str,
    user_text: str,
    candidate_node_ids: Iterable[str],
    preferred_refs: Sequence[Tuple[str, str]] = (),
    virtual_refs: Sequence[Tuple[str, str]] = (),
    contract_text: str = "",
    expected_count: Optional[int] = None,
) -> LocalReferenceContext:
    """Build a one-call, one-to-one ref mapping and redact displayed real IDs.

    Only candidate IDs that are literally present as complete tokens enter the
    mapping. Preferred entries are ordered first but must also be displayed.
    Virtual entries (for example NEW_BRIDGE) are explicit non-node enums.
    """

    raw_text = str(user_text)
    known_node_ids = tuple(sorted({
        str(value) for value in candidate_node_ids if str(value)
    }))
    known_node_ids_sha256 = _sha256(list(known_node_ids))
    ordered: List[LocalReferenceEntry] = []
    used_refs: set[str] = set()
    used_ids: set[str] = set()

    def add(ref: str, node_id: str, kind: str) -> None:
        if not LOCAL_REF_PATTERN.fullmatch(ref):
            raise ValueError(f"invalid local ref enum: {ref!r}")
        if ref in used_refs:
            raise ValueError(f"duplicate local ref enum: {ref}")
        if node_id in used_ids:
            raise ValueError(f"duplicate node target in local ref mapping: {node_id}")
        if _token_position(raw_text, node_id) is None:
            raise ValueError(f"local ref target is not displayed in this call: {node_id}")
        used_refs.add(ref)
        used_ids.add(node_id)
        ordered.append(LocalReferenceEntry(ref=ref, node_id=node_id, kind=kind))

    for ref, node_id in preferred_refs:
        add(str(ref), str(node_id), "node")
    for ref, literal in virtual_refs:
        add(str(ref), str(literal), "virtual")

    visible: List[Tuple[int, str]] = []
    for node_id in known_node_ids:
        if not node_id or node_id in used_ids:
            continue
        position = _token_position(raw_text, node_id)
        if position is not None:
            visible.append((position, node_id))
    visible.sort(key=lambda item: (item[0], item[1]))

    next_index = 0
    for _, node_id in visible:
        while f"N{next_index}" in used_refs:
            next_index += 1
        add(f"N{next_index}", node_id, "node")
        next_index += 1

    localized = raw_text
    for entry in sorted(ordered, key=lambda item: len(item.node_id), reverse=True):
        localized = _replace_exact_token(localized, entry.node_id, entry.ref)

    leaked = sorted(set(
        _known_id_hits(localized, known_node_ids)
        + _real_id_shape_hits(localized)
    ))
    if leaked:
        raise ValueError(f"real node IDs remain in localized prompt: {leaked}")

    schema_sha256 = _schema_sha256()
    runtime_contract_sha256 = _runtime_contract_sha256()
    system_sha256 = hashlib.sha256(str(contract_text).encode("utf-8")).hexdigest()
    contract_sha256 = _contract_sha256(contract_text, expected_count)
    context_payload = {
        "protocol_version": PROTOCOL_VERSION,
        "task": str(task),
        "contract_sha256": contract_sha256,
        "expected_count": expected_count,
        "known_node_ids_sha256": known_node_ids_sha256,
        "ref_mapping": [entry.audit_dict() for entry in ordered],
        "localized_user": localized,
    }
    context_sha256 = _sha256(context_payload)
    context_token = f"CTX_{context_sha256[:16].upper()}"
    allowed = ", ".join(entry.ref for entry in ordered) or "(none)"
    model_user = (
        "# Call-local reference ownership\n"
        f"context_token={context_token}\n"
        f"allowed_refs={allowed}\n"
        "Return the context_token exactly. Use only the listed refs in identity "
        "fields. Do not return real node IDs.\n\n"
        f"{localized}"
    )
    return LocalReferenceContext(
        task=str(task),
        context_token=context_token,
        context_sha256=context_sha256,
        contract_sha256=contract_sha256,
        schema_version=LOCAL_SCHEMA_VERSION,
        schema_sha256=schema_sha256,
        runtime_contract_sha256=runtime_contract_sha256,
        system_sha256=system_sha256,
        expected_count=expected_count,
        known_node_ids=known_node_ids,
        known_node_ids_sha256=known_node_ids_sha256,
        entries=tuple(ordered),
        localized_user=localized,
        model_user=model_user,
    )


def _validate_evidence(value: Any, path: str, errors: List[Dict[str, Any]]) -> None:
    required = EVIDENCE_REQUIRED_FIELDS
    if not isinstance(value, dict):
        errors.append(_error("EVIDENCE_NOT_OBJECT", "evidence must be an object", path=path))
        return
    missing = sorted(required - set(value))
    extra = sorted(set(value) - required)
    if missing:
        errors.append(_error("EVIDENCE_FIELD_MISSING", "evidence fields are missing", path=path, fields=missing))
    if extra:
        errors.append(_error("EVIDENCE_FIELD_EXTRA", "evidence has unknown fields", path=path, fields=extra))
    if "summary" in value and not isinstance(value.get("summary"), str):
        errors.append(_error("EVIDENCE_TYPE_INVALID", "evidence.summary must be a string", path=path))
    elif value.get("summary") == "":
        errors.append(_error(
            "EVIDENCE_VALUE_INVALID",
            "evidence.summary must be non-empty",
            path=path,
            repairable=False,
        ))
    if "warnings" in value and not (
        isinstance(value.get("warnings"), list)
        and all(isinstance(item, str) for item in value.get("warnings", []))
    ):
        errors.append(_error("EVIDENCE_TYPE_INVALID", "evidence.warnings must be a string list", path=path))
    elif isinstance(value.get("warnings"), list):
        warnings = value["warnings"]
        if any(item == "" for item in warnings) or len(warnings) != len(set(warnings)):
            errors.append(_error(
                "EVIDENCE_VALUE_INVALID",
                "evidence.warnings must contain unique non-empty strings",
                path=path,
                repairable=False,
            ))
    for key in (
        "target_represents_all_source_members",
        "pure_structural_redundancy",
        "cross_l1_authorized",
    ):
        if key in value and not isinstance(value.get(key), bool):
            errors.append(_error("EVIDENCE_TYPE_INVALID", f"evidence.{key} must be boolean", path=path))
    if "membership_basis" in value and not isinstance(value.get("membership_basis"), str):
        errors.append(_error("EVIDENCE_TYPE_INVALID", "evidence.membership_basis must be a string", path=path))


def bind_local_reference_payload(
    payload: Any,
    context: LocalReferenceContext,
    *,
    expected_count: Optional[int] = None,
) -> BindingResult:
    errors: List[Dict[str, Any]] = []
    if expected_count is not None and expected_count != context.expected_count:
        return BindingResult(
            ok=False,
            bound_payload=None,
            errors=[_error(
                "EXPECTED_COUNT_CONTRACT_MISMATCH",
                "binder expected_count conflicts with the locked call context",
                supplied=expected_count,
                locked=context.expected_count,
            )],
            local_payload=copy.deepcopy(payload),
        )
    locked_expected_count = context.expected_count
    if not isinstance(payload, dict):
        return BindingResult(
            ok=False,
            bound_payload=None,
            errors=[_error("ENVELOPE_NOT_OBJECT", "response root must be an object")],
        )

    # The model-facing contract is local-reference-only in every field, not
    # merely in executable identity slots.  Free-text evidence must not become
    # a side channel for persistent IDs, and a directly valid envelope must
    # not bypass the stricter repair-prompt check later in the call flow.
    response_id_hits = sorted(set(
        _known_id_hits(payload, context.known_node_ids)
        + _real_id_shape_hits(payload)
    ))
    if response_id_hits:
        errors.append(_error(
            "REAL_NODE_ID_IN_MODEL_RESPONSE",
            "model response cannot contain a known or real-ID-shaped value",
            repairable=False,
            hit_count=len(response_id_hits),
        ))

    envelope_keys = set(payload)
    missing_envelope = sorted({"context_token", "decisions"} - envelope_keys)
    extra_envelope = sorted(envelope_keys - {"context_token", "decisions"})
    if missing_envelope:
        errors.append(_error("ENVELOPE_FIELD_MISSING", "response envelope fields are missing", fields=missing_envelope))
    if extra_envelope:
        forbidden = sorted(set(extra_envelope) & FORBIDDEN_ID_FIELDS)
        if forbidden:
            errors.append(_error("FORBIDDEN_REAL_ID_FIELD", "real-ID fields are forbidden", fields=forbidden))
        remaining = sorted(set(extra_envelope) - set(forbidden))
        if remaining:
            errors.append(_error("ENVELOPE_FIELD_EXTRA", "response envelope has unknown fields", fields=remaining))

    token = payload.get("context_token")
    if token is None:
        errors.append(_error("CONTEXT_TOKEN_MISSING", "context_token is required"))
    elif token != context.context_token:
        errors.append(_error("CONTEXT_TOKEN_MISMATCH", "response belongs to a different call context"))

    decisions = payload.get("decisions")
    if not isinstance(decisions, list):
        errors.append(_error("DECISIONS_NOT_LIST", "decisions must be a list"))
        return BindingResult(False, None, errors, local_payload=copy.deepcopy(payload))
    if locked_expected_count is not None and len(decisions) != locked_expected_count:
        errors.append(_error(
            "DECISION_COUNT_MISMATCH",
            "unexpected decision count",
            expected=locked_expected_count,
            actual=len(decisions),
        ))

    mapping = context.mapping
    kinds = context.kinds
    bound_decisions: List[Dict[str, Any]] = []
    seen_decisions: set[str] = set()

    for index, raw_decision in enumerate(decisions):
        path = f"decisions[{index}]"
        if not isinstance(raw_decision, dict):
            errors.append(_error("DECISION_NOT_OBJECT", "decision must be an object", path=path))
            continue
        keys = set(raw_decision)
        forbidden = sorted(keys & FORBIDDEN_ID_FIELDS)
        if forbidden:
            errors.append(_error("FORBIDDEN_REAL_ID_FIELD", "real-ID fields are forbidden", path=path, fields=forbidden))
        missing = sorted(LOCAL_DECISION_FIELDS - keys)
        extra = sorted(keys - LOCAL_DECISION_FIELDS - FORBIDDEN_ID_FIELDS)
        for field in missing:
            errors.append(_error("DECISION_FIELD_MISSING", "decision field is missing", path=path, field=field))
        for field in extra:
            errors.append(_error("DECISION_FIELD_EXTRA", "decision field is unknown", path=path, field=field))

        for field in ("relation", "action"):
            if field in raw_decision and not isinstance(raw_decision.get(field), str):
                errors.append(_error("DECISION_SEMANTIC_TYPE_INVALID", f"{field} must be a string", path=path))
        if isinstance(raw_decision.get("relation"), str) and raw_decision["relation"] not in RELATIONS:
            errors.append(_error(
                "DECISION_SEMANTIC_VALUE_INVALID",
                "relation is outside the model-facing schema enum",
                path=path,
                field="relation",
                repairable=False,
            ))
        if isinstance(raw_decision.get("action"), str) and raw_decision["action"] not in ACTIONS:
            errors.append(_error(
                "DECISION_SEMANTIC_VALUE_INVALID",
                "action is outside the model-facing schema enum",
                path=path,
                field="action",
                repairable=False,
            ))
        if "new_label" in raw_decision and raw_decision.get("new_label") is not None and not isinstance(raw_decision.get("new_label"), str):
            errors.append(_error("DECISION_SEMANTIC_TYPE_INVALID", "new_label must be null or string", path=path))
        new_label = raw_decision.get("new_label")
        if isinstance(new_label, str) and (
            _known_id_hits(new_label, context.known_node_ids)
            or _real_id_shape_hits(new_label)
        ):
            errors.append(_error(
                "REAL_NODE_ID_IN_NEW_LABEL",
                "new_label cannot contain a real node ID",
                path=path,
            ))
        confidence = raw_decision.get("confidence")
        if "confidence" in raw_decision and (isinstance(confidence, bool) or not isinstance(confidence, (int, float))):
            errors.append(_error("DECISION_SEMANTIC_TYPE_INVALID", "confidence must be numeric", path=path))
        elif isinstance(confidence, (int, float)) and not 0 <= confidence <= 1:
            errors.append(_error(
                "DECISION_SEMANTIC_VALUE_INVALID",
                "confidence must be in the closed interval [0, 1]",
                path=path,
                field="confidence",
                repairable=False,
            ))
        if "evidence" in raw_decision:
            _validate_evidence(raw_decision.get("evidence"), f"{path}.evidence", errors)

        bound: Dict[str, Any] = {
            field: copy.deepcopy(raw_decision.get(field))
            for field in ("relation", "action", "new_label", "confidence", "evidence")
            if field in raw_decision
        }
        for local_field, bound_field, allow_virtual in (
            ("source_ref", "source_id", False),
            ("target_ref", "target_id", True),
        ):
            value = raw_decision.get(local_field)
            if local_field not in raw_decision:
                continue
            if not isinstance(value, str):
                errors.append(_error("REF_NOT_STRING", f"{local_field} must be a string enum", path=path))
                continue
            if value not in mapping:
                if value in context.known_node_ids or _looks_like_real_node_id(value):
                    errors.append(_error("REAL_NODE_ID_IN_REF", "real node ID supplied where a local ref is required", path=path, field=local_field))
                else:
                    errors.append(_error("UNKNOWN_REF", "ref is not in this call context", path=path, field=local_field, ref=value))
                continue
            if not allow_virtual and kinds.get(value) == "virtual":
                errors.append(_error("VIRTUAL_REF_NOT_ALLOWED", "virtual ref cannot be used as a source", path=path, field=local_field, ref=value))
                continue
            bound[bound_field] = mapping[value]

        raw_plan = raw_decision.get("child_plan")
        bound_plan: List[Dict[str, Any]] = []
        if "child_plan" in raw_decision and not isinstance(raw_plan, list):
            errors.append(_error("CHILD_PLAN_NOT_LIST", "child_plan must be a list", path=path))
        elif isinstance(raw_plan, list):
            seen_child_refs: set[str] = set()
            for child_index, raw_item in enumerate(raw_plan):
                item_path = f"{path}.child_plan[{child_index}]"
                if not isinstance(raw_item, dict):
                    errors.append(_error("CHILD_PLAN_ITEM_NOT_OBJECT", "child plan item must be an object", path=item_path))
                    continue
                item_keys = set(raw_item)
                forbidden_item = sorted(item_keys & FORBIDDEN_ID_FIELDS)
                if forbidden_item:
                    errors.append(_error("FORBIDDEN_REAL_ID_FIELD", "real-ID fields are forbidden", path=item_path, fields=forbidden_item))
                missing_item = sorted(LOCAL_CHILD_FIELDS - item_keys)
                extra_item = sorted(item_keys - LOCAL_CHILD_FIELDS - FORBIDDEN_ID_FIELDS)
                for field in missing_item:
                    errors.append(_error("CHILD_FIELD_MISSING", "child plan field is missing", path=item_path, field=field))
                for field in extra_item:
                    errors.append(_error("CHILD_FIELD_EXTRA", "child plan field is unknown", path=item_path, field=field))

                for field in ("disposition", "relation", "evidence"):
                    if field in raw_item and not isinstance(raw_item.get(field), str):
                        errors.append(_error("CHILD_SEMANTIC_TYPE_INVALID", f"{field} must be a string", path=item_path))
                if isinstance(raw_item.get("disposition"), str) and raw_item["disposition"] not in CHILD_DISPOSITIONS:
                    errors.append(_error(
                        "CHILD_SEMANTIC_VALUE_INVALID",
                        "child disposition is outside the model-facing schema enum",
                        path=item_path,
                        field="disposition",
                        repairable=False,
                    ))
                if isinstance(raw_item.get("relation"), str) and raw_item["relation"] not in RELATIONS:
                    errors.append(_error(
                        "CHILD_SEMANTIC_VALUE_INVALID",
                        "child relation is outside the model-facing schema enum",
                        path=item_path,
                        field="relation",
                        repairable=False,
                    ))
                if raw_item.get("evidence") == "":
                    errors.append(_error(
                        "CHILD_SEMANTIC_VALUE_INVALID",
                        "child evidence must be non-empty",
                        path=item_path,
                        field="evidence",
                        repairable=False,
                    ))
                if "same_domain" in raw_item and not isinstance(raw_item.get("same_domain"), bool):
                    errors.append(_error("CHILD_SEMANTIC_TYPE_INVALID", "same_domain must be boolean", path=item_path))

                bound_item: Dict[str, Any] = {
                    field: copy.deepcopy(raw_item.get(field))
                    for field in CHILD_SEMANTIC_FIELDS
                    if field in raw_item
                }
                for local_field, bound_field, allow_virtual in (
                    ("child_ref", "child_id", False),
                    ("target_parent_ref", "target_parent_id", True),
                ):
                    value = raw_item.get(local_field)
                    if local_field not in raw_item:
                        continue
                    if not isinstance(value, str):
                        errors.append(_error("REF_NOT_STRING", f"{local_field} must be a string enum", path=item_path))
                        continue
                    if value not in mapping:
                        if value in context.known_node_ids or _looks_like_real_node_id(value):
                            errors.append(_error("REAL_NODE_ID_IN_REF", "real node ID supplied where a local ref is required", path=item_path, field=local_field))
                        else:
                            errors.append(_error("UNKNOWN_REF", "ref is not in this call context", path=item_path, field=local_field, ref=value))
                        continue
                    if not allow_virtual and kinds.get(value) == "virtual":
                        errors.append(_error("VIRTUAL_REF_NOT_ALLOWED", "virtual ref cannot identify a child", path=item_path, field=local_field, ref=value))
                        continue
                    bound_item[bound_field] = mapping[value]
                    if local_field == "child_ref":
                        if value in seen_child_refs:
                            errors.append(_error("DUPLICATE_CHILD_REF", "child_ref occurs more than once in one plan", path=item_path, ref=value))
                        seen_child_refs.add(value)
                bound_plan.append(bound_item)
        if "child_plan" in raw_decision:
            bound["child_plan"] = bound_plan

        local_key = _canonical_json(raw_decision)
        if local_key in seen_decisions:
            errors.append(_error("DUPLICATE_DECISION", "duplicate decisions are forbidden", path=path))
        seen_decisions.add(local_key)
        bound_decisions.append(bound)

    if errors:
        return BindingResult(False, None, errors, local_payload=copy.deepcopy(payload))
    return BindingResult(
        True,
        {"decisions": bound_decisions},
        [],
        local_payload=copy.deepcopy(payload),
    )


def _candidate_decisions(payload: Any) -> Optional[List[Dict[str, Any]]]:
    if not isinstance(payload, dict):
        return None
    if isinstance(payload.get("decisions"), list):
        values = payload["decisions"]
    elif isinstance(payload.get("decision"), dict):
        values = [payload["decision"]]
    elif any(field in payload for field in DECISION_SEMANTIC_FIELDS):
        values = [payload]
    else:
        nested = [
            value.get("decisions")
            for value in payload.values()
            if isinstance(value, dict) and isinstance(value.get("decisions"), list)
        ]
        if len(nested) != 1:
            return None
        values = nested[0]
    if not all(isinstance(item, dict) for item in values):
        return None
    return list(values)


def semantic_projection(payload: Any) -> Optional[List[Dict[str, Any]]]:
    decisions = _candidate_decisions(payload)
    if decisions is None:
        return None
    projection: List[Dict[str, Any]] = []
    for decision in decisions:
        if any(field not in decision for field in DECISION_SEMANTIC_FIELDS):
            return None
        plan = decision.get("child_plan")
        if not isinstance(plan, list):
            return None
        projected_plan = []
        for item in plan:
            if not isinstance(item, dict) or any(field not in item for field in CHILD_SEMANTIC_FIELDS):
                return None
            projected_plan.append({field: copy.deepcopy(item[field]) for field in CHILD_SEMANTIC_FIELDS})
        projected = {field: copy.deepcopy(decision[field]) for field in DECISION_SEMANTIC_FIELDS}
        projected["child_plan"] = projected_plan
        projection.append(projected)
    return projection


def _opaque_evidence_id_hits(
    payload: Any,
    context: LocalReferenceContext,
) -> List[Dict[str, Any]]:
    decisions = _candidate_decisions(payload) or []
    hits: List[Dict[str, Any]] = []

    def inspect(path: str, value: Any) -> None:
        matched = sorted(set(
            _known_id_hits(value, context.known_node_ids)
            + _real_id_shape_hits(value)
        ))
        if matched:
            hits.append({
                "path": path,
                "value_sha256": _sha256(value),
                "hit_count": len(matched),
            })

    for index, decision in enumerate(decisions):
        inspect(f"decisions[{index}].evidence", decision.get("evidence"))
        plan = decision.get("child_plan")
        if not isinstance(plan, list):
            continue
        for child_index, item in enumerate(plan):
            if isinstance(item, Mapping):
                inspect(
                    f"decisions[{index}].child_plan[{child_index}].evidence",
                    item.get("evidence"),
                )
    return hits


def _stable_valid_refs(
    payload: Any,
    context: LocalReferenceContext,
    *,
    mutable_paths: Iterable[str] = (),
) -> Dict[str, str]:
    decisions = _candidate_decisions(payload) or []
    stable: Dict[str, str] = {}
    mapping = context.mapping
    mutable = {str(path) for path in mutable_paths}
    for index, decision in enumerate(decisions):
        for field in ("source_ref", "target_ref"):
            value = decision.get(field)
            path = f"decisions[{index}].{field}"
            if isinstance(value, str) and value in mapping and path not in mutable:
                stable[path] = value
        plan = decision.get("child_plan")
        if isinstance(plan, list):
            for child_index, item in enumerate(plan):
                if not isinstance(item, dict):
                    continue
                for field in ("child_ref", "target_parent_ref"):
                    value = item.get(field)
                    path = f"decisions[{index}].child_plan[{child_index}].{field}"
                    if isinstance(value, str) and value in mapping and path not in mutable:
                        stable[path] = value
    return stable


def _void_child_plan_indexes(errors: Sequence[Mapping[str, Any]]) -> set[int]:
    """Decision indexes whose only legal ``child_plan`` is the empty list.

    Authorisation comes from the stage scope validator via
    ``context[VOID_CHILD_PLAN_FLAG]``; see that constant for why it exists.
    """
    indexes: set[int] = set()
    for item in errors:
        details = item.get("context")
        if not isinstance(details, Mapping):
            continue
        if details.get(VOID_CHILD_PLAN_FLAG) is not True:
            continue
        index = details.get("decision_index")
        if isinstance(index, int) and not isinstance(index, bool) and index >= 0:
            indexes.add(index)
    return indexes


def _void_child_plan_ref_paths(payload: Any, indexes: Iterable[int]) -> set[str]:
    """Every child_plan ref path inside an authorised decision.

    Those refs are about to be deleted wholesale, so they must not be carried
    into ``stable_refs``: reading a deleted path yields ``None`` and would trip
    ``REPAIR_VALID_REF_DRIFT`` on a legitimate truncation.
    """
    decisions = _candidate_decisions(payload) or []
    authorised = {int(index) for index in indexes}
    paths: set[str] = set()
    for index, decision in enumerate(decisions):
        if index not in authorised:
            continue
        plan = decision.get("child_plan")
        if not isinstance(plan, list):
            continue
        for child_index, item in enumerate(plan):
            if not isinstance(item, dict):
                continue
            for field in ("child_ref", "target_parent_ref"):
                paths.add(f"decisions[{index}].child_plan[{child_index}].{field}")
    return paths


def _projection_preserved(
    projection: Optional[List[Dict[str, Any]]],
    repair_projection: Optional[List[Dict[str, Any]]],
    *,
    void_indexes: Iterable[int] = (),
) -> bool:
    """Frozen semantics survived the repair round.

    Byte equality is the rule.  The single tolerated exception is an authorised
    decision whose ``child_plan`` came back as exactly ``[]``; for that decision
    every other frozen field must still match byte for byte, and a non-empty
    repaired plan falls back to the strict comparison.
    """
    if _canonical_json(repair_projection) == _canonical_json(projection):
        return True
    authorised = {int(index) for index in void_indexes}
    if not authorised or projection is None or repair_projection is None:
        return False
    if len(projection) != len(repair_projection):
        return False
    for index, (before, after) in enumerate(zip(projection, repair_projection)):
        if _canonical_json(before) == _canonical_json(after):
            continue
        if index not in authorised or after.get("child_plan") != []:
            return False
        stripped_before = {
            field: value for field, value in before.items() if field != "child_plan"
        }
        stripped_after = {
            field: value for field, value in after.items() if field != "child_plan"
        }
        if _canonical_json(stripped_before) != _canonical_json(stripped_after):
            return False
    return True


def _mutable_ref_paths(errors: Sequence[Mapping[str, Any]]) -> set[str]:
    paths: set[str] = set()
    for item in errors:
        details = item.get("context")
        if not isinstance(details, Mapping):
            continue
        raw_paths = details.get("mutable_ref_paths")
        if isinstance(raw_paths, list):
            paths.update(str(path) for path in raw_paths if isinstance(path, str))
    return paths


def _run_bound_validator(
    validator: Optional[Callable[[Dict[str, Any]], Sequence[Mapping[str, Any]]]],
    bound_payload: Optional[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    if validator is None:
        return []
    if not isinstance(bound_payload, dict):
        return [_error(
            "BOUND_VALIDATOR_INPUT_INVALID",
            "bound validator requires an object payload",
            repairable=False,
        )]
    try:
        raw_errors = validator(copy.deepcopy(bound_payload))
    except Exception as exc:
        return [_error(
            "BOUND_VALIDATOR_FAILED",
            "bound scope validator raised an exception",
            repairable=False,
            exception_type=type(exc).__name__,
        )]
    if raw_errors is None:
        return []
    if not isinstance(raw_errors, Sequence) or isinstance(raw_errors, (str, bytes)):
        return [_error(
            "BOUND_VALIDATOR_RESULT_INVALID",
            "bound validator must return a sequence of error objects",
            repairable=False,
        )]
    normalized: List[Dict[str, Any]] = []
    for index, item in enumerate(raw_errors):
        if not isinstance(item, Mapping):
            return [_error(
                "BOUND_VALIDATOR_RESULT_INVALID",
                "bound validator returned a non-object error",
                repairable=False,
                error_index=index,
            )]
        copied = copy.deepcopy(dict(item))
        details = copied.get("context")
        if not isinstance(details, Mapping):
            return [_error(
                "BOUND_VALIDATOR_RESULT_INVALID",
                "bound validator errors require an object context",
                repairable=False,
                error_index=index,
            )]
        repairable = details.get("repairable")
        raw_paths = details.get("mutable_ref_paths")
        if not isinstance(repairable, bool) or not isinstance(raw_paths, list):
            return [_error(
                "BOUND_VALIDATOR_RESULT_INVALID",
                "bound validator errors require repairable and mutable_ref_paths",
                repairable=False,
                error_index=index,
            )]
        if repairable and not raw_paths:
            return [_error(
                "BOUND_VALIDATOR_RESULT_INVALID",
                "repairable scope errors require at least one mutable ref path",
                repairable=False,
                error_index=index,
            )]
        decisions = bound_payload.get("decisions", [])
        for path in raw_paths:
            match = re.fullmatch(
                r"decisions\[(\d+)\]\.(source_ref|target_ref)",
                path if isinstance(path, str) else "",
            )
            if match:
                if int(match.group(1)) < len(decisions):
                    continue
            child_match = re.fullmatch(
                r"decisions\[(\d+)\]\.child_plan\[(\d+)\]\.(child_ref|target_parent_ref)",
                path if isinstance(path, str) else "",
            )
            if child_match:
                decision_index = int(child_match.group(1))
                child_index = int(child_match.group(2))
                if decision_index < len(decisions):
                    plan = decisions[decision_index].get("child_plan", [])
                    if isinstance(plan, list) and child_index < len(plan):
                        continue
            return [_error(
                "BOUND_VALIDATOR_RESULT_INVALID",
                "bound validator named an invalid mutable ref path",
                repairable=False,
                error_index=index,
            )]
        normalized.append(copied)
    return normalized


def _read_ref_path(payload: Any, path: str) -> Any:
    decisions = _candidate_decisions(payload)
    if decisions is None:
        return None
    match = re.fullmatch(r"decisions\[(\d+)\]\.(source_ref|target_ref)", path)
    if match:
        index, field = int(match.group(1)), match.group(2)
        return decisions[index].get(field) if index < len(decisions) else None
    match = re.fullmatch(
        r"decisions\[(\d+)\]\.child_plan\[(\d+)\]\.(child_ref|target_parent_ref)",
        path,
    )
    if match:
        decision_index, child_index, field = int(match.group(1)), int(match.group(2)), match.group(3)
        if decision_index >= len(decisions):
            return None
        plan = decisions[decision_index].get("child_plan")
        if not isinstance(plan, list) or child_index >= len(plan) or not isinstance(plan[child_index], dict):
            return None
        return plan[child_index].get(field)
    return None


def _repairable(errors: Sequence[Mapping[str, Any]]) -> bool:
    hard_codes = {
        "CONTEXT_TOKEN_MISSING",
        "CONTEXT_TOKEN_MISMATCH",
        "FORBIDDEN_REAL_ID_FIELD",
        "REAL_NODE_ID_IN_MODEL_RESPONSE",
        "REAL_NODE_ID_IN_RAW_RESPONSE",
        "REAL_NODE_ID_IN_REF",
        "REAL_NODE_ID_IN_NEW_LABEL",
        "DUPLICATE_CHILD_REF",
        "DUPLICATE_DECISION",
        "DECISION_SEMANTIC_TYPE_INVALID",
        "DECISION_SEMANTIC_VALUE_INVALID",
        "CHILD_SEMANTIC_TYPE_INVALID",
        "CHILD_SEMANTIC_VALUE_INVALID",
        "EVIDENCE_NOT_OBJECT",
        "EVIDENCE_FIELD_MISSING",
        "EVIDENCE_TYPE_INVALID",
        "EVIDENCE_VALUE_INVALID",
        "VIRTUAL_REF_NOT_ALLOWED",
        "DECISION_COUNT_MISMATCH",
        "BOUND_VALIDATOR_INPUT_INVALID",
        "BOUND_VALIDATOR_FAILED",
        "BOUND_VALIDATOR_RESULT_INVALID",
    }
    explicitly_terminal = any(
        isinstance(item.get("context"), Mapping)
        and item.get("context", {}).get("repairable") is False
        for item in errors
    )
    return (
        bool(errors)
        and not explicitly_terminal
        and not any(str(item.get("code")) in hard_codes for item in errors)
    )


def _transport_history(
    result: Any,
    *,
    phase: str,
    global_offset: int,
    remaining: int,
) -> Tuple[Optional[List[Dict[str, Any]]], Optional[Dict[str, Any]]]:
    if not isinstance(result, dict):
        return None, _error("TRANSPORT_RESULT_INVALID", "transport result must be an object")
    attempts = result.get("attempts")
    history = result.get("attempt_history")
    if isinstance(attempts, bool) or not isinstance(attempts, int) or attempts < 1 or attempts > remaining:
        return None, _error("ATTEMPT_COUNT_INVALID", "transport attempts are outside the remaining budget", attempts=attempts, remaining=remaining)
    if not isinstance(history, list) or len(history) != attempts:
        return None, _error("ATTEMPT_HISTORY_INVALID", "attempt_history length must equal attempts", attempts=attempts, history_length=len(history) if isinstance(history, list) else None)
    normalized: List[Dict[str, Any]] = []
    for index, raw in enumerate(history, start=1):
        if not isinstance(raw, dict) or raw.get("attempt") != index:
            return None, _error("ATTEMPT_HISTORY_INVALID", "transport attempt indexes must be contiguous", expected=index)
        entry = copy.deepcopy(raw)
        entry["phase"] = phase
        entry["global_attempt"] = global_offset + index
        normalized.append(entry)
    return normalized, None


def _raw_response_id_hits(
    history: Sequence[Mapping[str, Any]],
    context: LocalReferenceContext,
) -> List[str]:
    """Scan every untrusted attempt for real node IDs, raw *and* decoded.

    Scanning only the raw transport text is not sufficient: JSON string escapes
    mean ``L1_\u004E762e6689`` contains no literal identifier byte-for-byte in
    the raw payload while decoding to a real ID.  C13RF21 therefore feeds both
    the raw text and the parsed structure (keys included) through the same
    detectors, so an escaped identifier cannot slip past the gate that a plain
    one would trip.
    """

    scanned: List[Any] = []
    for item in history:
        if not isinstance(item, Mapping):
            continue
        raw_text = item.get("raw")
        if isinstance(raw_text, str):
            scanned.append(raw_text)
            decoded = _decoded_json_payload(raw_text)
            if decoded is not None:
                scanned.append(decoded)
        parsed = item.get("json")
        if parsed is not None:
            scanned.append(parsed)
    return sorted(set(
        _known_id_hits(scanned, context.known_node_ids)
        + _real_id_shape_hits(scanned)
    ))


def _redact_real_ids(value: Any, known_node_ids: Sequence[str]) -> Any:
    """Replace real identifiers with HMAC-style fingerprints inside evidence.

    A response rejected for leaking a node ID must not be archived verbatim --
    otherwise the very identifier the gate exists to contain is written into the
    evidence bundle.  Only shape/known-ID matches are replaced; every other byte
    of the untrusted response is preserved so the audit trail stays faithful.
    """

    def fingerprint(token: str) -> str:
        digest = hashlib.sha256(token.encode("utf-8")).hexdigest()[:12]
        return f"<REDACTED_NODE_ID:{digest}>"

    def scrub(text: str) -> str:
        for node_id in sorted(set(known_node_ids), key=len, reverse=True):
            if node_id and node_id in text:
                text = text.replace(node_id, fingerprint(node_id))
        return REAL_NODE_ID_EMBEDDED_PATTERN.sub(
            lambda match: fingerprint(match.group(0)), text
        )

    if isinstance(value, str):
        return scrub(value)
    if isinstance(value, Mapping):
        return {scrub(str(k)): _redact_real_ids(v, known_node_ids)
                for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_redact_real_ids(v, known_node_ids) for v in value]
    return value


def _decoded_json_payload(text: str) -> Optional[Any]:
    """Best-effort JSON decode used purely to widen ID scanning coverage."""

    try:
        return json.loads(text)
    except (ValueError, TypeError):
        return None


def _repair_user(
    context: LocalReferenceContext,
    projection: List[Dict[str, Any]],
    stable_refs: Mapping[str, str],
    errors: Sequence[Mapping[str, Any]],
    *,
    void_indexes: Iterable[int] = (),
) -> str:
    safe_errors = []
    for item in errors:
        copied = copy.deepcopy(dict(item))
        context_payload = copied.get("context")
        if isinstance(context_payload, dict):
            context_payload.pop("ref", None)
        safe_errors.append(copied)
    # Decision indexes only: no node ID or ref value enters this clause, so the
    # repair-prompt ID scanners downstream stay unaffected.
    authorised = sorted({int(index) for index in void_indexes})
    void_clause = (
        "The source node of each decision index listed here has no children at "
        "all, so its corrected child_plan must be exactly []; emptying those "
        "plans is the required correction, not a semantic change.\n"
        f"child_plan_must_be_empty={_canonical_json(authorised)}\n"
        if authorised else ""
    )
    return (
        "# Schema/scope reference repair\n"
        f"context_token={context.context_token}\n"
        "Return one corrected local-reference envelope. Preserve the frozen semantic "
        "projection exactly; do not add, delete, reorder, or change semantic fields. "
        "Preserve every already-valid ref listed below. Use only allowed refs from the "
        "original call context.\n"
        f"{void_clause}"
        f"frozen_semantics={_canonical_json(projection)}\n"
        f"stable_valid_refs={_canonical_json(dict(stable_refs))}\n"
        f"schema_errors={_canonical_json(safe_errors)}\n\n"
        f"{context.model_user}"
    )


def _base_result(
    *,
    context: LocalReferenceContext,
    attempts: int,
    attempt_history: Sequence[Mapping[str, Any]],
    repair_count: int,
    initial_result: Any,
    repair_result: Any,
    initial_errors: Sequence[Mapping[str, Any]],
    repair_errors: Sequence[Mapping[str, Any]],
    projection: Optional[List[Dict[str, Any]]],
    final_local: Any,
    final_bound: Any,
    ok: bool,
    disposition: str,
    error: Optional[str],
    events: Sequence[Mapping[str, Any]],
    mutation_detected: bool = False,
) -> Dict[str, Any]:
    # C13RF21 item 4: when the call is rejected *because* it carried a real
    # identifier, the untrusted text must not be archived verbatim.  Redaction
    # is scoped to exactly that disposition so every other call keeps a
    # byte-faithful audit trail.
    if error == "REAL_NODE_ID_IN_RAW_RESPONSE":
        known = context.known_node_ids
        attempt_history = [
            _redact_real_ids(dict(item), known) for item in attempt_history
        ]
        initial_result = _redact_real_ids(initial_result, known)
        repair_result = _redact_real_ids(repair_result, known)
        final_local = _redact_real_ids(final_local, known)
    final_transport = repair_result if repair_result is not None else initial_result
    final_transport = final_transport if isinstance(final_transport, dict) else {}
    logical_call_id = (
        "CALL_"
        + context.context_sha256.upper()
        + f"_{next(_LOGICAL_CALL_SEQUENCE):08d}"
    )
    mutable_paths = sorted(_mutable_ref_paths(initial_errors))
    initial_scope_errors = [
        copy.deepcopy(item) for item in initial_errors
        if isinstance(item.get("context"), Mapping)
        and "mutable_ref_paths" in item.get("context", {})
    ]
    repair_scope_errors = [
        copy.deepcopy(item) for item in repair_errors
        if isinstance(item.get("context"), Mapping)
        and "mutable_ref_paths" in item.get("context", {})
    ]
    initial_payload = (
        initial_result.get("json") if isinstance(initial_result, Mapping) else None
    )
    stable_refs = (
        _stable_valid_refs(initial_payload, context, mutable_paths=mutable_paths)
        if isinstance(initial_payload, dict) else {}
    )
    event_names = {
        str(item.get("event")) for item in events if isinstance(item, Mapping)
    }
    repair_trigger = None
    if repair_count:
        repair_trigger = "scope" if mutable_paths else "binding"
    return {
        "ok": bool(ok),
        "json": copy.deepcopy(final_bound) if ok else None,
        "raw": final_transport.get("raw", ""),
        "error": error,
        "status": final_transport.get("status"),
        "latency_ms": sum(
            int(item.get("latency_ms") or 0) for item in attempt_history
        ),
        "profile": final_transport.get("profile"),
        "provider": final_transport.get("provider"),
        "model": final_transport.get("model"),
        "attempts": attempts,
        "attempt_history": [copy.deepcopy(item) for item in attempt_history],
        "repair_count": repair_count,
        "logical_call_id": logical_call_id,
        "protocol_version": PROTOCOL_VERSION,
        "context": context.audit_dict(),
        "initial_response": copy.deepcopy(initial_result),
        "repair_response": copy.deepcopy(repair_result),
        "initial_binding_errors": [copy.deepcopy(item) for item in initial_errors],
        "repair_binding_errors": [copy.deepcopy(item) for item in repair_errors],
        "initial_scope_errors": initial_scope_errors,
        "repair_scope_errors": repair_scope_errors,
        "repair_trigger": repair_trigger,
        "mutable_ref_paths": mutable_paths,
        "stable_valid_refs": stable_refs,
        "scope_validation_passed": bool(ok) and (
            "scope_validation_pass" in event_names
            or "repair_scope_validation_pass" in event_names
        ),
        "opaque_evidence_id_hits": _opaque_evidence_id_hits(
            initial_payload,
            context,
        ) if isinstance(initial_payload, dict) else [],
        # Non-blocking C13RF21 signal: field paths whose text matched only the
        # pre-narrowing wide shape.  Paths and counts, never matched text.
        "legacy_wide_shape_paths": _legacy_wide_shape_paths(initial_payload)
        if isinstance(initial_payload, dict) else [],
        "frozen_semantics": copy.deepcopy(projection),
        "frozen_semantics_sha256": _sha256(projection) if projection is not None else None,
        "final_local": copy.deepcopy(final_local),
        "final_bound": copy.deepcopy(final_bound),
        "final_disposition": disposition,
        "fuzzy_corrections": 0,
        "mutation_before_validation": bool(mutation_detected),
        "events": [copy.deepcopy(item) for item in events],
    }


def validated_scope_event_seq(call_audit: Mapping[str, Any]) -> int:
    """Return the unique final scope-validation event for a bound call."""

    if not isinstance(call_audit, Mapping) or call_audit.get("ok") is not True:
        raise ValueError("operation audit requires a successful logical call")
    events = call_audit.get("events")
    if not isinstance(events, list):
        raise ValueError("operation audit requires logical-call events")
    candidates = [
        item.get("seq")
        for item in events
        if isinstance(item, Mapping)
        and item.get("event") in {
            "scope_validation_pass",
            "repair_scope_validation_pass",
        }
    ]
    if (
        len(candidates) != 1
        or isinstance(candidates[0], bool)
        or not isinstance(candidates[0], int)
    ):
        raise ValueError("operation audit requires one final scope-validation event")
    return candidates[0]


def assert_call_audit_payload(
    call_audit: Optional[Mapping[str, Any]],
    payload: Mapping[str, Any],
) -> None:
    """Prevent a stage from executing a payload other than the audited binding."""

    if call_audit is None:
        return
    if call_audit.get("final_bound") != payload or call_audit.get("json") != payload:
        raise ValueError("stage payload does not match the logical call final_bound")
    validated_scope_event_seq(call_audit)


def attach_operation_audit(
    record: Dict[str, Any],
    *,
    call_audit: Optional[Mapping[str, Any]],
    decision_index: int,
    executed_decision: Mapping[str, Any],
    live_tree_before_sha256: Optional[str] = None,
    live_tree_after_sha256: Optional[str] = None,
    execution_transform: Optional[str] = None,
) -> Dict[str, Any]:
    """Bind a production operation record to its validated logical call.

    The caller invokes this only after the real single-operation mutation or the
    live-tree batch commit.  Candidate-tree work is intentionally not reported
    as a live mutation.
    """

    if call_audit is None:
        return record
    final_bound = call_audit.get("final_bound")
    decisions = (
        final_bound.get("decisions")
        if isinstance(final_bound, Mapping) else None
    )
    if (
        not isinstance(decisions, list)
        or isinstance(decision_index, bool)
        or not isinstance(decision_index, int)
        or not 0 <= decision_index < len(decisions)
    ):
        raise ValueError("operation audit decision index is outside final_bound")
    validation_seq = validated_scope_event_seq(call_audit)
    semantic_contract = record.get("semantic_contract")
    contract_passed = (
        isinstance(semantic_contract, Mapping)
        and semantic_contract.get("passed") is True
    )
    bound_sha256 = _sha256(decisions[decision_index])
    executed_sha256 = _sha256(executed_decision)
    if executed_sha256 != bound_sha256 and execution_transform not in {
        "materialize_new_bridge",
        "promote_safety_fail_closed_warning",
    }:
        raise ValueError("executed decision drift lacks an allowed transform attestation")
    if executed_sha256 == bound_sha256 and execution_transform is not None:
        raise ValueError("execution transform is present without decision drift")
    record.update({
        "logical_call_id": call_audit.get("logical_call_id"),
        "decision_index": decision_index,
        "bound_decision_sha256": bound_sha256,
        "executed_decision_sha256": executed_sha256,
        "execution_transform": execution_transform,
        "semantic_contract_passed": contract_passed,
        "validated_event_seq": validation_seq,
    })
    if record.get("status") == "applied":
        if (
            not isinstance(live_tree_before_sha256, str)
            or not isinstance(live_tree_after_sha256, str)
            or live_tree_before_sha256 == live_tree_after_sha256
        ):
            raise ValueError("applied operation audit requires an observed live-tree change")
        record.update({
            "mutation_seq": validation_seq + 1,
            "live_tree_sha256_before": live_tree_before_sha256,
            "live_tree_sha256_after": live_tree_after_sha256,
        })
    return record


def call_local_reference_json(
    *,
    transport: Callable[..., Dict[str, Any]],
    profile: str,
    system: str,
    context: LocalReferenceContext,
    task: str,
    expected_count: Optional[int] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    max_raw_attempts: int = 4,
    protected_manager: Any = None,
    bound_validator: Optional[
        Callable[[Dict[str, Any]], Sequence[Mapping[str, Any]]]
    ] = None,
) -> Dict[str, Any]:
    """Execute one bounded logical call with at most one reference-only repair."""

    if max_raw_attempts < 1 or max_raw_attempts > 4:
        raise ValueError("max_raw_attempts must be in 1..4")
    if str(task) != context.task:
        raise ValueError("transport task must match the locked local-reference context")
    if expected_count != context.expected_count:
        raise ValueError("expected_count must match the locked local-reference context")
    if _contract_sha256(system, expected_count) != context.contract_sha256:
        raise ValueError("system/schema contract does not match the locked context")

    tree_snapshot = copy.deepcopy(protected_manager.root) if protected_manager is not None else None
    snapshot_hash = _sha256(tree_snapshot) if tree_snapshot is not None else None
    events: List[Dict[str, Any]] = [{"seq": 1, "event": "context_locked"}]
    initial_prompt_hits = sorted(set(
        _known_id_hits([system, context.model_user], context.known_node_ids)
        + _real_id_shape_hits([system, context.model_user])
    ))
    if initial_prompt_hits:
        events.append({"seq": 2, "event": "model_prompt_rejected"})
        return _base_result(
            context=context, attempts=0, attempt_history=[], repair_count=0,
            initial_result=None, repair_result=None,
            initial_errors=[_error(
                "MODEL_PROMPT_REAL_ID_RISK",
                "model prompt contains a known or real-ID-shaped value",
                repairable=False,
                hit_count=len(initial_prompt_hits),
            )],
            repair_errors=[], projection=None, final_local=None, final_bound=None,
            ok=False, disposition="model_prompt_rejected",
            error="MODEL_PROMPT_REAL_ID_RISK", events=events,
        )

    def mutation_barrier() -> bool:
        if protected_manager is None:
            return False
        current_hash = _sha256(protected_manager.root)
        consistency_errors: List[str] = []
        validator = getattr(protected_manager, "validate_consistency", None)
        if callable(validator):
            try:
                consistency_errors = list(validator())
            except Exception as exc:
                consistency_errors = [f"consistency check failed: {type(exc).__name__}"]
        if current_hash == snapshot_hash and not consistency_errors:
            return False
        protected_manager._restore(copy.deepcopy(tree_snapshot))
        events.append({
            "seq": len(events) + 1,
            "event": "mutation_before_validation_rolled_back",
            "root_hash_changed": current_hash != snapshot_hash,
            "sidecar_inconsistency_count": len(consistency_errors),
        })
        return True

    call_kwargs: Dict[str, Any] = {
        "profile": profile,
        "system": system,
        "user": context.model_user,
        "task": task,
        "retries": max_raw_attempts - 1,
    }
    if temperature is not None:
        call_kwargs["temperature"] = temperature
    if max_tokens is not None:
        call_kwargs["max_tokens"] = max_tokens

    initial_result = transport(**call_kwargs)
    initial_history, transport_error = _transport_history(
        initial_result,
        phase="initial",
        global_offset=0,
        remaining=max_raw_attempts,
    )
    if initial_history is None:
        return _base_result(
            context=context, attempts=0, attempt_history=[], repair_count=0,
            initial_result=initial_result, repair_result=None,
            initial_errors=[transport_error or {}], repair_errors=[], projection=None,
            final_local=None, final_bound=None, ok=False,
            disposition="transport_audit_rejected", error=(transport_error or {}).get("code"),
            events=events,
        )
    event_base = len(events)
    events.extend(
        {"seq": event_base + index, "event": "raw_attempt", "phase": "initial", "global_attempt": item["global_attempt"]}
        for index, item in enumerate(initial_history, start=1)
    )
    if mutation_barrier():
        return _base_result(
            context=context, attempts=len(initial_history), attempt_history=initial_history,
            repair_count=0, initial_result=initial_result, repair_result=None,
            initial_errors=[_error("MUTATION_BEFORE_VALIDATION", "tree changed during the untrusted response phase")],
            repair_errors=[], projection=None, final_local=None, final_bound=None,
            ok=False, disposition="mutation_rejected", error="MUTATION_BEFORE_VALIDATION",
            events=events, mutation_detected=True,
        )

    initial_raw_id_hits = _raw_response_id_hits(initial_history, context)
    if initial_raw_id_hits:
        return _base_result(
            context=context, attempts=len(initial_history),
            attempt_history=initial_history, repair_count=0,
            initial_result=initial_result, repair_result=None,
            initial_errors=[_error(
                "REAL_NODE_ID_IN_RAW_RESPONSE",
                "raw model response contains a known or real-ID-shaped value",
                repairable=False,
                hit_count=len(initial_raw_id_hits),
            )],
            repair_errors=[], projection=None, final_local=None,
            final_bound=None, ok=False,
            disposition="raw_response_rejected",
            error="REAL_NODE_ID_IN_RAW_RESPONSE", events=events,
        )

    attempts_used = len(initial_history)
    initial_payload = initial_result.get("json") if isinstance(initial_result, dict) else None
    if not isinstance(initial_result, dict) or not initial_result.get("ok") or not isinstance(initial_payload, dict):
        return _base_result(
            context=context, attempts=attempts_used, attempt_history=initial_history,
            repair_count=0, initial_result=initial_result, repair_result=None,
            initial_errors=[_error("TRANSPORT_OR_JSON_FAILED", "no parseable JSON object was returned")],
            repair_errors=[], projection=None, final_local=initial_payload, final_bound=None,
            ok=False, disposition="transport_rejected", error="TRANSPORT_OR_JSON_FAILED",
            events=events,
        )

    initial_binding = bind_local_reference_payload(
        initial_payload,
        context,
        expected_count=expected_count,
    )
    initial_validation_errors: List[Dict[str, Any]] = []
    if initial_binding.ok:
        events.append({"seq": len(events) + 1, "event": "binding_pass"})
        initial_validation_errors = _run_bound_validator(
            bound_validator,
            initial_binding.bound_payload,
        )
        if initial_validation_errors:
            events.append({"seq": len(events) + 1, "event": "scope_validation_rejected"})
        else:
            events.append({"seq": len(events) + 1, "event": "scope_validation_pass"})
        if mutation_barrier():
            return _base_result(
                context=context, attempts=attempts_used, attempt_history=initial_history,
                repair_count=0, initial_result=initial_result, repair_result=None,
                initial_errors=[_error("MUTATION_BEFORE_VALIDATION", "tree changed before binding completed")],
                repair_errors=[], projection=None, final_local=initial_payload, final_bound=None,
                ok=False, disposition="mutation_rejected", error="MUTATION_BEFORE_VALIDATION",
                events=events, mutation_detected=True,
            )
        if not initial_validation_errors:
            return _base_result(
                context=context, attempts=attempts_used, attempt_history=initial_history,
                repair_count=0, initial_result=initial_result, repair_result=None,
                initial_errors=[], repair_errors=[], projection=semantic_projection(initial_payload),
                final_local=initial_payload, final_bound=initial_binding.bound_payload,
                ok=True, disposition="bound_without_repair", error=None, events=events,
            )
    else:
        events.append({"seq": len(events) + 1, "event": "initial_binding_rejected"})

    initial_errors = (
        initial_validation_errors if initial_binding.ok else initial_binding.errors
    )
    projection = semantic_projection(initial_payload)
    remaining = max_raw_attempts - attempts_used
    if projection is None or remaining < 1 or not _repairable(initial_errors):
        return _base_result(
            context=context, attempts=attempts_used, attempt_history=initial_history,
            repair_count=0, initial_result=initial_result, repair_result=None,
            initial_errors=initial_errors, repair_errors=[], projection=projection,
            final_local=initial_payload, final_bound=None, ok=False,
            disposition=("scope_rejected" if initial_binding.ok else "binding_rejected"),
            error=("BOUND_SCOPE_VALIDATION_FAILED" if initial_binding.ok else "LOCAL_REFERENCE_BINDING_FAILED"),
            events=events,
        )

    void_indexes = _void_child_plan_indexes(initial_errors)
    stable_refs = _stable_valid_refs(
        initial_payload,
        context,
        mutable_paths=(
            _mutable_ref_paths(initial_errors)
            | _void_child_plan_ref_paths(initial_payload, void_indexes)
        ),
    )
    projection_id_hits = sorted(set(
        _known_id_hits(projection, context.known_node_ids)
        + _real_id_shape_hits(projection)
    ))
    if projection_id_hits:
        return _base_result(
            context=context, attempts=attempts_used, attempt_history=initial_history,
            repair_count=0, initial_result=initial_result, repair_result=None,
            initial_errors=initial_errors,
            repair_errors=[_error(
                "REPAIR_PROMPT_REAL_ID_RISK",
                "frozen semantics contain a known real node ID and cannot be echoed",
                repairable=False,
                hit_count=len(projection_id_hits),
            )],
            projection=projection, final_local=initial_payload, final_bound=None,
            ok=False, disposition="repair_prompt_rejected",
            error="REPAIR_PROMPT_REAL_ID_RISK", events=events,
        )
    repair_user = _repair_user(
        context, projection, stable_refs, initial_errors, void_indexes=void_indexes,
    )
    repair_prompt_hits = sorted(set(
        _known_id_hits([system, repair_user], context.known_node_ids)
        + _real_id_shape_hits([system, repair_user])
    ))
    if repair_prompt_hits:
        return _base_result(
            context=context, attempts=attempts_used, attempt_history=initial_history,
            repair_count=0, initial_result=initial_result, repair_result=None,
            initial_errors=initial_errors,
            repair_errors=[_error(
                "REPAIR_PROMPT_REAL_ID_RISK",
                "repair prompt would contain a known real node ID",
                repairable=False,
                hit_count=len(repair_prompt_hits),
            )],
            projection=projection, final_local=initial_payload, final_bound=None,
            ok=False, disposition="repair_prompt_rejected",
            error="REPAIR_PROMPT_REAL_ID_RISK", events=events,
        )
    repair_kwargs = dict(call_kwargs)
    repair_kwargs.update({
        "system": system + "\n\nYou are performing one reference-only schema/scope repair. Semantic changes are forbidden.",
        "user": repair_user,
        "task": f"{task}:schema_repair",
        "retries": remaining - 1,
    })
    repair_result = transport(**repair_kwargs)
    repair_history, repair_transport_error = _transport_history(
        repair_result,
        phase="repair",
        global_offset=attempts_used,
        remaining=remaining,
    )
    if repair_history is None:
        return _base_result(
            context=context, attempts=attempts_used, attempt_history=initial_history,
            repair_count=1, initial_result=initial_result, repair_result=repair_result,
            initial_errors=initial_errors,
            repair_errors=[repair_transport_error or {}], projection=projection,
            final_local=None, final_bound=None, ok=False,
            disposition="repair_transport_audit_rejected",
            error=(repair_transport_error or {}).get("code"), events=events,
        )
    events.append({"seq": len(events) + 1, "event": "repair_started"})
    event_base = len(events)
    events.extend(
        {"seq": event_base + index, "event": "raw_attempt", "phase": "repair", "global_attempt": item["global_attempt"]}
        for index, item in enumerate(repair_history, start=1)
    )
    all_history = initial_history + repair_history
    attempts_used = len(all_history)
    if mutation_barrier():
        return _base_result(
            context=context, attempts=attempts_used, attempt_history=all_history,
            repair_count=1, initial_result=initial_result, repair_result=repair_result,
            initial_errors=initial_errors,
            repair_errors=[_error("MUTATION_BEFORE_VALIDATION", "tree changed during schema repair")],
            projection=projection, final_local=None, final_bound=None, ok=False,
            disposition="mutation_rejected", error="MUTATION_BEFORE_VALIDATION",
            events=events, mutation_detected=True,
        )

    repair_raw_id_hits = _raw_response_id_hits(repair_history, context)
    if repair_raw_id_hits:
        return _base_result(
            context=context, attempts=attempts_used,
            attempt_history=all_history, repair_count=1,
            initial_result=initial_result, repair_result=repair_result,
            initial_errors=initial_errors,
            repair_errors=[_error(
                "REAL_NODE_ID_IN_RAW_RESPONSE",
                "raw repair response contains a known or real-ID-shaped value",
                repairable=False,
                hit_count=len(repair_raw_id_hits),
            )],
            projection=projection, final_local=None, final_bound=None,
            ok=False, disposition="repair_raw_response_rejected",
            error="REAL_NODE_ID_IN_RAW_RESPONSE", events=events,
        )

    repair_payload = repair_result.get("json") if isinstance(repair_result, dict) else None
    if not isinstance(repair_result, dict) or not repair_result.get("ok") or not isinstance(repair_payload, dict):
        return _base_result(
            context=context, attempts=attempts_used, attempt_history=all_history,
            repair_count=1, initial_result=initial_result, repair_result=repair_result,
            initial_errors=initial_errors,
            repair_errors=[_error("REPAIR_TRANSPORT_OR_JSON_FAILED", "repair returned no parseable JSON object")],
            projection=projection, final_local=repair_payload, final_bound=None, ok=False,
            disposition="repair_rejected", error="REPAIR_TRANSPORT_OR_JSON_FAILED", events=events,
        )

    repair_projection = semantic_projection(repair_payload)
    if not _projection_preserved(
        projection, repair_projection, void_indexes=void_indexes,
    ):
        return _base_result(
            context=context, attempts=attempts_used, attempt_history=all_history,
            repair_count=1, initial_result=initial_result, repair_result=repair_result,
            initial_errors=initial_errors,
            repair_errors=[_error("REPAIR_SEMANTIC_DRIFT", "schema repair changed frozen semantic fields")],
            projection=projection, final_local=repair_payload, final_bound=None, ok=False,
            disposition="repair_semantic_drift", error="REPAIR_SEMANTIC_DRIFT", events=events,
        )
    changed_valid_refs = {
        path: {"before": value, "after": _read_ref_path(repair_payload, path)}
        for path, value in stable_refs.items()
        if _read_ref_path(repair_payload, path) != value
    }
    if changed_valid_refs:
        return _base_result(
            context=context, attempts=attempts_used, attempt_history=all_history,
            repair_count=1, initial_result=initial_result, repair_result=repair_result,
            initial_errors=initial_errors,
            repair_errors=[_error("REPAIR_VALID_REF_DRIFT", "schema repair changed an already-valid ref", changes=changed_valid_refs)],
            projection=projection, final_local=repair_payload, final_bound=None, ok=False,
            disposition="repair_ref_drift", error="REPAIR_VALID_REF_DRIFT", events=events,
        )

    repair_binding = bind_local_reference_payload(
        repair_payload,
        context,
        expected_count=expected_count,
    )
    if not repair_binding.ok:
        events.append({"seq": len(events) + 1, "event": "repair_binding_rejected"})
        return _base_result(
            context=context, attempts=attempts_used, attempt_history=all_history,
            repair_count=1, initial_result=initial_result, repair_result=repair_result,
            initial_errors=initial_errors, repair_errors=repair_binding.errors,
            projection=projection, final_local=repair_payload, final_bound=None, ok=False,
            disposition="repair_binding_rejected", error="REPAIR_BINDING_FAILED", events=events,
        )

    events.append({"seq": len(events) + 1, "event": "repair_binding_pass"})
    repair_validation_errors = _run_bound_validator(
        bound_validator,
        repair_binding.bound_payload,
    )
    if mutation_barrier():
        return _base_result(
            context=context, attempts=attempts_used, attempt_history=all_history,
            repair_count=1, initial_result=initial_result, repair_result=repair_result,
            initial_errors=initial_errors,
            repair_errors=[_error("MUTATION_BEFORE_VALIDATION", "tree changed during repaired scope validation")],
            projection=projection, final_local=repair_payload, final_bound=None,
            ok=False, disposition="mutation_rejected", error="MUTATION_BEFORE_VALIDATION",
            events=events, mutation_detected=True,
        )
    if repair_validation_errors:
        events.append({"seq": len(events) + 1, "event": "repair_scope_validation_rejected"})
        return _base_result(
            context=context, attempts=attempts_used, attempt_history=all_history,
            repair_count=1, initial_result=initial_result, repair_result=repair_result,
            initial_errors=initial_errors, repair_errors=repair_validation_errors,
            projection=projection, final_local=repair_payload, final_bound=None,
            ok=False, disposition="repair_scope_rejected",
            error="REPAIR_SCOPE_VALIDATION_FAILED", events=events,
        )
    events.append({"seq": len(events) + 1, "event": "repair_scope_validation_pass"})
    if mutation_barrier():
        return _base_result(
            context=context, attempts=attempts_used, attempt_history=all_history,
            repair_count=1, initial_result=initial_result, repair_result=repair_result,
            initial_errors=initial_errors,
            repair_errors=[_error("MUTATION_BEFORE_VALIDATION", "tree changed before repaired binding completed")],
            projection=projection, final_local=repair_payload, final_bound=None, ok=False,
            disposition="mutation_rejected", error="MUTATION_BEFORE_VALIDATION",
            events=events, mutation_detected=True,
        )
    return _base_result(
        context=context, attempts=attempts_used, attempt_history=all_history,
        repair_count=1, initial_result=initial_result, repair_result=repair_result,
        initial_errors=initial_errors, repair_errors=[], projection=projection,
        final_local=repair_payload, final_bound=repair_binding.bound_payload,
        ok=True, disposition="bound_after_repair", error=None, events=events,
    )


# --- Fail-closed stop signals (C13RF27, open debt D11) -----------------------
# The stage guard lives in the private harness (``run_stage_fail_closed.py`` and
# its v2 successor), which wraps ``call_local_reference_json`` and
# ``execute_semantic_decision`` and raises ``FailClosedCallError`` to stop the
# run at the first non-deferrable failure.  Production code cannot import that
# type -- the harness is not on the public import path, and stages must keep
# running unguarded -- so the signal is recognised by class name instead.  Both
# guard versions name it identically (v1 ``run_stage_fail_closed.py:16``, v2
# ``run_stage_fail_closed_v2.py:42``), and the contract is pinned by test.
#
# Why this is needed: 14b and 14d wrap their work in ``except Exception`` and
# convert *any* exception into a locally-handled outcome (a batch-abort record,
# or an empty decision list).  That is correct for real stage errors, but it
# also swallows the guard's stop signal -- so "stop at the first non-deferrable
# failure" silently became "log it and carry on".  C13RF26 measured the cost:
# the 14d fail-closed report was written at 10:17:37 and the stage then issued
# six more API calls (10:19:16 through 10:23:44) before E0 finally stopped it.
# 14a and 14c already re-raise after rolling back, so they were never affected.
#
# This predicate only decides *whether an exception keeps propagating*.  It
# changes no scope predicate, no defer criterion and no contract rule: a call
# the guard lets through (ok, or deferrable) never reaches here.
FAIL_CLOSED_SIGNAL_TYPE_NAMES = frozenset({"FailClosedCallError"})


def is_fail_closed_stop_signal(exc: BaseException) -> bool:
    """True when ``exc`` is the harness's fail-closed stop signal.

    Matched across the exception's own MRO by class name, so a future guard may
    subclass its error type without silently losing the stop semantics.
    """
    return any(
        klass.__name__ in FAIL_CLOSED_SIGNAL_TYPE_NAMES
        for klass in type(exc).__mro__
    )


__all__ = [
    "BindingResult",
    "FAIL_CLOSED_SIGNAL_TYPE_NAMES",
    "LocalReferenceContext",
    "LocalReferenceEntry",
    "PROTOCOL_VERSION",
    "VOID_CHILD_PLAN_FLAG",
    "bind_local_reference_payload",
    "build_local_reference_context",
    "call_local_reference_json",
    "is_fail_closed_stop_signal",
    "semantic_projection",
]
