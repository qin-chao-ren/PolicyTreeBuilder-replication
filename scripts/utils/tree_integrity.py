from __future__ import annotations

import csv
import json
import os
import tempfile
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple


class LineageError(ValueError):
    """Raised when redirects are conflicting, cyclic, or dangling."""


class E0ValidationError(RuntimeError):
    """Raised when a candidate tree is not eligible for publication."""


def canonical_exact_label(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    return " ".join(text.split())


def close_lineage(mapping: Mapping[str, str]) -> Dict[str, str]:
    """Resolve every redirect to its terminal target and reject cycles."""
    direct: Dict[str, str] = {}
    for raw_source, raw_target in mapping.items():
        source = str(raw_source).strip()
        target = str(raw_target).strip()
        if not source or not target:
            raise LineageError("lineage source and target must be non-empty")
        if source in direct and direct[source] != target:
            raise LineageError(
                f"conflicting lineage targets for {source}: {direct[source]} vs {target}"
            )
        direct[source] = target

    closed: Dict[str, str] = {}
    for source in direct:
        current = source
        path: List[str] = []
        positions: Dict[str, int] = {}
        while current in direct:
            if current in positions:
                cycle = path[positions[current]:] + [current]
                raise LineageError(f"lineage cycle: {' -> '.join(cycle)}")
            positions[current] = len(path)
            path.append(current)
            current = direct[current]
        for visited in path:
            closed[visited] = current
    return closed


def merge_lineage_maps(*mappings: Mapping[str, str]) -> Dict[str, str]:
    direct: Dict[str, str] = {}
    for mapping in mappings:
        if not isinstance(mapping, Mapping):
            raise LineageError("lineage payload must be an object mapping source to target")
        for raw_source, raw_target in mapping.items():
            source = str(raw_source).strip()
            target = str(raw_target).strip()
            if source in direct and direct[source] != target:
                raise LineageError(
                    f"conflicting lineage targets for {source}: {direct[source]} vs {target}"
                )
            direct[source] = target
    return close_lineage(direct)


# --- C13RF22: upstream lineage ledger integrity -------------------------------
#
# Every refinement stage that redirects nodes writes a "ledger" file mapping
# source -> target.  Downstream stages must consume every upstream ledger, or
# they will see membership rows pointing at nodes that no longer exist.
#
# Both producers write their ledger unconditionally at the end of a successful
# run (collapse_redundant_hierarchy.py and balance_tree_structure.py both call
# atomic_write_json outside any conditional), so an *absent* ledger file can
# only mean one of two things: the stage never ran, or its output was lost.
# Neither is safe to paper over with an empty dict -- that is exactly how
# C13RF21 died at export time with a diagnosis that pointed at the wrong layer.
#
# An *empty* ledger, by contrast, is perfectly legal: a stage may run and
# redirect nothing (RF11's 14b did precisely that, 28 operations and zero
# lineage targets).  The distinction that matters is "file missing" vs
# "file present and empty", never "map is falsy".

STAGE_LINEAGE_LEDGERS: Tuple[Tuple[str, str], ...] = (
    ("vertical_collapse", "vertical_collapse_trace.json"),
    ("structure_balancing", "structure_balancing_trace.json"),
)

# Operations records carry `step` values that identify the producing stage.
# 14b emits two distinct step names, both belonging to the same ledger.
_STEP_TO_LEDGER_STAGE: Dict[str, str] = {
    "vertical_collapse": "vertical_collapse",
    "structure_balancing": "structure_balancing",
    "structure_balancing_depth": "structure_balancing",
    "structure_balancing_fanout": "structure_balancing",
}


def stages_present_in_operations(records: Iterable[Mapping[str, Any]]) -> Set[str]:
    """Which upstream ledger-producing stages left records in the ops log."""
    seen: Set[str] = set()
    for record in records:
        if not isinstance(record, Mapping):
            continue
        stage = _STEP_TO_LEDGER_STAGE.get(str(record.get("step") or ""))
        if stage is not None:
            seen.add(stage)
    return seen


def load_stage_lineage_ledger(
    outdir: str | Path,
    stage: str,
    filename: str,
    *,
    required: bool,
) -> Dict[str, str]:
    """Load one upstream ledger, failing closed when a required one is absent.

    `required` must be decided from evidence that the stage actually ran (see
    `stages_present_in_operations`), not from whether the file happens to be
    there -- the latter is the fail-open bug this replaces.
    """
    path = Path(outdir) / filename
    if not path.exists():
        if required:
            raise LineageError(
                f"upstream stage {stage!r} left records in the operations log but its "
                f"lineage ledger is missing: {path}. Its redirects cannot be recovered, "
                f"so membership rows pointing at nodes it deleted would be unresolvable. "
                f"Stage a copy of {filename} from that stage's run before continuing."
            )
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as exc:
        raise LineageError(f"lineage ledger {path} is unreadable: {exc}") from exc
    if not isinstance(payload, dict):
        raise LineageError(f"lineage ledger must be an object: {path}")
    ledger: Dict[str, str] = {}
    for raw_source, raw_target in payload.items():
        source = str(raw_source).strip()
        target = str(raw_target).strip()
        if not source or not target:
            raise LineageError(f"lineage ledger {path} has an empty source or target")
        ledger[source] = target
    return ledger


def cross_validate_ledger_against_operations(
    ledger: Mapping[str, str],
    records: Iterable[Mapping[str, Any]],
    stage: str,
) -> Dict[str, Any]:
    """Cross-check a ledger against the operations log's own redirect record.

    Returns a verdict dict rather than raising: the operations schema only grew
    a `lineage_target` field in the current contract, so archives predating it
    carry applied redirects with no target recorded.  C13F's log is exactly
    that shape -- 32 vertical_collapse records, zero lineage_target fields --
    so treating "no targets in the log" as a mismatch would hard-reject every
    historical archive.  Absence of the field is reported as `not_comparable`;
    only a genuine disagreement between two populated sources is a mismatch.
    """
    stage_steps = {
        step for step, mapped in _STEP_TO_LEDGER_STAGE.items() if mapped == stage
    }
    applied = [
        record
        for record in records
        if isinstance(record, Mapping)
        and str(record.get("step") or "") in stage_steps
        and str(record.get("status") or "") == "applied"
    ]
    from_ops = {
        str(record.get("source_id") or "").strip(): str(record.get("lineage_target") or "").strip()
        for record in applied
        if record.get("lineage_target")
    }
    if not from_ops:
        return {
            "stage": stage,
            "comparable": False,
            "reason": "operations log records no lineage_target (pre-contract schema)",
            "applied_records": len(applied),
            "ledger_entries": len(ledger),
        }
    normalized = {str(k).strip(): str(v).strip() for k, v in ledger.items()}
    only_in_ops = {k: v for k, v in from_ops.items() if normalized.get(k) != v}
    return {
        "stage": stage,
        "comparable": True,
        "agrees": not only_in_ops,
        "applied_records": len(applied),
        "ledger_entries": len(normalized),
        "operations_redirects": len(from_ops),
        "disagreements": only_in_ops,
    }


def redirect_membership_rows(
    rows: Sequence[Mapping[str, Any]],
    lineage: Mapping[str, str],
    live_node_ids: Iterable[str],
) -> List[Dict[str, str]]:
    """Redirect membership without changing row identity or cardinality."""
    closed = close_lineage(lineage)
    live = {str(node_id) for node_id in live_node_ids}
    redirected: List[Dict[str, str]] = []
    for index, raw_row in enumerate(rows):
        row = {str(key): "" if value is None else str(value) for key, value in raw_row.items()}
        current = row.get("final_node_id", "").strip()
        if not current:
            raise LineageError(f"membership row {index} has no final_node_id")
        target = closed.get(current, current)
        if target not in live:
            raise LineageError(
                f"membership row {index} points to deleted node {target} (from {current})"
            )
        row["final_node_id"] = target
        redirected.append(row)
    if len(redirected) != len(rows):
        raise LineageError("membership row count changed during redirection")
    return redirected


def read_membership_csv(path: str | Path) -> Tuple[List[str], List[Dict[str, str]]]:
    source = Path(path)
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = [dict(row) for row in reader]
    return fieldnames, rows


def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"JSONL line {line_number} is not an object")
            records.append(value)
    return records


def atomic_write_bytes(path: str | Path, payload: bytes) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except Exception:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def atomic_write_text(path: str | Path, text: str, encoding: str = "utf-8") -> None:
    atomic_write_bytes(path, text.encode(encoding))


def atomic_write_json(path: str | Path, payload: Any) -> None:
    atomic_write_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def atomic_write_jsonl(
    path: str | Path,
    records: Sequence[Mapping[str, Any]],
) -> None:
    text = "".join(
        json.dumps(dict(record), ensure_ascii=False) + "\n"
        for record in records
    )
    atomic_write_text(path, text, encoding="utf-8")


def atomic_write_csv(
    path: str | Path,
    fieldnames: Sequence[str],
    rows: Sequence[Mapping[str, Any]],
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except Exception:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def _membership_identity(row: Mapping[str, Any]) -> Tuple[Tuple[str, str], ...]:
    return tuple(
        sorted(
            (str(key), "" if value is None else str(value))
            for key, value in row.items()
            if str(key) != "final_node_id"
        )
    )


def validate_tree_e0(
    root: Dict[str, Any],
    *,
    manager: Any = None,
    membership_rows: Optional[Sequence[Mapping[str, Any]]] = None,
    expected_membership_rows: Optional[Sequence[Mapping[str, Any]]] = None,
    lineage: Optional[Mapping[str, str]] = None,
    operations: Optional[Sequence[Mapping[str, Any]]] = None,
    require_membership: bool = False,
    parent_child_allowlist: Optional[Set[Tuple[str, str]]] = None,
) -> Dict[str, Any]:
    """Validate the E0 publication contract and return a reproducible report."""
    violations: List[Dict[str, Any]] = []
    parent_child_allowlist = parent_child_allowlist or set()

    def add(code: str, message: str, **context: Any) -> None:
        item: Dict[str, Any] = {
            "severity": "critical",
            "code": code,
            "message": message,
        }
        if context:
            item["context"] = context
        violations.append(item)

    raw_nodes: Dict[str, Dict[str, Any]] = {}
    raw_parents: Dict[str, str] = {}
    raw_depths: Dict[str, int] = {}
    seen_objects: Set[int] = set()
    active_objects: Set[int] = set()
    sibling_groups = 0
    parent_child_duplicates = 0
    level_mismatches = 0

    def walk(node: Any, parent_id: Optional[str], depth: int) -> None:
        nonlocal sibling_groups, parent_child_duplicates, level_mismatches
        if not isinstance(node, dict):
            add("NODE_NOT_OBJECT", "Every raw-tree node must be an object", depth=depth)
            return
        object_id = id(node)
        if object_id in active_objects:
            add("OBJECT_CYCLE", "Raw tree contains an object cycle", depth=depth)
            return
        if object_id in seen_objects:
            add("MULTIPLE_OBJECT_PARENT", "One node object is reachable by multiple paths", depth=depth)
            return
        seen_objects.add(object_id)
        active_objects.add(object_id)

        raw_id = node.get("node_id")
        node_id = "" if raw_id is None else str(raw_id).strip()
        if not node_id:
            add("MISSING_NODE_ID", "Node ID must be non-empty", depth=depth)
            node_id = f"<missing:{object_id}>"
        elif node_id in raw_nodes:
            add("DUPLICATE_NODE_ID", "node_id occurs more than once", node_id=node_id)
        else:
            raw_nodes[node_id] = node
            raw_depths[node_id] = depth
        if "tree_id" in node:
            add("LEGACY_TREE_ID", "tree_id must not survive canonicalization", node_id=node_id)
        if parent_id is not None:
            if node_id in raw_parents and raw_parents[node_id] != parent_id:
                add(
                    "MULTIPLE_ID_PARENT",
                    "One node ID has multiple parents",
                    node_id=node_id,
                    first_parent=raw_parents[node_id],
                    second_parent=parent_id,
                )
            raw_parents[node_id] = parent_id

        expected_level = "ROOT" if depth == 0 else f"L{depth}"
        actual_level = str(node.get("level", "")).upper()
        if actual_level != expected_level:
            level_mismatches += 1
            add(
                "LEVEL_DEPTH_MISMATCH",
                "Declared level does not match physical depth",
                node_id=node_id,
                declared=actual_level,
                expected=expected_level,
            )

        label = canonical_exact_label(node.get("label"))
        if depth > 0 and not label:
            add("EMPTY_LABEL", "Non-root node label must be non-empty", node_id=node_id)

        children = node.get("children", [])
        if not isinstance(children, list):
            add("CHILDREN_NOT_LIST", "children must be a list", node_id=node_id)
            active_objects.remove(object_id)
            return

        labels: Dict[str, List[str]] = defaultdict(list)
        for child in children:
            if isinstance(child, dict):
                child_id = str(child.get("node_id", "")).strip()
                child_label = canonical_exact_label(child.get("label"))
                labels[child_label].append(child_id)
                if (
                    label
                    and child_label == label
                    and (node_id, child_id) not in parent_child_allowlist
                ):
                    parent_child_duplicates += 1
                    add(
                        "PARENT_CHILD_EXACT_DUPLICATE",
                        "Parent and child have the same exact label",
                        parent_id=node_id,
                        child_id=child_id,
                        label=label,
                    )
            walk(child, node_id, depth + 1)
        for duplicate_label, node_ids in labels.items():
            if duplicate_label and len(node_ids) > 1:
                sibling_groups += 1
                add(
                    "SIBLING_EXACT_DUPLICATE",
                    "Siblings under one parent have the same exact label",
                    parent_id=node_id,
                    label=duplicate_label,
                    node_ids=node_ids,
                )
        active_objects.remove(object_id)

    walk(root, None, 0)
    raw_ids = set(raw_nodes)
    root_id = str(root.get("node_id", "")).strip() if isinstance(root, dict) else ""
    expected_parent_ids = raw_ids - ({root_id} if root_id else set())
    if set(raw_parents) != expected_parent_ids:
        add(
            "RAW_PARENT_COVERAGE",
            "Every non-root node must have exactly one raw parent",
            missing=sorted(expected_parent_ids - set(raw_parents)),
            extra=sorted(set(raw_parents) - expected_parent_ids),
        )

    if manager is not None:
        index_ids = {str(node_id) for node_id in manager.index}
        if index_ids != raw_ids:
            add(
                "RAW_INDEX_MISMATCH",
                "Manager index IDs differ from raw DFS IDs",
                raw_only=sorted(raw_ids - index_ids),
                index_only=sorted(index_ids - raw_ids),
            )
        manager_parents = {
            str(node_id): str(parent_id)
            for node_id, parent_id in manager.parent_map.items()
        }
        if manager_parents != raw_parents:
            add(
                "RAW_PARENT_MAP_MISMATCH",
                "Manager parent_map differs from raw child edges",
                raw_only=sorted(set(raw_parents.items()) - set(manager_parents.items())),
                map_only=sorted(set(manager_parents.items()) - set(raw_parents.items())),
            )
        for error in manager.validate_consistency():
            add("MANAGER_INVARIANT", error)

    closed_lineage: Dict[str, str] = {}
    if lineage is not None:
        try:
            closed_lineage = close_lineage(lineage)
        except LineageError as exc:
            add("LINEAGE_INVALID", str(exc))
        for source, target in closed_lineage.items():
            if target not in raw_ids:
                add(
                    "LINEAGE_DANGLING_TARGET",
                    "Lineage terminal target is not a live node",
                    source=source,
                    target=target,
                )
            if source in raw_ids and source != target:
                add(
                    "LINEAGE_SOURCE_STILL_LIVE",
                    "A redirected lineage source must not remain in the final tree",
                    source=source,
                    target=target,
                )

    if membership_rows is None:
        if require_membership:
            add("MEMBERSHIP_MISSING", "Final membership is required for publication")
    else:
        if require_membership and not membership_rows:
            add("MEMBERSHIP_EMPTY", "Final membership must contain at least one row")
        for index, row in enumerate(membership_rows):
            node_id = str(row.get("final_node_id", "")).strip()
            if not node_id:
                add(
                    "MEMBERSHIP_TARGET_MISSING",
                    "Membership row has no final_node_id",
                    row=index,
                )
            elif node_id not in raw_ids:
                add(
                    "MEMBERSHIP_DANGLING",
                    "Membership row points to a non-live node",
                    row=index,
                    sample_id=str(row.get("sample_id", "")),
                    final_node_id=node_id,
                )
        if expected_membership_rows is not None:
            expected = Counter(_membership_identity(row) for row in expected_membership_rows)
            actual = Counter(_membership_identity(row) for row in membership_rows)
            if actual != expected:
                add(
                    "MEMBERSHIP_NOT_CONSERVED",
                    "Membership row identities changed during finalization",
                    expected_rows=len(expected_membership_rows),
                    actual_rows=len(membership_rows),
                    missing=sum((expected - actual).values()),
                    extra=sum((actual - expected).values()),
                )

    applied_operations = 0
    if operations is not None:
        applied: List[Tuple[int, Mapping[str, Any], str]] = []
        for index, operation in enumerate(operations):
            if not isinstance(operation, Mapping):
                add(
                    "OPERATION_NOT_OBJECT",
                    "Every operation record must be an object",
                    operation_index=index,
                )
                continue
            status = str(operation.get("status", "")).lower()
            if status not in {"applied", "success"}:
                continue
            operation_type = str(operation.get("type") or operation.get("op") or "").lower()
            applied.append((index, operation, operation_type))

        applied_operations = len(applied)

        def source_id(operation: Mapping[str, Any], operation_type: str) -> str:
            if operation_type == "promote_child":
                return str(
                    operation.get("source_id")
                    or operation.get("parent")
                    or operation.get("parent_id")
                    or ""
                )
            return str(
                operation.get("source_id")
                or operation.get("loser")
                or operation.get("node_id")
                or operation.get("node")
                or ""
            )

        def target_id(operation: Mapping[str, Any], operation_type: str) -> str:
            if operation_type == "promote_child":
                return str(
                    operation.get("target_id")
                    or operation.get("child")
                    or operation.get("child_id")
                    or ""
                )
            return str(
                operation.get("target_id")
                or operation.get("merge_into")
                or operation.get("winner")
                or operation.get("target_parent_id")
                or operation.get("new_parent")
                or ""
            )

        last_move: Dict[str, int] = {}
        last_rename: Dict[str, int] = {}
        for index, operation, operation_type in applied:
            source = source_id(operation, operation_type)
            if operation_type in {"move", "lift_sibling"} and source:
                last_move[source] = index
            elif operation_type == "split_reparent":
                plan = operation.get("child_plan")
                if isinstance(plan, list):
                    for item in plan:
                        if not isinstance(item, Mapping) or item.get("disposition") != "move":
                            continue
                        child_id = str(item.get("child_id") or "")
                        if child_id:
                            last_move[child_id] = index
            elif operation_type == "rename" and source:
                last_rename[source] = index

        for index, operation, operation_type in applied:
            source = source_id(operation, operation_type)
            target = target_id(operation, operation_type)

            def terminal(node_id: str) -> str:
                return closed_lineage.get(node_id, node_id) if lineage is not None else node_id

            def lineage_matches_redirect() -> bool:
                if lineage is None:
                    return True
                return bool(source and target) and closed_lineage.get(source) == terminal(target)

            if operation_type in {"merge", "absorb", "promote_child"}:
                final_target = terminal(target)
                if (
                    not source
                    or source in raw_ids
                    or not target
                    or final_target not in raw_ids
                    or not lineage_matches_redirect()
                ):
                    add(
                        "APPLIED_MERGE_UNTRUE",
                        "Applied merge postcondition is false",
                        operation_index=index,
                        source=source,
                        target=target,
                        terminal_target=final_target,
                    )
            elif operation_type in {"remove", "flatten"}:
                if not source or source in raw_ids:
                    add(
                        "APPLIED_REMOVE_UNTRUE",
                        "Applied removal source still exists",
                        operation_index=index,
                        source=source,
                    )
                if operation_type == "flatten" and target and not lineage_matches_redirect():
                    add(
                        "APPLIED_FLATTEN_LINEAGE_UNTRUE",
                        "Applied flatten is not represented by closed lineage",
                        operation_index=index,
                        source=source,
                        target=target,
                    )
            elif operation_type in {"move", "lift_sibling"}:
                if index != last_move.get(source):
                    continue
                final_target = terminal(target)
                source_was_deleted_later = source not in raw_ids and source in closed_lineage
                if (
                    not source
                    or not target
                    or (
                        not source_was_deleted_later
                        and (source not in raw_ids or raw_parents.get(source) != final_target)
                    )
                ):
                    add(
                        "APPLIED_MOVE_UNTRUE",
                        "Applied move parent edge is false",
                        operation_index=index,
                        node_id=source,
                        target_parent_id=target,
                        actual_parent=raw_parents.get(source),
                    )
            elif operation_type == "split_reparent":
                plan = operation.get("child_plan")
                if not isinstance(plan, list):
                    add(
                        "APPLIED_SPLIT_PLAN_MISSING",
                        "Applied split_reparent has no child plan",
                        operation_index=index,
                    )
                    continue
                moved = 0
                for item in plan:
                    if not isinstance(item, Mapping) or item.get("disposition") != "move":
                        continue
                    child_id = str(item.get("child_id") or "")
                    planned_target = str(item.get("target_parent_id") or "")
                    if not child_id or not planned_target:
                        add(
                            "APPLIED_SPLIT_PLAN_INVALID",
                            "Applied split child or target is missing",
                            operation_index=index,
                        )
                        continue
                    if index != last_move.get(child_id):
                        continue
                    moved += 1
                    final_target = terminal(planned_target)
                    child_was_deleted_later = child_id not in raw_ids and child_id in closed_lineage
                    if (
                        not child_was_deleted_later
                        and (child_id not in raw_ids or raw_parents.get(child_id) != final_target)
                    ):
                        add(
                            "APPLIED_SPLIT_REPARENT_UNTRUE",
                            "Applied split child is not attached to its planned parent",
                            operation_index=index,
                            child_id=child_id,
                            target_parent_id=planned_target,
                            actual_parent=raw_parents.get(child_id),
                        )
                if moved == 0:
                    add(
                        "APPLIED_SPLIT_HAS_NO_MOVES",
                        "Applied split_reparent has no terminal move to validate",
                        operation_index=index,
                    )
            elif operation_type in {"create_bridge", "insert_bridge"}:
                bridge_id = str(operation.get("bridge_id") or operation.get("node_id") or "")
                parent_id = str(operation.get("parent_id") or operation.get("parent") or "")
                bridge_was_deleted_later = bridge_id not in raw_ids and bridge_id in closed_lineage
                if (
                    not bridge_id
                    or not parent_id
                    or (
                        not bridge_was_deleted_later
                        and (
                            bridge_id not in raw_ids
                            or raw_parents.get(bridge_id) != terminal(parent_id)
                        )
                    )
                ):
                    add(
                        "APPLIED_BRIDGE_UNTRUE",
                        "Applied bridge is missing or attached to the wrong parent",
                        operation_index=index,
                        bridge_id=bridge_id,
                        parent_id=parent_id,
                    )
            elif operation_type == "rename":
                if index != last_rename.get(source):
                    continue
                node_id = source
                expected_label = canonical_exact_label(operation.get("new_label"))
                actual_label = canonical_exact_label(
                    raw_nodes.get(node_id, {}).get("label") if node_id in raw_nodes else ""
                )
                node_was_deleted_later = node_id not in raw_ids and node_id in closed_lineage
                if (
                    not node_id
                    or (
                        not node_was_deleted_later
                        and (node_id not in raw_ids or actual_label != expected_label)
                    )
                ):
                    add(
                        "APPLIED_RENAME_UNTRUE",
                        "Applied rename label is false",
                        operation_index=index,
                        node_id=node_id,
                        expected_label=expected_label,
                        actual_label=actual_label,
                    )
            elif operation_type == "pending_auto_promote":
                node = raw_nodes.get(source)
                pending_fields = {"pending_as_l1", "original_level", "pending_parent"}
                if (
                    not node
                    or raw_parents.get(source) != root_id
                    or str(node.get("level", "")).upper() != "L1"
                    or pending_fields.intersection(node)
                ):
                    add(
                        "APPLIED_PENDING_PROMOTE_UNTRUE",
                        "Applied pending-root promotion postcondition is false",
                        operation_index=index,
                        node_id=source,
                        actual_parent=raw_parents.get(source),
                    )
            else:
                add(
                    "APPLIED_OPERATION_UNSUPPORTED",
                    "Applied operation type has no E0 postcondition validator",
                    operation_index=index,
                    operation_type=operation_type,
                )

    counts = Counter(item["code"] for item in violations)
    report: Dict[str, Any] = {
        "schema_version": "policy-tree-e0-v1",
        "generated_at_unix": int(time.time()),
        "passed": not violations,
        "critical_count": len(violations),
        "violation_counts": dict(sorted(counts.items())),
        "stats": {
            "raw_node_count": len(raw_nodes),
            "raw_edge_count": len(raw_parents),
            "sibling_duplicate_groups": sibling_groups,
            "parent_child_duplicates": parent_child_duplicates,
            "level_depth_mismatches": level_mismatches,
            "membership_rows": None if membership_rows is None else len(membership_rows),
            "lineage_redirects": len(closed_lineage),
            "applied_operations_checked": applied_operations,
        },
        "violations": violations,
    }
    return report


def publish_tree_if_valid(
    root: Dict[str, Any],
    output_path: str | Path,
    audit_path: str | Path,
    **validation_kwargs: Any,
) -> Dict[str, Any]:
    """Write the E0 report, then atomically publish the tree only on PASS."""
    report = validate_tree_e0(root, **validation_kwargs)
    atomic_write_json(audit_path, report)
    if not report["passed"]:
        raise E0ValidationError(
            f"E0 rejected candidate with {report['critical_count']} critical violation(s)"
        )
    atomic_write_json(output_path, root)
    return report


__all__ = [
    "E0ValidationError",
    "LineageError",
    "atomic_write_bytes",
    "atomic_write_csv",
    "atomic_write_json",
    "atomic_write_jsonl",
    "atomic_write_text",
    "canonical_exact_label",
    "close_lineage",
    "cross_validate_ledger_against_operations",
    "load_stage_lineage_ledger",
    "merge_lineage_maps",
    "publish_tree_if_valid",
    "read_jsonl",
    "read_membership_csv",
    "redirect_membership_rows",
    "stages_present_in_operations",
    "STAGE_LINEAGE_LEDGERS",
    "validate_tree_e0",
]
