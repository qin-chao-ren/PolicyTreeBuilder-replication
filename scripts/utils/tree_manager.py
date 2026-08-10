from __future__ import annotations

import copy
import unicodedata
from typing import Any, Callable, Dict, List, Optional, Set


class TreeInvariantError(ValueError):
    """Raised when the raw tree and its derived indexes disagree."""


def normalize_label(value: Any) -> str:
    """Canonical form used for deterministic bridge identity checks."""
    text = unicodedata.normalize("NFKC", str(value or ""))
    return " ".join(text.split()).casefold()


class TreeManager:
    """Mutable tree facade with rollback and structural postconditions.

    The nested ``root`` object is the source of truth. ``index`` and
    ``parent_map`` are derived views and are checked after every write.
    """

    def __init__(self, root: Dict[str, Any]):
        if not isinstance(root, dict):
            raise TreeInvariantError("Tree root must be a dictionary")
        self.root = root
        self.index: Dict[str, Dict[str, Any]] = {}
        self.parent_map: Dict[str, str] = {}
        self.last_error: Optional[str] = None
        self._rebuild_index()
        self.assert_consistent()

    # --- Index construction and validation ---

    @staticmethod
    def _node_id(node: Dict[str, Any]) -> str:
        raw_id = node.get("node_id")
        if raw_id is None or str(raw_id).strip() == "":
            raw_id = node.get("tree_id")
        if raw_id is None or str(raw_id).strip() == "":
            raise TreeInvariantError(f"Node is missing an ID: {node}")
        return str(raw_id).strip()

    def _rebuild_index(self) -> None:
        self.index = {}
        self.parent_map = {}
        seen_objects: Set[int] = set()
        self._build_index(self.root, None, seen_objects)

    def _build_index(
        self,
        node: Dict[str, Any],
        parent_id: Optional[str],
        seen_objects: Set[int],
    ) -> None:
        if not isinstance(node, dict):
            raise TreeInvariantError("Every tree node must be a dictionary")
        object_id = id(node)
        if object_id in seen_objects:
            raise TreeInvariantError("A node object is reachable through multiple paths")
        seen_objects.add(object_id)

        node_id = self._node_id(node)
        if node_id in self.index:
            raise TreeInvariantError(f"Duplicate node_id: {node_id}")

        node["node_id"] = node_id
        node.pop("tree_id", None)
        children = node.get("children", [])
        if children is None:
            children = []
        if not isinstance(children, list):
            raise TreeInvariantError(f"children must be a list for node {node_id}")
        node["children"] = children

        self.index[node_id] = node
        if parent_id is not None:
            self.parent_map[node_id] = parent_id
        for child in children:
            self._build_index(child, node_id, seen_objects)

    def validate_consistency(self) -> List[str]:
        """Return structural disagreements without trusting the derived indexes."""
        errors: List[str] = []
        raw_nodes: Dict[str, Dict[str, Any]] = {}
        raw_parents: Dict[str, str] = {}
        seen_objects: Set[int] = set()
        active_objects: Set[int] = set()

        def walk(node: Any, parent_id: Optional[str]) -> None:
            if not isinstance(node, dict):
                errors.append("raw tree contains a non-dictionary node")
                return
            object_id = id(node)
            if object_id in active_objects:
                errors.append("raw tree contains an object cycle")
                return
            if object_id in seen_objects:
                errors.append("raw tree contains a multiply-referenced node object")
                return
            seen_objects.add(object_id)
            active_objects.add(object_id)

            try:
                node_id = self._node_id(node)
            except TreeInvariantError as exc:
                errors.append(str(exc))
                active_objects.remove(object_id)
                return
            if node_id in raw_nodes:
                errors.append(f"raw tree contains duplicate node_id {node_id}")
            else:
                raw_nodes[node_id] = node
            if parent_id is not None:
                if node_id in raw_parents and raw_parents[node_id] != parent_id:
                    errors.append(f"node {node_id} has multiple raw parents")
                raw_parents[node_id] = parent_id

            children = node.get("children", [])
            if not isinstance(children, list):
                errors.append(f"children must be a list for node {node_id}")
            else:
                for child in children:
                    walk(child, node_id)
            active_objects.remove(object_id)

        walk(self.root, None)

        raw_ids = set(raw_nodes)
        index_ids = set(self.index)
        if raw_ids != index_ids:
            errors.append(
                "raw/index ID mismatch: "
                f"raw_only={sorted(raw_ids - index_ids)}, "
                f"index_only={sorted(index_ids - raw_ids)}"
            )
        for node_id in raw_ids & index_ids:
            if self.index[node_id] is not raw_nodes[node_id]:
                errors.append(f"index points to the wrong object for node {node_id}")

        if raw_parents != self.parent_map:
            raw_edges = set(raw_parents.items())
            map_edges = set(self.parent_map.items())
            errors.append(
                "raw/parent_map mismatch: "
                f"raw_only={sorted(raw_edges - map_edges)}, "
                f"map_only={sorted(map_edges - raw_edges)}"
            )
        try:
            root_id = self._node_id(self.root)
            if root_id in self.parent_map:
                errors.append(f"root node {root_id} must not have a parent")
        except TreeInvariantError:
            pass
        return errors

    def assert_consistent(self) -> None:
        errors = self.validate_consistency()
        if errors:
            raise TreeInvariantError("; ".join(errors))

    def _restore(self, snapshot: Dict[str, Any]) -> None:
        self.root.clear()
        self.root.update(snapshot)
        self._rebuild_index()

    def _atomic(
        self,
        mutation: Callable[[], None],
        postcondition: Optional[Callable[[], bool]] = None,
    ) -> bool:
        snapshot = copy.deepcopy(self.root)
        try:
            self.assert_consistent()
            mutation()
            self.assert_consistent()
            if postcondition is not None and not postcondition():
                raise TreeInvariantError("operation postcondition failed")
        except Exception as exc:
            self._restore(snapshot)
            self.last_error = str(exc)
            return False
        self.last_error = None
        return True

    # --- Query API ---

    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        return self.index.get(str(node_id))

    def exists(self, node_id: str) -> bool:
        return str(node_id) in self.index

    def get_parent_id(self, node_id: str) -> Optional[str]:
        return self.parent_map.get(str(node_id))

    def get_children(self, node_id: str) -> List[Dict[str, Any]]:
        node = self.index.get(str(node_id))
        return node.get("children", []) if node else []

    def get_all_node_ids(self) -> List[str]:
        return list(self.index)

    def is_descendant(self, node_id: str, ancestor_id: str) -> bool:
        node_id = str(node_id)
        ancestor_id = str(ancestor_id)
        if node_id not in self.index or ancestor_id not in self.index or node_id == ancestor_id:
            return False
        seen: Set[str] = set()
        current = self.parent_map.get(node_id)
        while current is not None:
            if current == ancestor_id:
                return True
            if current in seen:
                return False
            seen.add(current)
            current = self.parent_map.get(current)
        return False

    # --- Internal mutation helpers (caller owns the transaction) ---

    def _detach(self, node_id: str) -> tuple[str, int]:
        parent_id = self.parent_map.get(node_id)
        if parent_id is None or parent_id not in self.index:
            raise TreeInvariantError(f"node {node_id} has no detachable parent")
        siblings = self.index[parent_id]["children"]
        positions = [i for i, child in enumerate(siblings) if child.get("node_id") == node_id]
        if len(positions) != 1:
            raise TreeInvariantError(
                f"node {node_id} must occur exactly once under parent {parent_id}"
            )
        position = positions[0]
        siblings.pop(position)
        return parent_id, position

    def _move_unchecked(self, node_id: str, new_parent_id: str) -> None:
        old_parent_id = self.parent_map[node_id]
        if old_parent_id == new_parent_id:
            return
        self._detach(node_id)
        self.index[new_parent_id]["children"].append(self.index[node_id])
        self.parent_map[node_id] = new_parent_id

    # --- Atomic write API ---

    def move_node(self, node_id: str, new_parent_id: str) -> bool:
        node_id = str(node_id)
        new_parent_id = str(new_parent_id)
        if node_id not in self.index or new_parent_id not in self.index:
            self.last_error = "source or target node does not exist"
            return False
        if node_id not in self.parent_map:
            self.last_error = "root cannot be moved"
            return False
        if node_id == new_parent_id or self.is_descendant(new_parent_id, node_id):
            self.last_error = "move would create a cycle"
            return False
        old_parent_id = self.parent_map[node_id]
        if old_parent_id == new_parent_id:
            self.last_error = None
            return True

        def mutate() -> None:
            self._move_unchecked(node_id, new_parent_id)

        def postcondition() -> bool:
            old_count = sum(
                child.get("node_id") == node_id
                for child in self.index[old_parent_id]["children"]
            )
            new_count = sum(
                child.get("node_id") == node_id
                for child in self.index[new_parent_id]["children"]
            )
            return old_count == 0 and new_count == 1 and self.parent_map[node_id] == new_parent_id

        return self._atomic(mutate, postcondition)

    def move_nodes_atomically(self, moves: List[Dict[str, str]]) -> bool:
        """Move a complete reparent plan as one rollback-protected mutation."""
        if not isinstance(moves, list) or not moves:
            self.last_error = "atomic move plan must be a non-empty list"
            return False

        normalized: List[tuple[str, str]] = []
        seen_sources: Set[str] = set()
        for item in moves:
            if not isinstance(item, dict):
                self.last_error = "every atomic move item must be an object"
                return False
            node_id = str(item.get("node_id") or "").strip()
            new_parent_id = str(item.get("new_parent_id") or "").strip()
            if not node_id or not new_parent_id:
                self.last_error = "atomic move source and target must be non-empty"
                return False
            if node_id in seen_sources:
                self.last_error = f"duplicate atomic move source: {node_id}"
                return False
            if node_id not in self.index or new_parent_id not in self.index:
                self.last_error = f"atomic move source or target does not exist: {node_id}"
                return False
            if node_id not in self.parent_map:
                self.last_error = "root cannot be moved"
                return False
            if node_id == new_parent_id or self.is_descendant(new_parent_id, node_id):
                self.last_error = f"atomic move would create a cycle: {node_id}"
                return False
            seen_sources.add(node_id)
            normalized.append((node_id, new_parent_id))

        effective = [
            (node_id, new_parent_id)
            for node_id, new_parent_id in normalized
            if self.parent_map.get(node_id) != new_parent_id
        ]
        if not effective:
            self.last_error = None
            return True

        def mutate() -> None:
            for node_id, new_parent_id in effective:
                if node_id not in self.index or new_parent_id not in self.index:
                    raise TreeInvariantError("atomic move endpoint disappeared during mutation")
                if node_id == new_parent_id or self.is_descendant(new_parent_id, node_id):
                    raise TreeInvariantError("atomic move plan creates an interdependent cycle")
                self._move_unchecked(node_id, new_parent_id)

        def postcondition() -> bool:
            return all(
                self.parent_map.get(node_id) == new_parent_id
                for node_id, new_parent_id in normalized
            )

        return self._atomic(mutate, postcondition)

    def remove_node(self, node_id: str, keep_children_orphaned: bool = False) -> bool:
        """Remove a node atomically.

        The legacy ``keep_children_orphaned=True`` name is retained for API
        compatibility, but descendants are promoted to the removed node's
        parent. Keeping unreachable nodes in the indexes would violate the
        manager's source-of-truth contract.
        """
        node_id = str(node_id)
        if node_id not in self.index:
            self.last_error = "node does not exist"
            return False
        if node_id not in self.parent_map:
            self.last_error = "root cannot be removed"
            return False

        node = self.index[node_id]
        parent_id = self.parent_map[node_id]
        descendants: Set[str] = set()
        stack = [node]
        while stack:
            current = stack.pop()
            current_id = self._node_id(current)
            if current_id in descendants:
                self.last_error = "subtree contains a duplicate or cycle"
                return False
            descendants.add(current_id)
            stack.extend(current.get("children", []))

        promoted_ids = [self._node_id(child) for child in node["children"]]

        def mutate() -> None:
            _, position = self._detach(node_id)
            parent_children = self.index[parent_id]["children"]
            if keep_children_orphaned:
                promoted = list(node["children"])
                node["children"] = []
                parent_children[position:position] = promoted
                for child_id in promoted_ids:
                    self.parent_map[child_id] = parent_id
                del self.index[node_id]
                del self.parent_map[node_id]
            else:
                for removed_id in descendants:
                    self.index.pop(removed_id, None)
                    self.parent_map.pop(removed_id, None)

        def postcondition() -> bool:
            if keep_children_orphaned:
                return (
                    node_id not in self.index
                    and node_id not in self.parent_map
                    and all(self.parent_map.get(child_id) == parent_id for child_id in promoted_ids)
                )
            return all(
                removed_id not in self.index and removed_id not in self.parent_map
                for removed_id in descendants
            )

        return self._atomic(mutate, postcondition)

    def promote_child_safe(self, child_id: str) -> bool:
        child_id = str(child_id)
        parent_id = self.get_parent_id(child_id)
        grandparent_id = self.get_parent_id(parent_id) if parent_id else None
        if not parent_id or not grandparent_id:
            self.last_error = "child has no grandparent"
            return False
        return self.move_node(child_id, grandparent_id)

    def promote_child_and_remove_parent(
        self,
        child_id: str,
        new_label: Optional[str] = None,
    ) -> bool:
        """Atomically replace a single-child parent with its child."""
        child_id = str(child_id)
        parent_id = self.get_parent_id(child_id)
        grandparent_id = self.get_parent_id(parent_id) if parent_id else None
        if not parent_id or not grandparent_id:
            self.last_error = "child has no removable parent and grandparent"
            return False
        if [self._node_id(child) for child in self.get_children(parent_id)] != [child_id]:
            self.last_error = "parent must contain exactly the promoted child"
            return False
        if new_label is not None and not normalize_label(new_label):
            self.last_error = "promoted label must be non-empty"
            return False

        def mutate() -> None:
            _, position = self._detach(parent_id)
            parent = self.index[parent_id]
            child = self.index[child_id]
            parent["children"] = []
            self.index[grandparent_id]["children"].insert(position, child)
            self.parent_map[child_id] = grandparent_id
            del self.index[parent_id]
            del self.parent_map[parent_id]
            if new_label is not None:
                child["label"] = str(new_label).strip()

        def postcondition() -> bool:
            structurally_valid = (
                parent_id not in self.index
                and parent_id not in self.parent_map
                and self.parent_map.get(child_id) == grandparent_id
            )
            return structurally_valid and (
                new_label is None or not self._has_label_conflict(child_id)
            )

        return self._atomic(mutate, postcondition)

    def flatten_node(self, node_id: str) -> bool:
        """Atomically replace a node with its children under the same parent."""
        node_id = str(node_id)
        if node_id not in self.index or node_id not in self.parent_map:
            self.last_error = "root or missing node cannot be flattened"
            return False
        parent_id = self.parent_map[node_id]
        child_ids = [self._node_id(child) for child in self.get_children(node_id)]

        def mutate() -> None:
            _, position = self._detach(node_id)
            node = self.index[node_id]
            children = list(node["children"])
            node["children"] = []
            self.index[parent_id]["children"][position:position] = children
            for child_id in child_ids:
                self.parent_map[child_id] = parent_id
            del self.index[node_id]
            del self.parent_map[node_id]

        def postcondition() -> bool:
            return (
                node_id not in self.index
                and node_id not in self.parent_map
                and all(self.parent_map.get(child_id) == parent_id for child_id in child_ids)
            )

        return self._atomic(mutate, postcondition)

    def absorb_node(
        self,
        winner_id: str,
        loser_id: str,
        new_label: Optional[str] = None,
    ) -> bool:
        winner_id = str(winner_id)
        loser_id = str(loser_id)
        if winner_id not in self.index or loser_id not in self.index:
            self.last_error = "winner or loser does not exist"
            return False
        if winner_id == loser_id or loser_id not in self.parent_map:
            self.last_error = "winner and loser must be distinct non-root nodes"
            return False
        if self.is_descendant(winner_id, loser_id):
            self.last_error = "winner cannot be inside the loser subtree"
            return False
        if new_label is not None and not normalize_label(new_label):
            self.last_error = "winner label must be non-empty"
            return False
        loser_children = [self._node_id(child) for child in self.get_children(loser_id)]

        def mutate() -> None:
            for child_id in loser_children:
                self._move_unchecked(child_id, winner_id)
            self._detach(loser_id)
            self.index[loser_id]["children"] = []
            del self.index[loser_id]
            del self.parent_map[loser_id]
            if new_label is not None:
                self.index[winner_id]["label"] = str(new_label).strip()

        def postcondition() -> bool:
            structurally_valid = (
                loser_id not in self.index
                and loser_id not in self.parent_map
                and all(self.parent_map.get(child_id) == winner_id for child_id in loser_children)
            )
            return structurally_valid and (
                new_label is None or not self._has_label_conflict(winner_id)
            )

        return self._atomic(mutate, postcondition)

    def _has_label_conflict(self, node_id: str) -> bool:
        canonical = normalize_label(self.index[node_id].get("label"))
        parent_id = self.parent_map.get(node_id)
        if parent_id:
            if normalize_label(self.index[parent_id].get("label")) == canonical:
                return True
            for sibling in self.get_children(parent_id):
                sibling_id = self._node_id(sibling)
                if sibling_id != node_id and normalize_label(sibling.get("label")) == canonical:
                    return True
        return any(
            normalize_label(child.get("label")) == canonical
            for child in self.get_children(node_id)
        )

    def rename_node(self, node_id: str, new_label: str) -> bool:
        node_id = str(node_id)
        if node_id not in self.index or not normalize_label(new_label):
            self.last_error = "node is missing or label is empty"
            return False
        def mutate() -> None:
            self.index[node_id]["label"] = str(new_label).strip()

        success = self._atomic(
            mutate,
            lambda: not self._has_label_conflict(node_id),
        )
        if not success and self.last_error == "operation postcondition failed":
            self.last_error = "rename would create an exact parent-child or sibling duplicate"
        return success

    def add_child_node(self, parent_id: str, new_node_data: Dict[str, Any]) -> bool:
        parent_id = str(parent_id)
        if parent_id not in self.index or not isinstance(new_node_data, dict):
            self.last_error = "parent is missing or payload is invalid"
            return False
        try:
            node_id = self._node_id(new_node_data)
        except TreeInvariantError as exc:
            self.last_error = str(exc)
            return False
        if node_id in self.index:
            self.last_error = f"node_id conflict: {node_id}"
            return False
        payload = copy.deepcopy(new_node_data)

        def mutate() -> None:
            self.index[parent_id]["children"].append(payload)
            self._build_index(payload, parent_id, {id(node) for node in self.index.values()})

        return self._atomic(
            mutate,
            lambda: self.parent_map.get(node_id) == parent_id and node_id in self.index,
        )

    def create_or_reuse_bridge(
        self,
        parent_id: str,
        bridge_data: Dict[str, Any],
        child_ids: List[str],
    ) -> Optional[str]:
        """Create/reuse one deterministic bridge and move new children atomically."""
        parent_id = str(parent_id)
        if parent_id not in self.index or not isinstance(bridge_data, dict):
            self.last_error = "bridge parent or payload is invalid"
            return None
        try:
            requested_id = self._node_id(bridge_data)
        except TreeInvariantError as exc:
            self.last_error = str(exc)
            return None
        label_key = normalize_label(bridge_data.get("label"))
        if not label_key:
            self.last_error = "bridge label is empty"
            return None

        bridge_id: Optional[str] = None
        existing = self.index.get(requested_id)
        if existing is not None:
            bridge_id = requested_id
        else:
            matching = [
                child for child in self.get_children(parent_id)
                if normalize_label(child.get("label")) == label_key
                and (child.get("synthetic_bridge") or "_BR_" in self._node_id(child))
            ]
            if len(matching) == 1:
                bridge_id = self._node_id(matching[0])
            elif len(matching) > 1:
                self.last_error = "multiple bridges already exist for the same parent and label"
                return None

        if bridge_id is not None:
            bridge = self.index[bridge_id]
            if self.parent_map.get(bridge_id) != parent_id:
                self.last_error = "bridge ID is attached to a different parent"
                return None
            if normalize_label(bridge.get("label")) != label_key:
                self.last_error = "bridge ID conflicts with a different label"
                return None
            expected_level = bridge_data.get("level")
            if expected_level is not None and str(bridge.get("level")) != str(expected_level):
                self.last_error = "bridge ID conflicts with a different level"
                return None
            for key, expected_value in bridge_data.items():
                if key in {"node_id", "tree_id", "children", "label", "level"}:
                    continue
                if bridge.get(key) != expected_value:
                    self.last_error = f"bridge ID conflicts on payload field {key}"
                    return None
        elif requested_id in self.index:
            self.last_error = "bridge ID conflicts with a different payload"
            return None

        unique_child_ids = list(dict.fromkeys(str(child_id) for child_id in child_ids))
        if bridge_id is None and not unique_child_ids:
            self.last_error = "a new bridge needs at least one child"
            return None
        for child_id in unique_child_ids:
            if child_id not in self.index or child_id == (bridge_id or requested_id):
                self.last_error = f"invalid bridge child: {child_id}"
                return None
            current_parent = self.parent_map.get(child_id)
            if current_parent not in {parent_id, bridge_id}:
                self.last_error = f"bridge child {child_id} belongs to another parent"
                return None

        actual_bridge_id = bridge_id or requested_id
        payload = copy.deepcopy(bridge_data)
        payload["node_id"] = requested_id
        payload.pop("tree_id", None)
        payload["children"] = []
        payload["synthetic_bridge"] = True

        def mutate() -> None:
            if bridge_id is None:
                self.index[parent_id]["children"].append(payload)
                self.index[requested_id] = payload
                self.parent_map[requested_id] = parent_id
            for child_id in unique_child_ids:
                if self.parent_map.get(child_id) == parent_id:
                    self._move_unchecked(child_id, actual_bridge_id)

        def postcondition() -> bool:
            return (
                actual_bridge_id in self.index
                and self.parent_map.get(actual_bridge_id) == parent_id
                and all(self.parent_map.get(child_id) == actual_bridge_id for child_id in unique_child_ids)
            )

        if not self._atomic(mutate, postcondition):
            return None
        return actual_bridge_id


__all__ = ["TreeManager", "TreeInvariantError", "normalize_label"]
