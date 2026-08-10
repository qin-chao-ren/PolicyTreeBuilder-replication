from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from utils.tree_manager import TreeInvariantError, TreeManager  # noqa: E402


def node(node_id, label, level, children=None):
    return {
        "node_id": node_id,
        "label": label,
        "level": level,
        "children": list(children or []),
    }


class TreeManagerTests(unittest.TestCase):
    def test_duplicate_ids_fail_during_index_build(self):
        root = node(
            "ROOT", "ROOT", "ROOT",
            [node("X", "one", "L1"), node("X", "two", "L1")],
        )
        with self.assertRaises(TreeInvariantError):
            TreeManager(root)

    def test_remove_leaf_updates_raw_index_parent_map_and_dump(self):
        root = node("ROOT", "ROOT", "ROOT", [node("P", "parent", "L1", [node("L", "leaf", "L2")])])
        manager = TreeManager(root)

        self.assertTrue(manager.remove_node("L"), manager.last_error)
        self.assertEqual(manager.get_children("P"), [])
        self.assertNotIn("L", manager.index)
        self.assertNotIn("L", manager.parent_map)
        self.assertNotIn('"node_id": "L"', json.dumps(root, ensure_ascii=False, indent=2))
        self.assertEqual(manager.validate_consistency(), [])

    def test_remove_subtree_and_promote_children_modes_are_consistent(self):
        subtree = node("S", "subtree", "L1", [node("C", "child", "L2")])
        root = node("ROOT", "ROOT", "ROOT", [subtree])
        manager = TreeManager(root)
        self.assertTrue(manager.remove_node("S"), manager.last_error)
        self.assertEqual(set(manager.index), {"ROOT"})

        subtree = node("S", "subtree", "L1", [node("C", "child", "L2")])
        root = node("ROOT", "ROOT", "ROOT", [subtree])
        manager = TreeManager(root)
        self.assertTrue(manager.remove_node("S", keep_children_orphaned=True), manager.last_error)
        self.assertEqual([child["node_id"] for child in manager.get_children("ROOT")], ["C"])
        self.assertEqual(manager.get_parent_id("C"), "ROOT")
        self.assertEqual(manager.validate_consistency(), [])

    def test_absorb_moves_all_children_and_removes_loser(self):
        winner = node("W", "winner", "L1")
        loser = node("L", "loser", "L1", [node("C1", "one", "L2"), node("C2", "two", "L2")])
        root = node("ROOT", "ROOT", "ROOT", [winner, loser])
        manager = TreeManager(root)

        self.assertTrue(manager.absorb_node("W", "L"), manager.last_error)
        self.assertEqual([child["node_id"] for child in manager.get_children("W")], ["C1", "C2"])
        self.assertFalse(manager.exists("L"))
        self.assertEqual(manager.get_parent_id("C1"), "W")
        self.assertEqual(manager.get_parent_id("C2"), "W")
        self.assertEqual(manager.validate_consistency(), [])

    def test_absorb_rolls_back_after_partial_failure(self):
        winner = node("W", "winner", "L1")
        loser = node("L", "loser", "L1", [node("C1", "one", "L2"), node("C2", "two", "L2")])
        root = node("ROOT", "ROOT", "ROOT", [winner, loser])
        manager = TreeManager(root)
        before = copy.deepcopy(root)
        original_move = manager._move_unchecked
        calls = 0

        def fail_on_second_move(node_id, parent_id):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("injected failure")
            original_move(node_id, parent_id)

        manager._move_unchecked = fail_on_second_move
        self.assertFalse(manager.absorb_node("W", "L"))
        self.assertEqual(root, before)
        self.assertEqual(manager.validate_consistency(), [])

    def test_promote_and_flatten_are_single_atomic_operations(self):
        child = node("C", "child", "L3")
        parent = node("P", "parent", "L2", [child])
        grandparent = node("G", "grand", "L1", [parent])
        root = node("ROOT", "ROOT", "ROOT", [grandparent])
        manager = TreeManager(root)

        self.assertTrue(manager.promote_child_and_remove_parent("C"), manager.last_error)
        self.assertFalse(manager.exists("P"))
        self.assertEqual(manager.get_parent_id("C"), "G")

        wrapper = node("F", "wrapper", "L2", [node("F1", "one", "L3"), node("F2", "two", "L3")])
        root = node("ROOT", "ROOT", "ROOT", [node("G", "grand", "L1", [wrapper])])
        manager = TreeManager(root)
        self.assertTrue(manager.flatten_node("F"), manager.last_error)
        self.assertEqual([child["node_id"] for child in manager.get_children("G")], ["F1", "F2"])
        self.assertFalse(manager.exists("F"))

    def test_move_rejects_cycle_without_mutation(self):
        root = node("ROOT", "ROOT", "ROOT", [node("A", "a", "L1", [node("B", "b", "L2")])])
        manager = TreeManager(root)
        before = copy.deepcopy(root)
        self.assertFalse(manager.move_node("A", "B"))
        self.assertEqual(root, before)

    def test_batch_reparent_is_atomic_and_rejects_interdependent_cycle(self):
        root = node(
            "ROOT", "ROOT", "ROOT",
            [node("P", "parent", "L1", [node("A", "a", "L2"), node("B", "b", "L2")])],
        )
        manager = TreeManager(root)
        before = copy.deepcopy(root)
        self.assertFalse(manager.move_nodes_atomically([
            {"node_id": "A", "new_parent_id": "B"},
            {"node_id": "B", "new_parent_id": "A"},
        ]))
        self.assertEqual(root, before)
        self.assertEqual(manager.validate_consistency(), [])

    def test_bridge_creation_is_deterministic_idempotent_and_conflict_closed(self):
        parent = node("P", "parent", "L1", [node("C1", "one", "L2"), node("C2", "two", "L2")])
        root = node("ROOT", "ROOT", "ROOT", [parent])
        manager = TreeManager(root)
        payload = {"node_id": "P_BR_fixed", "label": "Shared Group", "level": "L2", "children": []}

        first = manager.create_or_reuse_bridge("P", payload, ["C1"])
        second = manager.create_or_reuse_bridge("P", payload, ["C1", "C2"])
        self.assertEqual(first, "P_BR_fixed")
        self.assertEqual(second, first)
        bridges = [child for child in manager.get_children("P") if child["node_id"] == first]
        self.assertEqual(len(bridges), 1)
        self.assertEqual([child["node_id"] for child in manager.get_children(first)], ["C1", "C2"])

        before = copy.deepcopy(root)
        conflict = {**payload, "label": "Different Meaning"}
        self.assertIsNone(manager.create_or_reuse_bridge("P", conflict, ["C1"]))
        self.assertEqual(root, before)

    def test_id_conflicts_do_not_receive_random_suffixes(self):
        root = node("ROOT", "ROOT", "ROOT", [node("P", "parent", "L1"), node("X", "existing", "L1")])
        manager = TreeManager(root)
        self.assertFalse(manager.add_child_node("P", node("X", "conflict", "L2")))
        self.assertEqual(set(manager.index), {"ROOT", "P", "X"})

    def test_rename_rejects_new_exact_duplicate(self):
        root = node(
            "ROOT", "ROOT", "ROOT",
            [node("P", "parent", "L1", [node("A", "alpha", "L2"), node("B", "beta", "L2")])],
        )
        manager = TreeManager(root)
        self.assertFalse(manager.rename_node("B", "alpha"))
        self.assertEqual(manager.get_node("B")["label"], "beta")


if __name__ == "__main__":
    unittest.main()
