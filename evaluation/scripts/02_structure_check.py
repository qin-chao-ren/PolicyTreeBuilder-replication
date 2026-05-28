#!/usr/bin/env python3
"""Run deterministic structure checks for the policy tree evaluation."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from eval_paths import default_output_dir, default_tree_path, resolve_repo_path

LEVEL_TO_DEPTH = {"ROOT": 0, "L1": 1, "L2": 2, "L3": 3, "L4": 4, "L5": 5, "L6": 6}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tree",
        type=Path,
        default=default_tree_path(),
        help="Input tree JSON. Defaults to data/final_tree/policy_tree_final.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_dir(),
        help="Directory containing nodes.csv and edges.csv.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tree_path = resolve_repo_path(args.tree)
    out_dir = resolve_repo_path(args.output_dir)

    with tree_path.open("r", encoding="utf-8") as f:
        raw_tree = json.load(f)

    parent_of_seen: defaultdict[str, list[str]] = defaultdict(list)
    all_ids_seen: list[str] = []
    empty_label_ids: list[str] = []
    bad_children_field: list[str] = []

    def check_raw(node: dict, parent_id: str | None = None) -> None:
        node_id = node.get("node_id")
        label = node.get("label", "")
        all_ids_seen.append(node_id)
        if not label or label.strip() == "":
            empty_label_ids.append(node_id)
        if "children" in node and not isinstance(node["children"], list):
            bad_children_field.append(node_id)
        if parent_id is not None:
            parent_of_seen[node_id].append(parent_id)
        for child in node.get("children", []) or []:
            check_raw(child, parent_id=node_id)

    def detect_cycles(node: dict, ancestors: set[str] | None = None) -> list[str]:
        ancestors = ancestors or set()
        node_id = node.get("node_id")
        if node_id in ancestors:
            return [node_id]
        circular: list[str] = []
        next_ancestors = ancestors | {node_id}
        for child in node.get("children", []) or []:
            circular.extend(detect_cycles(child, next_ancestors))
        return circular

    check_raw(raw_tree)
    duplicate_ids = [key for key, value in Counter(all_ids_seen).items() if value > 1]
    multi_parent = {key: value for key, value in parent_of_seen.items() if len(set(value)) > 1}
    circular_ids = sorted(set(detect_cycles(raw_tree)))

    nodes = []
    with (out_dir / "nodes.csv").open("r", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            for key in ["depth", "children_count", "sibling_count", "is_leaf", "label_len"]:
                row[key] = int(row[key])
            nodes.append(row)

    orphan = [node for node in nodes if node["level"] != "ROOT" and not node["parent_id"]]

    mismatch = []
    for node in nodes:
        expected_depth = LEVEL_TO_DEPTH.get(node["level"])
        if expected_depth is not None and expected_depth != node["depth"]:
            mismatch.append({
                "node_id": node["node_id"],
                "label": node["label"],
                "level": node["level"],
                "depth": node["depth"],
                "expected_depth": expected_depth,
                "parent_label": node["parent_label"],
                "path_labels": node["path_labels"],
            })

    oversized = [node for node in nodes if node["children_count"] > 8]
    oversized_sorted = sorted(oversized, key=lambda item: -item["children_count"])
    long_labels = [node for node in nodes if node["label_len"] > 25]

    by_parent: defaultdict[str, list[dict]] = defaultdict(list)
    for node in nodes:
        by_parent[node["parent_id"]].append(node)

    sibling_dup_label = []
    for parent_id, group in by_parent.items():
        label_counts = Counter(item["label"] for item in group)
        for label, count in label_counts.items():
            if count > 1 and label.strip():
                sibling_dup_label.append({
                    "parent_id": parent_id,
                    "parent_label": group[0]["parent_label"] if group else "",
                    "duplicated_label": label,
                    "count": count,
                    "node_ids": [item["node_id"] for item in group if item["label"] == label],
                })

    parent_child_same = [
        {"node_id": node["node_id"], "label": node["label"], "parent_id": node["parent_id"]}
        for node in nodes
        if node["parent_label"] and node["label"] == node["parent_label"]
    ]

    label_to_nodes: defaultdict[str, list[dict]] = defaultdict(list)
    for node in nodes:
        if node["level"] != "ROOT":
            label_to_nodes[node["label"]].append(node)

    cross_branch_same_label = []
    for label, group in label_to_nodes.items():
        distinct_parents = {item["parent_label"] for item in group}
        if len(group) > 1 and len(distinct_parents) > 1:
            cross_branch_same_label.append({
                "label": label,
                "count": len(group),
                "occurrences": [
                    {
                        "node_id": item["node_id"],
                        "parent_label": item["parent_label"],
                        "level": item["level"],
                    }
                    for item in group
                ],
            })

    internal = [node for node in nodes if node["children_count"] > 0]
    avg_children = sum(node["children_count"] for node in internal) / max(1, len(internal))
    max_depth = max(node["depth"] for node in nodes)
    level_dist = dict(Counter(node["level"] for node in nodes))
    depth_dist = dict(Counter(node["depth"] for node in nodes))
    l1_dist = dict(Counter(node["l1_label"] for node in nodes if node["l1_label"]))

    l1_leaf: defaultdict[str, int] = defaultdict(int)
    l1_max_depth: defaultdict[str, int] = defaultdict(int)
    for node in nodes:
        if not node["l1_label"]:
            continue
        if node["is_leaf"]:
            l1_leaf[node["l1_label"]] += 1
        l1_max_depth[node["l1_label"]] = max(l1_max_depth[node["l1_label"]], node["depth"])

    with (out_dir / "edges.csv").open("r", encoding="utf-8-sig") as f:
        total_edges = sum(1 for _ in f) - 1

    report = {
        "tree_basic_stats": {
            "total_nodes": len(nodes),
            "total_edges": total_edges,
            "total_leaf_nodes": sum(1 for node in nodes if node["is_leaf"]),
            "max_depth": max_depth,
            "level_distribution": level_dist,
            "depth_distribution": depth_dist,
            "l1_node_count": l1_dist,
            "l1_leaf_count": dict(l1_leaf),
            "l1_max_depth": dict(l1_max_depth),
            "avg_children_per_internal_node": round(avg_children, 2),
            "internal_node_count": len(internal),
            "oversized_parent_count": len(oversized),
        },
        "rule_based_issues": {
            "duplicate_node_id_count": len(duplicate_ids),
            "duplicate_node_ids": duplicate_ids,
            "orphan_node_count": len(orphan),
            "orphan_node_ids": [node["node_id"] for node in orphan],
            "multi_parent_node_count": len(multi_parent),
            "multi_parent_nodes": multi_parent,
            "circular_reference_count": len(circular_ids),
            "circular_reference_ids": circular_ids,
            "empty_label_count": len(empty_label_ids),
            "empty_label_ids": empty_label_ids,
            "bad_children_field_count": len(bad_children_field),
            "bad_children_field_ids": bad_children_field,
            "level_depth_mismatch_count": len(mismatch),
            "level_depth_mismatch_examples": mismatch[:20],
        },
        "structural_warnings": {
            "oversized_parents": [
                {
                    "node_id": node["node_id"],
                    "label": node["label"],
                    "level": node["level"],
                    "children_count": node["children_count"],
                }
                for node in oversized_sorted
            ],
            "long_labels_count": len(long_labels),
            "long_labels_examples": [
                {"node_id": node["node_id"], "label": node["label"], "label_len": node["label_len"]}
                for node in sorted(long_labels, key=lambda item: -item["label_len"])[:20]
            ],
            "sibling_duplicate_label_count": len(sibling_dup_label),
            "sibling_duplicate_label_examples": sibling_dup_label[:20],
            "parent_child_same_label_count": len(parent_child_same),
            "parent_child_same_label_examples": parent_child_same[:20],
            "cross_branch_same_label_count": len(cross_branch_same_label),
            "cross_branch_same_label_examples": sorted(
                cross_branch_same_label, key=lambda item: -item["count"]
            )[:20],
        },
    }

    with (out_dir / "structure_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(json.dumps({
        "tree_basic_stats": report["tree_basic_stats"],
        "rule_based_issues_summary": {
            key: value
            for key, value in report["rule_based_issues"].items()
            if key.endswith("_count")
        },
        "structural_warnings_summary": {
            "oversized_parent_count": len(oversized),
            "long_labels_count": len(long_labels),
            "sibling_duplicate_label_count": len(sibling_dup_label),
            "parent_child_same_label_count": len(parent_child_same),
            "cross_branch_same_label_count": len(cross_branch_same_label),
        },
    }, ensure_ascii=False, indent=2))
    print(f"\n[OK] -> {out_dir / 'structure_report.json'}")


if __name__ == "__main__":
    main()
