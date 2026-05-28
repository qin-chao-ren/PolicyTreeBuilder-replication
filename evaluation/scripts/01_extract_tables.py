#!/usr/bin/env python3
"""Expand the final policy tree into node, edge, and root-to-leaf path tables."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from eval_paths import default_output_dir, default_tree_path, resolve_repo_path


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
        help="Directory for nodes.csv, edges.csv, and paths.jsonl.",
    )
    return parser.parse_args()


def get_l1_label(path_labels: str) -> str:
    parts = path_labels.split(">")
    return parts[1] if len(parts) >= 2 else ""


def main() -> None:
    args = parse_args()
    tree_path = resolve_repo_path(args.tree)
    out_dir = resolve_repo_path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with tree_path.open("r", encoding="utf-8") as f:
        tree = json.load(f)

    nodes: list[dict] = []
    edges: list[dict] = []
    node_by_id: dict[str, dict] = {}

    def walk(node: dict, parent: dict | None = None, depth: int = 0,
             path_ids: list[str] | None = None, path_labels: list[str] | None = None) -> None:
        path_ids = path_ids or []
        path_labels = path_labels or []

        node_id = node.get("node_id", "")
        label = node.get("label", "")
        level = node.get("level", "")
        children = node.get("children", []) or []

        new_path_ids = path_ids + [node_id]
        new_path_labels = path_labels + [label]

        row = {
            "node_id": node_id,
            "label": label,
            "level": level,
            "depth": depth,
            "parent_id": parent["node_id"] if parent else "",
            "parent_label": parent["label"] if parent else "",
            "parent_level": parent["level"] if parent else "",
            "children_count": len(children),
            "is_leaf": 1 if not children else 0,
            "path_ids": ">".join(new_path_ids),
            "path_labels": ">".join(new_path_labels),
            "label_len": len(label),
        }
        nodes.append(row)
        node_by_id[node_id] = row

        if parent is not None:
            edges.append({
                "parent_id": parent["node_id"],
                "parent_label": parent["label"],
                "parent_level": parent["level"],
                "child_id": node_id,
                "child_label": label,
                "child_level": level,
                "edge_depth": depth,
            })

        for child in children:
            walk(child, parent=node, depth=depth + 1,
                 path_ids=new_path_ids, path_labels=new_path_labels)

    walk(tree)

    siblings_count_by_parent: dict[str, int] = {}
    for node in nodes:
        parent_id = node["parent_id"]
        siblings_count_by_parent[parent_id] = siblings_count_by_parent.get(parent_id, 0) + 1
    for node in nodes:
        node["sibling_count"] = max(0, siblings_count_by_parent.get(node["parent_id"], 1) - 1)
        node["l1_label"] = get_l1_label(node["path_labels"])

    nodes_csv = out_dir / "nodes.csv"
    node_fields = [
        "node_id", "label", "level", "depth", "parent_id", "parent_label",
        "parent_level", "children_count", "sibling_count", "is_leaf",
        "label_len", "l1_label", "path_ids", "path_labels",
    ]
    with nodes_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=node_fields)
        writer.writeheader()
        for node in nodes:
            writer.writerow({key: node.get(key, "") for key in node_fields})

    edges_csv = out_dir / "edges.csv"
    edge_fields = [
        "parent_id", "parent_label", "parent_level",
        "child_id", "child_label", "child_level", "edge_depth",
    ]
    with edges_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=edge_fields)
        writer.writeheader()
        writer.writerows(edges)

    paths_jsonl = out_dir / "paths.jsonl"
    path_count = 0
    with paths_jsonl.open("w", encoding="utf-8") as f:
        for node in nodes:
            if node["is_leaf"] != 1:
                continue
            ids = node["path_ids"].split(">")
            labels = node["path_labels"].split(">")
            record = {
                "path_id": f"P{path_count:04d}_{node['node_id']}",
                "leaf_id": node["node_id"],
                "leaf_label": node["label"],
                "l1_label": labels[1] if len(labels) >= 2 else "",
                "path_node_ids": ids,
                "path_labels": labels,
                "path_levels": [node_by_id[node_id]["level"] for node_id in ids],
                "path_depths": [node_by_id[node_id]["depth"] for node_id in ids],
                "path_length": len(ids),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            path_count += 1

    print(f"[OK] nodes.csv  : {len(nodes)} rows -> {nodes_csv}")
    print(f"[OK] edges.csv  : {len(edges)} rows -> {edges_csv}")
    print(f"[OK] paths.jsonl: {path_count} paths -> {paths_jsonl}")

    level_dist = Counter(node["level"] for node in nodes)
    depth_dist = Counter(node["depth"] for node in nodes)
    l1_dist = Counter(node["l1_label"] for node in nodes if node["l1_label"])
    print("\nlevel distribution:", dict(level_dist))
    print("depth distribution:", dict(depth_dist))
    print("L1 node counts:")
    for key, value in sorted(l1_dist.items(), key=lambda item: -item[1]):
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
