#!/usr/bin/env python3
"""Sample nodes and root-to-leaf paths for model-based tree evaluation."""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path

from eval_paths import default_output_dir, resolve_repo_path

DEFAULT_SEED = 20260430


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_dir(),
        help="Directory containing nodes.csv, paths.jsonl, and structure_report.json.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed used for deterministic sampling.",
    )
    return parser.parse_args()


def stratified_sample(rng: random.Random, level_nodes: list[dict], ratio: float) -> list[dict]:
    if not level_nodes:
        return []
    sample_size = max(1, math.ceil(len(level_nodes) * ratio))
    return rng.sample(level_nodes, min(sample_size, len(level_nodes)))


def main() -> None:
    args = parse_args()
    out_dir = resolve_repo_path(args.output_dir)
    rng = random.Random(args.seed)

    nodes = []
    with (out_dir / "nodes.csv").open("r", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            for key in ["depth", "children_count", "sibling_count", "is_leaf", "label_len"]:
                row[key] = int(row[key])
            nodes.append(row)

    nodes_by_id = {node["node_id"]: node for node in nodes}

    paths = []
    with (out_dir / "paths.jsonl").open("r", encoding="utf-8") as f:
        for line in f:
            paths.append(json.loads(line))

    with (out_dir / "structure_report.json").open("r", encoding="utf-8") as f:
        report = json.load(f)

    high_risk_ids: set[str] = set()

    for item in report["structural_warnings"]["sibling_duplicate_label_examples"]:
        high_risk_ids.update(item["node_ids"])

    for item in report["structural_warnings"]["cross_branch_same_label_examples"]:
        for occurrence in item["occurrences"]:
            high_risk_ids.add(occurrence["node_id"])

    oversized_parent_ids = {
        item["node_id"]
        for item in report["structural_warnings"]["oversized_parents"]
    }
    for node in nodes:
        if node["parent_id"] in oversized_parent_ids:
            high_risk_ids.add(node["node_id"])
    high_risk_ids.update(oversized_parent_ids)

    for item in report["structural_warnings"]["long_labels_examples"]:
        high_risk_ids.add(item["node_id"])

    for item in report["rule_based_issues"]["level_depth_mismatch_examples"]:
        high_risk_ids.add(item["node_id"])

    for node in nodes:
        if node["depth"] >= 5:
            high_risk_ids.add(node["node_id"])

    high_risk_ids.discard("ROOT")
    high_risk_ids = {
        node_id
        for node_id in high_risk_ids
        if node_id in nodes_by_id and nodes_by_id[node_id]["level"] != "ROOT"
    }
    print(f"high-risk nodes: {len(high_risk_ids)}")

    by_level: defaultdict[str, list[dict]] = defaultdict(list)
    for node in nodes:
        if node["level"] != "ROOT":
            by_level[node["level"]].append(node)

    sampled: list[dict] = []
    sampled_ids: set[str] = set()

    for level, ratio in [
        ("L1", 1.0),
        ("L2", 1.0),
        ("L3", 0.5),
        ("L4", 0.3),
        ("L5", 0.3),
        ("L6", 0.3),
    ]:
        chosen = by_level.get(level, []) if ratio == 1.0 else stratified_sample(rng, by_level.get(level, []), ratio)
        for node in chosen:
            if node["node_id"] not in sampled_ids:
                sampled.append(node)
                sampled_ids.add(node["node_id"])

    for node_id in sorted(high_risk_ids):
        if node_id not in sampled_ids:
            sampled.append(nodes_by_id[node_id])
            sampled_ids.add(node_id)

    sampled_by_level: defaultdict[str, int] = defaultdict(int)
    total_by_level: defaultdict[str, int] = defaultdict(int)
    for node in nodes:
        if node["level"] != "ROOT":
            total_by_level[node["level"]] += 1
    for node in sampled:
        sampled_by_level[node["level"]] += 1

    print(f"sampled nodes: {len(sampled)}")
    print("level coverage:")
    for level in ["L1", "L2", "L3", "L4", "L5", "L6"]:
        sampled_count = sampled_by_level[level]
        total_count = total_by_level[level]
        pct = sampled_count / total_count * 100 if total_count else 0
        print(f"  {level}: {sampled_count}/{total_count} ({pct:.0f}%)")

    children_by_parent: defaultdict[str, list[dict]] = defaultdict(list)
    for node in nodes:
        children_by_parent[node["parent_id"]].append(node)

    def get_children(node_id: str) -> list[dict]:
        return children_by_parent.get(node_id, [])

    def get_siblings(node_id: str) -> list[dict]:
        node = nodes_by_id[node_id]
        return [
            item
            for item in children_by_parent.get(node["parent_id"], [])
            if item["node_id"] != node_id
        ]

    samples_out = []
    sibling_duplicate_examples = report["structural_warnings"]["sibling_duplicate_label_examples"]
    cross_branch_examples = report["structural_warnings"]["cross_branch_same_label_examples"]
    mismatch_examples = report["rule_based_issues"]["level_depth_mismatch_examples"]

    for node in sampled:
        parent = nodes_by_id.get(node["parent_id"]) if node["parent_id"] else None
        risk_reasons = []
        if node["node_id"] in high_risk_ids:
            if any(node["node_id"] in item["node_ids"] for item in sibling_duplicate_examples):
                risk_reasons.append("sibling_duplicate_label")
            if any(
                node["node_id"] in [occ["node_id"] for occ in item["occurrences"]]
                for item in cross_branch_examples
            ):
                risk_reasons.append("cross_branch_same_label")
            if node["parent_id"] in oversized_parent_ids:
                risk_reasons.append("under_oversized_parent")
            if node["node_id"] in oversized_parent_ids:
                risk_reasons.append("oversized_parent")
            if node["label_len"] > 25:
                risk_reasons.append("label_too_long")
            if any(item["node_id"] == node["node_id"] for item in mismatch_examples):
                risk_reasons.append("level_depth_mismatch")
            if node["depth"] >= 5:
                risk_reasons.append("very_deep")

        samples_out.append({
            "sample_id": f"N{len(samples_out):04d}_{node['node_id']}",
            "current_node": {
                "node_id": node["node_id"],
                "label": node["label"],
                "level": node["level"],
                "depth": node["depth"],
            },
            "parent_node": {
                "node_id": parent["node_id"] if parent else "",
                "label": parent["label"] if parent else "",
                "level": parent["level"] if parent else "",
                "depth": parent["depth"] if parent else -1,
            } if parent else None,
            "children_nodes": [
                {"node_id": child["node_id"], "label": child["label"], "level": child["level"]}
                for child in get_children(node["node_id"])
            ],
            "sibling_nodes": [
                {"node_id": sibling["node_id"], "label": sibling["label"], "level": sibling["level"]}
                for sibling in get_siblings(node["node_id"])
            ],
            "path_from_root": node["path_labels"].split(">"),
            "meta": {
                "l1_label": node["l1_label"],
                "children_count": node["children_count"],
                "sibling_count": node["sibling_count"],
                "label_len": node["label_len"],
                "is_leaf": bool(node["is_leaf"]),
                "high_risk": node["node_id"] in high_risk_ids,
                "risk_reasons": risk_reasons,
            },
        })

    with (out_dir / "sampled_nodes_for_judge.jsonl").open("w", encoding="utf-8") as f:
        for record in samples_out:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"[OK] sampled_nodes_for_judge.jsonl: {len(samples_out)} samples")

    paths_by_l1: defaultdict[str, list[dict]] = defaultdict(list)
    for path in paths:
        paths_by_l1[path["l1_label"]].append(path)

    target_paths = 40
    l1_path_count = {l1: len(items) for l1, items in paths_by_l1.items()}
    total_paths = sum(l1_path_count.values())
    l1_quota = {
        l1: max(3, round(target_paths * count / total_paths))
        for l1, count in l1_path_count.items()
    }

    selected_paths: list[dict] = []
    selected_path_ids: set[str] = set()

    for l1, l1_paths in paths_by_l1.items():
        quota = l1_quota[l1]
        buckets: defaultdict[int, list[dict]] = defaultdict(list)
        for path in l1_paths:
            buckets[path["path_length"]].append(path)

        longest = max(path["path_length"] for path in l1_paths)
        pick_long = rng.sample(buckets[longest], min(2, len(buckets[longest])))
        for path in pick_long:
            if path["path_id"] not in selected_path_ids:
                selected_paths.append(path)
                selected_path_ids.add(path["path_id"])

        remaining = quota - len(pick_long)
        other_paths = [path for path in l1_paths if path["path_id"] not in selected_path_ids]
        if remaining > 0 and other_paths:
            for path in rng.sample(other_paths, min(remaining, len(other_paths))):
                selected_paths.append(path)
                selected_path_ids.add(path["path_id"])

    risk_paths = [
        path
        for path in paths
        if any(node_id in high_risk_ids for node_id in path["path_node_ids"])
    ]
    rng.shuffle(risk_paths)
    for path in risk_paths:
        if len(selected_paths) >= 50:
            break
        if path["path_id"] not in selected_path_ids:
            selected_paths.append(path)
            selected_path_ids.add(path["path_id"])

    global_longest = max(paths, key=lambda item: item["path_length"])
    if global_longest["path_id"] not in selected_path_ids:
        selected_paths.append(global_longest)
        selected_path_ids.add(global_longest["path_id"])

    selected_by_l1: defaultdict[str, int] = defaultdict(int)
    for path in selected_paths:
        selected_by_l1[path["l1_label"]] += 1

    print(f"sampled paths: {len(selected_paths)}")
    print("L1 path quota vs actual:")
    for l1 in l1_path_count:
        print(f"  {l1}: quota {l1_quota[l1]}, actual {selected_by_l1[l1]}, total {l1_path_count[l1]}")

    path_samples_out = []
    for path in selected_paths:
        path_nodes = [
            {"node_id": node_id, "label": label, "level": level, "depth": depth}
            for node_id, label, level, depth in zip(
                path["path_node_ids"],
                path["path_labels"],
                path["path_levels"],
                path["path_depths"],
            )
        ]
        path_samples_out.append({
            "sample_id": path["path_id"],
            "l1_label": path["l1_label"],
            "path_length": path["path_length"],
            "path": path_nodes,
            "meta": {
                "leaf_id": path["leaf_id"],
                "leaf_label": path["leaf_label"],
                "contains_high_risk": any(node["node_id"] in high_risk_ids for node in path_nodes),
                "is_global_longest": path["path_id"] == global_longest["path_id"],
            },
        })

    with (out_dir / "sampled_paths_for_judge.jsonl").open("w", encoding="utf-8") as f:
        for record in path_samples_out:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"[OK] sampled_paths_for_judge.jsonl: {len(path_samples_out)} paths")

    sampling_meta = {
        "random_seed": args.seed,
        "node_sample": {
            "total_population": len([node for node in nodes if node["level"] != "ROOT"]),
            "total_sampled": len(samples_out),
            "by_level": dict(sampled_by_level),
            "high_risk_included": len([sample for sample in samples_out if sample["meta"]["high_risk"]]),
            "rules": {
                "L1": "100%",
                "L2": "100%",
                "L3": "50%",
                "L4": "30%",
                "L5": "30%",
                "L6": "30%",
                "force_include": [
                    "sibling_duplicate_label",
                    "cross_branch_same_label",
                    "oversized_parent_and_children",
                    "label_too_long",
                    "level_depth_mismatch",
                    "depth_>=_5",
                ],
            },
        },
        "path_sample": {
            "total_population": len(paths),
            "total_sampled": len(path_samples_out),
            "by_l1": dict(selected_by_l1),
            "l1_quota": l1_quota,
            "rules": {
                "target": "~40 (cap 50)",
                "weighting": "by L1 path count, min 3 per L1",
                "force_include": ["longest path globally", "paths containing high-risk nodes"],
            },
        },
    }
    with (out_dir / "sampling_meta.json").open("w", encoding="utf-8") as f:
        json.dump(sampling_meta, f, ensure_ascii=False, indent=2)
    print("[OK] sampling_meta.json")


if __name__ == "__main__":
    main()
