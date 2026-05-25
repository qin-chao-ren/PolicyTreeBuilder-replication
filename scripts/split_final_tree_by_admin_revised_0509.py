#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Split a final policy action tree by administrative level/name.

The script can also apply a node_id-based label map before splitting.
This is useful when the same Chinese tree structure should be exported with
revised academic English labels without changing node ids or memberships.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_TREE = PROJECT_ROOT / "data" / "final_tree" / "v4_tree_final.json"
DEFAULT_OUTDIR = PROJECT_ROOT / "data" / "final_tree"
DEFAULT_MEMBERSHIP_DIR = PROJECT_ROOT / "data" / "intermediate_outputs"
DEFAULT_ADMIN_MAP = PROJECT_ROOT / "data" / "source" / "admin_mapping" / "roundA_final_overview_scored_selected1120.csv"

PROVINCE_LEVEL = "省级"
CITY_LEVEL = "市级"


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def apply_label_map(root: dict[str, Any], label_map: dict[str, str]) -> dict[str, Any]:
    """Return a copy of root whose labels are overwritten by node_id-based label_map."""
    node = deepcopy(root)
    node_id = str(node.get("node_id", ""))
    if node_id in label_map:
        node["label"] = label_map[node_id]
    node["children"] = [apply_label_map(child, label_map) for child in node.get("children") or []]
    return node


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def collect_tree_node_ids(node: dict[str, Any], out: set[str]) -> None:
    node_id = node.get("node_id")
    if node_id:
        out.add(str(node_id))
    for child in node.get("children") or []:
        collect_tree_node_ids(child, out)


def prune_tree(node: dict[str, Any], valid_node_ids: set[str]) -> dict[str, Any] | None:
    node_id = str(node.get("node_id", ""))
    kept_children = []
    for child in node.get("children") or []:
        pruned = prune_tree(child, valid_node_ids)
        if pruned is not None:
            kept_children.append(pruned)

    if node_id in valid_node_ids or kept_children:
        new_node = deepcopy(node)
        new_node["children"] = kept_children
        return new_node
    return None


def safe_filename(name: str) -> str:
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name).strip()
    cleaned = re.sub(r"\s+", "_", cleaned)
    return cleaned or "UNKNOWN"


def build_sample_records(
    membership_dir: Path,
    admin_map_path: Path,
    tree_node_ids: set[str],
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    admin_rows = read_csv_rows(admin_map_path)
    membership_rows: list[dict[str, str]] = []
    membership_counts: dict[str, int] = {}
    for level in ["L4", "L3", "L2"]:
        path = membership_dir / f"v4_membership_{level}.csv"
        rows = read_csv_rows(path)
        membership_counts[level] = len(rows)
        for row in rows:
            normalized = dict(row)
            normalized["membership_level"] = level
            membership_rows.append(normalized)

    admin_by_doc = {
        row.get("doc_id", "").strip(): row
        for row in admin_rows
        if row.get("doc_id", "").strip()
    }
    membership_doc_ids = {
        ((row.get("member_id", "") or row.get("sample_id", "")).strip().split("_", 1)[0])
        for row in membership_rows
        if (row.get("member_id", "") or row.get("sample_id", "")).strip()
    }
    admin_docs_without_membership = [
        {
            "doc_id": row.get("doc_id", "").strip(),
            "admin_level": row.get("admin_level", "").strip(),
            "admin_name": row.get("admin_name", "").strip(),
            "title": row.get("title", "").strip(),
        }
        for row in admin_rows
        if row.get("doc_id", "").strip()
        and row.get("doc_id", "").strip() not in membership_doc_ids
    ]

    stats = {
        "membership_dir": str(membership_dir),
        "membership_rows": len(membership_rows),
        "membership_rows_by_level": membership_counts,
        "admin_rows": len(admin_rows),
        "admin_doc_count": len(admin_by_doc),
        "membership_doc_count": len(membership_doc_ids),
        "admin_docs_without_membership": admin_docs_without_membership,
        "admin_units_without_membership": sorted(
            {
                item["admin_name"]
                for item in admin_docs_without_membership
                if item["admin_name"]
            }
        ),
        "samples_missing_admin": 0,
        "membership_rows_missing_tree_node": 0,
        "linked_samples_on_tree": 0,
        "linked_membership_records_on_tree": 0,
    }

    records: list[dict[str, str]] = []
    linked_sample_ids: set[str] = set()
    for row in membership_rows:
        sample_id = row.get("member_id", "").strip() or row.get("sample_id", "").strip()
        if not sample_id:
            continue
        node_id = row.get("node_id", "").strip()
        doc_id = sample_id.split("_", 1)[0]
        admin = admin_by_doc.get(doc_id)
        if not admin:
            stats["samples_missing_admin"] += 1
            continue

        if node_id not in tree_node_ids:
            stats["membership_rows_missing_tree_node"] += 1
            continue

        linked_sample_ids.add(sample_id)
        records.append(
            {
                "sample_id": sample_id,
                "mapped_node_id": node_id,
                "membership_level": row.get("membership_level", ""),
                "doc_id": doc_id,
                "admin_level": admin.get("admin_level", "").strip(),
                "admin_name": admin.get("admin_name", "").strip(),
            }
        )
    stats["linked_samples_on_tree"] = len(linked_sample_ids)
    stats["linked_membership_records_on_tree"] = len(records)
    return records, stats


def write_split_tree(
    root: dict[str, Any],
    node_ids: set[str],
    output_path: Path,
) -> bool:
    pruned = prune_tree(root, node_ids)
    if pruned is None:
        return False
    write_json(output_path, pruned)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Split v4 final tree by province/city/admin name.")
    parser.add_argument("--tree", default=str(DEFAULT_TREE), help="Input v4_tree_final.json")
    parser.add_argument("--outdir", default=str(DEFAULT_OUTDIR), help="Output directory")
    parser.add_argument("--membership-dir", default=str(DEFAULT_MEMBERSHIP_DIR), help="Directory containing v4_membership_L4/L3/L2.csv")
    parser.add_argument("--admin-map", default=str(DEFAULT_ADMIN_MAP), help="RoundA admin metadata CSV")
    parser.add_argument("--label-map", default="", help="Optional node_id -> English label JSON. Use this to split the Chinese tree while writing revised English labels.")
    parser.add_argument("--output-stem", default="v4_tree_final", help="Output file stem. Default preserves the original file names.")
    parser.add_argument("--output-suffix", default="", help="Optional suffix before .json, e.g. _en_academic_0509.")
    args = parser.parse_args()

    tree_path = Path(args.tree)
    outdir = Path(args.outdir)
    membership_dir = Path(args.membership_dir)
    admin_map_path = Path(args.admin_map)

    print(f"[1/5] Loading final tree: {tree_path}")
    root = load_json(tree_path)
    label_map_path = Path(args.label_map) if args.label_map else None
    if label_map_path:
        print(f"      Applying label map: {label_map_path}")
        raw_label_map = load_json(label_map_path)
        if not isinstance(raw_label_map, dict):
            raise TypeError("--label-map must point to a JSON object keyed by node_id")
        root = apply_label_map(root, {str(k): str(v) for k, v in raw_label_map.items()})
    tree_node_ids: set[str] = set()
    collect_tree_node_ids(root, tree_node_ids)
    print(f"      Tree nodes: {len(tree_node_ids)}")

    print("[2/5] Building sample/admin/membership-node records")
    records, stats = build_sample_records(membership_dir, admin_map_path, tree_node_ids)
    missing_admin_units = stats.get("admin_units_without_membership") or []
    if missing_admin_units:
        print(
            "      [WARN] Admin units with no membership rows: "
            + ", ".join(missing_admin_units)
        )

    province_node_ids = {r["mapped_node_id"] for r in records if PROVINCE_LEVEL in r["admin_level"]}
    city_node_ids = {r["mapped_node_id"] for r in records if CITY_LEVEL in r["admin_level"]}

    province_sample_ids = {r["sample_id"] for r in records if PROVINCE_LEVEL in r["admin_level"]}
    city_sample_ids = {r["sample_id"] for r in records if CITY_LEVEL in r["admin_level"]}
    province_records = sum(1 for r in records if PROVINCE_LEVEL in r["admin_level"])
    city_records = sum(1 for r in records if CITY_LEVEL in r["admin_level"])
    print(f"      Linked unique samples on tree: {stats['linked_samples_on_tree']}")
    print(f"      Linked membership records on tree: {stats['linked_membership_records_on_tree']}")
    print(f"      Province samples/records/nodes: {len(province_sample_ids)}/{province_records}/{len(province_node_ids)}")
    print(f"      City samples/records/nodes: {len(city_sample_ids)}/{city_records}/{len(city_node_ids)}")

    print(f"[3/5] Writing level trees to: {outdir}")
    outdir.mkdir(parents=True, exist_ok=True)
    output_stem = safe_filename(args.output_stem)
    output_suffix = safe_filename(args.output_suffix) if args.output_suffix else ""
    level_outputs = {
        "province_tree": str(outdir / f"{output_stem}_provincial{output_suffix}.json"),
        "city_tree": str(outdir / f"{output_stem}_city{output_suffix}.json"),
    }
    wrote_province = write_split_tree(root, province_node_ids, Path(level_outputs["province_tree"]))
    wrote_city = write_split_tree(root, city_node_ids, Path(level_outputs["city_tree"]))

    print("[4/5] Writing one tree per administrative unit")
    by_admin: dict[str, set[str]] = defaultdict(set)
    admin_levels: dict[str, str] = {}
    admin_sample_counts: dict[str, int] = defaultdict(int)
    for record in records:
        level = record["admin_level"]
        name = record["admin_name"]
        if not name:
            continue
        if PROVINCE_LEVEL not in level and CITY_LEVEL not in level:
            continue
        by_admin[name].add(record["mapped_node_id"])
        admin_levels[name] = level
        admin_sample_counts[name] += 1

    admin_dir = outdir / "trees_by_admin"
    admin_dir.mkdir(parents=True, exist_ok=True)
    admin_outputs = []
    for admin_name in sorted(by_admin):
        output_path = admin_dir / f"tree_{safe_filename(admin_name)}{output_suffix}.json"
        if write_split_tree(root, by_admin[admin_name], output_path):
            admin_outputs.append(
                {
                    "admin_name": admin_name,
                    "admin_level": admin_levels.get(admin_name, ""),
                    "membership_record_count": admin_sample_counts[admin_name],
                    "node_count": len(by_admin[admin_name]),
                    "path": str(output_path),
                }
            )

    print("[5/5] Writing summary")
    summary = {
        "input_tree": str(tree_path),
        "output_dir": str(outdir),
        "label_map": str(label_map_path) if label_map_path else "",
        "output_stem": args.output_stem,
        "output_suffix": args.output_suffix,
        "tree_nodes": len(tree_node_ids),
        "stats": stats,
        "province": {
            "wrote": wrote_province,
            "sample_count": len(province_sample_ids),
            "membership_record_count": province_records,
            "node_count": len(province_node_ids),
            "path": level_outputs["province_tree"],
        },
        "city": {
            "wrote": wrote_city,
            "sample_count": len(city_sample_ids),
            "membership_record_count": city_records,
            "node_count": len(city_node_ids),
            "path": level_outputs["city_tree"],
        },
        "admin_tree_count": len(admin_outputs),
        "admin_outputs": admin_outputs,
    }
    write_json(outdir / "split_final_tree_by_admin_summary.json", summary)

    print("[DONE]")
    print(f"      Province tree written: {wrote_province}")
    print(f"      City tree written: {wrote_city}")
    print(f"      Admin trees written: {len(admin_outputs)}")
    print(f"      Summary: {outdir / 'split_final_tree_by_admin_summary.json'}")


if __name__ == "__main__":
    main()
