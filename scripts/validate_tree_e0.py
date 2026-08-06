#!/usr/bin/env python3
"""Deterministic E0 validator for a policy-tree candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict

from utils.tree_integrity import (
    atomic_write_json,
    read_jsonl,
    read_membership_csv,
    validate_tree_e0,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_lineage(path: Path) -> Dict[str, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("lineage JSON must be an object mapping source IDs to target IDs")
    return {str(source): str(target) for source, target in payload.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the PolicyTreeBuilder E0 contract")
    parser.add_argument("--tree", required=True, help="Candidate tree JSON")
    parser.add_argument("--membership", help="Final membership CSV")
    parser.add_argument("--lineage", help="Closed lineage JSON")
    parser.add_argument("--operations", help="Applied-operation JSONL")
    parser.add_argument("--report", help="Machine-readable E0 report JSON")
    parser.add_argument(
        "--require-membership",
        action="store_true",
        help="Fail when --membership is omitted",
    )
    args = parser.parse_args()

    tree_path = Path(args.tree)
    root = json.loads(tree_path.read_text(encoding="utf-8"))
    membership_rows = None
    if args.membership:
        _, membership_rows = read_membership_csv(args.membership)
    lineage = load_lineage(Path(args.lineage)) if args.lineage else None
    operations = read_jsonl(args.operations) if args.operations else None

    report = validate_tree_e0(
        root,
        membership_rows=membership_rows,
        lineage=lineage,
        operations=operations,
        require_membership=args.require_membership,
    )
    report["inputs"] = {
        "tree": str(tree_path),
        "tree_sha256": sha256_file(tree_path),
        "membership": args.membership,
        "lineage": args.lineage,
        "operations": args.operations,
    }
    if args.report:
        atomic_write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
