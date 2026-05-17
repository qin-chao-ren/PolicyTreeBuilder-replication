#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 3 · 构建全局粗树（v4）
- 汇总 v4_nodes/v4_links 生成 data/intermediate_outputs/v4_tree_coarse_global.json。
- 可选 --emit-samples yes：在 L4 之下展开样本叶子（默认只到 L4）。
- 额外导出 v4_tree_flat.csv、v4_tree_levels.csv 供 Step 4 / 复核使用。
"""
from __future__ import annotations

import argparse
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

from typing import Dict, List
import pandas as pd
import yaml
from common_utils import ensure_outdir, safe_write_csv, safe_write_json, read_corpus


def _load_nodes(outdir: Path, level: str) -> pd.DataFrame:
    path = outdir / f"v4_nodes_{level}.csv"
    if not path.exists():
        return pd.DataFrame(columns=["node_id", "label", "level"])
    return pd.read_csv(path, dtype=str)


def _load_links(outdir: Path, from_level: str, to_level: str) -> pd.DataFrame:
    path = outdir / f"v4_links_{from_level}_to_{to_level}.csv"
    if not path.exists():
        return pd.DataFrame(columns=["child_id", "parent_id"])
    return pd.read_csv(path, dtype=str)


def build_tree(outdir: Path, corpus_path: Path | None, emit_samples: bool = False) -> Dict:
    nL1 = _load_nodes(outdir, "L1")
    nL2 = _load_nodes(outdir, "L2")
    nL3 = _load_nodes(outdir, "L3")
    nL4 = _load_nodes(outdir, "L4")
    mL4_path = outdir / "v4_membership_L4.csv"
    mL4 = pd.read_csv(mL4_path, dtype=str) if mL4_path.exists() else pd.DataFrame(columns=["node_id", "member_id"])

    links_43 = _load_links(outdir, "L4", "L3")
    links_32 = _load_links(outdir, "L3", "L2")
    links_21 = _load_links(outdir, "L2", "L1")

    L1 = {r["node_id"]: {"node_id": r["node_id"], "label": r.get("label", ""), "level": "L1", "children": []} for _, r in nL1.iterrows()}
    L2 = {r["node_id"]: {"node_id": r["node_id"], "label": r.get("label", ""), "level": "L2", "children": []} for _, r in nL2.iterrows()}
    L3 = {r["node_id"]: {"node_id": r["node_id"], "label": r.get("label", ""), "level": "L3", "children": []} for _, r in nL3.iterrows()}
    L4 = {r["node_id"]: {"node_id": r["node_id"], "label": r.get("label", ""), "level": "L4", "children": []} for _, r in nL4.iterrows()}

    corpus = read_corpus(corpus_path) if emit_samples and corpus_path and corpus_path.exists() else None
    sid2title = {str(r["sample_id"]): str(r["cleaned_title"]) for _, r in corpus.iterrows()} if corpus is not None else {}
    members_L4 = {nid: grp["member_id"].astype(str).tolist() for nid, grp in mL4.groupby("node_id")}

    for _, row in links_21.iterrows():
        parent = L1.get(row["parent_id"])
        child = L2.get(row["child_id"])
        if parent and child:
            parent.setdefault("children", []).append(child)

    for _, row in links_32.iterrows():
        parent = L2.get(row["parent_id"])
        child = L3.get(row["child_id"])
        if parent and child:
            parent.setdefault("children", []).append(child)

    for _, row in links_43.iterrows():
        parent = L3.get(row["parent_id"])
        child = L4.get(row["child_id"])
        if not parent or not child:
            continue
        if emit_samples:
            child_samples = []
            for sid in members_L4.get(child["node_id"], []):
                # [修复 1]：为 Sample 节点添加 node_id，下游脚本需要它
                child_samples.append({"node_id": sid, "level": "Sample", "sample_id": sid, "label": sid2title.get(sid, "")})
            child["children"] = child_samples
        parent.setdefault("children", []).append(child)

    # [修复 2]：为 ROOT 节点添加 node_id，下游脚本需要它
    root = {"node_id": "ROOT", "level": "ROOT", "label": "ROOT", "children": list(L1.values())}
    return root

def write_flat(tree: Dict, outdir: Path, emit_samples: bool):
    rows = []

    def dfs(node: Dict, path_ids: List[str], path_labels: List[str]):
        level = node.get("level")
        if emit_samples:
            if level == "Sample":
                rows.append(
                    {
                        "leaf_id": node.get("sample_id"),
                        "path_ids": "/".join(path_ids + [node.get("sample_id") or ""]),
                        "path_labels": "/".join(path_labels + [node.get("label") or ""]),
                    }
                )
                return
        else:
            if level == "L4":
                rows.append(
                    {
                        "leaf_id": node.get("node_id"),
                        "path_ids": "/".join(path_ids + [node.get("node_id") or ""]),
                        "path_labels": "/".join(path_labels + [node.get("label") or ""]),
                    }
                )
                return
        for ch in node.get("children", []) or []:
            if ch.get("level") in {"L1", "L2", "L3"}:
                dfs(ch, path_ids + [ch.get("node_id")], path_labels + [ch.get("label") or ""])
            else:
                dfs(ch, path_ids, path_labels)

    for l1 in tree.get("children", []) or []:
        dfs(l1, [l1.get("node_id")], [l1.get("label") or ""])

    flat = pd.DataFrame(rows)
    safe_write_csv(flat, outdir / "v4_tree_flat.csv")

    lev_rows = []
    for _, r in flat.iterrows():
        labels = (r["path_labels"] or "").split("/")
        lev_rows.append(
            {
                ("sample_id" if emit_samples else "node_id"): r["leaf_id"],
                "L1": labels[0] if len(labels) > 0 else "",
                "L2": labels[1] if len(labels) > 1 else "",
                "L3": labels[2] if len(labels) > 2 else "",
            }
        )
    safe_write_csv(pd.DataFrame(lev_rows), outdir / "v4_tree_levels.csv")


def main():
    parser = argparse.ArgumentParser(description="Round C v4 · Step3 build coarse tree")
    parser.add_argument("--config", required=True)
    parser.add_argument("--emit-samples", choices=["yes", "no"], default="no")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    outdir = Path(cfg.get("paths", {}).get("outdir", ROOT / "outputs"))
    ensure_outdir(outdir)
    corpus_path = Path(cfg.get("paths", {}).get("corpus", outdir / "v4_corpus_calibrated.csv"))
    emit_samples = args.emit_samples == "yes"

    tree = build_tree(outdir, corpus_path, emit_samples=emit_samples)
    safe_write_json(outdir / "v4_tree_coarse_global.json", tree)
    write_flat(tree, outdir, emit_samples=emit_samples)
    log_path = ROOT / "data" / "intermediate_outputs" / "logs" / "step3_build_coarse_tree.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(f"[TREE] emit_samples={emit_samples} written\n")
    print(f"[WRITE] v4_tree_coarse_global.json in {outdir}")


if __name__ == "__main__":
    main()
