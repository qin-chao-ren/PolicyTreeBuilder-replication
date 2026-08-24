#!/usr/bin/env python3
"""Offline probe for the 14c sibling-merge prefilter (debt item D12).

Reproduces, with zero API calls, the measurements recorded in
KNOWN_LIMITATIONS.md section 2:

  * the similarity scores of the known escaping pair;
  * whether the node label is actually absent from the node vector;
  * threshold sensitivity over every sibling pair of the 14c input tree;
  * the section-heading observation, with its own control comparison.

Everything is computed from committed artifacts plus a precomputed embedding
parquet. No model or embedding service is contacted.

Usage
-----
    python probe_d12_prefilter.py \
        --scripts   /path/to/repo/scripts \
        --embeddings /path/to/policy_corpus_embeddings.parquet \
        --intermediate /path/to/intermediate_outputs \
        --tree policy_tree_after_structure_balancing.json \
        --out-csv sibling_pairs.csv

`--intermediate` must contain the tree named by `--tree` (the output of 14b,
which is what 14c consumes) and the `tree_node_membership_L*.csv` rosters.

Requires pandas, numpy and pyarrow, and imports the package's own
`common_utils` / `step4_shared` so that the tokenizer, the centroid helper and
the cosine function are the production ones rather than reimplementations.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scripts", required=True, type=Path,
                   help="repository scripts/ directory (for common_utils and utils/step4_shared)")
    p.add_argument("--embeddings", required=True, type=Path,
                   help="precomputed policy_corpus_embeddings.parquet")
    p.add_argument("--intermediate", required=True, type=Path,
                   help="directory holding the 14c input tree and membership CSVs")
    p.add_argument("--tree", default="policy_tree_after_structure_balancing.json",
                   help="filename of the 14c input tree inside --intermediate")
    p.add_argument("--out-csv", type=Path, default=None,
                   help="optional path to write the full per-pair score table")
    p.add_argument("--thr-jac", type=float, default=0.75, help="production lexical threshold")
    p.add_argument("--thr-cos", type=float, default=0.85, help="production vector threshold")
    return p.parse_args()


# The pair that halted an earlier run: same parent, semantically equivalent,
# rejected by both prefilter channels.
KNOWN_PAIR_LABELS = ("构建全球国际航线网络", "构建覆盖全球的货运航线网络")

# The near-synonym family this project exists to collapse.
FAMILY_LABELS = (
    "构建全球国际航线网络",
    "构建覆盖全球的货运航线网络",
    "构建全球航空货运网络",
    "构建优质航线网络",
)

LEVELS = ("L4", "L3", "L2", "L1")


def main() -> int:
    args = parse_args()

    sys.path.insert(0, str(args.scripts))
    sys.path.insert(0, str(args.scripts / "utils"))
    import pandas as pd
    from common_utils import cosine_sim as mapped_cosine, jaccard_overlap, tokenize_label
    from step4_shared import EmbeddingHelper, read_membership_map

    emb = EmbeddingHelper(args.embeddings)
    membership = {lvl: read_membership_map(args.intermediate, lvl) for lvl in LEVELS}
    tree = json.loads((args.intermediate / args.tree).read_text(encoding="utf-8"))

    embedded_text = {
        str(row["sample_id"]): str(row["text_for_embed"])
        for _, row in pd.read_parquet(args.embeddings).iterrows()
    }

    def members(node: dict) -> list[str]:
        return membership.get(node.get("level"), {}).get(node.get("node_id"), [])

    def cosine(node_a: dict, node_b: dict) -> float:
        return EmbeddingHelper.cosine_sim(emb.get_centroid(members(node_a)),
                                          emb.get_centroid(members(node_b)))

    nodes: list[dict] = []
    pairs: list[tuple[str, dict, dict]] = []

    def walk(node: dict) -> None:
        nodes.append(node)
        children = node.get("children") or []
        for i in range(len(children)):
            for j in range(i + 1, len(children)):
                pairs.append((node.get("node_id"), children[i], children[j]))
        for child in children:
            walk(child)

    walk(tree)

    print(f"input tree      : {args.tree}")
    print(f"nodes           : {len(nodes)}")
    print(f"sibling pairs   : {len(pairs)}")
    print(f"embedding cache : {len(emb.cache)} vectors")

    # --- 1. the known pair, and whether its vector carries the label ----------
    print("\n=== 1. known escaping pair ===")
    by_label = {n.get("label"): n for n in nodes}
    node_a, node_b = (by_label.get(lbl) for lbl in KNOWN_PAIR_LABELS)
    if node_a and node_b:
        jac = jaccard_overlap(node_a["label"], node_b["label"])
        raw = cosine(node_a, node_b)
        print(f"  tokens A : {tokenize_label(node_a['label'])}")
        print(f"  tokens B : {tokenize_label(node_b['label'])}")
        print(f"  jaccard  : {jac:.4f}  (threshold {args.thr_jac}) -> "
              f"{'PASS' if jac >= args.thr_jac else 'FAIL'}")
        print(f"  cosine   : {raw:.6f}  (threshold {args.thr_cos}) -> "
              f"{'PASS' if raw >= args.thr_cos else 'FAIL'}")
        # The mapped variant rescales to [0,1] and would suggest the opposite
        # conclusion; production uses the bare cosine above.
        print(f"  cosine, [0,1]-mapped variant (NOT the production channel): "
              f"{mapped_cosine(emb.get_centroid(members(node_a)), emb.get_centroid(members(node_b))):.6f}")
        print(f"  verdict  : {'reaches model' if (jac >= args.thr_jac or raw >= args.thr_cos) else 'NEVER reaches model'}")

        print("\n  is the label absent from the vector?")
        for node in (node_a, node_b):
            member_ids = members(node)
            if len(member_ids) == 1:
                text = embedded_text.get(str(member_ids[0]), "")
                print(f"    {node['node_id']}: label == embedded text -> {text == node['label']}")
                if text != node["label"]:
                    print(f"      label : {node['label']}")
                    print(f"      text  : {text}")
    else:
        print("  known pair not present in this tree")

    # --- 2. label/member-text alignment across the whole tree ----------------
    print("\n=== 2. label vs embedded text, all nodes ===")
    tally = {"no members": 0, "single, identical": 0, "single, differs": 0, "multiple members": 0}
    for node in nodes:
        member_ids = members(node)
        if not member_ids:
            tally["no members"] += 1
        elif len(member_ids) > 1:
            tally["multiple members"] += 1
        elif embedded_text.get(str(member_ids[0])) == node.get("label"):
            tally["single, identical"] += 1
        else:
            tally["single, differs"] += 1
    for key, value in tally.items():
        print(f"  {key:20}: {value}")

    # --- 3. every sibling pair, scored --------------------------------------
    rows = []
    for parent_id, child_a, child_b in pairs:
        vec_a = emb.get_centroid(members(child_a))
        vec_b = emb.get_centroid(members(child_b))
        rows.append({
            "parent": parent_id,
            "id_a": child_a.get("node_id"),
            "id_b": child_b.get("node_id"),
            "label_a": child_a.get("label", ""),
            "label_b": child_b.get("label", ""),
            "jaccard": jaccard_overlap(child_a.get("label", ""), child_b.get("label", "")),
            "cosine": EmbeddingHelper.cosine_sim(vec_a, vec_b),
            "vectors_computable": vec_a is not None and vec_b is not None,
        })
    table = pd.DataFrame(rows)
    print(f"\n=== 3. threshold sensitivity (thr_jac fixed at {args.thr_jac}) ===")
    print(f"  pairs with vectors on both sides: {int(table.vectors_computable.sum())} / {len(table)}")

    def admitted(thr_cos: float) -> int:
        return int(((table.jaccard >= args.thr_jac) | (table.cosine >= thr_cos)).sum())

    baseline = admitted(args.thr_cos)
    for thr in (0.85, 0.84, 0.83, 0.82, 0.80):
        count = admitted(thr)
        delta = f"+{count - baseline}" if thr != args.thr_cos else "baseline"
        print(f"  thr_cos={thr:.2f} -> {count:3d} pairs reach the model   ({delta})")

    newly = table[(table.cosine >= 0.83) & (table.cosine < args.thr_cos)
                  & (table.jaccard < args.thr_jac)].sort_values("cosine", ascending=False)
    print(f"\n  pairs admitted by {args.thr_cos} -> 0.83 ({len(newly)}); judge these by eye:")
    for _, row in newly.iterrows():
        print(f"    cos={row.cosine:.4f} jac={row.jaccard:.2f} | {row.label_a}  ||  {row.label_b}")

    # --- 4. section-heading observation, with control ------------------------
    print("\n=== 4. section-heading observation (an association, not a cause) ===")

    def heading_of(node_id: str, level_hint: str | None = None) -> str | None:
        """Trailing '[SEP] <heading>' of a single-member node, '' if none."""
        for lvl in LEVELS:
            member_ids = membership[lvl].get(node_id)
            if member_ids:
                if len(member_ids) != 1:
                    return None
                text = embedded_text.get(str(member_ids[0]), "")
                return text.split("[SEP]")[1].strip() if "[SEP]" in text else ""
        return None

    table["heading_a"] = table.id_a.map(heading_of)
    table["heading_b"] = table.id_b.map(heading_of)
    single = table[table.heading_a.notna() & table.heading_b.notna()]
    shared = single[(single.heading_a != "") & (single.heading_a == single.heading_b)]
    distinct = single[single.heading_a != single.heading_b]
    print(f"  single-member pairs           : {len(single)}")
    print(f"  sharing a section heading     : {len(shared):3d}  mean cosine {shared.cosine.mean():.4f}")
    print(f"  with different headings       : {len(distinct):3d}  mean cosine {distinct.cosine.mean():.4f}")
    print("  (offline data cannot separate section context from action semantics;")
    print("   this is reported as an association only.)")

    # --- 5. the near-synonym family ----------------------------------------
    print("\n=== 5. the near-synonym family, pairwise ===")
    family = table[table.label_a.isin(FAMILY_LABELS) & table.label_b.isin(FAMILY_LABELS)]
    for _, row in family.iterrows():
        reaches = row.jaccard >= args.thr_jac or row.cosine >= args.thr_cos
        print(f"  cos={row.cosine:.4f} jac={row.jaccard:.2f} "
              f"{'reaches model' if reaches else 'withheld     '} | {row.label_a} || {row.label_b}")

    if args.out_csv:
        table.to_csv(args.out_csv, index=False)
        print(f"\nper-pair table written to {args.out_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
