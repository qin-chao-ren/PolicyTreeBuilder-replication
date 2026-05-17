#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import pandas as pd

from common_utils import trimmed_mean, p75, safe_write_csv


def contract_edges(
    node_members: Dict[str, Iterable[str]],
    pairs_df: pd.DataFrame,
    id_a_col: str = "id_a",
    id_b_col: str = "id_b",
    score_col: str = "rerank_score",
) -> pd.DataFrame:
    """
    将样本对相似度收缩为“节点对”相似度。
    node_members: node_id -> {sample_id,...}
    pairs_df: 含 id_a/id_b/score 的 DataFrame
    """
    s2n: Dict[str, str] = {}
    for nid, members in node_members.items():
        for sid in members:
            s2n[str(sid)] = nid

    sub = pairs_df[[id_a_col, id_b_col, score_col]].copy()
    sub[id_a_col] = sub[id_a_col].astype(str)
    sub[id_b_col] = sub[id_b_col].astype(str)
    sub = sub[sub[id_a_col].isin(s2n) & sub[id_b_col].isin(s2n)]
    if sub.empty:
        return pd.DataFrame(columns=["node_a", "node_b", "score_mean", "score_p75", "score_max", "pair_count"])

    na = sub[id_a_col].map(s2n)
    nb = sub[id_b_col].map(s2n)
    mask = na != nb
    sub = sub[mask]
    na = na[mask]
    nb = nb[mask]

    node_pair = pd.DataFrame(
        {
            "node_a": np.where(na < nb, na, nb),
            "node_b": np.where(na < nb, nb, na),
            "score": sub[score_col].astype(float).values,
        }
    )

    rows = []
    for (a, b), grp in node_pair.groupby(["node_a", "node_b"]):
        scores = grp["score"].tolist()
        rows.append(
            {
                "node_a": a,
                "node_b": b,
                "score_mean": trimmed_mean(scores, 0.1),
                "score_p75": p75(scores),
                "score_max": float(np.max(scores)),
                "pair_count": int(len(scores)),
            }
        )
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description="Contract sample-level edges to node-level edges.")
    ap.add_argument("--membership", required=True, help="v4_membership_L{n}.csv")
    ap.add_argument("--pairs", required=True, help="data/intermediate_outputs/v4_rerank_edges.csv")
    ap.add_argument("--out", required=True, help="v4_edges_L{n}.csv")
    args = ap.parse_args()

    mem = pd.read_csv(args.membership, dtype=str)
    node_members: Dict[str, set[str]] = {}
    for _, r in mem.iterrows():
        node_members.setdefault(str(r["node_id"]), set()).add(str(r["member_id"]))

    pairs = pd.read_csv(args.pairs)
    out = contract_edges(node_members, pairs)
    safe_write_csv(out, args.out)


if __name__ == "__main__":
    main()
