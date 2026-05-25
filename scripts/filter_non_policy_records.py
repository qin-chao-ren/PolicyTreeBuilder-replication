#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Non-policy filter · T0 过滤（public replication package）

- 输入（默认）: data/intermediate_outputs/policy_corpus_cleaned.csv
  必需列: sample_id, cleaned_title, final_level, doc_id
- 输出（默认）: data/intermediate_outputs/policy_corpus_filtered.csv
- 规则: 与以下关键词完全相同，或标题长度 ≤4 且包含关键词，则视为章首/T0：
  主要任务/重点任务/工作措施/保障措施/总体要求/基本原则/组织实施/附则
"""

import argparse
import re
from pathlib import Path
from typing import Dict, List

import pandas as pd

HERE = Path(__file__).resolve().parent  # 指向 .../scripts 目录
ROOT = HERE.parent
OUT_DIR_DEFAULT = ROOT / "data" / "intermediate_outputs"
DEFAULT_CORPUS = OUT_DIR_DEFAULT / "policy_corpus_cleaned.csv"
DEFAULT_OUT = OUT_DIR_DEFAULT / "policy_corpus_filtered.csv"

DEFAULT_PATTERNS = [
    '重要举措', '若干举措', '若干措施', '工作方案', '工作举措', '主要任务', '主要举措', '方案', '工作任务', '通知', '意见', '工作要点',
    '组织实施', '有关事项', '实施细则', '实施意见', '实施方案', '重点任务', '工作要求', '总体要求', '基本原则', '保障措施',
]


def log(msg: str) -> None:
    from time import strftime

    print(f"[{strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def ensure_columns(df: pd.DataFrame, cols: List[str], ctx: str = "dataframe") -> None:
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise ValueError(f"missing columns {miss} in {ctx}")


def is_t0(title: str, patterns: List[str]) -> bool:
    txt = (title or "").strip()
    if not txt:
        return False
    if txt in patterns:
        return True
    if len(txt) <= 4:
        return any(p in txt for p in patterns)
    return False


def _transform_group(group: pd.DataFrame, level_re: re.Pattern[str]) -> List[Dict]:
    group = group.sort_values("_order")
    stack: List[int] = []
    out_rows: List[Dict] = []
    for _, row in group.iterrows():
        lvl_str = str(row.get("final_level") or "")
        m = level_re.match(lvl_str)
        if not m:
            if bool(row.get("is_t0", False)):
                continue
            new_row = row.to_dict()
            new_row["final_level_origin"] = row.get("final_level")
            out_rows.append(new_row)
            continue

        lvl_num = int(m.group(1))
        row_is_t0 = bool(row.get("is_t0", False))
        if row_is_t0:
            while stack and stack[-1] >= lvl_num:
                stack.pop()
            stack.append(lvl_num)
            continue

        while stack and stack[-1] >= lvl_num:
            stack.pop()
        shift = len(stack)
        new_lvl_num = max(1, lvl_num - shift)

        new_row = row.to_dict()
        new_row["final_level_origin"] = row.get("final_level")
        new_row["final_level"] = f"H{new_lvl_num}"
        out_rows.append(new_row)
    return out_rows


def main() -> None:
    ap = argparse.ArgumentParser(description="PolicyTreeBuilder final replication · Non-policy filter T0 Filter")
    ap.add_argument("--corpus", type=str, default=str(DEFAULT_CORPUS))
    ap.add_argument("--outdir", type=str, default=str(OUT_DIR_DEFAULT))
    ap.add_argument(
        "--mode",
        type=str,
        default="delete",
        choices=["delete", "mark"],
        help="delete: 删除 T0；mark: 仅标记 is_t0",
    )
    ap.add_argument("--patterns", type=str, default=",".join(DEFAULT_PATTERNS))
    args = ap.parse_args()

    corpus_path = Path(args.corpus)
    if not corpus_path.exists():
        raise FileNotFoundError(f"corpus not found: {corpus_path}")

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "policy_corpus_filtered.csv"

    df = pd.read_csv(corpus_path, dtype=str)
    ensure_columns(
        df,
        ["sample_id", "cleaned_title", "final_level", "doc_id", "block_idx"],
        "corpus",
    )

    pats = [p.strip() for p in args.patterns.split(",") if p.strip()]
    df["is_t0"] = df["cleaned_title"].apply(lambda s: is_t0(str(s), pats))

    if args.mode == "mark":
        df.to_csv(out_path, index=False, encoding="utf-8-sig")
        log(f"[WRITE] {out_path} rows={len(df)} (marked is_t0)")
        return

    df["_order"] = range(len(df))
    level_re = re.compile(r"H(\d+)")
    new_rows: List[Dict] = []
    for _, grp in df.groupby("doc_id", sort=False):
        new_rows.extend(_transform_group(grp, level_re))

    if new_rows:
        kept = pd.DataFrame(new_rows)
    else:
        cols = df.columns.tolist() + ["final_level_origin"]
        kept = pd.DataFrame(columns=cols)

    if "_order" in kept.columns:
        kept = kept.drop(columns="_order")

    base_cols = [c for c in df.columns if c != "_order"]
    if "final_level_origin" not in base_cols:
        insert_pos = base_cols.index("final_level") + 1
        base_cols = base_cols[:insert_pos] + ["final_level_origin"] + base_cols[insert_pos:]
    kept = kept.reindex(columns=base_cols, fill_value="")

    kept.to_csv(out_path, index=False, encoding="utf-8-sig")
    deleted = int(df["is_t0"].sum())
    log(f"[WRITE] {out_path} rows={len(kept)} (deleted {deleted} T0 rows)")


if __name__ == "__main__":
    main()
