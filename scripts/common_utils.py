#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd


def ensure_outdir(path: str | Path):
    Path(path).mkdir(parents=True, exist_ok=True)


def read_corpus(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str)
    required = {"sample_id", "cleaned_title", "calibrated_level"}
    miss = required - set(df.columns)
    if miss:
        raise ValueError(f"Corpus missing columns: {sorted(miss)} in {path}")
    return df


def read_pairs(path: str | Path, only_cols: Sequence[str] | None = None) -> pd.DataFrame:
    # 支持 csv/csv.gz
    p = Path(path)
    if p.suffix == ".gz":
        df = pd.read_csv(p, compression="gzip")
    else:
        df = pd.read_csv(p)
    # 标准化列名
    cols = {c.lower(): c for c in df.columns}
    need_map = {
        "id_a": None,
        "id_b": None,
        "rerank_score": None,
        "is_mutual": None,
        "kept_by_rerank": None,
    }
    for k in list(need_map):
        for cand in (k, k.upper(), k.capitalize()):
            if cand in df.columns:
                need_map[k] = cand
                break
    if not need_map["id_a"] or not need_map["id_b"] or not need_map["rerank_score"]:
        raise ValueError(f"Pairs file missing id_a/id_b/rerank_score columns: {path}")
    df = df.rename(columns={need_map[k]: k for k in need_map if need_map[k]})
    if only_cols:
        df = df[[c for c in only_cols if c in df.columns]]
    return df


def read_embeddings(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if p.suffix == ".parquet":
        df = pd.read_parquet(p)
    elif p.suffixes[-2:] == [".csv", ".gz"]:
        df = pd.read_csv(p, compression="gzip")
    else:
        df = pd.read_csv(p)
    # 期望列： sample_id + 嵌入列
    if "sample_id" not in df.columns:
        raise ValueError("embeddings must contain column 'sample_id'")
    # 检测向量列
    vec_col = None
    for c in ("embedding", "vector", "emb", "embeddings"):
        if c in df.columns:
            vec_col = c
            break
    if vec_col is not None:
        # 保持为 numpy 数组
        def _to_vec(x):
            if isinstance(x, (list, tuple, np.ndarray)):
                return np.asarray(x, dtype=np.float32)
            # 若是字符串如 "[0.1, 0.2]"
            try:
                v = json.loads(x)
                return np.asarray(v, dtype=np.float32)
            except Exception:
                return np.asarray([], dtype=np.float32)
        df["_vec"] = df[vec_col].apply(_to_vec)
    else:
        # 以所有数值列为向量
        num_cols = [c for c in df.columns if c != "sample_id" and pd.api.types.is_numeric_dtype(df[c])]
        if not num_cols:
            raise ValueError("No embedding columns found")
        df["_vec"] = df[num_cols].astype(np.float32).values.tolist()
        df["_vec"] = df["_vec"].apply(lambda x: np.asarray(x, dtype=np.float32))
    return df[["sample_id", "_vec"]]


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return 0.0
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    v = float(np.dot(a, b) / (na * nb))
    # 映射到 [0,1]
    return max(0.0, min(1.0, (v + 1.0) / 2.0))


_CN_CHAR_RE = re.compile(r"[\u4e00-\u9fa5]")
_ALNUM_RE = re.compile(r"[A-Za-z0-9]+")
_STOPWORDS = set(["的", "与", "及", "和", "及其", "等", "与否", "有关", "相关", "工作", "方面"])  # 简易版


def tokenize_label(s: str) -> List[str]:
    s = (s or "").strip()
    cn = _CN_CHAR_RE.findall(s)
    en = _ALNUM_RE.findall(s)
    toks = [t for t in cn + en if t and t not in _STOPWORDS]
    return toks


def jaccard_overlap(a: str, b: str) -> float:
    ta = set(tokenize_label(a))
    tb = set(tokenize_label(b))
    if not ta and not tb:
        return 0.0
    inter = len(ta & tb)
    union = len(ta | tb)
    return inter / union if union > 0 else 0.0


def trimmed_mean(values: Sequence[float], trim_ratio: float = 0.1) -> float:
    if not values:
        return float("nan")
    arr = np.asarray(list(values), dtype=np.float32)
    n = len(arr)
    if n == 1:
        return float(arr[0])
    arr.sort()
    k = int(math.floor(n * trim_ratio))
    sl = arr[k : n - k] if n - k > k else arr
    return float(sl.mean()) if len(sl) else float(arr.mean())


def p75(values: Sequence[float]) -> float:
    if not values:
        return float("nan")
    return float(np.percentile(np.asarray(values, dtype=np.float32), 75))


def vector_centroid(sample_ids: Iterable[str], emb_df: pd.DataFrame) -> np.ndarray:
    ids = set(str(x) for x in sample_ids)
    sub = emb_df[emb_df["sample_id"].astype(str).isin(ids)]
    if not len(sub):
        return np.asarray([], dtype=np.float32)
    vs = np.stack(sub["_vec"].values, axis=0)
    return vs.mean(axis=0)


def safe_write_json(path: str | Path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def safe_write_csv(df: pd.DataFrame, path: str | Path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")


__all__ = [
    "ensure_outdir",
    "read_corpus",
    "read_pairs",
    "read_embeddings",
    "cosine_sim",
    "tokenize_label",
    "jaccard_overlap",
    "trimmed_mean",
    "p75",
    "vector_centroid",
    "safe_write_json",
    "safe_write_csv",
]

