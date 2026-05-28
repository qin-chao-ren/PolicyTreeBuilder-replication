#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Step 2 - Embeddings + candidate edges (+ optional rerank) for the public replication package.

Outputs:
- policy_corpus_embeddings.parquet (with csv.gz fallback)
- policy_similarity_pairs_initial.csv
- policy_similarity_rerank_edges.csv (optional)
- policy_embedding_index_stats.json
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

from llm_runtime import call_embedding, call_rerank, load_env_file


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT_DIR_DEFAULT = ROOT / "data" / "intermediate_outputs"
LOG_DIR_DEFAULT = OUT_DIR_DEFAULT / "logs"


def log_write(log_path: Path, msg: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(msg.rstrip() + "\n")


def embed_call_openai(
    texts: List[str],
    model: str,
    log_path: Path,
    timeout_s: int = 60,
    max_retries: int = 3,
    batch_size: int = 10,
) -> np.ndarray:
    out: List[List[float]] = []
    n = len(texts)
    bs = max(1, batch_size)
    for start in range(0, n, bs):
        end = min(start + bs, n)
        batch = texts[start:end]
        log_write(log_path, f"[EMB API] Attempting batch {start+1}-{end} (size {len(batch)})")
        resp = call_embedding(
            profile="embedding_default",
            texts=batch,
            model_override=model,
            timeout_s=timeout_s,
            retries=max_retries,
        )
        vectors = resp.get("vectors") or []
        if resp.get("ok") and len(vectors) == len(batch):
            out.extend(vectors)
            log_write(log_path, f"[EMB API] Batch {start+1}-{end} SUCCESS")
            continue
        err = resp.get("error") or "embedding request failed"
        log_write(log_path, f"[EMB API] Batch {start+1}-{end} FAILED. Error: {err}. Filling with zeros.")
        dim = len(out[0]) if out else 1536
        out.extend([[0.0] * dim for _ in batch])
    return np.asarray(out, dtype=np.float32)


def _ensure_np_vector(val) -> np.ndarray:
    if isinstance(val, np.ndarray):
        return val.astype(np.float32)
    if isinstance(val, (list, tuple)):
        return np.asarray(val, dtype=np.float32)
    if isinstance(val, str):
        s = val.strip()
        if not s:
            return np.asarray([], dtype=np.float32)
        try:
            parsed = json.loads(s)
        except Exception:
            from ast import literal_eval

            try:
                parsed = literal_eval(s)
            except Exception:
                return np.asarray([], dtype=np.float32)
        return np.asarray(parsed, dtype=np.float32)
    return np.asarray([], dtype=np.float32)


def save_emb_cache(df: pd.DataFrame, path: Path, log_path: Path) -> None:
    df_save = df.copy()
    df_save["vector"] = [np.asarray(v, dtype=np.float32).tolist() for v in df_save["vector"]]
    df_save["embedding"] = [json.dumps(v, ensure_ascii=False) for v in df_save["vector"]]
    try:
        df_save.to_parquet(path, index=False)
        log_write(log_path, f"[CACHE] saved parquet: {path}")
    except Exception as exc:
        alt = path.with_suffix(".csv.gz")
        log_write(log_path, f"[CACHE WARN] parquet unavailable ({exc}); fallback -> {alt.name}")
        df_save.to_csv(alt, index=False, encoding="utf-8-sig", compression="gzip")


def load_emb_cache(path: Path) -> Optional[pd.DataFrame]:
    df: Optional[pd.DataFrame]
    if path.exists():
        df = pd.read_parquet(path)
    else:
        alt = path.with_suffix(".csv.gz")
        if alt.exists():
            df = pd.read_csv(alt, dtype=str, compression="gzip")
        else:
            df = None
    if df is None:
        return None
    if not len(df):
        return df
    if "vector" in df.columns:
        df["vector"] = df["vector"].apply(_ensure_np_vector)
    elif "embedding" in df.columns:
        df["vector"] = df["embedding"].apply(_ensure_np_vector)
    else:
        cols = [c for c in df.columns if c not in {"sample_id", "text_for_embed"}]
        if not cols:
            return None
        arr = df[cols].astype(float).values
        df["vector"] = [np.asarray(row, dtype=np.float32) for row in arr]
    df["sample_id"] = df["sample_id"].astype(str)
    if "text_for_embed" not in df.columns:
        df["text_for_embed"] = ""
    return df


def _to_str(val) -> str:
    if val is None:
        return ""
    try:
        if isinstance(val, float) and np.isnan(val):
            return ""
    except Exception:
        pass
    s = str(val).strip()
    return "" if s.lower() == "nan" else s


def build_text_for_embed(cleaned: str, path_text: str, template: str) -> str:
    cleaned = _to_str(cleaned)
    path_text = _to_str(path_text)
    if template == "title_only":
        return cleaned
    if template == "path_only":
        return path_text
    return f"{cleaned} [SEP] {path_text}" if path_text else cleaned


def rerank_api_call(
    endpoint: str,
    api_key: str,
    model: str,
    query: str,
    documents: List[str],
    timeout_s: int = 60,
    max_retries: int = 3,
) -> Optional[List[float]]:
    if endpoint:
        os.environ["RERANK_API_ENDPOINT"] = endpoint
    if api_key:
        os.environ["RERANK_API_KEY"] = api_key
    resp = call_rerank(
        profile="rerank_default",
        query=query,
        documents=documents,
        model_override=model,
        endpoint_override=endpoint,
        timeout_s=timeout_s,
        retries=max_retries,
    )
    if not resp.get("ok"):
        print(f"\n[ERROR] Rerank API Error: {resp.get('error')}")
        return None
    return [float(x) for x in (resp.get("scores") or [])]


def main() -> None:
    ap = argparse.ArgumentParser(description="Embed policy corpus and build similarity edges")
    ap.add_argument("--corpus", default=None, help="corpus csv; defaults to policy_corpus_filtered/policy_corpus_cleaned")
    ap.add_argument("--env", default=str(ROOT / "configs" / ".env"))
    ap.add_argument("--outdir", default=str(OUT_DIR_DEFAULT))

    ap.add_argument("--embed-text-template", choices=["title_plus_path", "title_only", "path_only"], default="title_plus_path")
    ap.add_argument("--embed-model", default=None)
    ap.add_argument("--timeout", type=int, default=60)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=10)
    ap.add_argument("--recompute", choices=["yes", "no"], default="no")
    ap.add_argument("--refresh-zero", choices=["yes", "no"], default="yes")

    ap.add_argument("--nn-topk", type=int, default=50)
    ap.add_argument("--mutual-knn", choices=["yes", "no"], default="yes")
    ap.add_argument("--local-k", type=int, default=20)
    ap.add_argument("--min-sim", type=float, default=0.58)

    ap.add_argument("--use-reranker", choices=["yes", "no"], default="yes")
    ap.add_argument("--rerank-model", default=None)
    ap.add_argument("--rerank-endpoint", default=None)
    ap.add_argument("--rerank-api-key", default=None)
    ap.add_argument("--rerank-topm", type=int, default=10)
    ap.add_argument("--rerank-threshold", type=float, default=0.55)
    ap.add_argument("--rerank-scope", choices=["topm", "suspicious", "all"], default="topm")
    ap.add_argument("--rerank-mode", choices=["soft", "strict"], default="soft")
    ap.add_argument("--keep-only-mutual", choices=["yes", "no"], default="no")
    ap.add_argument("--suspicious-margin", type=float, default=0.02)
    ap.add_argument("--suspicious-min-rank", type=int, default=10)

    ap.add_argument("--prefer-filtered", choices=["yes", "no"], default="yes")
    args = ap.parse_args()

    load_env_file(args.env)

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR_DEFAULT / "step2_embed_and_nn.log"

    if args.corpus:
        corpus_path = Path(args.corpus)
    else:
        prefer = args.prefer_filtered == "yes"
        cand1 = out_dir / "policy_corpus_filtered.csv"
        cand2 = out_dir / "policy_corpus_cleaned.csv"
        if prefer and cand1.exists():
            corpus_path = cand1
        elif cand2.exists():
            corpus_path = cand2
        else:
            corpus_path = cand1 if cand1.exists() else cand2
    if not corpus_path.exists():
        raise FileNotFoundError(f"corpus not found: {corpus_path}")
    log_write(log_path, f"[INPUT] corpus: {corpus_path}")

    df = pd.read_csv(corpus_path, dtype=str)
    required_cols = {"sample_id", "cleaned_title", "path_text"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"missing column(s) in corpus: {sorted(missing)}")

    df["text_for_embed"] = [
        build_text_for_embed(ct, pt, args.embed_text_template)
        for ct, pt in zip(df["cleaned_title"], df["path_text"])
    ]
    df = df[df["text_for_embed"].astype(str).str.strip() != ""].copy()
    ids = df["sample_id"].astype(str).tolist()
    texts = df["text_for_embed"].tolist()
    n = len(ids)
    if n == 0:
        log_write(log_path, "[WARN] no rows to embed after filtering")
        return

    embed_model = args.embed_model or os.getenv("EMBED_MODEL_NAME") or "text-embedding-v4"
    log_write(log_path, f"[EMB] model={embed_model} n={n}")

    emb_cache_path = out_dir / "policy_corpus_embeddings.parquet"
    emb_df = None if args.recompute == "yes" else load_emb_cache(emb_cache_path)
    text_map = dict(zip(ids, texts))
    if emb_df is not None and len(emb_df):
        if "text_for_embed" not in emb_df.columns:
            emb_df["text_for_embed"] = ""
        emb_df.loc[emb_df["sample_id"].isin(text_map), "text_for_embed"] = emb_df["sample_id"].map(text_map)
    else:
        emb_df = None

    if emb_df is not None:
        have = set(emb_df["sample_id"].astype(str).tolist())
        bad_ids = set()
        if args.refresh_zero == "yes":
            try:
                sub = emb_df[emb_df["sample_id"].isin(ids)]
                arr = (
                    np.stack([np.asarray(v, dtype=np.float32) for v in sub["vector"].tolist()])
                    if len(sub)
                    else np.zeros((0, 1), dtype=np.float32)
                )
                norms = np.linalg.norm(arr, axis=1) if arr.size else np.array([])
                for sid, nv in zip(sub["sample_id"].tolist(), norms.tolist()):
                    if not nv or nv < 1e-8:
                        bad_ids.add(sid)
                if bad_ids:
                    log_write(log_path, f"[EMB CACHE] detected {len(bad_ids)} zero vectors; refreshing")
            except Exception:
                pass
        need_pairs = [(i, text_map[i]) for i in ids if (i not in have) or (i in bad_ids)]
        if need_pairs:
            log_write(log_path, f"[EMB CACHE] {len(have)} cached; new/refresh {len(need_pairs)}")
            vec_new = embed_call_openai(
                [t for _, t in need_pairs],
                embed_model,
                log_path,
                timeout_s=args.timeout,
                max_retries=args.retries,
                batch_size=args.batch_size,
            )
            add_df = pd.DataFrame(
                {
                    "sample_id": [i for i, _ in need_pairs],
                    "text_for_embed": [t for _, t in need_pairs],
                    "vector": [np.asarray(v, dtype=np.float32) for v in vec_new],
                }
            )
            keep_mask = ~emb_df["sample_id"].isin(add_df["sample_id"])
            emb_df = pd.concat([emb_df[keep_mask], add_df], ignore_index=True)
            save_emb_cache(emb_df, emb_cache_path, log_path)
    else:
        log_write(log_path, "[EMB] computing all embeddings")
        vec = embed_call_openai(
            texts,
            embed_model,
            log_path,
            timeout_s=args.timeout,
            max_retries=args.retries,
            batch_size=args.batch_size,
        )
        emb_df = pd.DataFrame(
            {
                "sample_id": ids,
                "text_for_embed": texts,
                "vector": [np.asarray(v, dtype=np.float32) for v in vec],
            }
        )
        save_emb_cache(emb_df, emb_cache_path, log_path)

    emb_map = {str(r["sample_id"]): np.asarray(r["vector"], dtype=np.float32) for _, r in emb_df.iterrows()}
    missing_ids = [sid for sid in ids if sid not in emb_map]
    if missing_ids:
        raise RuntimeError(f"missing embeddings for {len(missing_ids)} sample(s)")

    X = np.stack([emb_map[sid] for sid in ids], axis=0)
    X = normalize(X, norm="l2", copy=False)
    dim = int(X.shape[1])
    log_write(log_path, f"[EMB] vectors ready: n={n}, dim={dim}")

    tk = max(1, min(int(args.nn_topk), max(1, n - 1)))
    nn = NearestNeighbors(n_neighbors=tk + 1, metric="cosine", algorithm="auto")
    nn.fit(X)
    dist, ind = nn.kneighbors(X, return_distance=True)
    dist = dist[:, 1:]
    ind = ind[:, 1:]
    sim = 1.0 - dist

    local_k = max(1, min(int(args.local_k), tk))
    local_thr = np.maximum(
        args.min_sim,
        (sim[:, :local_k].mean(axis=1) - sim[:, :local_k].std(axis=1)),
    )
    idx_to_pos = [{int(ind[i, j]): j for j in range(tk)} for i in range(n)]

    rows: List[Dict[str, object]] = []
    for i in range(n):
        th = float(local_thr[i])
        for jpos in range(tk):
            j = int(ind[i, jpos])
            s = float(sim[i, jpos])
            if s < args.min_sim or s < th:
                continue
            is_mut = bool(i in idx_to_pos[j]) if args.mutual_knn == "yes" else False
            rows.append(
                {
                    "id_a": ids[i],
                    "id_b": ids[j],
                    "sim": s,
                    "is_mutual": is_mut,
                    "rank_pos": int(jpos),
                    "local_thr": th,
                }
            )

    edge_df = pd.DataFrame(rows)
    if len(edge_df) == 0:
        log_write(log_path, "[WARN] no candidate edges after kNN thresholding")
    if args.keep_only_mutual == "yes" and not edge_df.empty:
        edge_df = edge_df[edge_df["is_mutual"] == True].copy()  # noqa: E712

    rerank_dump: List[Dict[str, object]] = []
    if args.use_reranker == "yes":
        api_ep = (
            args.rerank_endpoint
            or os.getenv("RERANK_API_ENDPOINT")
            or os.getenv("RERANK_API_URL")
        )
        api_key = args.rerank_api_key or os.getenv("RERANK_API_KEY")
        model = args.rerank_model or os.getenv("RERANK_MODEL_NAME") or "gte-rerank-v2"
        if not api_ep and os.getenv("RERANK_BASE_URL"):
            base = os.getenv("RERANK_BASE_URL")
            if base:
                api_ep = base.rstrip("/") + "/api/v1/services/rerank/text-rerank/text-rerank"
        if not api_ep or not api_key or not model:
            log_write(log_path, "[RERANK] skipped: endpoint/api_key/model missing")
        elif edge_df.empty:
            log_write(log_path, "[RERANK] skipped: no edges to rerank")
        else:
            by_a_rows: Dict[str, List[Dict[str, object]]] = {}
            for _, r in edge_df.iterrows():
                by_a_rows.setdefault(r["id_a"], []).append(
                    {
                        "id_b": r["id_b"],
                        "sim": float(r["sim"]),
                        "is_mutual": bool(r["is_mutual"]),
                        "rank_pos": int(r.get("rank_pos", 0)),
                        "local_thr": float(r.get("local_thr", args.min_sim)),
                    }
                )
            id_to_idx = {sid: idx for idx, sid in enumerate(ids)}
            text_lookup = dict(zip(df["sample_id"], df["text_for_embed"]))
            for a, lst_rows in by_a_rows.items():
                if args.rerank_scope == "all":
                    cand_rows = list(lst_rows)
                elif args.rerank_scope == "suspicious":
                    cand_rows = []
                    for rr in lst_rows:
                        suspicious = (
                            (not rr["is_mutual"])
                            or (rr["sim"] - rr["local_thr"] <= args.suspicious_margin)
                            or (rr["rank_pos"] >= args.suspicious_min_rank)
                        )
                        if suspicious:
                            cand_rows.append(rr)
                    cand_rows = (
                        sorted(cand_rows, key=lambda x: -x["sim"])[: max(1, args.rerank_topm)]
                        if cand_rows
                        else []
                    )
                else:
                    cand_rows = sorted(lst_rows, key=lambda x: -x["sim"])[: max(1, args.rerank_topm)]
                if not cand_rows:
                    continue
                Bs = [cr["id_b"] for cr in cand_rows]
                docs_idx_all = [id_to_idx[b] for b in Bs if b in id_to_idx]
                if not docs_idx_all:
                    continue
                docs_all = [texts[idx] for idx in docs_idx_all]
                query = text_lookup.get(a, "")
                chunk = 20
                merged_scores: List[float] = []
                offset = 0
                while offset < len(docs_all):
                    docs_chunk = docs_all[offset : offset + chunk]
                    scores = rerank_api_call(
                        api_ep,
                        api_key,
                        model,
                        query,
                        docs_chunk,
                        timeout_s=args.timeout,
                        max_retries=args.retries,
                    )
                    if scores is None:
                        merged_scores.extend([-1.0] * len(docs_chunk))
                    else:
                        merged_scores.extend([float(s) for s in scores])
                    offset += chunk
                for b, sc in zip(Bs, merged_scores):
                    keep = bool(sc >= args.rerank_threshold)
                    rerank_dump.append(
                        {"id_a": a, "id_b": b, "rerank_score": float(sc), "kept_by_rerank": keep}
                    )
            if rerank_dump:
                rer_df = pd.DataFrame(rerank_dump)
                edge_df = edge_df.merge(rer_df, on=["id_a", "id_b"], how="left")
                edge_df["kept_by_rerank"] = edge_df["kept_by_rerank"].fillna(False)
                if args.rerank_mode == "strict":
                    edge_df = edge_df[edge_df["kept_by_rerank"] == True].copy()  # noqa: E712
                else:
                    edge_df = edge_df[
                        (edge_df["kept_by_rerank"]) | (edge_df["rerank_score"].isna())
                    ].copy()
            log_write(log_path, f"[RERANK] completed rows={len(rerank_dump)}")

    pairs_path = out_dir / "policy_similarity_pairs_initial.csv"
    edge_df.to_csv(pairs_path, index=False, encoding="utf-8-sig")
    log_write(log_path, f"[EDGES] {pairs_path} rows={len(edge_df)}")

    if rerank_dump:
        rerank_path = out_dir / "policy_similarity_rerank_edges.csv"
        pd.DataFrame(rerank_dump).to_csv(rerank_path, index=False, encoding="utf-8-sig")
        log_write(log_path, f"[RERANK] dump -> {rerank_path}")

    if not edge_df.empty:
        deg = edge_df.groupby("id_a")["id_b"].count().reindex(ids).fillna(0).astype(int).values
    else:
        deg = np.zeros(n, dtype=np.int32)
    stats = {
        "n_samples": int(n),
        "embed_dim": int(dim),
        "nn_topk": int(tk),
        "local_k": int(local_k),
        "min_sim": float(args.min_sim),
        "use_reranker": args.use_reranker,
        "rerank_topm": int(args.rerank_topm),
        "deg_mean": float(np.mean(deg)) if len(deg) else 0.0,
        "deg_median": float(np.median(deg)) if len(deg) else 0.0,
        "deg_p05": float(np.percentile(deg, 5)) if len(deg) else 0.0,
        "deg_p95": float(np.percentile(deg, 95)) if len(deg) else 0.0,
    }
    stats_path = out_dir / "policy_embedding_index_stats.json"
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    log_write(log_path, f"[STATS] {stats_path}")
    log_write(log_path, "[DONE] step2_embed_and_nn")

    print(f"[WRITE] {emb_cache_path}")
    print(f"[WRITE] {pairs_path}")


if __name__ == "__main__":
    main()
