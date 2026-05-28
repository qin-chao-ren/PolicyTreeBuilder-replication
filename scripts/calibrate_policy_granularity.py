#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Granularity calibration · 文档级层级校准（public）

输入优先：data/intermediate_outputs/policy_corpus_filtered.csv（若不存在则回退 policy_corpus_cleaned.csv）
必需列：sample_id, doc_id, final_level, cleaned_title, path_text

LLM：按 doc_id 汇总结构后调用 1 次，让模型判断 granularity=macro/meso/micro。
再依据映射表将 H 层级映射为 T 层级：
  - macro: H1→T1, H2→T2, H3→T3, H4→T4
  - meso : H1→T2, H2→T3, H3→T4, H4→T4
  - micro: H1→T3, H2→T4, H3→T4, H4→T4

输出：
  - data/intermediate_outputs/policy_corpus_calibrated.csv（新增列：doc_granularity, calibrated_level, calibration_confidence）
  - data/intermediate_outputs/policy_granularity_calibration_report.json
  - data/intermediate_outputs/policy_granularity_calibration_review.csv（低置信度样本，人工复核）
日志：data/intermediate_outputs/logs/calibrate_policy_granularity.log
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT_DIR_DEFAULT = ROOT / "data" / "intermediate_outputs"
LOG_DIR_DEFAULT = OUT_DIR_DEFAULT / "logs"
PROMPT_DEFAULT = ROOT / "prompts" / "calibrate_policy_granularity.md"

if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from llm_runtime import call_llm_json, load_env_file, profiles_from_config  # noqa: E402


def log_write(path: Path, msg: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(msg.rstrip() + "\n")


def ensure_columns(df: pd.DataFrame, cols: List[str], ctx: str) -> None:
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise ValueError(f"missing columns {miss} in {ctx}")


SYSTEM_PROMPT_FALLBACK = (
    "你是航空物流政策分析专家。请根据提供的 H1/H2 结构判断文档整体粒度（macro/meso/micro）。"
    "严格以 JSON 返回：{\"granularity\":\"macro|meso|micro\",\"confidence\":0-1,\"reasoning\":\"1-2 句理由\"}"
)


def read_system_prompt(path: str | Path) -> str:
    p = Path(path)
    if p.exists():
        try:
            text = p.read_text(encoding="utf-8").strip()
            if text:
                return text
        except Exception:
            pass
    return SYSTEM_PROMPT_FALLBACK


def build_user_prompt(
    doc_id: str,
    h1_titles: List[str],
    h2_samples: Dict[str, List[str]],
    max_h1: int,
    max_h2_per_h1: int,
    max_h1_with_samples: int,
) -> str:
    h1_titles = (h1_titles or [])[:max_h1]
    h1_list = "\n".join([f"{idx+1}. {title}" for idx, title in enumerate(h1_titles)]) or "（无 H1 标题）"
    sample_pairs = list(h2_samples.items())[:max_h1_with_samples]
    h2_blocks: List[str] = []
    for h1, subs in sample_pairs:
        subs = (subs or [])[:max_h2_per_h1]
        if subs:
            sub_lines = "\n  ".join(f"- {s}" for s in subs)
            h2_blocks.append(f"【{h1}】的子标题示例：\n  {sub_lines}\n")
    h2_examples = "\n".join(h2_blocks) if h2_blocks else "（该文档尚无可供展示的 H2/H3 子标题）"
    return (
        f"# 判定文档整体粒度\n"
        f"- 文档 ID: {doc_id}\n"
        f"- H1 标题数量: {len(h1_titles)}\n\n"
        f"## H1 标题（按照出现顺序）\n{h1_list}\n\n"
        f"## 子标题示例（部分 H1 的 H2/H3）\n{h2_examples}\n"
        "\n---\n\n"
        "## 任务\n"
        "请综合判断该文档的粒度（macro/meso/micro），并仅以 JSON 形式输出：\n"
        "{\n"
        '  "granularity": "macro|meso|micro",\n'
        '  "confidence": 0.0-1.0,\n'
        '  "reasoning": "1-2 句中文理由"\n'
        "}\n"
    )


MAPPING_RULES: Dict[str, Dict[str, str]] = {
    "macro": {"H1": "T1", "H2": "T2", "H3": "T3", "H4": "T4"},
    "meso": {"H1": "T2", "H2": "T3", "H3": "T4", "H4": "T4"},
    "micro": {"H1": "T3", "H2": "T4", "H3": "T4", "H4": "T4"},
}


def map_level(granularity: str, final_level: str) -> str:
    rules = MAPPING_RULES.get((granularity or "").strip().lower(), MAPPING_RULES["meso"])
    return rules.get(final_level, "T4")


def call_llm_granularity(
    *,
    profile: str,
    model_override: str | None,
    system_text: str,
    user_text: str,
    timeout_s: int,
    retries: int,
) -> Tuple[str, float, str, List[Dict[str, str]]]:
    """
    返回 (majority_granularity, confidence, raw_text, exceptions)；失败时回退为 ('meso', 0.4, err, [])。
    """
    result = call_llm_json(
        profile=profile,
        model_override=model_override,
        system=system_text,
        user=user_text,
        task="calibrate_policy_granularity",
        timeout_s=timeout_s,
        retries=retries,
    )
    obj = result.get("json") or {}
    if isinstance(obj, dict) and result.get("ok"):
        gran_raw = str(obj.get("majority_granularity") or obj.get("granularity") or "").strip().lower()
        gran = gran_raw if gran_raw in ("macro", "meso", "micro") else "meso"
        try:
            conf = float(obj.get("confidence") or 0.5)
        except Exception:
            conf = 0.5
        conf = max(0.0, min(1.0, conf))
        exceptions_raw = obj.get("exceptions") or []
        exceptions: List[Dict[str, str]] = []
        if isinstance(exceptions_raw, list):
            for item in exceptions_raw:
                if not isinstance(item, dict):
                    continue
                title = str(item.get("h1_title") or "").strip()
                egran_raw = str(item.get("exception_granularity") or "").strip().lower()
                reason = str(item.get("reasoning") or "").strip()
                if title and egran_raw in ("macro", "meso", "micro"):
                    exceptions.append(
                        {
                            "h1_title": title,
                            "exception_granularity": egran_raw,
                            "reasoning": reason,
                        }
                    )
        return gran, conf, str(result.get("raw") or ""), exceptions
    err = str(result.get("error") or result.get("raw") or "llm_call_failed")
    return "meso", 0.4, err, []


def extract_h2_samples(
    group: pd.DataFrame,
    h1_titles: List[str],
    max_h2_per_h1: int,
) -> Dict[str, List[str]]:
    """
    依据 H1 的出现顺序，截取其下方的 H2/H3，便于 prompt 展示。
    """
    samples: Dict[str, List[str]] = {}
    if not h1_titles:
        return samples
    grp = group.reset_index(drop=False)
    grp["final_level_norm"] = grp["final_level"].astype(str).str.upper()
    for h1 in h1_titles:
        mask = (grp["final_level_norm"] == "H1") & (grp["cleaned_title"] == h1)
        if not mask.any():
            continue
        h1_idx = int(grp.loc[mask, "index"].iloc[0])
        next_h1 = grp[(grp["index"] > h1_idx) & (grp["final_level_norm"] == "H1")]
        if len(next_h1):
            end_idx = int(next_h1.iloc[0]["index"])
            window = group[(group.index > h1_idx) & (group.index < end_idx)]
        else:
            window = group[group.index > h1_idx]
        subs = (
            window[window["final_level"].astype(str).str.upper().isin(["H2", "H3"])]
            ["cleaned_title"]
            .fillna("")
            .astype(str)
            .tolist()
        )
        subs = [s for s in subs if s][:max_h2_per_h1]
        if subs:
            samples[h1] = subs
    return samples


def aggregate_doc_context(
    df: pd.DataFrame,
    max_h1: int,
    max_h2_per_h1: int,
    max_h1_with_samples: int,
) -> Tuple[Dict[str, List[str]], Dict[str, Dict[str, List[str]]]]:
    """
    返回 doc_id → H1 列表、及 doc_id → {H1: H2 样本列表} 映射。
    """
    h1_map: Dict[str, List[str]] = {}
    h2_map: Dict[str, Dict[str, List[str]]] = {}
    for doc_id, group in df.groupby("doc_id", sort=False):
        key = str(doc_id)
        grp = group.copy()
        grp["final_level_norm"] = grp["final_level"].astype(str).str.upper()
        h1_titles = grp[grp["final_level_norm"] == "H1"]["cleaned_title"].fillna("").astype(str).tolist()
        if not h1_titles:
            h1_titles = grp[grp["final_level_norm"] == "H2"]["cleaned_title"].fillna("").astype(str).tolist()
        h1_titles = [t for t in h1_titles if t][:max_h1]
        h1_map[key] = h1_titles
        if h1_titles:
            selection = h1_titles[:max_h1_with_samples]
            h2_map[key] = extract_h2_samples(grp, selection, max_h2_per_h1)
        else:
            h2_map[key] = {}
    return h1_map, h2_map


def heuristic_granularity(h1_titles: List[str]) -> Tuple[str, float]:
    joined = " ".join(h1_titles).strip()
    if any(key in joined for key in ("体系", "框架", "战略", "布局")):
        return "macro", 0.7
    if any(joined.startswith(prefix) for prefix in ("完善", "提升", "加强", "优化", "统筹", "推进", "发展", "建设")):
        return "meso", 0.7
    return "micro", 0.6


def main() -> None:
    ap = argparse.ArgumentParser(description="PolicyTreeBuilder final replication · Granularity calibration Calibrate Levels")
    ap.add_argument("--corpus", type=str, default=str(OUT_DIR_DEFAULT / "policy_corpus_filtered.csv"))
    ap.add_argument("--outdir", type=str, default=str(OUT_DIR_DEFAULT))
    ap.add_argument("--env", type=str, default=str(ROOT / "configs" / ".env"))
    ap.add_argument("--mode", type=str, choices=("llm", "rule"), default="llm")
    ap.add_argument("--model", type=str, default=None, help="LLM model override for the configured primary profile")
    ap.add_argument("--prompt", type=str, default=str(PROMPT_DEFAULT))
    ap.add_argument("--threshold", type=float, default=0.7, help="低置信度复核阈值")
    ap.add_argument("--timeout", type=int, default=60)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--max-h1", type=int, default=10)
    ap.add_argument("--max-h1-with-samples", type=int, default=5)
    ap.add_argument("--max-h2-per-h1", type=int, default=3)
    args = ap.parse_args()

    if args.env:
        try:
            load_env_file(args.env)
        except FileNotFoundError:
            print(f"[WARN] env file not found: {args.env}")

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR_DEFAULT / "calibrate_policy_granularity.log"

    corpus_path = Path(args.corpus)
    if not corpus_path.exists():
        fallback = OUT_DIR_DEFAULT / "policy_corpus_cleaned.csv"
        if fallback.exists():
            corpus_path = fallback
        else:
            raise FileNotFoundError(f"corpus not found: {args.corpus}")

    df = pd.read_csv(corpus_path, dtype=str)
    ensure_columns(df, ["sample_id", "doc_id", "final_level", "cleaned_title", "path_text"], "corpus")

    llm_profile, _ = profiles_from_config({})

    system_text = read_system_prompt(args.prompt)
    h1_map, h2_map = aggregate_doc_context(df, args.max_h1, args.max_h2_per_h1, args.max_h1_with_samples)

    doc_ids = list(dict.fromkeys(df["doc_id"].astype(str).tolist()))
    results: List[pd.DataFrame] = []
    review_rows: List[Dict[str, object]] = []
    gran_dist = {"macro": 0, "meso": 0, "micro": 0}

    total = len(doc_ids)
    for idx, doc_id in enumerate(doc_ids, start=1):
        if idx == 1 or idx % 10 == 0 or idx == total:
            log_write(log_path, f"[INFO] calibrating doc {idx}/{total}: {doc_id}")
        g = "meso"
        conf = 0.6
        raw = ""
        exceptions: List[Dict[str, str]] = []
        if args.mode == "llm":
            prompt = build_user_prompt(
                doc_id,
                h1_map.get(doc_id, []),
                h2_map.get(doc_id, {}),
                args.max_h1,
                args.max_h2_per_h1,
                args.max_h1_with_samples,
            )
            g, conf, raw, exceptions = call_llm_granularity(
                profile=llm_profile,
                model_override=args.model,
                system_text=system_text,
                user_text=prompt,
                timeout_s=args.timeout,
                retries=args.retries,
            )
        else:
            g, conf = heuristic_granularity(h1_map.get(doc_id, []))
            exceptions = []

        gran_dist[g] = gran_dist.get(g, 0) + 1
        sub = df[df["doc_id"].astype(str) == doc_id].copy()
        sub["doc_granularity"] = g
        sub["calibration_confidence"] = float(conf)
        sub["calibrated_level"] = [
            map_level(g, str(lv).strip().upper()) for lv in sub["final_level"]
        ]
        if exceptions:
            for exc in exceptions:
                title = exc.get("h1_title") or ""
                egran = exc.get("exception_granularity") or ""
                title = str(title).strip()
                egran = str(egran).strip().lower()
                if not title or egran not in ("macro", "meso", "micro"):
                    continue
                new_level = map_level(egran, "H1")
                mask = (
                    sub["final_level"].astype(str).str.upper().eq("H1")
                    & sub["cleaned_title"].astype(str).eq(title)
                )
                if mask.any():
                    sub.loc[mask, "calibrated_level"] = new_level
        if exceptions:
            log_write(
                log_path,
                f"[INFO] doc {doc_id} overrides {len(exceptions)} H1 exceptions",
            )
        results.append(sub)

        if conf < float(args.threshold):
            review_rows.append(
                {
                    "doc_id": doc_id,
                    "confidence": float(conf),
                    "llm_raw": raw,
                    "exceptions": json.dumps(exceptions, ensure_ascii=False),
                }
            )

    out_df = pd.concat(results, ignore_index=True) if results else df.copy()
    calibrated_path = out_dir / "policy_corpus_calibrated.csv"
    out_df.to_csv(calibrated_path, index=False, encoding="utf-8-sig")
    log_write(log_path, f"[WRITE] {calibrated_path} rows={len(out_df)}")

    report = {
        "total_docs": len(doc_ids),
        "granularity_distribution": {k: int(v) for k, v in gran_dist.items()},
        "low_confidence_docs": review_rows[:100],
        "input_corpus": str(corpus_path),
    }
    report_path = out_dir / "policy_granularity_calibration_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    log_write(log_path, f"[WRITE] {report_path}")

    if review_rows:
        review_path = out_dir / "policy_granularity_calibration_review.csv"
        pd.DataFrame(review_rows).to_csv(review_path, index=False, encoding="utf-8-sig")
        log_write(log_path, f"[WRITE] {review_path}")

    print(f"[WRITE] {calibrated_path}")
    print(f"[WRITE] {report_path}")
    if review_rows:
        print(f"[WRITE] {out_dir / 'policy_granularity_calibration_review.csv'}")


if __name__ == "__main__":
    main()
