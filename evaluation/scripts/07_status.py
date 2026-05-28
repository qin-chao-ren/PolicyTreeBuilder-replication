"""
查看所有 judge 的进度和失败情况。

用法:
  python scripts/07_status.py
  python scripts/07_status.py --show-failures   # 列出失败样本及最后一次错误
  python scripts/07_status.py --show-failures --judge C_gemini

不调用 API，只读 outputs/ 下的 jsonl 文件。
"""
import json
import argparse
from pathlib import Path
from collections import Counter, defaultdict

from eval_paths import default_output_dir, resolve_repo_path

OUT = default_output_dir()


def load_jsonl(p):
    out = []
    if not p.exists():
        return out
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    return out


def discover_judges(kind):
    """kind in {node, path}"""
    keys = set()
    for p in OUT.glob(f"judge_*_{kind}_scores.jsonl"):
        name = p.name
        prefix = "judge_"
        suffix = f"_{kind}_scores.jsonl"
        if name.startswith(prefix) and name.endswith(suffix):
            keys.add(name[len(prefix):-len(suffix)])
    for p in OUT.glob(f"judge_*_{kind}_raw.jsonl"):
        name = p.name
        prefix = "judge_"
        suffix = f"_{kind}_raw.jsonl"
        if name.startswith(prefix) and name.endswith(suffix):
            keys.add(name[len(prefix):-len(suffix)])
    return sorted(keys)


def load_sample_ids(jsonl_path):
    """读 sampled_*.jsonl 拿到全量 sample_id"""
    ids = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            ids.append(json.loads(line)["sample_id"])
    return ids


def report_kind(kind, total_ids, judges, show_failures=False, only_judge=None):
    print(f"\n{'='*60}")
    print(f"  {kind.upper()}-LEVEL  (全量样本: {len(total_ids)})")
    print(f"{'='*60}")

    total_set = set(total_ids)

    for jk in judges:
        if only_judge and jk != only_judge:
            continue

        scores = load_jsonl(OUT / f"judge_{jk}_{kind}_scores.jsonl")
        raws = load_jsonl(OUT / f"judge_{jk}_{kind}_raw.jsonl")

        success_ids = {r["sample_id"] for r in scores}
        # raws 中包含成功+失败; 失败行带 "error" 字段
        raw_failed_ids = {r["sample_id"] for r in raws if "error" in r and r["sample_id"] not in success_ids}
        # 既没成功也没在 raw 里 (从未尝试 / 中途中断)
        not_attempted_ids = total_set - success_ids - raw_failed_ids

        print(f"\n  [{jk}]")
        print(f"    成功    : {len(success_ids):>4} / {len(total_set)}")
        print(f"    失败    : {len(raw_failed_ids):>4}  (有 error 记录)")
        print(f"    未尝试  : {len(not_attempted_ids):>4}  (未出现在 scores 或 raw)")
        completed = len(success_ids)
        pct = completed / len(total_set) * 100 if total_set else 0
        bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
        print(f"    进度    : [{bar}] {pct:.1f}%")

        if show_failures and raw_failed_ids:
            print(f"\n    -- 失败样本及错误 --")
            err_counter = Counter()
            for r in raws:
                if r["sample_id"] in raw_failed_ids:
                    err = (r.get("error") or "").split("|||")[0][:120]
                    err_counter[err] += 1
                    raw_preview = (r.get("raw") or "")[:80].replace("\n", "\\n")
                    print(f"      {r['sample_id']}")
                    print(f"        err: {err}")
                    if raw_preview:
                        print(f"        raw: {raw_preview!r}")
            print(f"\n    -- 错误类型分布 --")
            for err, cnt in err_counter.most_common():
                print(f"      {cnt:>3}x  {err[:100]}")


def main():
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--show-failures", action="store_true",
                    help="列出失败样本和它们的错误信息")
    ap.add_argument("--judge", default=None, help="只看某个 judge")
    ap.add_argument("--kind", choices=["node", "path", "both"], default="both")
    ap.add_argument("--output-dir", type=Path, default=default_output_dir(),
                    help="Directory containing sampled files and judge outputs.")
    args = ap.parse_args()
    OUT = resolve_repo_path(args.output_dir)

    if args.kind in ("node", "both"):
        node_ids = load_sample_ids(OUT / "sampled_nodes_for_judge.jsonl")
        node_judges = discover_judges("node")
        report_kind("node", node_ids, node_judges,
                    show_failures=args.show_failures, only_judge=args.judge)

    if args.kind in ("path", "both"):
        path_ids = load_sample_ids(OUT / "sampled_paths_for_judge.jsonl")
        path_judges = discover_judges("path")
        report_kind("path", path_ids, path_judges,
                    show_failures=args.show_failures, only_judge=args.judge)

    print("\n" + "="*60)
    print("  下一步建议:")
    print("="*60)
    print("""
  - 全量再跑一次 (会自动跳过已成功的，只补失败和未尝试):
      python evaluation/scripts/04_run_node_judge.py --judge <key>
      python evaluation/scripts/05_run_path_judge.py --judge <key>

  - 看具体失败原因:
      python evaluation/scripts/07_status.py --show-failures --judge <key>

  - 全部数据 OK 后:
      python evaluation/scripts/06_aggregate.py
""")


if __name__ == "__main__":
    main()
