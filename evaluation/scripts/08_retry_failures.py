"""
清理某个 judge 的 raw 文件中失败的记录, 让下次重跑可以从干净状态重新写入。

为什么需要这个工具:
  - 04/05 脚本的断点续跑只看 _scores.jsonl, 不看 _raw.jsonl
  - 不清理也能续跑, 但 raw 文件会累积每次失败的记录, 排查时混乱
  - 此脚本只删除 raw 中的失败行, 保留成功对应的 raw, 保留 scores

用法:
  python scripts/08_retry_failures.py --judge C_gemini --kind node
  python scripts/08_retry_failures.py --judge C_gemini --kind path
  python scripts/08_retry_failures.py --judge C_gemini --kind both
  python scripts/08_retry_failures.py --judge all --kind both    # 清理所有 judge

  --dry-run: 只打印将要清理多少条, 不实际改文件
"""
import json
import argparse
import shutil
from pathlib import Path
from datetime import datetime

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
    keys = set()
    for p in OUT.glob(f"judge_*_{kind}_*.jsonl"):
        name = p.name
        for suffix in (f"_{kind}_scores.jsonl", f"_{kind}_raw.jsonl"):
            if name.endswith(suffix):
                key = name[len("judge_"):-len(suffix)]
                keys.add(key)
                break
    return sorted(keys)


def clean_raw(judge_key, kind, dry_run=False):
    raw_path = OUT / f"judge_{judge_key}_{kind}_raw.jsonl"
    scores_path = OUT / f"judge_{judge_key}_{kind}_scores.jsonl"
    if not raw_path.exists():
        print(f"  [{judge_key}/{kind}] raw 文件不存在, 跳过")
        return

    raws = load_jsonl(raw_path)
    success_ids = {r["sample_id"] for r in load_jsonl(scores_path)}

    keep = []
    drop_failed = 0
    drop_dup_success = 0
    seen_success = set()
    for r in raws:
        sid = r["sample_id"]
        is_failure = "error" in r
        if is_failure:
            drop_failed += 1
            continue
        # 同一个成功样本若有多条 raw（比如重跑后旧记录还在），只保留最后一条
        if sid in success_ids:
            if sid in seen_success:
                drop_dup_success += 1
                # 不 keep，直接覆盖前面的
                # 但因为我们是顺序遍历，需要在 keep 里替换前面的
                # 简化处理：从 keep 里移除旧的同 sid 项
                keep = [k for k in keep if k["sample_id"] != sid]
            seen_success.add(sid)
            keep.append(r)
        else:
            # 在 raw 里但不在 scores 里且没标 error,
            # 多半是历史脚本落盘逻辑 bug，保留
            keep.append(r)

    print(f"  [{judge_key}/{kind}]")
    print(f"    raw 总行数 : {len(raws)}")
    print(f"    将删除失败 : {drop_failed}")
    print(f"    将删除重复 : {drop_dup_success}")
    print(f"    保留       : {len(keep)}")

    if dry_run:
        print(f"    (dry-run, 未实际修改)")
        return

    if drop_failed == 0 and drop_dup_success == 0:
        print(f"    无需清理")
        return

    # 备份原文件
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = raw_path.with_suffix(f".jsonl.bak_{ts}")
    shutil.copy2(raw_path, backup)
    print(f"    已备份至: {backup.name}")

    with raw_path.open("w", encoding="utf-8") as f:
        for r in keep:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"    已重写 {raw_path.name}")


def main():
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", required=True,
                    help="judge_key 或 'all'")
    ap.add_argument("--kind", choices=["node", "path", "both"], default="both")
    ap.add_argument("--dry-run", action="store_true",
                    help="只打印, 不修改文件")
    ap.add_argument("--output-dir", type=Path, default=default_output_dir(),
                    help="Directory containing judge raw and score JSONL files.")
    args = ap.parse_args()
    OUT = resolve_repo_path(args.output_dir)

    kinds = ["node", "path"] if args.kind == "both" else [args.kind]

    if args.judge == "all":
        judges = set()
        for k in kinds:
            judges |= set(discover_judges(k))
        judges = sorted(judges)
    else:
        judges = [args.judge]

    print(f"目标 judges: {judges}")
    print(f"目标 kinds : {kinds}")
    if args.dry_run:
        print("模式: DRY-RUN\n")
    else:
        print()

    for jk in judges:
        for k in kinds:
            clean_raw(jk, k, dry_run=args.dry_run)
        print()

    print("完成。下一步可重跑:")
    for jk in judges:
        if "node" in kinds:
            print(f"  python evaluation/scripts/04_run_node_judge.py --judge {jk}")
        if "path" in kinds:
            print(f"  python evaluation/scripts/05_run_path_judge.py --judge {jk}")


if __name__ == "__main__":
    main()
