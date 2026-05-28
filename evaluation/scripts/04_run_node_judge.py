#!/usr/bin/env python3
"""Run one or more model judges on sampled policy-tree nodes."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from eval_paths import default_output_dir, resolve_repo_path
from prompts import NODE_SYSTEM, NODE_USER_TEMPLATE

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from llm_runtime import available_judges, call_llm_json, judge_profile_name, resolve_profile  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--judge", required=True, help="Judge key from configs/llm_profiles, or 'all'.")
    parser.add_argument("--limit", type=int, default=None, help="Only evaluate the first N pending samples.")
    parser.add_argument("--no-resume", action="store_true", help="Do not skip samples already in scores JSONL.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_dir(),
        help="Directory containing sampled_nodes_for_judge.jsonl and receiving judge outputs.",
    )
    return parser.parse_args()


def build_user_prompt(sample: dict) -> str:
    current = sample["current_node"]
    parent = sample.get("parent_node") or {"node_id": "", "label": "(none)", "level": "", "depth": -1}
    return NODE_USER_TEMPLATE.format(
        current_node_json=json.dumps(current, ensure_ascii=False, indent=2),
        parent_node_json=json.dumps(parent, ensure_ascii=False, indent=2),
        sibling_nodes_json=json.dumps(sample["sibling_nodes"], ensure_ascii=False, indent=2),
        children_nodes_json=json.dumps(sample["children_nodes"], ensure_ascii=False, indent=2),
        path_from_root_json=json.dumps(sample["path_from_root"], ensure_ascii=False),
        node_id=current["node_id"],
        node_label=current["label"].replace('"', '\\"'),
    )


def load_done_ids(path: Path) -> set[str]:
    done_ids: set[str] = set()
    if not path.exists():
        return done_ids
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                done_ids.add(json.loads(line)["sample_id"])
            except Exception:
                pass
    return done_ids


def run_one_judge(judge_key: str, samples: list[dict], output_dir: Path,
                  limit: int | None = None, resume: bool = True) -> None:
    profile_name = judge_profile_name(judge_key)
    profile_cfg = resolve_profile(profile_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"judge_{judge_key}_node_scores.jsonl"
    raw_path = output_dir / f"judge_{judge_key}_node_raw.jsonl"

    done_ids = load_done_ids(out_path) if resume else set()
    if done_ids:
        print(f"  already completed: {len(done_ids)}; resuming")

    todo = [sample for sample in samples if sample["sample_id"] not in done_ids]
    if limit:
        todo = todo[:limit]
    print(f"  pending: {len(todo)} / {len(samples)}")

    fail_count = 0
    with out_path.open("a", encoding="utf-8") as f_score, raw_path.open("a", encoding="utf-8") as f_raw:
        for index, sample in enumerate(todo, start=1):
            user_prompt = build_user_prompt(sample)
            resp = call_llm_json(
                profile=profile_name,
                system=NODE_SYSTEM,
                user=user_prompt,
                task="evaluation_node_judge",
            )
            if not resp.get("ok"):
                err_msg = resp.get("error") or "judge call failed"
                raw_dump = resp.get("raw") or ""
                print(f"  [{index}/{len(todo)}] FAIL {sample['sample_id']}: {err_msg}")
                fail_count += 1
                f_raw.write(json.dumps({
                    "sample_id": sample["sample_id"],
                    "error": err_msg,
                    "raw": raw_dump,
                }, ensure_ascii=False) + "\n")
                f_raw.flush()
                continue
            result = resp.get("json") or {}
            raw_text = resp.get("raw") or ""

            record = {
                "sample_id": sample["sample_id"],
                "judge_key": judge_key,
                "judge_provider": profile_cfg.provider,
                "judge_model": resp.get("model") or profile_cfg.model,
                "judge_base_url": profile_cfg.base_url,
                "node_id": sample["current_node"]["node_id"],
                "level": sample["current_node"]["level"],
                "depth": sample["current_node"]["depth"],
                "l1_label": sample["meta"]["l1_label"],
                "high_risk": sample["meta"]["high_risk"],
                "result": result,
            }
            f_score.write(json.dumps(record, ensure_ascii=False) + "\n")
            f_score.flush()
            f_raw.write(json.dumps({"sample_id": sample["sample_id"], "raw": raw_text}, ensure_ascii=False) + "\n")
            f_raw.flush()

            if index % 10 == 0:
                print(f"  [{index}/{len(todo)}] OK; failures so far: {fail_count}")

    print(f"  completed judge={judge_key}, failures={fail_count}, output={out_path}")


def main() -> None:
    args = parse_args()
    output_dir = resolve_repo_path(args.output_dir)

    samples = []
    with (output_dir / "sampled_nodes_for_judge.jsonl").open("r", encoding="utf-8") as f:
        for line in f:
            samples.append(json.loads(line))
    print(f"loaded {len(samples)} node samples")

    judges = available_judges() if args.judge == "all" else [args.judge]
    for judge in judges:
        if judge not in available_judges() and judge_profile_name(judge) == judge:
            print(f"unknown judge: {judge}; available: {available_judges()}")
            continue
        profile = resolve_profile(judge_profile_name(judge))
        print(f"\n===== running judge: {judge} ({profile.model}) =====")
        run_one_judge(judge, samples, output_dir, limit=args.limit, resume=not args.no_resume)


if __name__ == "__main__":
    main()
