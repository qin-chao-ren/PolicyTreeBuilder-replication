#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Finalization · Overall Structure (Fixed Version)
功能：L1 级整体结构审计 (Semantic Audit) + 最终层级强制对齐

【修复说明】
1. _call_llm 添加防御性检查，防止 resp 为 None 时崩溃
2. _export_flat_csv 添加防御性检查，防止幽灵节点
3. 添加更详细的错误日志
"""

import argparse
import time
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Generator
from difflib import SequenceMatcher

# 保持与你提供的一致
from llm_runtime import call_llm_json
from utils.step4_shared import (
    Step4Env,
    load_tree,
    append_jsonl
)
from utils.tree_manager import TreeManager
from utils.tree_integrity import (
    E0ValidationError,
    LineageError,
    atomic_write_bytes,
    atomic_write_csv,
    atomic_write_json,
    atomic_write_jsonl,
    merge_lineage_maps,
    read_jsonl,
    read_membership_csv,
    redirect_membership_rows,
    validate_tree_e0,
)

# --- 1. 路径锚点 (Path Anchors) ---
HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent

# --- 2. 关键资源路径 (基于锚点) ---
ENV_PATH = PROJECT_ROOT / "configs" / ".env"

# --- 3. Prompt 路径 ---
PROMPT_PATH = PROJECT_ROOT / "prompts" / "finalize_tree_structure.md"

# --- 4. 默认输出/输入路径 ---
DEFAULT_L1_DEF = PROJECT_ROOT / "data" / "intermediate_outputs" / "top_level_categories.json"
DEFAULT_AUDIT = PROJECT_ROOT / "data" / "intermediate_outputs" / "policy_tree_final_audit.json"
DEFAULT_FLAT = PROJECT_ROOT / "data" / "intermediate_outputs" / "policy_tree_final_flat.csv"

BATCH_SIZE_LIMIT = 200

# --- Helper Functions ---

def get_l1_ancestor(tm: TreeManager, node_id: str) -> Optional[str]:
    curr = node_id
    while curr:
        node = tm.get_node(curr)
        if not node: return None
        if str(node.get("level", "")).upper() == "L1":
            return curr
        curr = tm.get_parent_id(curr)
    return None

def render_subtree(tm: TreeManager, node_id: str, prefix_path: List[str]) -> List[str]:
    node = tm.get_node(node_id)
    if not node: return []

    level = str(node.get("level", "")).upper()
    label = str(node.get("label", "") or "")
    path_str = ' / '.join(prefix_path + [label])

    children = tm.get_children(node_id)
    structural_children = [c for c in children if str(c.get("level", "")).upper().startswith("L")]

    markers = []
    if len(structural_children) == 1:
        markers.append('⚡SingleChild')

    parent_label = prefix_path[-1] if prefix_path else ''
    if parent_label:
        sim = SequenceMatcher(None, str(parent_label).lower(), label.lower()).ratio()
        if sim > 0.6:
            markers.append(f'⚡Repetitive({int(sim * 100)}%)')

    suffix = f" << {' '.join(markers)} >>" if markers else ''
    lines = [f"- [{level}] {path_str} ({node_id}){suffix}"]

    next_path = prefix_path + [label]
    for ch in structural_children:
        lines.extend(render_subtree(tm, ch["node_id"], next_path))

    return lines

def generate_l1_batches(tm: TreeManager, l1_id: str, definition: str) -> Generator[str, None, None]:
    l1_node = tm.get_node(l1_id)
    header_lines = [
        f"# L1 节点\nID={l1_id} · label={l1_node.get('label','')}\n定义：{definition or '未提供'}",
        "# Tree 展开 (L2-L4)",
    ]
    header_text = '\n'.join(header_lines)

    l2_nodes = [ch for ch in tm.get_children(l1_id) if str(ch.get("level", "")).upper() == "L2"]

    if not l2_nodes:
        yield header_text
        return

    current_batch_lines = []
    for l2 in l2_nodes:
        l2_block = render_subtree(tm, l2["node_id"], [])
        current_size = len(header_lines) + len(current_batch_lines)
        block_size = len(l2_block)

        if current_size + block_size > BATCH_SIZE_LIMIT and current_batch_lines:
            yield header_text + "\n" + "\n".join(current_batch_lines)
            current_batch_lines = []

        if block_size > BATCH_SIZE_LIMIT:
            if current_batch_lines:
                yield header_text + "\n" + "\n".join(current_batch_lines)
                current_batch_lines = []
            yield header_text + "\n" + "\n".join(l2_block[:BATCH_SIZE_LIMIT]) + "\n... (截断: 节点过大)"
        else:
            current_batch_lines.extend(l2_block)

    if current_batch_lines:
        yield header_text + "\n" + "\n".join(current_batch_lines)

# --- Main Logic Class ---

class OverallStructureAudit:
    def __init__(self, env: Step4Env, tm: TreeManager, args):
        self.env = env
        self.llm_profile = env.primary_llm_profile()
        self.tm = tm
        self.args = args

        self.ops_log = env.outdir / "policy_tree_final_operations.jsonl"
        self.llm_log = env.log_dir / "llm_finalize_policy_tree.jsonl"
        self.audit_log = env.outdir / "policy_tree_final_audit.json"

        self.redirect_map: Dict[str, str] = {}
        self.audit_entries = []
        self.operation_records: List[Dict] = []

        # 统计
        self.stats = {
            "llm_calls": 0,
            "llm_failures": 0,
            "ops_applied": 0,
            "ops_skipped": 0,
        }

        membership_input_arg = getattr(self.args, "membership_input", "")
        self.membership_input_explicit = bool(membership_input_arg)
        self.membership_input = Path(
            membership_input_arg or (self.env.outdir / "policy_tree_final_membership.csv")
        )
        self.membership_output = Path(
            getattr(self.args, "membership_out", "")
            or (Path(self.args.output).parent / "policy_tree_final_membership.csv")
        )
        self.lineage_input = Path(
            getattr(self.args, "lineage_in", "")
            or (self.env.outdir / "policy_tree_lineage.json")
        )
        self.lineage_output = Path(
            getattr(self.args, "lineage_out", "")
            or (Path(self.args.output).parent / "policy_tree_final_lineage.json")
        )
        self.operations_output = Path(
            getattr(self.args, "operations_out", "")
            or (Path(self.args.output).parent / "policy_tree_final_operations.jsonl")
        )
        self.operations_input = Path(
            getattr(self.args, "operations_input", "")
            or (self.env.outdir / "tree_refinement_operations.jsonl")
        )

    def run(self):
        self.ops_log.parent.mkdir(parents=True, exist_ok=True)
        self.ops_log.write_text("", encoding="utf-8")

        # 1. 解决挂在 Root 下的非 L1 游离节点 (Pending L1)
        self._resolve_pending_root_nodes()

        # 2. 遍历所有 L1 进行审计
        l1_nodes = [n for n in self.tm.get_children(self.tm.root["node_id"])
                    if str(n.get("level", "")).upper() == "L1"]

        l1_defs = self._load_l1_defs()
        print(f"[Finalization] Auditing {len(l1_nodes)} L1 categories...")

        for l1 in l1_nodes:
            l1_id = l1["node_id"]
            definition = l1_defs.get(l1_id, "")

            for batch_ctx in generate_l1_batches(self.tm, l1_id, definition):
                ops = self._call_llm(batch_ctx, l1_id)
                if not ops:
                    continue

                applied = self._apply_operations(ops, l1_id)
                self.operation_records.extend(applied)
                self.audit_entries.append({
                    "l1_id": l1_id,
                    "label": l1.get("label"),
                    "raw_ops": ops,
                    "applied": applied
                })
                for item in applied:
                    append_jsonl(self.ops_log, item)

        # 关键修复：在导出前强制重算层级
        self._realign_tree_levels()

        # 3. Build candidate auxiliary outputs without touching formal outputs.
        candidate_operations = list(self.operation_records)
        try:
            source_fieldnames, source_membership = self._load_membership_source()
            lineage = self._load_lineage()
            candidate_operations = self._load_operation_history() + self.operation_records
            candidate_membership = redirect_membership_rows(
                source_membership, lineage, self.tm.get_all_node_ids()
            )
            flat_fieldnames, flat_rows = self._export_flat_csv()
        except Exception as exc:
            e0_report = validate_tree_e0(
                self.tm.root,
                manager=self.tm,
                operations=candidate_operations,
                require_membership=True,
            )
            e0_report["violations"].append({
                "severity": "critical",
                "code": "CANDIDATE_PREPARATION_FAILED",
                "message": str(exc),
                "context": {"exception_type": type(exc).__name__},
            })
            e0_report["passed"] = False
            e0_report["critical_count"] = len(e0_report["violations"])
            e0_report["violation_counts"] = dict(sorted(Counter(
                item["code"] for item in e0_report["violations"]
            ).items()))
            atomic_write_json(self.args.audit_out, self._audit_payload(e0_report, {}))
            raise E0ValidationError(
                f"E0 candidate preparation failed: {exc}"
            ) from exc

        # 4. E0 is evaluated against the candidate state before publication.
        e0_report = validate_tree_e0(
            self.tm.root,
            manager=self.tm,
            membership_rows=candidate_membership,
            expected_membership_rows=source_membership,
            lineage=lineage,
            operations=candidate_operations,
            require_membership=True,
        )
        audit_payload = self._audit_payload(e0_report, lineage)
        atomic_write_json(self.args.audit_out, audit_payload)
        if not e0_report["passed"]:
            raise E0ValidationError(
                f"E0 rejected final candidate with {e0_report['critical_count']} critical violation(s)"
            )

        # 5. Publish auxiliary files first and the formal tree last.
        try:
            self._publish_candidate_outputs(
                source_fieldnames=source_fieldnames,
                candidate_membership=candidate_membership,
                flat_fieldnames=flat_fieldnames,
                flat_rows=flat_rows,
                lineage=lineage,
                candidate_operations=candidate_operations,
            )
        except Exception as exc:
            e0_report["violations"].append({
                "severity": "critical",
                "code": "PUBLICATION_FAILED",
                "message": str(exc),
                "context": {"exception_type": type(exc).__name__},
            })
            e0_report["passed"] = False
            e0_report["critical_count"] = len(e0_report["violations"])
            e0_report["violation_counts"] = dict(sorted(Counter(
                item["code"] for item in e0_report["violations"]
            ).items()))
            atomic_write_json(self.args.audit_out, self._audit_payload(e0_report, lineage))
            raise E0ValidationError(f"Candidate publication failed: {exc}") from exc

        # 打印统计
        print(f"\n[Finalization Stats]")
        print(f"  LLM calls: {self.stats['llm_calls']} (failures: {self.stats['llm_failures']})")
        print(f"  Operations: applied={self.stats['ops_applied']}, skipped={self.stats['ops_skipped']}")
        print(f"  E0: PASS (critical=0)")
        print(f"[DONE] Audit Completed. Final Tree published: {self.args.output}")

    def _audit_payload(self, e0_report: Dict, lineage: Dict[str, str]) -> Dict:
        return {
            "ts": int(time.time()),
            "entries": self.audit_entries,
            "stats": self.stats,
            "lineage": lineage,
            "e0": e0_report,
        }

    def _publish_candidate_outputs(
        self,
        *,
        source_fieldnames,
        candidate_membership,
        flat_fieldnames,
        flat_rows,
        lineage,
        candidate_operations,
    ):
        writes = [
            (
                self.membership_output,
                lambda: atomic_write_csv(
                    self.membership_output, source_fieldnames, candidate_membership
                ),
            ),
            (
                Path(self.args.flat_csv),
                lambda: atomic_write_csv(self.args.flat_csv, flat_fieldnames, flat_rows),
            ),
            (
                self.lineage_output,
                lambda: atomic_write_json(self.lineage_output, lineage),
            ),
            (
                self.operations_output,
                lambda: atomic_write_jsonl(self.operations_output, candidate_operations),
            ),
            (
                Path(self.args.output),
                lambda: atomic_write_json(self.args.output, self.tm.root),
            ),
        ]
        snapshots = {
            path: path.read_bytes() if path.exists() else None
            for path, _ in writes
        }
        try:
            for _, write in writes:
                write()
        except Exception as exc:
            rollback_errors = []
            for path, _ in reversed(writes):
                try:
                    previous = snapshots[path]
                    if previous is None:
                        path.unlink(missing_ok=True)
                    else:
                        atomic_write_bytes(path, previous)
                except Exception as rollback_exc:
                    rollback_errors.append(f"{path}: {rollback_exc}")
            detail = ""
            if rollback_errors:
                detail = f" Rollback errors: {'; '.join(rollback_errors)}"
            raise RuntimeError(f"formal output write failed: {exc}.{detail}") from exc

    def _realign_tree_levels(self):
        """
        DFS 遍历树，根据物理深度强制重写 level 属性。
        """
        print("[Finalization] Re-aligning node levels based on physical topology...")

        def dfs(node_id, current_depth):
            node = self.tm.get_node(node_id)
            if not node: return

            node["level"] = "ROOT" if current_depth == 0 else f"L{current_depth}"

            children = self.tm.get_children(node_id)
            for child in children:
                dfs(child["node_id"], current_depth + 1)

        dfs(self.tm.root["node_id"], 0)
        print("[Finalization] Level realignment completed.")

    def _load_l1_defs(self) -> Dict[str, str]:
        p = Path(self.args.l1_def)
        if not p.exists(): return {}
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            return {str(item.get("id")): item.get("definition", "")
                    for item in data.get("categories", [])}
        except:
            return {}

    def _resolve_pending_root_nodes(self):
        root_children = self.tm.get_children(self.tm.root["node_id"])
        pending = [ch for ch in root_children if ch.get("pending_as_l1")]

        if not pending: return
        print(f"[Finalization] Resolving {len(pending)} pending root nodes...")

        for node in pending:
            node["level"] = "L1"
            node.pop("pending_as_l1", None)
            node.pop("original_level", None)
            node.pop("pending_parent", None)
            record = {
                "op": "pending_auto_promote", "type": "pending_auto_promote",
                "node_id": node["node_id"], "reason": "Finalization Auto Fix",
                "status": "applied"
            }
            self.operation_records.append(record)
            append_jsonl(self.ops_log, record)

    def _call_llm(self, context: str, l1_id: str) -> List[Dict]:
        """
        调用 LLM 进行审计

        【修复】添加防御性检查，防止 resp 为 None 时崩溃
        """
        self.stats["llm_calls"] += 1

        try:
            instruction = PROMPT_PATH.read_text(encoding="utf-8")
        except FileNotFoundError:
            print(f"[ERROR] Prompt file not found: {PROMPT_PATH}")
            self.stats["llm_failures"] += 1
            return []

        # 调用 LLM
        try:
            resp = call_llm_json(
                profile=self.llm_profile,
                system=instruction,
                user=context,
                task="finalize_policy_tree",
            )
        except Exception as e:
            print(f"[ERROR] LLM call raised exception: {e}")
            self.stats["llm_failures"] += 1
            append_jsonl(self.llm_log, {
                "ts": int(time.time()), "l1_id": l1_id,
                "error": str(e), "resp": None
            })
            return []

        # 记录日志
        append_jsonl(self.llm_log, {
            "ts": int(time.time()), "l1_id": l1_id, "resp": resp
        })

        # =====================================================
        # 【核心修复】防御性检查：防止 resp 为 None 时崩溃
        # =====================================================
        if resp is None:
            print(f"[WARN] LLM call returned None for L1: {l1_id}. Skipping this batch.")
            self.stats["llm_failures"] += 1
            return []

        if not isinstance(resp, dict):
            print(f"[WARN] LLM response is not a dict: {type(resp)}. Skipping.")
            self.stats["llm_failures"] += 1
            return []

        if "json" not in resp:
            # 可能 LLM 返回了纯文本或解析失败
            print(f"[WARN] LLM response missing 'json' key. Raw snippet: {str(resp)[:200]}")
            self.stats["llm_failures"] += 1
            return []

        json_data = resp.get("json")
        if json_data is None:
            print(f"[WARN] resp['json'] is None. Skipping.")
            self.stats["llm_failures"] += 1
            return []

        if not isinstance(json_data, dict):
            print(f"[WARN] resp['json'] is not a dict: {type(json_data)}. Skipping.")
            self.stats["llm_failures"] += 1
            return []

        operations = json_data.get("operations", [])
        if not isinstance(operations, list):
            print(f"[WARN] 'operations' is not a list: {type(operations)}. Skipping.")
            self.stats["llm_failures"] += 1
            return []

        return operations

    def _apply_operations(self, ops: List[Dict], l1_id: str) -> List[Dict]:
        applied: List[Dict] = []
        for op in ops:
            if not isinstance(op, dict):
                self.stats["ops_skipped"] += 1
                applied.append({"status": "skipped", "message": "operation is not an object"})
                continue
            typ = op.get("type", "").lower()
            node_id = str(op.get("node_id") or "")

            if not node_id or not self.tm.exists(node_id):
                applied.append({**op, "type": typ, "status": "skipped", "message": "node not found"})
                self.stats["ops_skipped"] += 1
                continue

            status = "skipped"
            message = ""
            record = {**op, "type": typ, "node_id": node_id}
            try:
                if typ == "rename":
                    new_label = op.get("new_label")
                    if new_label and self.tm.rename_node(node_id, new_label):
                        status = "applied"
                        record["source_id"] = node_id
                    else:
                        message = self.tm.last_error or "missing new_label"

                elif typ == "move":
                    target = str(op.get("target_parent_id") or "")
                    record.update({"source_id": node_id, "target_parent_id": target})
                    if not target or not self.tm.exists(target):
                        message = "target not found"
                    elif self.tm.is_descendant(target, node_id):
                        message = "target is descendant"
                    else:
                        t_l1 = get_l1_ancestor(self.tm, target)
                        if t_l1 != l1_id:
                            message = "cross-L1 move forbidden"
                        elif self.tm.move_node(node_id, target):
                            status = "applied"
                        else:
                            message = self.tm.last_error or "move failed"

                elif typ == "merge":
                    target = str(op.get("merge_into") or "")
                    record.update({"source_id": node_id, "target_id": target})
                    if not target or not self.tm.exists(target):
                        message = "target not found"
                    elif target == node_id:
                        message = "merge into self"
                    else:
                        t_l1 = get_l1_ancestor(self.tm, target)
                        if t_l1 != l1_id:
                            message = "cross-L1 merge forbidden"
                        else:
                            if self.tm.absorb_node(
                                target, node_id, new_label=op.get("new_label")
                            ):
                                self.redirect_map[node_id] = target
                                status = "applied"
                            else:
                                status = "failed"
                                message = self.tm.last_error or "absorb failed"
                else:
                    message = "unknown type"
            except Exception as e:
                status, message = "error", str(e)

            if status == "applied":
                self.stats["ops_applied"] += 1
            else:
                self.stats["ops_skipped"] += 1
            applied.append({**record, "status": status, "message": message})
        return applied

    def _load_membership_source(self):
        if self.membership_input.exists():
            fieldnames, rows = read_membership_csv(self.membership_input)
            if "final_node_id" not in fieldnames:
                raise E0ValidationError(
                    f"membership input lacks final_node_id: {self.membership_input}"
                )
            return fieldnames, rows

        if self.membership_input_explicit:
            raise E0ValidationError(
                f"explicit membership input does not exist: {self.membership_input}"
            )

        fieldnames = ["sample_id", "final_node_id", "original_node_id", "original_level"]
        rows = []
        for level in ["L4", "L3", "L2", "L1"]:
            path = self.env.outdir / f"tree_node_membership_{level}.csv"
            if not path.exists():
                continue
            _, level_rows = read_membership_csv(path)
            for row in level_rows:
                original_node_id = str(row.get("node_id", ""))
                rows.append({
                    "sample_id": str(row.get("member_id", "")),
                    "final_node_id": original_node_id,
                    "original_node_id": original_node_id,
                    "original_level": level,
                })
        return fieldnames, rows

    def _load_lineage(self) -> Dict[str, str]:
        existing: Dict[str, str] = {}
        if self.lineage_input.exists():
            payload = json.loads(self.lineage_input.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise LineageError("lineage input must be an object")
            existing = {str(source): str(target) for source, target in payload.items()}
        return merge_lineage_maps(existing, self.redirect_map)

    def _load_operation_history(self) -> List[Dict]:
        if not self.operations_input.exists():
            return []
        return read_jsonl(self.operations_input)

    def _export_flat_csv(self):
        rows = []
        cols = ["L1_id", "L1_label", "L2_id", "L2_label", "L3_id", "L3_label", "L4_id", "L4_label"]

        def dfs(node_id, path_stack):
            node = self.tm.get_node(node_id)

            # 【修复】防御性检查：防止幽灵节点
            if not node:
                return

            new_stack = path_stack + [node]

            children = [c for c in self.tm.get_children(node_id)
                        if str(c.get("level", "")).upper().startswith("L")]

            if not children:
                row = {c: "" for c in cols}
                for n in new_stack:
                    if not n: continue  # 再次防御
                    nlvl = str(n.get("level", "")).upper()
                    if f"{nlvl}_id" in row:
                        row[f"{nlvl}_id"] = n["node_id"]
                        row[f"{nlvl}_label"] = n.get("label", "")
                rows.append(row)

            for ch in children:
                dfs(ch["node_id"], new_stack)

        l1_nodes = [n for n in self.tm.get_children(self.tm.root["node_id"])
                    if str(n.get("level", "")).upper() == "L1"]
        for l1 in l1_nodes:
            dfs(l1["node_id"], [])

        return cols, rows

def main():
    parser = argparse.ArgumentParser(description="PolicyTreeBuilder final replication · Finalization Overall Structure (Fixed)")
    parser.add_argument("--input", required=True, help="policy_tree_refined.json")
    parser.add_argument("--output", required=True, help="policy_tree_final.json")
    parser.add_argument("--config", required=True, help="tree_refinement_config.yaml")

    parser.add_argument("--l1-def", default=str(DEFAULT_L1_DEF))
    parser.add_argument("--audit-out", default=str(DEFAULT_AUDIT))
    parser.add_argument("--flat-csv", default=str(DEFAULT_FLAT))
    parser.add_argument("--membership-input", help="Pre-finalization membership CSV")
    parser.add_argument("--membership-out", help="Published final membership CSV")
    parser.add_argument("--lineage-in", help="Consolidated pre-finalization lineage JSON")
    parser.add_argument("--lineage-out", help="Published closed lineage JSON")
    parser.add_argument("--operations-input", help="Pre-finalization refinement operation JSONL")
    parser.add_argument("--operations-out", help="Published finalization operation JSONL")

    args = parser.parse_args()

    # 1. 初始化
    env = Step4Env(args.config, str(ENV_PATH))

    # 2. 加载数据
    raw_tree = load_tree(Path(args.input))
    tm = TreeManager(raw_tree)

    # 3. 执行
    audit = OverallStructureAudit(env, tm, args)
    audit.run()

if __name__ == "__main__":
    main()
