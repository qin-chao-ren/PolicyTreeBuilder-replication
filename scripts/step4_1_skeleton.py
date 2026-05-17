#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 4.1 · Skeleton (Refactored)
功能：纵向坍缩 (Vertical Collapse) 与 兄弟重置 (Rehome)
核心改进：使用 TreeManager 管理动态拓扑，使用 Fail-Safe Promote 逻辑。
"""

import argparse
import time
from pathlib import Path
from common_utils import jaccard_overlap, call_json
from utils.step4_shared import Step4Env, EmbeddingHelper, load_tree, dump_tree, append_jsonl, read_membership_map, read_title_map
from utils.tree_manager import TreeManager

PROMPT_COLLAPSE = Path("prompts/step4_vertical_collapse.md")
PROMPT_REHOME = Path("prompts/step4_rehome_siblings.md")


def collect_titles(sample_ids, title_map, top_k=5):
    return [title_map[sid] for sid in sample_ids if sid in title_map][:top_k]


def describe_node(node_id, manager: TreeManager, membership, title_map):
    """生成 Prompt 用的节点描述"""
    node = manager.get_node(node_id)
    if not node:
        return "Node Not Found"

    level = node.get("level", "")
    members = membership.get(level, {}).get(node_id, [])
    titles = collect_titles(members, title_map)
    children_count = len(manager.get_children(node_id))

    return (
        f"ID: {node_id} · level={level} · label={node.get('label','')}\n"
        f"成员数={len(members)} · 子节点数={children_count}\n"
        f"示例标题: {', '.join(titles) if titles else '无'}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    # 1. 初始化环境与共享组件
    env = Step4Env(args.config, "configs/roundC_v4.env")
    llm_cfg = env.build_llm_config()
    emb_helper = EmbeddingHelper(Path(env.config["paths"]["embeddings"]))

    # 2. 加载数据
    raw_tree = load_tree(Path(args.input))
    tm = TreeManager(raw_tree)  # 这里会自动清洗 tree_id

    corpus_path = Path(env.config["paths"]["corpus"])
    title_map = read_title_map(corpus_path)
    membership = {
        lvl: read_membership_map(env.outdir, lvl)
        for lvl in ["L4", "L3", "L2", "L1"]
    }

    # 3. 准备日志与追溯表
    ops_log = env.outdir / "v4_operations_log.jsonl"
    llm_log = env.log_dir / "llm_step4_vertical_collapse.jsonl"
    redirect_map = {}  # 记录 ID 变更 trace

    # 4. 遍历与处理 (为了避免遍历时修改的 Stale Index，我们先快照所有父节点 ID)
    candidate_parents = [nid for nid in tm.get_all_node_ids() if tm.get_children(nid)]

    for parent_id in candidate_parents:
        # 【关键】动态检查：如果父节点在之前的操作中被移除了，跳过
        if not tm.exists(parent_id):
            continue

        parent_node = tm.get_node(parent_id)
        # 获取子节点快照 (处理时子节点可能会变)
        children = list(tm.get_children(parent_id))

        for child_node in children:
            child_id = child_node["node_id"]
            if not tm.exists(child_id):
                continue  # 防御性检查

            # --- 4.1 构建 Evidence ---
            # 计算向量与相似度
            p_mems = membership.get(parent_node.get("level"), {}).get(parent_id, [])
            c_mems = membership.get(child_node.get("level"), {}).get(child_id, [])

            vec_p = emb_helper.get_centroid(p_mems)
            vec_c = emb_helper.get_centroid(c_mems)

            jac = jaccard_overlap(parent_node.get("label", ""), child_node.get("label", ""))
            cos = emb_helper.cosine_sim(vec_p, vec_c)

            # 如果相似度太低，直接跳过 LLM 以节省 Token
            if jac < 0.3 and cos < 0.5:
                continue

            evidence = (
                f"# 父节点\n{describe_node(parent_id, tm, membership, title_map)}\n\n"
                f"# 子节点\n{describe_node(child_id, tm, membership, title_map)}\n\n"
                f"相似度: Jaccard={jac:.2f}, Cosine={cos:.2f}\n"
            )

            # --- 4.2 LLM Call: Collapse Decision ---
            prompt_text = PROMPT_COLLAPSE.read_text(encoding="utf-8")
            resp = call_json(llm_cfg.primary, prompt_text, evidence + "\n请决策:", temperature=0.0)

            # 记录 LLM 日志
            append_jsonl(llm_log, {
                "ts": int(time.time()), "parent": parent_id, "child": child_id,
                "metrics": {"jac": jac, "cos": cos}, "resp": resp
            })

            result = resp.get("json", {})
            decision = result.get("decision", "keep")
            new_label = result.get("new_label")

            # --- 4.3 执行操作 (通过 TreeManager) ---
            op_record = {
                "step": "step4_1", "parent": parent_id, "child": child_id,
                "decision": decision, "reason": result.get("reason")
            }

            if decision == "rename_then_keep" and new_label:
                parent_node["label"] = new_label
                append_jsonl(ops_log, {**op_record, "op": "rename"})

            elif decision == "absorb_child":
                # 父吞噬子
                success = tm.absorb_node(parent_id, child_id)
                if success:
                    if new_label:
                        parent_node["label"] = new_label
                    redirect_map[child_id] = parent_id
                    append_jsonl(ops_log, {**op_record, "op": "absorb"})

            elif decision == "promote_child":
                # 子上位
                # 【Fail-Safe】如果 promote 失败（例如没有爷爷），则降级为 keep
                success = tm.promote_child_safe(child_id)

                if not success:
                    print(f"[WARN] Promote failed for {child_id} (No grandparent?), keeping structure.")
                    append_jsonl(ops_log, {**op_record, "op": "promote_failed_kept"})
                    continue

                # 成功提升后：
                if new_label:
                    tm.get_node(child_id)["label"] = new_label
                redirect_map[parent_id] = child_id  # 父 ID 指向子 ID

                # 处理原父节点的“其他兄弟” (Rehome Siblings)
                siblings = [c for c in tm.get_children(parent_id) if c["node_id"] != child_id]

                if siblings:
                    # 简单示例：默认全部挂给刚刚上位的 child (继承)
                    for sib in siblings:
                        tm.move_node(sib["node_id"], child_id)
                    append_jsonl(ops_log, {**op_record, "op": "promote_and_rehome_orphans"})

                # 最后移除空的父节点
                tm.remove_node(parent_id)
                append_jsonl(ops_log, {**op_record, "op": "promote_success"})

    # 5. 导出结果
    dump_tree(Path(args.output), tm.root)

    # 导出 Trace Map (用于 Step 4.3 合并)
    trace_path = env.outdir / "v4_trace_4_1.json"
    trace_path.write_text(json.dumps(redirect_map, indent=2), encoding="utf-8")

    print(f"[DONE] Skeleton Refined. Tree saved to {args.output}")


if __name__ == "__main__":
    main()
