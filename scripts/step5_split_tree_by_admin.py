#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 5 · Split Tree by Administration (Tree Pruning) - BOM Fix Version
"""
import json
import csv
import argparse
from pathlib import Path
from typing import Dict, List, Set, Optional
import pandas as pd

# === 1. 路径配置 ===
DEFAULT_OUTDIR = Path(r"data/intermediate_outputs")
ADMIN_MAP_FILE = Path(r"data/source/admin_mapping/roundA_final_overview_scored_selected1120.csv")

GLOBAL_TREE_FILE = DEFAULT_OUTDIR / "v4_tree_coarse_global.json"
CORPUS_FILE = DEFAULT_OUTDIR / "v4_cluster_corpus_cleaned.csv"
MEMBERSHIP_FILES = {
    "L4": DEFAULT_OUTDIR / "v4_membership_L4.csv",
    "L3": DEFAULT_OUTDIR / "v4_membership_L3.csv",
    "L2": DEFAULT_OUTDIR / "v4_membership_L2.csv"
}

def load_json(path: Path) -> Dict:
    if not path.exists():
        print(f"[ERROR] 树文件不存在: {path}")
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(path: Path, data: Dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_admin_mapping() -> Dict[str, Dict]:
    """加载属性映射"""
    print(">>> 1. 加载属性映射...")
    doc_admin_map = {}

    # 1. 读取 RoundA 映射表
    with open(ADMIN_MAP_FILE, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # 兼容带空格的列名
            did = row.get("doc_id", "").strip()
            if did:
                doc_admin_map[did] = {
                    "level": row.get("admin_level", ""),
                    "name": row.get("admin_name", "")
                }

    # 2. 读取语料库 (关键修改：使用 utf-8-sig 防止 BOM 导致列名错误)
    sample_admin_map = {}
    with open(CORPUS_FILE, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)

        # --- 调试信息：检查列名 ---
        headers = reader.fieldnames
        if "sample_id" not in headers:
            print(f"    [警告] 语料库找不到 'sample_id' 列！实际列名是: {headers}")
            # 尝试模糊匹配修复
            actual_sid_col = next((h for h in headers if "sample_id" in h), None)
        else:
            actual_sid_col = "sample_id"

        actual_doc_col = "doc_id" if "doc_id" in headers else next((h for h in headers if "doc_id" in h), None)
        # ------------------------

        for row in reader:
            sid = row.get(actual_sid_col)
            did = row.get(actual_doc_col)

            if sid: sid = sid.strip()
            if did: did = did.strip()

            if sid and did and did in doc_admin_map:
                sample_admin_map[sid] = doc_admin_map[did]

    print(f"    成功关联 {len(sample_admin_map)} 个样本的属性信息。")
    return sample_admin_map

def load_all_memberships() -> Dict[str, Dict[str, Set[str]]]:
    """加载树成员"""
    print(">>> 2. 加载树节点成员...")
    all_members = {}
    total_samples_in_tree = set()

    for level, path in MEMBERSHIP_FILES.items():
        if not path.exists():
            print(f"    [WARN] {level} 成员表缺失: {path}")
            all_members[level] = {}
            continue

        # pandas 读取通常比较稳健
        try:
            df = pd.read_csv(path, dtype=str)
            node_samples = {}
            count = 0

            # 自动清洗列名空格
            df.columns = df.columns.str.strip()

            nid_col = "node_id"
            mid_col = "member_id" if "member_id" in df.columns else "sample_id"

            if nid_col in df.columns and mid_col in df.columns:
                for nid, grp in df.groupby(nid_col):
                    sids = set(grp[mid_col].astype(str).str.strip().tolist())
                    node_samples[str(nid)] = sids
                    total_samples_in_tree.update(sids)
                    count += len(sids)
            all_members[level] = node_samples
            print(f"    {level}: 加载了 {len(node_samples)} 个节点，包含 {count} 个样本关系。")
        except Exception as e:
            print(f"    [ERROR] 读取 {level} 失败: {e}")
            all_members[level] = {}

    print(f"    树结构中总共包含 {len(total_samples_in_tree)} 个唯一的样本 ID。")
    return all_members, total_samples_in_tree

def prune_tree(node: Dict, valid_samples: Set[str], all_members: Dict[str, Dict[str, Set[str]]]) -> Optional[Dict]:
    """递归剪枝"""
    node_level = str(node.get("level", "")).upper()
    node_id = str(node.get("node_id"))

    # 1. 自身是否命中
    is_direct_hit = False
    if node_level in all_members:
        members = all_members[node_level].get(node_id, set())
        # 检查交集
        if not members.isdisjoint(valid_samples):
            is_direct_hit = True

    # 2. 递归检查子节点
    valid_children = []
    original_children = node.get("children", []) or []

    for child in original_children:
        result = prune_tree(child, valid_samples, all_members)
        if result:
            valid_children.append(result)

    # 3. 决定是否保留
    if is_direct_hit or valid_children:
        new_node = node.copy()
        new_node["children"] = valid_children
        return new_node
    else:
        return None

def main():
    print(f"=== 开始生成分层树 (输出目录: {DEFAULT_OUTDIR}) ===")

    # 1. 加载数据
    global_tree = load_json(GLOBAL_TREE_FILE)
    if not global_tree: return

    sample_map = load_admin_mapping()
    all_members, tree_samples = load_all_memberships()

    # 2. 准备过滤器
    print(">>> 3. 准备过滤器...")
    prov_samples = {sid for sid, info in sample_map.items() if "省级" in info["level"]}
    city_samples = {sid for sid, info in sample_map.items() if "市级" in info["level"]}

    prov_intersect = prov_samples.intersection(tree_samples)
    city_intersect = city_samples.intersection(tree_samples)

    print(f"    [统计] 原始映射中 '省级' 样本数: {len(prov_samples)}")
    print(f"    [统计] 实际上树的 '省级' 样本数: {len(prov_intersect)}")
    print(f"    [统计] 原始映射中 '市级' 样本数: {len(city_samples)}")
    print(f"    [统计] 实际上树的 '市级' 样本数: {len(city_intersect)}")

    if len(prov_intersect) == 0 and len(city_intersect) == 0:
        print("\n[警告] 树结构里的样本 和 映射表里的样本 没有交集！")
        return

    # 3. 生成省级树
    print("\n>>> 4. 生成省级树...")
    if prov_intersect:
        prov_tree = prune_tree(global_tree, prov_samples, all_members)
        if prov_tree:
            out_p = DEFAULT_OUTDIR / "v4_tree_provincial.json"
            save_json(out_p, prov_tree)
            print(f"    ✅ 已保存: {out_p}")
        else:
            print("    ❌ 剪枝后结果为空 (结构未命中)")
    else:
        print("    ⚠️ 跳过：没有有效的省级样本在树中。")

    # 4. 生成市级树
    print("\n>>> 5. 生成市级树...")
    if city_intersect:
        city_tree = prune_tree(global_tree, city_samples, all_members)
        if city_tree:
            out_c = DEFAULT_OUTDIR / "v4_tree_city.json"
            save_json(out_c, city_tree)
            print(f"    ✅ 已保存: {out_c}")
        else:
            print("    ❌ 剪枝后结果为空")
    else:
        print("    ⚠️ 跳过：没有有效的市级样本在树中。")

    # 5. 生成具体行政单位树
    print("\n>>> 6. 生成行政单位独立树...")
    name_groups = {}
    for sid, info in sample_map.items():
        name = info["name"]
        if name:
            name_groups.setdefault(name, set()).add(sid)

    sub_dir = DEFAULT_OUTDIR / "trees_by_admin"
    sub_dir.mkdir(exist_ok=True)
    count = 0

    for name, sids in name_groups.items():
        if not sids.intersection(tree_samples):
            continue

        safe_name = "".join([c for c in name if c.isalnum() or c in (' ','_','-')])
        admin_tree = prune_tree(global_tree, sids, all_members)
        if admin_tree:
            save_json(sub_dir / f"tree_{safe_name}.json", admin_tree)
            count += 1

    print(f"    ✅ 已生成 {count} 个具体行政单位的树文件")
    print("\n=== 全部完成 ===")

if __name__ == "__main__":
    main()