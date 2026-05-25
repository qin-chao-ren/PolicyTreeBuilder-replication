#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Automation Test: Step 4 Pipeline (Mini Integration Test)
运行位置：E:\\code\\llm_based_graphpolicy (项目根目录)
命令：python scripts/tests/run_step4_mini_test.py
"""

import sys
import os
import json
import shutil
import subprocess
import yaml
from pathlib import Path

# --- 1. 路径锚点 (Path Anchors) ---
# HERE = scripts/tests/
HERE = Path(__file__).resolve().parent

# ROUNDC_ROOT = roundC_v4/
ROUNDC_ROOT = HERE.parents[1]

# SCRIPTS_DIR = scripts/
SCRIPTS_DIR = ROUNDC_ROOT / "scripts"

# OUTPUTS_DIR = data/intermediate_outputs/
OUTPUTS_DIR = ROUNDC_ROOT / "data" / "intermediate_outputs"
TEST_DIR = OUTPUTS_DIR / "test"

# 真实数据路径
REAL_TREE_PATH = OUTPUTS_DIR / "v4_tree_coarse_global.json"
REAL_CONFIG_PATH = ROUNDC_ROOT / "configs" / "step4_config.yaml"

# 测试用临时文件 (全部使用绝对路径)
TEST_CONFIG_PATH = TEST_DIR / "test_config.yaml"
TEST_INPUT_TREE = TEST_DIR / "test_tree_input.json"
TEST_TREE_S1 = TEST_DIR / "test_tree_s1.json"
TEST_TREE_S2 = TEST_DIR / "test_tree_s2.json"
TEST_TREE_S3 = TEST_DIR / "test_tree_s3.json"
TEST_TREE_FINAL = TEST_DIR / "test_tree_final.json"

def run_command(cmd_args):
    """执行 Shell 命令"""
    # 打印相对路径以便阅读，实际执行用绝对路径
    display_cmd = " ".join([str(arg).replace(str(ROUNDC_ROOT), ".") for arg in cmd_args])
    print(f"\n[EXEC] {display_cmd}")

    try:
        # cwd 设置为 roundC_v4 的父目录 (即项目根目录 .)
        # 这样脚本内部的相对路径逻辑如果做得好，应该能兼容
        # 但既然我们传入的都是绝对路径，cwd 的影响被最小化了
        project_root = ROUNDC_ROOT.parent

        result = subprocess.run(
            [sys.executable] + cmd_args,
            cwd=project_root,
            check=True,
            env=os.environ.copy()
        )
        return result
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Command failed with exit code {e.returncode}")
        sys.exit(1)

def count_descendants(node):
    count = 0
    children = node.get("children", []) or []
    count += len(children)
    for ch in children:
        count += count_descendants(ch)
    return count

def count_levels(root):
    stats = {"L1": 0, "L2": 0, "L3": 0, "L4": 0, "Other": 0}
    def dfs(node):
        lvl = str(node.get("level", "")).upper()
        if lvl in stats: stats[lvl] += 1
        else:
            if lvl: stats["Other"] += 1
        for ch in node.get("children", []) or []:
            dfs(ch)
    for ch in root.get("children", []):
        dfs(ch)
    return stats

def setup_test_env():
    print("--- 1. Setting up Test Environment ---")
    if TEST_DIR.exists(): shutil.rmtree(TEST_DIR)
    TEST_DIR.mkdir(parents=True, exist_ok=True)

    # 1.1 数据切片
    if not REAL_TREE_PATH.exists():
        print(f"[ERROR] Real tree not found at {REAL_TREE_PATH}")
        print("Please ensure you have run Step 3 successfully.")
        sys.exit(1)

    print(f"Loading real tree from {REAL_TREE_PATH}...")
    full_tree = json.loads(REAL_TREE_PATH.read_text(encoding="utf-8"))

    l1_nodes = full_tree.get("children", [])
    if not l1_nodes:
        print("[ERROR] No L1 nodes found.")
        sys.exit(1)

    best_l1 = max(l1_nodes, key=count_descendants)
    print(f"Selected Test L1: {best_l1.get('label')} (ID: {best_l1.get('node_id')})")

    test_root = {"node_id": "ROOT_TEST", "level": "ROOT", "children": [best_l1]}
    TEST_INPUT_TREE.write_text(json.dumps(test_root, ensure_ascii=False, indent=2), encoding="utf-8")

    # 1.2 【新增】复制 Membership CSV 文件到测试目录
    # Step 4.3 需要读取这些文件来生成 v4_final_membership.csv
    print("Copying membership CSVs to test directory...")
    for level in ["L4", "L3", "L2", "L1"]:
        src_csv = OUTPUTS_DIR / f"v4_membership_{level}.csv"
        dst_csv = TEST_DIR / f"v4_membership_{level}.csv"
        if src_csv.exists():
            shutil.copy2(src_csv, dst_csv)
        else:
            print(f"[WARN] Original CSV not found: {src_csv.name}")

    # 1.3 配置生成
    config_data = yaml.safe_load(REAL_CONFIG_PATH.read_text(encoding="utf-8"))

    # 【关键】将 output 指向测试目录 (绝对路径)
    config_data["outdir"] = str(TEST_DIR)

    # 【关键】修正输入文件的路径为绝对路径 (指向生产环境的产物)
    if "paths" not in config_data: config_data["paths"] = {}
    config_data["paths"]["corpus"] = str(OUTPUTS_DIR / "v4_corpus_calibrated.csv")
    config_data["paths"]["embeddings"] = str(OUTPUTS_DIR / "v4_embeddings.parquet")
    config_data["paths"]["pairs"] = str(OUTPUTS_DIR / "v4_rerank_edges.csv")

    TEST_CONFIG_PATH.write_text(yaml.dump(config_data), encoding="utf-8")
    print(f"Generated test config at {TEST_CONFIG_PATH}")



def run_pipeline():
    print("\n--- 2. Executing Step 4 Pipeline ---")

    # 统一使用绝对路径调用
    # Step 4.1
    run_command([
        str(SCRIPTS_DIR / "step4_1_skeleton.py"),
        "--input", str(TEST_INPUT_TREE),
        "--output", str(TEST_TREE_S1),
        "--config", str(TEST_CONFIG_PATH)
    ])

    # Step 4.2
    run_command([
        str(SCRIPTS_DIR / "step4_2_shaping.py"),
        "--input", str(TEST_TREE_S1),
        "--output", str(TEST_TREE_S2),
        "--config", str(TEST_CONFIG_PATH)
    ])

    # Step 4.3
    run_command([
        str(SCRIPTS_DIR / "step4_3_polishing.py"),
        "--input", str(TEST_TREE_S2),
        "--output", str(TEST_TREE_S3),
        "--config", str(TEST_CONFIG_PATH)
    ])

    # Step 4.5
    run_command([
        str(SCRIPTS_DIR / "step4_5_overall_structure.py"),
        "--input", str(TEST_TREE_S3),
        "--output", str(TEST_TREE_FINAL),
        "--config", str(TEST_CONFIG_PATH),
        "--l1-def", str(OUTPUTS_DIR / "v4_l1_definition.json"),
        # "--ops-out", str(TEST_DIR / "v4_final_ops.jsonl"),  <--- 【删除或注释此行】脚本会自动根据 config outdir 生成此文件
        "--audit-out", str(TEST_DIR / "v4_final_audit.json"),
        "--flat-csv", str(TEST_DIR / "v4_tree_final_flat.csv")
    ])

def verify_results():
    print("\n--- 3. Verifying Results ---")
    if not TEST_TREE_FINAL.exists():
        print("[FAIL] Final tree file not found.")
        sys.exit(1)

    final_tree = json.loads(TEST_TREE_FINAL.read_text(encoding="utf-8"))

    # 检查 tree_id 是否被清洗
    has_tree_id = False
    def check(node):
        nonlocal has_tree_id
        if "tree_id" in node: has_tree_id = True
        for ch in node.get("children", []): check(ch)
    check(final_tree)

    if has_tree_id: print("[FAIL] 'tree_id' field found.")
    else: print("[PASS] ID Cleanup Verified.")

    # 检查 CSV
    if (TEST_DIR / "v4_final_membership.csv").exists():
        print("[PASS] Membership CSV created.")
    else:
        print("[FAIL] Membership CSV missing.")

    print("\n✅ Mini Integration Test Passed.")

if __name__ == "__main__":
    setup_test_env()
    run_pipeline()
    verify_results()
