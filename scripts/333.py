#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import csv
import os
from pathlib import Path
import pandas as pd

# === 配置 ===
# 这里是你刚才提供的路径
DEFAULT_OUTDIR = Path(r"<LOCAL_SOURCE_ROOT>/data\intermediate_outputs")
ADMIN_MAP_FILE = Path(r"data/source/admin_mapping/roundA_final_overview_scored_selected1120.csv")

CORPUS_FILE = DEFAULT_OUTDIR / "v4_cluster_corpus_cleaned.csv"
MEMBERSHIP_FILE = DEFAULT_OUTDIR / "v4_membership_L4.csv"

def check_file(name, path):
    exists = path.exists()
    status = "✅ 存在" if exists else "❌ 不存在 (请检查路径!)"
    print(f"[{name}] {path}\n   状态: {status}")
    return exists

def debug_main():
    print("=== 1. 文件路径检查 ===")
    f1 = check_file("输出目录", DEFAULT_OUTDIR)
    f2 = check_file("映射表 (RoundA)", ADMIN_MAP_FILE)
    f3 = check_file("语料库 (Corpus)", CORPUS_FILE)
    f4 = check_file("成员表 (L4)", MEMBERSHIP_FILE)
    
    if not (f1 and f2 and f3 and f4):
        print("\n💥 致命错误：有核心文件找不到，程序停止。")
        return

    print("\n=== 2. ID 匹配检查 ===")
    
    # 2.1 检查 Admin Map (RoundA)
    doc_admin_map = {}
    print(f"正在读取映射表: {ADMIN_MAP_FILE.name} ...")
    try:
        with open(ADMIN_MAP_FILE, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames
            print(f"   表头字段: {headers}")
            
            # 检查关键列名
            key_doc = "doc_id" if "doc_id" in headers else None
            key_level = "admin_level" if "admin_level" in headers else None
            
            if not key_doc or not key_level:
                print(f"   ❌ 列名错误！找不到 'doc_id' 或 'admin_level'。请检查表头。")
                return

            sample_rows = []
            for i, row in enumerate(reader):
                if i < 3: sample_rows.append(row[key_doc])
                doc_id = row.get(key_doc, "").strip()
                if doc_id:
                    doc_admin_map[doc_id] = row.get(key_level, "")
            
            print(f"   已加载 {len(doc_admin_map)} 个文档 ID。")
            print(f"   前3个 ID 示例: {sample_rows}")
    except Exception as e:
        print(f"   ❌ 读取失败: {e}")
        return

    # 2.2 检查 Corpus
    print(f"\n正在读取语料库: {CORPUS_FILE.name} ...")
    corpus_doc_ids = set()
    matched_count = 0
    sample_rows = []
    
    try:
        with open(CORPUS_FILE, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                did = row.get("doc_id", "").strip()
                if i < 3: sample_rows.append(did)
                if did:
                    corpus_doc_ids.add(did)
                    if did in doc_admin_map:
                        matched_count += 1
    except Exception as e:
        print(f"   ❌ 读取失败: {e}")
        return

    print(f"   语料库包含 {len(corpus_doc_ids)} 个唯一的 doc_id。")
    print(f"   前3个 ID 示例: {sample_rows}")
    
    print(f"\n=== 3. 匹配结果 ===")
    print(f"   映射成功的行数: {matched_count}")
    
    if matched_count == 0:
        print("   ❌ 失败原因：两个表里的 doc_id 格式对不上！")
        print("   (例如：一个叫 '1'，另一个叫 'doc_1'，或者列名不对)")
    else:
        # 2.3 检查筛选条件
        prov_cnt = sum(1 for did in corpus_doc_ids if did in doc_admin_map and "省级" in doc_admin_map[did])
        city_cnt = sum(1 for did in corpus_doc_ids if did in doc_admin_map and "市级" in doc_admin_map[did])
        print(f"   其中包含 '省级': {prov_cnt}")
        print(f"   其中包含 '市级': {city_cnt}")
        
        if prov_cnt == 0 and city_cnt == 0:
             print("   ❌ 失败原因：匹配成功了，但 admin_level 字段里没有发现 '省级' 或 '市级' 这两个词。")

    print("\n=== 4. 成员表检查 ===")
    # 检查 L4 是否有内容
    try:
        df = pd.read_csv(MEMBERSHIP_FILE, dtype=str)
        print(f"   L4 成员表行数: {len(df)}")
        if len(df) == 0:
            print("   ❌ L4 表是空的！树当然也是空的。")
    except Exception as e:
         print(f"   ❌ L4 读取失败: {e}")

if __name__ == "__main__":
    debug_main()