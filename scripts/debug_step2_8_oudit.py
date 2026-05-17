#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import pandas as pd
from pathlib import Path
import sys

# 假设你的目录结构
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT_DIR_DEFAULT = ROOT / 'outputs'

def main():
    print("--- 开始诊断 Step 2.8 T4 丢失问题 ---")
    
    corpus_path = OUT_DIR_DEFAULT / 'v4_corpus_calibrated.csv'
    if not corpus_path.exists():
        print(f"[Error] 找不到语料文件: {corpus_path}")
        return

    # 1. 读取语料
    print(f"正在读取语料: {corpus_path} ...")
    df = pd.read_csv(corpus_path, dtype=str)
    
    # 2. 检查是否存在 T4
    t4_rows = df[df['calibrated_level'] == 'T4']
    count_t4 = len(t4_rows)
    print(f"语料中 T4 总行数: {count_t4}")
    
    if count_t4 == 0:
        print("[结论] 语料本身没有 T4，所以结果里没有 T4。请检查上一步 (Step 2.7) 的输出。")
        return

    # 3. 挑选一个包含 T4 的典型文档进行追踪
    # 获取第一个含有 T4 的 doc_id
    target_doc_id = t4_rows.iloc[0]['doc_id']
    print(f"\n>>> 锁定目标文档进行追踪: {target_doc_id}")
    
    # 4. 模拟 Step 2.8 的切分逻辑
    g = df[df['doc_id'] == target_doc_id].copy()
    
    def _to_int(x):
        try: return int(str(x).strip())
        except: return 0
        
    g['_blk'] = g['block_idx'].apply(_to_int)
    g = g.sort_values('_blk')
    
    print(f"该文档总行数: {len(g)}")
    print(f"该文档 T4 行数: {len(g[g['calibrated_level'] == 'T4'])}")
    
    idxs = g.index.tolist()
    
    # 模拟 H1 切分 (Step 2.8 Round 1 Logic)
    heads = g[g['final_level'] == 'H1'].index.tolist()
    if not heads:
        # 兜底逻辑
        heads = [idxs[0]] if idxs else []
        print("该文档没有 H1 标签，使用首行作为 H1。")
        
    print(f"\n[Round 1 模拟] 找到 {len(heads)} 个 H1 头")
    
    # 遍历每个 H1 段落
    for k, h in enumerate(heads):
        start_pos = idxs.index(h)
        end_pos = idxs.index(heads[k + 1]) if k + 1 < len(heads) else len(idxs)
        sec_idxs = idxs[start_pos:end_pos]
        
        # 检查这个段落里有没有 T4
        sec_df = df.loc[sec_idxs]
        t4_in_sec = sec_df[sec_df['calibrated_level'] == 'T4']
        
        print(f"  - H1段落[{k+1}]: ID={df.loc[h]['sample_id']}, 包含行数={len(sec_idxs)}")
        print(f"    内部 T4 数量: {len(t4_in_sec)}")
        
        if len(t4_in_sec) > 0:
            print("    -> 发现 T4！追踪它在 Round 2 的命运...")
            
            # 模拟 Round 2 切分 (Step 2.8 Round 2 Logic)
            # 获取当前段的顶级 T
            hrow = df.loc[h]
            sec_top_level = str(hrow.get('calibrated_level') or '')
            
            # 计算下一级
            def tnum(lvl): return int(lvl.replace('T','')) if 'T' in lvl else 99
            next_tn = tnum(sec_top_level) + 1
            next_t = f'T{next_tn}'
            
            print(f"    当前头等级: {sec_top_level}, Round 2 寻找: {next_t}")
            
            if next_tn > 4:
                print("    [结论] 等级太深 (T5+)，脚本逻辑跳过。")
                continue
                
            cheads = sec_df[sec_df['calibrated_level'] == next_t].index.tolist()
            
            if not cheads:
                print("    [Round 2] 没有找到下一级子头。")
                print("    [逻辑路径] -> 进入 Round 3 (Task)")
                print("    [验证] 如果 Round 3 正常工作，这些 T4 应该被 Round 3 的逻辑捕获。")
            else:
                print(f"    [Round 2] 找到 {len(cheads)} 个子头 ({next_t})")
                sub_idxs = sec_df.index.tolist()
                
                # 遍历 Round 2 子段
                found_t4_in_sub = False
                for h_i, ch in enumerate(cheads):
                    s_pos = sub_idxs.index(ch)
                    e_pos = sub_idxs.index(cheads[h_i+1]) if (h_i+1) < len(cheads) else len(sub_idxs)
                    current_sub_sec_idxs = sub_idxs[s_pos:e_pos]
                    
                    # 检查子段里的 T4
                    sub_df_check = df.loc[current_sub_sec_idxs]
                    sub_t4 = sub_df_check[sub_df_check['calibrated_level'] == 'T4']
                    
                    if len(sub_t4) > 0:
                        found_t4_in_sub = True
                        print(f"      -> 子头[{h_i}] (ID={df.loc[ch]['sample_id']}) 下辖范围包含 {len(sub_t4)} 个 T4")
                        print("      [验证] 这些 T4 应该被写入 CSV。")
                        print(f"      [数据快照] T4 样本ID: {sub_t4.iloc[0]['sample_id']}")
                
                if not found_t4_in_sub:
                    # 可能是 T4 位于第一个子头之前？（虽然不合规）
                    print("      [警告] 子头逻辑运行完毕，但在子头的辖区内未发现 T4。可能 T4 位于第一个子头之前的“前言”区？")

    print("\n--- 诊断结束 ---")
    print("请将以上输出发送给我。")

if __name__ == '__main__':
    main()