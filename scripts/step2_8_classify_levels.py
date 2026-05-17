#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Step 2.8 · 文档级分类至固定 L1（三轮判定 · v4 · 动态新建ID版 · Hash修正版）

修改重点：
1. [ID生成] 废弃数字递增逻辑，统一使用 Hash ID (L1_N...)。自动识别 create_new 并通过名称生成稳定 ID。
2. [物理跟随] 严格遵循父节点（Head）的物理范围，子节点无条件继承。
3. [数据完整] Round 3 产生的新 ID 和所有子节点都会写入 node_assignments.csv，确保 Step 3 可读。
"""

import argparse
import json
import os
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from common_llm import call_json
from utils.llm_client import load_env_file

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT_DIR_DEFAULT = ROOT / 'outputs'
LOG_DIR_DEFAULT = OUT_DIR_DEFAULT / 'logs'


def log_write(p: Path, msg: str):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('a', encoding='utf-8') as f:
        f.write(msg.rstrip() + '\n')


def md5_8(s: str) -> str:
    """生成与 step2_8_define_l1.py 一致的 8位 Hash ID"""
    return hashlib.md5((s or '').encode('utf-8')).hexdigest()[:8]


def main():
    ap = argparse.ArgumentParser(description='Round C v4 · Step 2.8 classify documents to L1 (3-round)')
    ap.add_argument('--l1-def', type=str, default=str(OUT_DIR_DEFAULT / 'v4_l1_definition.json'))
    ap.add_argument('--corpus', type=str, default=str(OUT_DIR_DEFAULT / 'v4_corpus_calibrated.csv'))
    ap.add_argument('--env', type=str, default=str(ROOT / 'configs' / 'roundC_v4.env'))
    ap.add_argument('--outdir', type=str, default=str(OUT_DIR_DEFAULT))
    ap.add_argument('--limit', type=int, default=0, help='仅处理前 N 行，0 表示全部')
    args = ap.parse_args()

    load_env_file(args.env)
    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR_DEFAULT / 'step2_8_classify_l1.log'

    # 注入 OPENAI_* 以兼容 v3 封装
    model = os.getenv('PRIMARY_LLM_MODEL') or os.getenv('PRIMARY_LLM')
    base = os.getenv('PRIMARY_LLM_BASE_URL')
    key = os.getenv('PRIMARY_LLM_API_KEY')
    if base:
        os.environ['OPENAI_BASE_URL'] = base
    if key:
        os.environ['OPENAI_API_KEY'] = key

    # --- 1. 读取 L1 定义 ---
    l1_obj = json.loads(Path(args.l1_def).read_text(encoding='utf-8'))
    l1_list: List[Dict[str, object]] = list(l1_obj.get('categories') or [])
    
    # 记录本次运行中生成的 {新名称: 新ID} 映射，防止同名新建多次分配不同 ID
    created_l1_map: Dict[str, str] = {}

    l1_json = json.dumps([
        {
            'id': c.get('id'),
            'name': c.get('name'),
            'keywords': c.get('keywords'),
            'definition': c.get('definition'),
        }
        for c in l1_list
    ], ensure_ascii=False)

    # 读取语料
    df = pd.read_csv(args.corpus, dtype=str)
    required = ['sample_id', 'doc_id', 'block_idx', 'calibrated_level', 'cleaned_title']
    for c in required:
        if c not in df.columns:
            raise ValueError(f'missing column in corpus: {c}')
    if args.limit and args.limit > 0:
        df = df.head(args.limit).copy()

    def _to_int(x: str) -> int:
        try:
            return int(str(x).strip())
        except Exception:
            return 0

    df['_blk'] = df['block_idx'].apply(_to_int)
    df.sort_values(['doc_id', '_blk'], inplace=True)

    # 工具：调用 LLM 判定某标题的 L1
    sys_prompt = (ROOT / 'prompts' / 'step2_8_classify_l1.md').read_text(encoding='utf-8')

    def judge_l1(title: str) -> Tuple[str | None, float, Dict]:
        user_text = '标题: ' + (title or '') + '\n\n# L1 列表（JSON）\n' + l1_json
        res = call_json(
            model=model,
            system_text=sys_prompt,
            user_text=user_text,
            temperature=float(os.getenv('LLM_TEMPERATURE', '0.2')),
            max_tokens=int(os.getenv('LLM_MAX_TOKENS', '800')),
            response_format='json_object',
            timeout=float(os.getenv('TIMEOUT', '120')),
            retries=int(os.getenv('RETRIES', '2')),
            backoff=float(os.getenv('BACKOFF', '1.5')),
        )
        obj = res.get('json') or {}
        return obj.get('best_l1_id'), float(obj.get('confidence') or 0.0), obj

    # 结果收集
    review_rows: List[Dict[str, object]] = []
    assign_rows: List[Dict[str, object]] = []
    node_assign_rows: List[Dict[str, object]] = []

    def tnum(level: str) -> int:
        try:
            return int(str(level).replace('T', '').strip())
        except Exception:
            return 99

    # 逐文档三轮
    for doc_id, g in df.groupby('doc_id', sort=False):
        g = g.sort_values('_blk')
        
        # [结构判定 H1]
        heads = g[g['final_level'] == 'H1'].index.tolist()
        if not heads:
            if g.index.tolist():
                heads = [g.index.tolist()[0]]
            else:
                continue 

        idxs = g.index.tolist()
        sections: List[List[int]] = []
        for k, h in enumerate(heads):
            start_pos = idxs.index(h)
            end_pos = idxs.index(heads[k + 1]) if k + 1 < len(heads) else len(idxs)
            sections.append(idxs[start_pos:end_pos])

        unresolved: List[List[int]] = []
        
        # --- Round 1: 顶层判定 ---
        for sec in sections:
            hrow = df.loc[sec[0]]
            sec_top_level = str(hrow.get('calibrated_level') or '')
            
            best, conf, obj = judge_l1(str(hrow.get('cleaned_title') or ''))
            
            if best:
                if sec_top_level == 'T1':
                    review_rows.append({
                        'sample_id': hrow.get('sample_id', ''),
                        'suggested_l1': 'repeat',
                        'confidence': float(f"{conf:.4f}"),
                        'human_decision': 'repeat',
                        'notes': 'Round1: T1 命中 L1，标记 repeat',
                    })
                    # 子节点继承 best
                    for idx in sec[1:]:
                        row = df.loc[idx]
                        node_assign_rows.append({
                            'doc_id': row.get('doc_id',''),
                            'sample_id': row.get('sample_id',''),
                            'assigned_l1_id': best,
                            'round': 1,
                            'source_head': hrow.get('sample_id',''),
                            'head_level': sec_top_level,
                            'confidence': float(f"{conf:.4f}"),
                            'notes': '',
                        })
                else:
                    assign_rows.append({
                        'doc_id': doc_id,
                        'sample_id': hrow.get('sample_id', ''),
                        'head_level': sec_top_level,
                        'assigned_l1_id': best,
                        'round': 1,
                        'confidence': float(f"{conf:.4f}"),
                        'notes': '',
                    })
                    for idx in sec:
                        row = df.loc[idx]
                        node_assign_rows.append({
                            'doc_id': row.get('doc_id',''),
                            'sample_id': row.get('sample_id',''),
                            'assigned_l1_id': best,
                            'round': 1,
                            'source_head': hrow.get('sample_id',''),
                            'head_level': sec_top_level,
                            'confidence': float(f"{conf:.4f}"),
                            'notes': '',
                        })
            else:
                unresolved.append(sec)

        # --- Round 2 & 3 准备 ---
        # 存储 (Head_Index, List_of_All_Children_Indices)
        round3_tasks: List[Tuple[int, List[int]]] = []

        for sec in unresolved:
            hrow = df.loc[sec[0]] 
            sec_top_level = str(hrow.get('calibrated_level') or '')
            next_tn = tnum(sec_top_level) + 1
            next_t = f'T{next_tn}' if next_tn <= 4 else None

            if not next_t:
                round3_tasks.append((sec[0], sec))
                continue
            
            sub = df.loc[sec]
            cheads = sub[sub['calibrated_level'] == next_t].index.tolist()
            
            if not cheads:
                round3_tasks.append((sec[0], sec))
                continue
            
            sub_idxs = sub.index.tolist()
            for h_i, h in enumerate(cheads):
                # [H范围] 物理锁定
                start_pos = sub_idxs.index(h)
                end_pos = sub_idxs.index(cheads[h_i+1]) if (h_i+1) < len(cheads) else len(sub_idxs)
                current_sub_sec_idxs = sub_idxs[start_pos:end_pos]

                crow = df.loc[h]
                best, conf, obj = judge_l1(str(crow.get('cleaned_title') or ''))
                
                if best:
                    assign_rows.append({
                        'doc_id': doc_id,
                        'sample_id': crow.get('sample_id', ''),
                        'head_level': str(next_t),
                        'assigned_l1_id': best,
                        'round': 2,
                        'confidence': float(f"{conf:.4f}"),
                        'notes': '',
                    })
                    for idx in current_sub_sec_idxs:
                        row = df.loc[idx]
                        node_assign_rows.append({
                            'doc_id': row.get('doc_id',''),
                            'sample_id': row.get('sample_id',''),
                            'assigned_l1_id': best,
                            'round': 2,
                            'source_head': crow.get('sample_id',''),
                            'head_level': str(next_t),
                            'confidence': float(f"{conf:.4f}"),
                            'notes': '',
                        })
                else:
                    round3_tasks.append((h, current_sub_sec_idxs))

        # --- Round 3: 复核/新建 (使用 Hash ID)/丢弃 ---
        round3_tasks.sort(key=lambda x: x[0])

        for h_idx, section_idxs in round3_tasks:
            row = df.loc[h_idx]
            sec_top_level = str(row.get('calibrated_level') or '')
            
            best, conf, obj = judge_l1(str(row.get('cleaned_title') or ''))
            
            create_new = bool((obj or {}).get('create_new_l1'))
            discard = bool((obj or {}).get('discard'))
            notes = str((obj or {}).get('not_match_reason') or '')
            
            # 决策逻辑变量
            final_decision = ""        # review.csv 中的 suggested_l1
            assigned_id_for_csv = ""   # node_assignments.csv 中的 assigned_l1_id
            
            # [逻辑分支 1: 新建]
            if create_new:
                new_name = str((obj or {}).get('new_l1_name') or '').strip()
                new_kws = str((obj or {}).get('new_l1_keywords') or '')
                
                if not new_name:
                    # 如果 LLM 没给名字，退化为 best 或 Unknown
                    new_name = "Unknown_New_L1"

                # 检查是否已经为这个新名字分配过 ID
                if new_name in created_l1_map:
                    new_id = created_l1_map[new_name]
                else:
                    # 【核心修改】使用 Hash 生成，确保稳定且格式统一 (L1_N{hash})
                    new_id = f"L1_N{md5_8(new_name)}"
                    created_l1_map[new_name] = new_id
                
                # 设置输出
                final_decision = new_id # Review 表直接显示新 ID
                assigned_id_for_csv = new_id
                notes = (notes + f"; New L1: {new_name} (ID={new_id})").strip('; ')
                
                if sec_top_level == 'T1':
                    notes += " (T1 head)"

            # [逻辑分支 2: 丢弃]
            elif discard:
                final_decision = 'discard'
                assigned_id_for_csv = '' # 丢弃的不写进 node_assignments

            # [逻辑分支 3: 低置信度匹配 / 兜底]
            else:
                if best:
                    final_decision = best
                    assigned_id_for_csv = best
                    notes += " (Low confidence match)"
                else:
                    # 实在没辙，当做 Create New Unknown 处理
                    new_name = "Unknown_Cluster"
                    if new_name in created_l1_map:
                        new_id = created_l1_map[new_name]
                    else:
                        new_id = f"L1_N{md5_8(new_name)}"
                        created_l1_map[new_name] = new_id
                    
                    final_decision = new_id
                    assigned_id_for_csv = new_id
                    notes += f"; Auto-created Unknown (ID={new_id})"

            # 1. 写入复核表 (Review CSV)
            review_rows.append({
                'sample_id': row.get('sample_id', ''),
                'suggested_l1': final_decision, # 这里现在是 ID 或 'discard'
                'confidence': float(f"{conf:.4f}"),
                'human_decision': final_decision, # 默认填好新 ID
                'notes': notes,
            })

            # 2. 写入 Assignment 表 (Node CSV)
            if final_decision != 'discard' and assigned_id_for_csv:
                
                # 记录头 (Doc Assignment)
                assign_rows.append({
                    'doc_id': row.get('doc_id',''),
                    'sample_id': row.get('sample_id', ''),
                    'head_level': sec_top_level,
                    'assigned_l1_id': assigned_id_for_csv,
                    'round': 3,
                    'confidence': float(f"{conf:.4f}"),
                    'notes': f"Review: {notes}",
                })
                
                # 记录物理范围内的所有子节点
                for idx in section_idxs:
                    r = df.loc[idx]
                    node_assign_rows.append({
                        'doc_id': r.get('doc_id',''),
                        'sample_id': r.get('sample_id',''),
                        'assigned_l1_id': assigned_id_for_csv, # 使用新生成的 ID
                        'round': 3,
                        'source_head': row.get('sample_id',''),
                        'head_level': sec_top_level,
                        'confidence': float(f"{conf:.4f}"),
                        'notes': f"Review: {notes}",
                    })

    # 写出
    out_review = out_dir / 'v4_l1_classification_review.csv'
    pd.DataFrame(review_rows).to_csv(out_review, index=False, encoding='utf-8-sig')
    log_write(log_path, f"[WRITE] {out_review} rows={len(review_rows)}")
    
    out_assign = out_dir / 'v4_l1_doc_assignments.csv'
    pd.DataFrame(assign_rows).to_csv(out_assign, index=False, encoding='utf-8-sig')
    log_write(log_path, f"[WRITE] {out_assign} rows={len(assign_rows)}")
    
    out_nodes = out_dir / 'v4_l1_node_assignments.csv'
    pd.DataFrame(node_assign_rows).to_csv(out_nodes, index=False, encoding='utf-8-sig')
    log_write(log_path, f"[WRITE] {out_nodes} rows={len(node_assign_rows)}")
    
    print(f'[WRITE] {out_review}')
    print(f'[WRITE] {out_assign}')
    print(f'[WRITE] {out_nodes}')


if __name__ == '__main__':
    main()
