#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Step 2.8 · 定义 L1 顶层类别（v4）

输入：
- --calibrated data/intermediate_outputs/v4_corpus_calibrated.csv（含 calibrated_level 与 cleaned_title/path_text/doc_id）
- --assets-dir assets/l1_samples/（若存在若干 csv，列至少包含 title/cleaned_title）

处理：
- 聚合 calibrated_level==T1 的标题作为候选母集（抽样上限）
- 读取 assets 的引导样本（若存在）
- 调用 LLM（默认开启，PRIMARY_LLM_*）按 prompts/step2_8_define_l1.md 生成 categories（name/keywords/definition）
- 程序侧为每个 name 计算稳定 id=L1_N{md5(name)[:8]}

输出：
- data/intermediate_outputs/v4_l1_definition.json -> {"categories":[{"id","name","keywords","definition"},…]}
- 日志：data/intermediate_outputs/logs/step2_8_define_l1.log

python scripts/step2_8_define_l1.py `
--env configs/roundC_v4.env `
--calibrated data/intermediate_outputs/v4_corpus_calibrated.csv `
--assets-dir assets/l1_samples `
--outdir data/intermediate_outputs `
--max-t1 80
"""

import argparse
import csv
import hashlib
import json
import os
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
    return hashlib.md5((s or '').encode('utf-8')).hexdigest()[:8]


def read_assets_titles(assets_dir: Path, limit_per_file: int = 50) -> Dict[str, List[str]]:
    """读取已评级样本。支持两种形式：
    1) 仅含 cleaned_title/title 列 → 以文件名为组名
    2) 含 l1/label/category/L1 任一列 → 按该列分组展示
    返回：group_name -> [titles]
    """
    out: Dict[str, List[str]] = {}
    if not assets_dir.exists():
        return out
    for p in assets_dir.glob('*.csv'):
        rows: List[Tuple[str, str]] = []  # (group, title)
        try:
            with p.open('r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                # 识别分组列
                cols = [c.lower() for c in (reader.fieldnames or [])]
                group_col = None
                for cand in ['l1','label','category','L1','Label','Category']:
                    if cand.lower() in cols:
                        group_col = cand
                        break
                for i, r in enumerate(reader):
                    title = r.get('cleaned_title') or r.get('title') or ''
                    title = str(title or '').strip()
                    if title:
                        if group_col:
                            gname = str(r.get(group_col) or '').strip() or p.stem
                        else:
                            gname = p.stem
                        rows.append((gname, title))
            if rows:
                # 限制每组最多 limit_per_file 条
                by_group: Dict[str, List[str]] = {}
                for gname, title in rows:
                    arr = by_group.setdefault(gname, [])
                    if len(arr) < limit_per_file:
                        arr.append(title)
                for gname, arr in by_group.items():
                    out[gname] = arr
        except Exception:
            continue
    return out


def main():
    ap = argparse.ArgumentParser(description='Round C v4 · Step 2.8 Define L1')
    ap.add_argument('--calibrated', type=str, default=str(OUT_DIR_DEFAULT / 'v4_corpus_calibrated.csv'))
    ap.add_argument('--assets-dir', type=str, default=str(ROOT / 'assets' / 'l1_samples'))
    ap.add_argument('--env', type=str, default=str(ROOT / 'configs' / 'roundC_v4.env'))
    ap.add_argument('--outdir', type=str, default=str(OUT_DIR_DEFAULT))
    ap.add_argument('--model', type=str, default=None)
    ap.add_argument('--version', type=str, default=None, help='版本号（未提供则使用YYYYMMDDHHMM）')
    ap.add_argument('--max-t1', type=int, default=80)
    args = ap.parse_args()

    load_env_file(args.env)
    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR_DEFAULT / 'step2_8_define_l1.log'

    # 注入 OPENAI_* 以兼容 v3 封装
    model = args.model or os.getenv('PRIMARY_LLM_MODEL') or os.getenv('PRIMARY_LLM')
    base = os.getenv('PRIMARY_LLM_BASE_URL')
    key = os.getenv('PRIMARY_LLM_API_KEY')
    if base:
        os.environ['OPENAI_BASE_URL'] = base
    if key:
        os.environ['OPENAI_API_KEY'] = key

    # 读取 calibrated 语料，收集 T1 标题
    cal_path = Path(args.calibrated)
    if not cal_path.exists():
        raise FileNotFoundError(f'calibrated not found: {cal_path}')
    df = pd.read_csv(cal_path, dtype=str)
    t1 = df[df['calibrated_level'].astype(str) == 'T1']
    candidates = t1['cleaned_title'].fillna('').astype(str).drop_duplicates().tolist()[: args.max_t1]

    # 读取 assets 引导样本
    assets = read_assets_titles(Path(args.assets_dir))

    # 组织 prompt
    sys_p = (ROOT / 'prompts' / 'step2_8_define_l1.md').read_text(encoding='utf-8')
    user_lines: List[str] = []
    user_lines.append('# 候选母集（来自 Step 2.5 的 T1 标题，节选）')
    for i, t in enumerate(candidates, start=1):
        user_lines.append(f'{i}. {t}')
    if assets:
        user_lines.append('\n# 引导样本（assets/l1_samples/*，含已评级分组）')
        for name, arr in assets.items():
            user_lines.append(f'【{name}】示例（≤{len(arr)}）:')
            for j, s in enumerate(arr[:20], start=1):
                user_lines.append(f'  - {s}')
    user_lines.append('\n# 设计约束')
    user_lines.append('- 目标类别数：6–12；互斥、可执行；名称≤10字；keywords 3–8 个；definition 1–2 句。')
    user_text = '\n'.join(user_lines)

    log_write(log_path, f'[INFO] calling LLM model={model} t1={len(candidates)} seed_groups={len(assets)}')
    res = call_json(
        model=model,
        system_text=sys_p,
        user_text=user_text,
        temperature=float(os.getenv('LLM_TEMPERATURE', '0.2')),
        max_tokens=int(os.getenv('LLM_MAX_TOKENS', '2000')),
        response_format='json_object',
        timeout=float(os.getenv('TIMEOUT', '120')),
        retries=int(os.getenv('RETRIES', '2')),
        backoff=float(os.getenv('BACKOFF', '1.5')),
    )
    obj = res.get('json') or {}
    cats_in = obj.get('categories') or []
    cats_out: List[Dict[str, object]] = []
    seen = set()
    for c in cats_in:
        name = str((c.get('name') or '')).strip()
        if not name or name in seen:
            continue
        seen.add(name)
        cid = f"L1_N{md5_8(name)}"
        kws = c.get('keywords') or []
        if isinstance(kws, list):
            kws = [str(x).strip() for x in kws if str(x).strip()]
        else:
            kws = [str(kws).strip()] if str(kws).strip() else []
        defi = str(c.get('definition') or '').strip()
        cats_out.append({'id': cid, 'name': name, 'keywords': kws, 'definition': defi})

    # 版本号
    version = args.version
    if not version:
        from datetime import datetime
        version = datetime.now().strftime('%Y%m%d%H%M')

    out_path = Path(args.outdir) / 'v4_l1_definition.json'
    out_obj = {
        'version': str(version),
        'categories': cats_out,
    }
    out_path.write_text(json.dumps(out_obj, ensure_ascii=False, indent=2), encoding='utf-8')
    log_write(log_path, f'[WRITE] {out_path} cats={len(cats_out)}')
    print(f'[WRITE] {out_path}')


if __name__ == '__main__':
    main()
