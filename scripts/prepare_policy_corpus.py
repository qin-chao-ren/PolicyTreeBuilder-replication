#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Step 1 · 标题清洗（public 迁移实现，贴近 v2 行为）

改动点：
- 默认开启 LLM 清洗（--llm-clean yes），prompt 改为 prompts/clean_policy_titles.md；
- LLM 清洗统一通过 scripts/llm_runtime.py 的 profile 配置调用；
- 输出对应：主输出写入 data/intermediate_outputs/policy_corpus_cleaned.csv；同时保留 v2 风格的 policy_corpus_cleaned.csv 作为兼容（可选）。

python scripts/prepare_policy_corpus.py `
--source data/source/policy_action_segments.csv `
--env configs/.env `
--outdir data/intermediate_outputs `
--llm-clean yes `
--debug-dump-llm no
"""

import argparse
import os
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import pandas as pd

from llm_runtime import call_llm_json, load_env_file, profiles_from_config


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT_DIR_DEFAULT = ROOT / 'data' / 'intermediate_outputs'
LOG_DIR_DEFAULT = OUT_DIR_DEFAULT / 'logs'


def log_write(log_path: Path, msg: str):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open('a', encoding='utf-8') as f:
        f.write(msg.rstrip() + '\n')


STOP_PATTERNS = [
    '重要举措', '若干举措', '若干措施', '工作方案', '工作举措', '主要任务', '主要举措', '方案', '工作任务', '通知', '意见', '工作要点',
    '组织实施', '有关事项', '实施细则', '实施意见', '实施方案', '重点任务', '工作要求', '总体要求', '基本原则', '保障措施',
]


def simple_clean(title: str) -> str:
    s = (title or '').strip()
    for pat in STOP_PATTERNS:
        if s.startswith(pat):
            s = s[len(pat):].lstrip('：: ，, ')
    # 去除尾部括注
    for tail in ['（试行）', '(试行)', '（通知）', '(通知)']:
        if s.endswith(tail):
            s = s[:-len(tail)].rstrip('：: ')
    return s or (title or '')


def llm_clean_titles(profile: str, title: str, path_text: str, prompt_tmpl: str,
                     model_override: str | None = None,
                     timeout_s: int = 60, max_retries: int = 3, max_new: int = 3) -> Tuple[List[str], str, int, str]:
    """调用 LLM 返回精简标题列表。"""
    system_prompt = "You are an expert Chinese policy editor who outputs JSON."
    user_prompt = (prompt_tmpl or '').replace('{TITLE}', title).replace('{PATH_TEXT}', path_text or '').replace('{MAX_NEW}', str(max_new))
    res = call_llm_json(
        profile=profile,
        model_override=model_override,
        system=system_prompt,
        user=user_prompt,
        task='prepare_policy_corpus_clean_titles',
        timeout_s=timeout_s,
        retries=max_retries,
    )
    raw = str(res.get('raw') or '')
    obj = res.get('json') or {}
    if res.get('ok') and isinstance(obj, dict):
        arr = obj.get('clean_titles') or obj.get('titles') or []
        uniq, seen = [], set()
        for s in arr:
            if isinstance(s, str):
                s = s.strip()
                if s and s not in seen:
                    seen.add(s)
                    uniq.append(s)
        return uniq[:max_new], 'ok', int(res.get('attempts') or 0), raw
    err = str(res.get('error') or '').lower()
    status = 'timeout' if 'timeout' in err else 'error'
    return [], status, int(res.get('attempts') or 0), raw

import re

def strQ2B(ustring: str) -> str:
    res = []
    for uchar in ustring:
        inside_code = ord(uchar)
        if inside_code == 0x3000:
            inside_code = 32
        elif 0xFF01 <= inside_code <= 0xFF5E:
            inside_code -= 0xFEE0
        res.append(chr(inside_code))
    return ''.join(res)


STOP_SLOGAN_PREFIX = [
    '重点任务', '主要任务', '工作举措', '组织实施', '总体要求', '发展目标',
    '总体思路', '基本原则', '总体部署', '工作安排', '工作目标', '总体目标与任务',
]


def normalize_spaces(s: str) -> str:
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def strip_numbering_prefix(s: str) -> str:
    t = s
    patterns = [
        r"^[\(（]?[一二三四五六七八九十百千]+[\)）][、\s]*",
        r"^[一二三四五六七八九十百千]+、",
        r"^\d+[\.、\)]\s*",
        r"^[\(（]\d+[\)）]",
        r"^第[一二三四五六七八九十百千零〇]+[章节条]\s*",
    ]
    for p in patterns:
        t = re.sub(p, "", t)
    return t.strip()


def strip_suffixes(s: str) -> str:
    t = s
    suffixes = ['（试行）', '(试行)', '（暂行）', '(暂行)', '实施方案', '工作方案', '若干措施']
    for suf in suffixes:
        if t.endswith(suf):
            t = t[: -len(suf)]
    return t.strip()


def remove_slogan_prefix(s: str) -> Tuple[str, Optional[str]]:
    t = s
    for pref in STOP_SLOGAN_PREFIX:
        if t == pref:
            return t, 'slogan_only'
        if t.startswith(pref):
            t = t[len(pref):].lstrip(' ：:')
            break
    return t, None


def verb_normalize(s: str) -> str:
    VERB_NORMALIZE = {'促进': '推进', '推动': '推进'}
    t = s
    for k, v in VERB_NORMALIZE.items():
        t = re.sub(re.escape(k), v, t)
    return t


def keep_zh_alnum(s: str) -> str:
    # keep Chinese, letters, digits (去标点/空格)
    return re.sub(r"[^\u4e00-\u9fa5A-Za-z0-9]", "", s or "")


def post_sanitize(title: str, max_len: int) -> str:
    """
    智能截断：优先在语义边界处截断，保留完整的动宾结构。
    """
    s = keep_zh_alnum(title).strip()

    # 如果本身就不超长，直接返回
    if len(s) <= max_len:
        return s

    # 定义语义边界词（优先在这些位置截断）
    boundary_markers = [
        ('的', 1),      # "推进XX的YY" → 保留"推进XX的"
        ('与', 0),      # "XX与YY" → 保留"XX"
        ('和', 0),      # "XX和YY" → 保留"XX"
        ('及', 0),      # "XX及YY" → 保留"XX"
        ('或', 0),      # "XX或YY" → 保留"XX"
        ('、', 0),      # "XX、YY" → 保留"XX"
        ('等', 1),      # "XX等YY" → 保留"XX等"
        ('相关', 2),    # "XX相关YY" → 保留"XX相关"
    ]

    # 尝试在语义边界处截断
    for marker, offset in boundary_markers:
        search_text = s[:max_len + 5]
        last_pos = search_text.rfind(marker)

        if last_pos > 0:
            cut_pos = last_pos + offset
            candidate = s[:cut_pos].strip()

            if 8 <= len(candidate) <= max_len:
                has_verb = any(v in candidate for v in ['推进', '加强', '完善', '优化', '建设', '提升', '发展', '实施', '支持', '鼓励', '促进', '构建', '打造', '开展'])
                if has_verb:
                    return candidate

    # 保留动宾结构
    common_verbs = ['推进', '加强', '完善', '优化', '建设', '提升', '发展', '实施', '支持', '鼓励', '促进', '构建', '打造', '开展', '建立', '健全', '落实', '深化', '新设立', '新增', '新建']

    for verb in common_verbs:
        verb_pos = s.find(verb)
        if verb_pos >= 0:
            candidate = s[verb_pos:verb_pos + max_len]
            if len(candidate) >= 8:
                return candidate

    # 兜底
    min_len = min(12, max_len)
    if len(s) < min_len:
        return s
    return s[:max_len]


def build_cleaned_title_rule(raw_title: str) -> Tuple[str, Optional[str]]:
    if not raw_title:
        return '', 'empty'
    t = raw_title.strip()
    t = strQ2B(t)
    t = normalize_spaces(t)
    t = strip_numbering_prefix(t)
    t, skip = remove_slogan_prefix(t)
    if skip:
        return t, skip
    t = strip_suffixes(t)
    t = verb_normalize(t)
    t = normalize_spaces(t)
    core = re.sub(r"[\d\s\W_]+", "", keep_zh_alnum(t))
    if len(core) < 2:
        return t, 'too_short'
    return t, None

def detect_iscluster_col(df: pd.DataFrame) -> Optional[str]:
    for c in df.columns:
        if c.strip().lower() == 'to_cluster':
            return c
    if '是否聚类' in df.columns:
        return '是否聚类'
    for c in df.columns:
        if '聚类' in c:
            return c
    return None


def to_bool_cluster(v: str) -> bool:
    s = str(v or '').strip().lower()
    return s in {'1','y','yes','true','是','對','对'}

def main():
    ap = argparse.ArgumentParser(description='PolicyTreeBuilder final replication · Step 1 Prepare Corpus')
    ap.add_argument('--source', required=True, help='输入CSV，至少包含 sample_id 与 title 或 cleaned_title 字段')
    ap.add_argument('--env', default=str(ROOT / 'configs' / '.env'))
    ap.add_argument('--outdir', default=str(OUT_DIR_DEFAULT))
    ap.add_argument('--title-col', default='cleaned_title')
    ap.add_argument('--raw-title-col', default='title')
    ap.add_argument('--llm-clean', choices=['yes','no'], default='yes')
    ap.add_argument('--llm-clean-model', default=None)
    ap.add_argument('--llm-clean-max-new-titles', type=int, default=3)
    ap.add_argument('--llm-clean-prompt', default=str(ROOT / 'prompts' / 'clean_policy_titles.md'))
    ap.add_argument('--llm-timeout', type=int, default=None)
    ap.add_argument('--llm-retries', type=int, default=None)
    ap.add_argument('--llm-defer-on-fail', choices=['yes','no'], default='yes')
    ap.add_argument('--debug-dump-llm', choices=['yes','no'], default='no')
    ap.add_argument('--title-max-len', type=int, default=15, help='Sanitized title max length')
    ap.add_argument('--skip-keywords', default='附则,附录,名词解释,前言,序言,执行日期,附件清单', help='V2: Keywords to skip for path/cleaning')
    args = ap.parse_args()

    load_env_file(args.env)

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR_DEFAULT / 'step1_prepare_corpus.log'

    df = pd.read_csv(args.source, dtype=str)

    # 1. V2 - 增加 iscol (to_cluster) 过滤
    required = ['doc_id','block_idx','block_text','final_level']
    for c in required:
        if c not in df.columns:
            raise ValueError(f'missing required column: {c}')

    iscol = detect_iscluster_col(df) # 依赖步骤一的全局函数
    if iscol:
        log_write(log_path, f"[INFO] Detected 'to_cluster' column: {iscol}")
        mask_cluster = df[iscol].apply(to_bool_cluster) # 依赖步骤一的全局函数
    else:
        log_write(log_path, f"[WARN] 'to_cluster' column not found, processing all rows.")
        mask_cluster = pd.Series([True] * len(df))

    mask_level = df['final_level'].str.match(r'^H[1-4]$', na=False)
    dfh = df[mask_cluster & mask_level].copy()
    if dfh.empty:
        log_write(log_path, f"[WARN] DataFrame is empty after H1-H4 and 'to_cluster' filtering.")

    # 2. V2 - 增加 _should_skip 逻辑
    skip_kw = [x.strip() for x in args.skip_keywords.split(',') if x.strip()]

    def _should_skip(title: str) -> bool:
        t = (title or '').strip()
        if not t:
            return True
        t = strip_numbering_prefix(t) # 依赖你上次粘贴的全局函数
        for kw in skip_kw:
            if t == kw or t.startswith(kw):
                return True
        for pref in STOP_SLOGAN_PREFIX: # 依赖你上次粘贴的全局函数
            if t == pref or t.startswith(pref):
                return True
        return False

    dfh['block_idx_num'] = dfh['block_idx'].astype(str).str.extract(r'(\d+)').fillna('0').astype(int)
    dfh.sort_values(['doc_id','block_idx_num'], inplace=True)

    # 为 V2 的 build_path_text_filtered 准备 _should_skip 标记
    dfh['_should_skip'] = dfh['block_text'].apply(_should_skip)

    rows = dfh.to_dict('records')

    # 3. V2 - 替换 build_path_text_filtered 实现
    def build_path_text_filtered(rows: List[Dict[str,str]]) -> List[str]:
        import re as _re
        stack: Dict[int,str] = {}
        current_doc_id: Optional[str] = None
        out: List[str] = []
        for r in rows:
            doc_id = str(r.get('doc_id') or '').strip()
            if doc_id != current_doc_id:
                stack = {}
                current_doc_id = doc_id

            lvl = str(r.get('final_level') or '').strip()
            title = str(r.get('block_text') or '').strip()
            should_skip = r.get('_should_skip', False) # V2 ????

            m = _re.match(r'H(\d)', lvl)
            if not m: # ?? dfh ????, ?????
                out.append('')
                continue

            d = int(m.group(1))
            parents = [stack[i] for i in range(1, d) if i in stack and stack[i]]
            path = ' > '.join(parents)
            out.append(path)

            # ??: ??????????
            if not should_skip:
                stack[d] = title
            else:
                stack[d] = ''

            for j in list(stack.keys()):
                if j > d:
                    stack.pop(j, None)
        return out

    paths = build_path_text_filtered(rows)

    base_rows = []
    for r, path in zip(rows, paths):
        base_rows.append({
            'doc_id': r['doc_id'],
            'block_idx': r['block_idx'],
            'final_level': r['final_level'],
            'raw_title': str(r['block_text'] or '').strip(),
            'path_text': path,
        })
    base_df = pd.DataFrame(base_rows)
    base_df['sample_id'] = base_df.apply(lambda rr: f"{rr['doc_id']}_{str(rr['block_idx']).zfill(5)}", axis=1)

    # LLM 清洗（默认开启）
    cleaned_rows = []
    llm_on = (args.llm_clean == 'yes')
    llm_profile, _ = profiles_from_config({})
    llm_model_override = args.llm_clean_model
    llm_timeout = int(args.llm_timeout or 60)
    llm_retries = int(args.llm_retries or 3)
    prompt_tmpl = Path(args.llm_clean_prompt).read_text(encoding='utf-8') if Path(args.llm_clean_prompt).exists() else ''

    total = len(base_df)
    pending: List[Dict[str,str]] = []
    raw_dump_path = out_dir / 'policy_title_cleaning_llm_raw.jsonl'
    raw_fp = open(raw_dump_path, 'a', encoding='utf-8') if (llm_on and args.debug_dump_llm == 'yes') else None

    for idx, r in enumerate(base_df.itertuples(index=False), start=1):
        if idx == 1 or idx % 10 == 0 or idx == total:
            log_write(log_path, f"[INFO] Processing {idx}/{total} {r.sample_id}")
        base_row = {
            'sample_id': '',
            'source_sample_id': r.sample_id,
            'doc_id': r.doc_id,
            'block_idx': r.block_idx,
            'final_level': r.final_level,
            'raw_title': r.raw_title,
            'path_text': r.path_text,
            'cleaned_title': '',
            'skip_reason': '',
            'clean_source': '',
            'variant_idx': 0,
            'llm_status': 'off',
            'llm_attempts': 0,
        }
        titles: List[str] = []
        llm_status = 'off'
        llm_attempts = 0
        llm_raw_text = ''
        clean_source = 'rule'
        if llm_on:
            try:
                titles, llm_status, llm_attempts, llm_raw_text = llm_clean_titles(
                    llm_profile, r.raw_title, r.path_text, prompt_tmpl,
                    model_override=llm_model_override,
                    timeout_s=llm_timeout, max_retries=max(1, llm_retries), max_new=int(args.llm_clean_max_new_titles)
                )
                if llm_status == 'ok' and titles:
                    clean_source = 'llm'
            except Exception as e:
                titles, llm_status, llm_attempts, llm_raw_text = [], 'error', 0, str(e)
                if args.llm_defer_on_fail == 'yes':
                    pending.append({'source_sample_id': r.sample_id, 'doc_id': r.doc_id, 'raw_title': r.raw_title, 'path_text': r.path_text, 'status': llm_status})
        if raw_fp is not None and (llm_raw_text or llm_status != 'off'):
            raw_rec = {
                'sample_id': r.sample_id, 'doc_id': r.doc_id, 'raw_title': r.raw_title, 'path_text': r.path_text,
                'llm_status': llm_status, 'attempts': llm_attempts, 'response': llm_raw_text,
            }
            raw_fp.write(json.dumps(raw_rec, ensure_ascii=False) + '\n')

        if clean_source == 'rule':
            t, reason = build_cleaned_title_rule(r.raw_title)
            titles = [t] if t else []
            if not titles:
                base_row.update({'sample_id': r.sample_id + '_00', 'skip_reason': reason or 'empty', 'clean_source': 'rule', 'llm_status': llm_status, 'llm_attempts': int(llm_attempts)})
                cleaned_rows.append(base_row)
                continue

        for k, ct in enumerate(titles, start=1):
            ct2 = post_sanitize(ct, int(os.environ.get('TITLE_MAX_LEN', args.title_max_len)))
            core = keep_zh_alnum(ct2) # 直接调用全局函数，不再需要 _re
            reason = '' if len(core) >= 2 else 'too_short'
            final_row = base_row.copy()
            final_row.update({
                'sample_id': f"{r.sample_id}_{str(k).zfill(2)}",
                'cleaned_title': ct2,
                'skip_reason': reason,
                'clean_source': clean_source,
                'variant_idx': int(k),
                'llm_status': llm_status,
                'llm_attempts': int(llm_attempts),
            })
            cleaned_rows.append(final_row)

    if raw_fp is not None:
        raw_fp.close()
        log_write(log_path, f"[WRITE] {raw_dump_path}")

    cleaned_df = pd.DataFrame(cleaned_rows)
    try:
        counts = cleaned_df.groupby('source_sample_id')['cleaned_title'].apply(lambda s: int((s.fillna('') != '').sum())).to_dict()
        cleaned_df['num_variants_per_source'] = cleaned_df['source_sample_id'].map(lambda x: int(counts.get(x, 0)))
    except Exception:
        cleaned_df['num_variants_per_source'] = 0

    # public 主输出
    cleaned_out = out_dir / 'policy_corpus_cleaned.csv'
    cleaned_df.to_csv(cleaned_out, index=False, encoding='utf-8-sig')
    log_write(log_path, f"[WRITE] {cleaned_out} rows={len(cleaned_df)}")
    print(f"[WRITE] {cleaned_out}")


    if pending:
        pd.DataFrame(pending).to_csv(out_dir / 'policy_title_cleaning_llm_pending.csv', index=False, encoding='utf-8-sig')
        log_write(log_path, f"[WRITE] {out_dir / 'policy_title_cleaning_llm_pending.csv'} rows={len(pending)}")

    meta = {
        'source': str(Path(args.source).resolve()),
        'encoding': 'utf-8',
        'title_max_len': int(args.title_max_len),
        'llm_clean': args.llm_clean,
        'llm_clean_profile': llm_profile,
        'llm_clean_model_override': llm_model_override or '',
        'llm_clean_max_new_titles': int(args.llm_clean_max_new_titles),
        'llm_clean_prompt_path': args.llm_clean_prompt,
        'llm_timeout': int(llm_timeout),
        'llm_retries': int(llm_retries),
        'llm_defer_on_fail': args.llm_defer_on_fail,
        'debug_dump_llm': args.debug_dump_llm,
    }
    (out_dir / 'run_metadata.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
    log_write(log_path, f"[WRITE] {out_dir / 'run_metadata.json'}")


if __name__ == '__main__':
    main()
