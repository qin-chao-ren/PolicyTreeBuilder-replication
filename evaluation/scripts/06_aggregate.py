"""
后处理 - 读取所有 judge_*_node_scores.jsonl 和 judge_*_path_scores.jsonl，
计算:
  - 节点级综合分 NodeScore_adjusted
  - 路径级综合分 PathScore_adjusted
  - 各维度均分
  - 模型间一致性: 两两 quadratic weighted Cohen kappa；3+ 模型时计算 ordinal Krippendorff alpha
  - flags: per-model rate、full agreement、majority agreement、pairwise Cohen kappa
  - 生成 9.1 / 9.2 / 9.3 / 9.4 / 9.5 报告

输出:
  outputs/final_node_scores.csv          # 9.2
  outputs/final_path_scores.csv          # 9.3
  outputs/agreement_node.csv             # 9.4 节点级
  outputs/agreement_path.csv             # 9.4 路径级
  outputs/agreement_flags.csv            # 9.4 flags
  outputs/final_summary.json             # 9.1 + 综合分 + 摘要
  outputs/final_summary_zh.txt           # 9.5 中文摘要
"""
import json
import csv
import math
import itertools
import argparse
from pathlib import Path
from collections import defaultdict, Counter
from statistics import mean, stdev

from eval_paths import default_output_dir, resolve_repo_path

OUT = default_output_dir()


def parse_args():
    parser = argparse.ArgumentParser(description="Aggregate node/path judge scores into evaluation reports.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_dir(),
        help="Directory containing judge score JSONL files and receiving aggregate outputs.",
    )
    return parser.parse_args()

# ============== 配置 ==============
NODE_DIM_WEIGHTS = {"vertical_coherence": 0.40, "horizontal_coherence": 0.35, "label_quality": 0.25}
PATH_DIM_WEIGHTS = {"path_coherence": 0.55, "granularity_progression": 0.45}

NODE_FLAG_PENALTY = {
    # critical
    "PARENT_CHILD_IDENTICAL": 1.0,
    "ORPHAN_NODE": 1.0,
    "CIRCULAR_REFERENCE": 1.0,
    # moderate
    "SIBLING_SEMANTIC_DUPLICATE": 0.5,
    "LOW_ABSTRACTION_PARENT": 0.5,
    "SEMANTIC_DRIFT_CHILD": 0.5,
    "UNBALANCED_BRANCHING": 0.5,
    # minor
    "GRANULARITY_JUMP": 0.2,
    "LABEL_TOO_LONG": 0.2,
    "MIXED_ABSTRACTION_SIBLINGS": 0.2,
}
PATH_FLAG_PENALTY = {
    "PATH_SEMANTIC_DRIFT": 0.8,
    "PATH_GRANULARITY_JUMP": 0.6,
    "PATH_REDUNDANT_CHAIN": 0.5,
    "PATH_OVER_DEEP": 0.4,
    "PATH_UNDER_STRUCTURED": 0.4,
}

FRAMEWORK_NODE_W = 0.65
FRAMEWORK_PATH_W = 0.35

# ============== 工具 ==============
def load_jsonl(path):
    out = []
    if not path.exists(): return out
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try: out.append(json.loads(line))
            except: pass
    return out

def safe_int(x, lo=1, hi=5):
    try:
        v = int(round(float(x)))
        return max(lo, min(hi, v))
    except: return None

def quad_weighted_kappa(y1, y2, K=5):
    """Quadratic weighted Cohen's kappa, ratings in 1..K"""
    n = len(y1)
    if n == 0: return None
    O = [[0]*K for _ in range(K)]
    for a, b in zip(y1, y2):
        if 1 <= a <= K and 1 <= b <= K:
            O[a-1][b-1] += 1
    r = [sum(O[i]) for i in range(K)]
    c = [sum(O[i][j] for i in range(K)) for j in range(K)]
    E = [[r[i]*c[j]/n for j in range(K)] for i in range(K)]
    W = [[((i-j)**2)/((K-1)**2) for j in range(K)] for i in range(K)]
    num = sum(W[i][j]*O[i][j] for i in range(K) for j in range(K))
    den = sum(W[i][j]*E[i][j] for i in range(K) for j in range(K))
    if den == 0: return None
    return 1 - num/den

def krippendorff_alpha_ordinal(matrix, K=5):
    """
    matrix: list of length-N items, each item is a list of M ratings (or None)
    ordinal alpha following Krippendorff (interval distance on rank-converted scores
    is approximated here by squared difference -- equivalent to interval alpha for
    integer 1..K. For strict ordinal you'd use cumulative-rank distances; we report
    interval-style alpha which is what most papers actually report).
    """
    # Pairs within each unit
    sum_within = 0.0
    n_pairs_within = 0
    all_vals = []
    for ratings in matrix:
        valid = [r for r in ratings if r is not None]
        m = len(valid)
        if m < 2: continue
        for a, b in itertools.combinations(valid, 2):
            sum_within += (a-b)**2
            n_pairs_within += 1
        all_vals.extend(valid)
    if n_pairs_within == 0 or len(all_vals) < 2:
        return None
    Do = sum_within / n_pairs_within
    # Expected disagreement: all pairs from pooled values
    sum_all = 0.0
    n_pairs_all = 0
    for a, b in itertools.combinations(all_vals, 2):
        sum_all += (a-b)**2
        n_pairs_all += 1
    De = sum_all / n_pairs_all
    if De == 0: return None
    return 1 - Do/De

def cohen_kappa_binary(y1, y2):
    """Standard Cohen's kappa for binary 0/1 ratings"""
    n = len(y1)
    if n == 0: return None
    a = sum(1 for x,y in zip(y1,y2) if x==y)
    Po = a/n
    p1 = sum(y1)/n; q1 = 1-p1
    p2 = sum(y2)/n; q2 = 1-p2
    Pe = p1*p2 + q1*q2
    if Pe == 1: return None
    return (Po - Pe) / (1 - Pe)

# ============== 读评分 ==============
def discover_judges(kind):
    """kind in {'node','path'}; returns list of judge_keys based on existing files"""
    keys = []
    for p in OUT.glob(f"judge_*_{kind}_scores.jsonl"):
        # judge_<key>_<kind>_scores.jsonl
        name = p.name
        prefix = "judge_"
        suffix = f"_{kind}_scores.jsonl"
        if name.startswith(prefix) and name.endswith(suffix):
            key = name[len(prefix):-len(suffix)]
            keys.append(key)
    return sorted(keys)

def load_node_scores(judge_keys):
    """
    return:
      records[judge_key][sample_id] = {
        "scores": {...}, "flags": [...], "node_id":..., "level":...,
        "node_score_raw":..., "node_score_adjusted":...
      }
    """
    out = {}
    for jk in judge_keys:
        recs = load_jsonl(OUT / f"judge_{jk}_node_scores.jsonl")
        d = {}
        for r in recs:
            res = r.get("result", {}) or {}
            scores = res.get("scores", {}) or {}
            v = safe_int(scores.get("vertical_coherence"))
            h = safe_int(scores.get("horizontal_coherence"))
            l = safe_int(scores.get("label_quality"))
            flags = res.get("flags", []) or []
            if not isinstance(flags, list): flags = []
            # raw
            if v is not None and h is not None and l is not None:
                raw = NODE_DIM_WEIGHTS["vertical_coherence"]*v \
                    + NODE_DIM_WEIGHTS["horizontal_coherence"]*h \
                    + NODE_DIM_WEIGHTS["label_quality"]*l
                pen = sum(NODE_FLAG_PENALTY.get(f, 0) for f in flags)
                adj = max(1.0, raw - pen)
            else:
                raw = None; adj = None
            d[r["sample_id"]] = {
                "judge_key": jk,
                "node_id": r.get("node_id"),
                "level": r.get("level"),
                "depth": r.get("depth"),
                "l1_label": r.get("l1_label"),
                "high_risk": r.get("high_risk"),
                "vertical_coherence": v,
                "horizontal_coherence": h,
                "label_quality": l,
                "flags": flags,
                "node_score_raw": round(raw, 3) if raw is not None else None,
                "node_score_adjusted": round(adj, 3) if adj is not None else None,
                "issue_summary": res.get("issue_summary",""),
                "suggested_fix_type": res.get("suggested_fix_type",""),
                "suggested_fix_note": res.get("suggested_fix_note",""),
                "node_label": res.get("node_label",""),
            }
        out[jk] = d
    return out

def load_path_scores(judge_keys):
    out = {}
    for jk in judge_keys:
        recs = load_jsonl(OUT / f"judge_{jk}_path_scores.jsonl")
        d = {}
        for r in recs:
            res = r.get("result", {}) or {}
            scores = res.get("scores", {}) or {}
            pc = safe_int(scores.get("path_coherence"))
            gp = safe_int(scores.get("granularity_progression"))
            flags = res.get("flags", []) or []
            if not isinstance(flags, list): flags = []
            if pc is not None and gp is not None:
                raw = PATH_DIM_WEIGHTS["path_coherence"]*pc \
                    + PATH_DIM_WEIGHTS["granularity_progression"]*gp
                pen = sum(PATH_FLAG_PENALTY.get(f, 0) for f in flags)
                adj = max(1.0, raw - pen)
            else:
                raw = None; adj = None
            d[r["sample_id"]] = {
                "judge_key": jk,
                "l1_label": r.get("l1_label"),
                "path_length": r.get("path_length"),
                "path_coherence": pc,
                "granularity_progression": gp,
                "flags": flags,
                "path_score_raw": round(raw,3) if raw is not None else None,
                "path_score_adjusted": round(adj,3) if adj is not None else None,
                "issue_summary": res.get("issue_summary",""),
                "suggested_fix_type": res.get("suggested_fix_type",""),
                "path_labels": res.get("path_labels", []),
            }
        out[jk] = d
    return out

# ============== 9.2 节点级表 ==============
def write_final_node_scores(node_scores):
    rows = []
    for jk, d in node_scores.items():
        for sid, r in d.items():
            rows.append({
                "model_name": jk,
                "sample_id": sid,
                "node_id": r["node_id"],
                "node_label": r["node_label"],
                "level": r["level"],
                "depth": r["depth"],
                "l1_label": r["l1_label"],
                "high_risk": r["high_risk"],
                "vertical_coherence": r["vertical_coherence"],
                "horizontal_coherence": r["horizontal_coherence"],
                "label_quality": r["label_quality"],
                "node_score_raw": r["node_score_raw"],
                "node_score_adjusted": r["node_score_adjusted"],
                "flags": ";".join(r["flags"]),
                "issue_summary": r["issue_summary"],
                "suggested_fix_type": r["suggested_fix_type"],
            })
    if not rows:
        print("  (no node scores yet)")
        return
    fields = list(rows[0].keys())
    with (OUT / "final_node_scores.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    print(f"[OK] final_node_scores.csv: {len(rows)} rows")

# ============== 9.3 路径级表 ==============
def write_final_path_scores(path_scores):
    rows = []
    for jk, d in path_scores.items():
        for sid, r in d.items():
            rows.append({
                "model_name": jk,
                "sample_id": sid,
                "l1_label": r["l1_label"],
                "path_length": r["path_length"],
                "path_labels": ">".join(r["path_labels"]) if r["path_labels"] else "",
                "path_coherence": r["path_coherence"],
                "granularity_progression": r["granularity_progression"],
                "path_score_raw": r["path_score_raw"],
                "path_score_adjusted": r["path_score_adjusted"],
                "flags": ";".join(r["flags"]),
                "issue_summary": r["issue_summary"],
                "suggested_fix_type": r["suggested_fix_type"],
            })
    if not rows:
        print("  (no path scores yet)")
        return
    fields = list(rows[0].keys())
    with (OUT / "final_path_scores.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    print(f"[OK] final_path_scores.csv: {len(rows)} rows")

# ============== 9.4 一致性 ==============
def agreement_for_dim(scores_by_judge, dim, K=5):
    """
    scores_by_judge: dict[judge_key] -> dict[sample_id] -> record
    dim: e.g. 'vertical_coherence'
    return: list of (jk_a, jk_b, kappa, n) and overall alpha (if 3+ judges)
    """
    judges = list(scores_by_judge.keys())
    sample_ids = set()
    for jk in judges:
        sample_ids |= set(scores_by_judge[jk].keys())
    sample_ids = sorted(sample_ids)

    pairwise = []
    for a, b in itertools.combinations(judges, 2):
        ya, yb = [], []
        for sid in sample_ids:
            ra = scores_by_judge[a].get(sid); rb = scores_by_judge[b].get(sid)
            if ra and rb and ra.get(dim) is not None and rb.get(dim) is not None:
                ya.append(ra[dim]); yb.append(rb[dim])
        kw = quad_weighted_kappa(ya, yb, K=K) if ya else None
        pairwise.append({"judge_a": a, "judge_b": b, "n": len(ya),
                         "weighted_kappa": round(kw, 3) if kw is not None else None})

    matrix = []
    for sid in sample_ids:
        row = []
        for jk in judges:
            r = scores_by_judge[jk].get(sid)
            row.append(r[dim] if r and r.get(dim) is not None else None)
        matrix.append(row)
    alpha = krippendorff_alpha_ordinal(matrix, K=K) if len(judges) >= 2 else None

    valid_kappas = [p["weighted_kappa"] for p in pairwise if p["weighted_kappa"] is not None]
    mean_kappa = round(mean(valid_kappas), 3) if valid_kappas else None

    return {
        "dimension": dim,
        "judges": judges,
        "pairwise": pairwise,
        "mean_pairwise_weighted_kappa": mean_kappa,
        "krippendorff_alpha_ordinal": round(alpha, 3) if alpha is not None else None,
    }

def agreement_for_flags(scores_by_judge, flag_universe):
    """对每个 flag 计算 per-model rate / full agreement / majority agreement / mean Cohen kappa"""
    judges = list(scores_by_judge.keys())
    sample_ids = set()
    for jk in judges:
        sample_ids |= set(scores_by_judge[jk].keys())
    sample_ids = sorted(sample_ids)

    out = []
    for flag in flag_universe:
        # 二元向量
        vecs = {jk: [] for jk in judges}
        rates = {jk: 0 for jk in judges}
        n_total = 0
        full_agree = 0
        majority_agree = 0
        for sid in sample_ids:
            present = []
            for jk in judges:
                r = scores_by_judge[jk].get(sid)
                if r is None:
                    present.append(None)
                else:
                    present.append(1 if flag in r.get("flags", []) else 0)
            if any(p is None for p in present):
                continue
            n_total += 1
            for jk, v in zip(judges, present):
                vecs[jk].append(v); rates[jk] += v
            if all(v == present[0] for v in present):
                full_agree += 1
            # majority: most common value count > len/2
            cnt = Counter(present)
            if cnt.most_common(1)[0][1] > len(present)/2:
                majority_agree += 1

        if n_total == 0:
            continue

        # pairwise Cohen kappa（二元）
        pair_kappas = []
        for a, b in itertools.combinations(judges, 2):
            k = cohen_kappa_binary(vecs[a], vecs[b])
            if k is not None:
                pair_kappas.append(k)
        mean_kappa = round(mean(pair_kappas), 3) if pair_kappas else None

        rec = {
            "flag": flag,
            "n_evaluated": n_total,
            "full_agreement_rate": round(full_agree / n_total, 3),
            "majority_agreement_rate": round(majority_agree / n_total, 3),
            "mean_pairwise_cohen_kappa": mean_kappa,
        }
        for jk in judges:
            rec[f"rate_{jk}"] = round(rates[jk] / n_total, 3)
        out.append(rec)
    return out

def write_agreement_table(rows_dim, path_csv):
    """rows_dim: list of dicts produced by agreement_for_dim"""
    if not rows_dim:
        print(f"  (no agreement rows for {path_csv.name})")
        return
    judges = rows_dim[0]["judges"]
    fields = ["dimension"]
    for a, b in itertools.combinations(judges, 2):
        fields.append(f"weighted_kappa_{a}_VS_{b}")
        fields.append(f"n_{a}_VS_{b}")
    fields += ["mean_pairwise_weighted_kappa", "krippendorff_alpha_ordinal"]
    with path_csv.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for r in rows_dim:
            row = {"dimension": r["dimension"]}
            for p in r["pairwise"]:
                row[f"weighted_kappa_{p['judge_a']}_VS_{p['judge_b']}"] = p["weighted_kappa"]
                row[f"n_{p['judge_a']}_VS_{p['judge_b']}"] = p["n"]
            row["mean_pairwise_weighted_kappa"] = r["mean_pairwise_weighted_kappa"]
            row["krippendorff_alpha_ordinal"] = r["krippendorff_alpha_ordinal"]
            w.writerow(row)
    print(f"[OK] {path_csv.name}")

def write_flag_agreement(rows, path_csv, judges):
    if not rows: return
    fields = ["flag", "n_evaluated"]
    for jk in judges: fields.append(f"rate_{jk}")
    fields += ["full_agreement_rate", "majority_agreement_rate", "mean_pairwise_cohen_kappa"]
    with path_csv.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    print(f"[OK] {path_csv.name}")

# ============== 9.1 / 综合分 / 9.5 ==============
def per_judge_means(scores_by_judge, dims):
    out = {}
    for jk, d in scores_by_judge.items():
        means = {}
        for dim in dims:
            vals = [r[dim] for r in d.values() if r.get(dim) is not None]
            means[dim] = round(mean(vals), 3) if vals else None
        out[jk] = means
    return out

def per_judge_adjusted(scores_by_judge, key):
    """key in {'node_score_adjusted','path_score_adjusted'}"""
    out = {}
    for jk, d in scores_by_judge.items():
        vals = [r[key] for r in d.values() if r.get(key) is not None]
        out[jk] = round(mean(vals), 3) if vals else None
    return out

def divergent_samples_node(node_scores, threshold=2):
    """某指标在不同模型间最大分差 >= threshold 的样本"""
    judges = list(node_scores.keys())
    if len(judges) < 2: return []
    sids = set()
    for jk in judges: sids |= set(node_scores[jk].keys())
    out = []
    dims = ["vertical_coherence","horizontal_coherence","label_quality"]
    for sid in sorted(sids):
        rs = [node_scores[jk].get(sid) for jk in judges]
        if any(r is None for r in rs): continue
        max_diff = 0
        worst_dim = None
        for dim in dims:
            vals = [r[dim] for r in rs if r.get(dim) is not None]
            if len(vals) >= 2:
                diff = max(vals) - min(vals)
                if diff > max_diff:
                    max_diff = diff; worst_dim = dim
        # flag 不一致
        flag_sets = [set(r.get("flags", [])) for r in rs]
        flag_disagree = any(s != flag_sets[0] for s in flag_sets[1:])
        if max_diff >= threshold or flag_disagree:
            out.append({
                "sample_id": sid,
                "node_id": rs[0]["node_id"],
                "node_label": rs[0]["node_label"],
                "level": rs[0]["level"],
                "max_score_diff": max_diff,
                "worst_dim": worst_dim,
                "flag_disagree": flag_disagree,
                "scores_by_judge": {jk: {d: rs[i].get(d) for d in dims} for i, jk in enumerate(judges)},
            })
    return out

def divergent_samples_path(path_scores, threshold=2):
    judges = list(path_scores.keys())
    if len(judges) < 2: return []
    sids = set()
    for jk in judges: sids |= set(path_scores[jk].keys())
    out = []
    dims = ["path_coherence","granularity_progression"]
    for sid in sorted(sids):
        rs = [path_scores[jk].get(sid) for jk in judges]
        if any(r is None for r in rs): continue
        max_diff = 0
        for dim in dims:
            vals = [r[dim] for r in rs if r.get(dim) is not None]
            if len(vals) >= 2:
                diff = max(vals) - min(vals)
                if diff > max_diff: max_diff = diff
        flag_sets = [set(r.get("flags", [])) for r in rs]
        flag_disagree = any(s != flag_sets[0] for s in flag_sets[1:])
        if max_diff >= threshold or flag_disagree:
            out.append({"sample_id": sid, "max_score_diff": max_diff,
                        "flag_disagree": flag_disagree})
    return out

# ============== 主流程 ==============
def main():
    global OUT
    args = parse_args()
    OUT = resolve_repo_path(args.output_dir)
    OUT.mkdir(parents=True, exist_ok=True)

    structure_report = json.load((OUT / "structure_report.json").open("r", encoding="utf-8"))

    node_judges = discover_judges("node")
    path_judges = discover_judges("path")
    print(f"detected node judges: {node_judges}")
    print(f"detected path judges: {path_judges}")

    node_scores = load_node_scores(node_judges) if node_judges else {}
    path_scores = load_path_scores(path_judges) if path_judges else {}

    # 9.2 / 9.3
    write_final_node_scores(node_scores)
    write_final_path_scores(path_scores)

    # ---------- 9.4 一致性 ----------
    summary = {
        "tree_basic_stats": structure_report["tree_basic_stats"],
        "rule_based_issues_summary": {
            k: structure_report["rule_based_issues"][k]
            for k in structure_report["rule_based_issues"] if k.endswith("_count")
        },
        "structural_warnings_summary": {
            k: structure_report["structural_warnings"][k]
            for k in structure_report["structural_warnings"] if k.endswith("_count")
        },
        "node_judges": node_judges,
        "path_judges": path_judges,
    }

    if len(node_judges) >= 2:
        node_dims = ["vertical_coherence","horizontal_coherence","label_quality"]
        rows = [agreement_for_dim(node_scores, d, K=5) for d in node_dims]
        write_agreement_table(rows, OUT / "agreement_node.csv")
        summary["agreement_node"] = rows

        # flags
        flag_universe = set()
        for jk in node_judges:
            for r in node_scores[jk].values():
                flag_universe.update(r.get("flags", []))
        flag_rows = agreement_for_flags(node_scores, sorted(flag_universe))
        write_flag_agreement(flag_rows, OUT / "agreement_node_flags.csv", node_judges)
        summary["agreement_node_flags"] = flag_rows

        # divergent
        div = divergent_samples_node(node_scores)
        with (OUT / "divergent_nodes.json").open("w", encoding="utf-8") as f:
            json.dump(div, f, ensure_ascii=False, indent=2)
        print(f"[OK] divergent_nodes.json: {len(div)} samples")

    if len(path_judges) >= 2:
        path_dims = ["path_coherence","granularity_progression"]
        rows = [agreement_for_dim(path_scores, d, K=5) for d in path_dims]
        write_agreement_table(rows, OUT / "agreement_path.csv")
        summary["agreement_path"] = rows

        flag_universe = set()
        for jk in path_judges:
            for r in path_scores[jk].values():
                flag_universe.update(r.get("flags", []))
        flag_rows = agreement_for_flags(path_scores, sorted(flag_universe))
        write_flag_agreement(flag_rows, OUT / "agreement_path_flags.csv", path_judges)
        summary["agreement_path_flags"] = flag_rows

        div = divergent_samples_path(path_scores)
        with (OUT / "divergent_paths.json").open("w", encoding="utf-8") as f:
            json.dump(div, f, ensure_ascii=False, indent=2)
        print(f"[OK] divergent_paths.json: {len(div)} samples")

    # ---------- 综合分 ----------
    framework_per_model = {}
    for jk in set(node_judges) | set(path_judges):
        node_d = node_scores.get(jk, {})
        path_d = path_scores.get(jk, {})
        n_vals = [r["node_score_adjusted"] for r in node_d.values() if r.get("node_score_adjusted") is not None]
        p_vals = [r["path_score_adjusted"] for r in path_d.values() if r.get("path_score_adjusted") is not None]
        mean_n = mean(n_vals) if n_vals else None
        mean_p = mean(p_vals) if p_vals else None
        if mean_n is not None and mean_p is not None:
            fw = FRAMEWORK_NODE_W * mean_n + FRAMEWORK_PATH_W * mean_p
        elif mean_n is not None:
            fw = mean_n
        elif mean_p is not None:
            fw = mean_p
        else:
            fw = None
        framework_per_model[jk] = {
            "mean_node_score_adjusted": round(mean_n, 3) if mean_n is not None else None,
            "mean_path_score_adjusted": round(mean_p, 3) if mean_p is not None else None,
            "framework_score": round(fw, 3) if fw is not None else None,
            "n_node_samples": len(n_vals),
            "n_path_samples": len(p_vals),
        }
    summary["framework_per_model"] = framework_per_model

    fw_vals = [v["framework_score"] for v in framework_per_model.values() if v["framework_score"] is not None]
    summary["final_framework_score"] = round(mean(fw_vals), 3) if fw_vals else None

    # 维度均分
    summary["dim_means_node"] = per_judge_means(node_scores,
        ["vertical_coherence","horizontal_coherence","label_quality"]) if node_scores else {}
    summary["dim_means_path"] = per_judge_means(path_scores,
        ["path_coherence","granularity_progression"]) if path_scores else {}

    with (OUT / "final_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[OK] final_summary.json")

    # ---------- 9.5 中文摘要 ----------
    write_zh_summary(summary)

def write_zh_summary(s):
    lines = []
    bs = s["tree_basic_stats"]
    lines.append("# 政策行动树质量评估摘要\n")
    lines.append(f"## 一、树规模\n")
    lines.append(f"全树共 {bs['total_nodes']} 个节点（含 ROOT），其中叶节点 {bs['total_leaf_nodes']} 个，最大深度 {bs['max_depth']}。"
                 f"L1 共 {bs['level_distribution'].get('L1', 0)} 个主题域，"
                 f"内部节点平均分支数 {bs['avg_children_per_internal_node']}，"
                 f"子节点数超过 8 的父节点 {bs['oversized_parent_count']} 个。\n")

    lines.append(f"## 二、规则检查问题\n")
    rb = s["rule_based_issues_summary"]
    lines.append(f"- 重复 node_id：{rb.get('duplicate_node_id_count',0)}")
    lines.append(f"- 孤立节点：{rb.get('orphan_node_count',0)}")
    lines.append(f"- 多父引用：{rb.get('multi_parent_node_count',0)}")
    lines.append(f"- 循环引用：{rb.get('circular_reference_count',0)}")
    lines.append(f"- 空标签：{rb.get('empty_label_count',0)}")
    lines.append(f"- level 与实际 depth 不一致：{rb.get('level_depth_mismatch_count',0)}\n")

    sw = s["structural_warnings_summary"]
    lines.append(f"## 三、结构性提示\n")
    lines.append(f"- 子节点数 > 8 的父节点：{sw.get('oversized_parent_count',0)}")
    lines.append(f"- 标签长度 > 25 字：{sw.get('long_labels_count',0)}")
    lines.append(f"- 兄弟标签字面重复：{sw.get('sibling_duplicate_label_count',0)}")
    lines.append(f"- 父子标签字面相同：{sw.get('parent_child_same_label_count',0)}")
    lines.append(f"- 跨分支同名节点：{sw.get('cross_branch_same_label_count',0)}\n")

    if s.get("node_judges"):
        lines.append(f"## 四、节点级评价\n")
        lines.append(f"评价模型：{', '.join(s['node_judges'])}")
        for jk, m in s.get("dim_means_node", {}).items():
            lines.append(f"- {jk}: vertical={m.get('vertical_coherence')}, "
                         f"horizontal={m.get('horizontal_coherence')}, "
                         f"label={m.get('label_quality')}")
        if "agreement_node" in s and s["agreement_node"]:
            lines.append("\n模型间一致性（节点级）：")
            for r in s["agreement_node"]:
                lines.append(f"- {r['dimension']}: 均 weighted κ = {r['mean_pairwise_weighted_kappa']}, "
                             f"Krippendorff α = {r['krippendorff_alpha_ordinal']}")

    if s.get("path_judges"):
        lines.append(f"\n## 五、路径级评价\n")
        lines.append(f"评价模型：{', '.join(s['path_judges'])}")
        for jk, m in s.get("dim_means_path", {}).items():
            lines.append(f"- {jk}: coherence={m.get('path_coherence')}, "
                         f"granularity={m.get('granularity_progression')}")
        if "agreement_path" in s and s["agreement_path"]:
            lines.append("\n模型间一致性（路径级）：")
            for r in s["agreement_path"]:
                lines.append(f"- {r['dimension']}: 均 weighted κ = {r['mean_pairwise_weighted_kappa']}, "
                             f"Krippendorff α = {r['krippendorff_alpha_ordinal']}")

    if s.get("framework_per_model"):
        lines.append(f"\n## 六、综合质量评分\n")
        for jk, v in s["framework_per_model"].items():
            lines.append(f"- {jk}: 节点均分={v['mean_node_score_adjusted']}, "
                         f"路径均分={v['mean_path_score_adjusted']}, "
                         f"FrameworkScore={v['framework_score']}")
        lines.append(f"\n**最终框架质量评分（多模型平均）：{s.get('final_framework_score')}**")

    txt = "\n".join(lines)
    (OUT / "final_summary_zh.txt").write_text(txt, encoding="utf-8")
    print(f"[OK] final_summary_zh.txt")

if __name__ == "__main__":
    main()
