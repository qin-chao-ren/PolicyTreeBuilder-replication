import pandas as pd
import numpy as np
from pathlib import Path

# 配置路径 (根据你的 config 调整)
pairs_path = Path("data/intermediate_outputs/v4_rerank_edges.csv")

def check_scores():
    if not pairs_path.exists():
        print(f"❌ 找不到文件: {pairs_path}")
        return

    print(f"正在读取: {pairs_path} ...")
    df = pd.read_csv(pairs_path)
    
    if "rerank_score" not in df.columns:
        print("❌ CSV 中没有 'rerank_score' 列！请检查列名。")
        return

    # 强制转为 float，看看有没有报错（检查是否有非数字字符）
    try:
        scores = df["rerank_score"].astype(float)
    except Exception as e:
        print(f"❌ 数据类型转换失败，可能包含非数字字符: {e}")
        return

    total = len(scores)
    print(f"\n✅ 数据总行数: {total}")
    print(f"📊 分数统计 (Min/Mean/Max):")
    print(f"   Min:  {scores.min():.4f}")
    print(f"   Mean: {scores.mean():.4f}")
    print(f"   Max:  {scores.max():.4f}")
    
    print("\n🔍 阈值命中测试:")
    
    # 测试 High Threshold (0.90)
    high_count = (scores >= 0.90).sum()
    print(f"   >= 0.90 (High Thr): {high_count} 条 ({high_count/total:.2%})")
    
    # 测试 Mid Threshold (0.80)
    mid_count = (scores >= 0.80).sum()
    print(f"   >= 0.80 (Mid Thr):  {mid_count} 条 ({mid_count/total:.2%})")
    
    # 测试更低一点的阈值，看看数据到底在哪
    low_count = (scores >= 0.70).sum()
    print(f"   >= 0.70 (For Ref):  {low_count} 条 ({low_count/total:.2%})")

    if high_count == 0 and mid_count == 0:
        print("\n❌ [结论]: 阈值设置过高！没有任何边能通过筛选。")
        print("   -> 建议：降低 high_thr 和 mid_thr_low。")
    elif high_count > 0 and mid_count == 0:
        print("\n⚠️ [结论]: 只有高分边，没有中间候选区。")
        print("   -> 建议：降低 mid_thr_low。")
    else:
        print("\n✅ [结论]: 数据分布看起来正常，如果主程序仍不行，可能是代码里的 ID 匹配或逻辑 bug。")

if __name__ == "__main__":
    check_scores()