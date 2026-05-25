#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Production Tree Visualization (Horizontal Layout)
全量生产数据可视化
功能：将 v4 流程产生的 5 个关键阶段 JSON 树渲染为高清大图。
运行：python scripts/visualize_production.py
"""

import json
import sys
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import platform
from pathlib import Path

# --- [修改点 1] 增加递归深度限制，防止全量数据过大导致报错 ---
sys.setrecursionlimit(5000)

# --- [修改点 2] 路径锚点调整 ---
# 假设本脚本放在 scripts/ 目录下
HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent  # roundC_v4/
OUTPUT_DIR = PROJECT_ROOT / "data" / "intermediate_outputs"

# --- [修改点 3] 生产环境的文件名列表 ---
TARGET_FILES = [
    "v4_tree_coarse_global.json",  # Step 3 产出的粗树 (对应 test_tree_input)
    "v4_tree_s4_1.json",           # Step 4.1 Skeleton 产物
    "v4_tree_s4_2.json",           # Step 4.2 Shaping 产物
    "v4_tree_refined.json",        # Step 4.3 Polishing 产物
    "v4_tree_final.json"           # Step 4.5 最终产物
]

# --- 下面的代码基本保持不变 ---

# 字体设置
system_name = platform.system()
if system_name == 'Windows':
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'sans-serif']
elif system_name == 'Darwin':
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC', 'sans-serif']
else:
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

def load_json_data(filepath):
    if not filepath.exists():
        return None
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_tree_stats(node, current_depth=0):
    """统计最大深度和叶子节点总数"""
    node['_depth'] = current_depth
    max_d = current_depth
    leaf_count = 0

    children = node.get('children', [])
    if not children:
        leaf_count = 1
    else:
        for child in children:
            d, l = get_tree_stats(child, current_depth + 1)
            max_d = max(max_d, d)
            leaf_count += l

    return max_d, leaf_count

def calculate_positions(node, current_leaf_y, x_step=1.0, y_step=1.0):
    """递归计算坐标 (Left-to-Right)"""
    node['_x'] = node['_depth'] * x_step
    children = node.get('children', [])

    if not children:
        node['_y'] = current_leaf_y[0] * y_step
        current_leaf_y[0] += 1
    else:
        child_ys = []
        for child in children:
            calculate_positions(child, current_leaf_y, x_step, y_step)
            child_ys.append(child['_y'])
        node['_y'] = sum(child_ys) / len(child_ys)

def collect_draw_data(node, nodes_list, edges_list, max_depth):
    """收集绘图所需的扁平化数据"""
    color_val = node['_depth'] / max_depth if max_depth > 0 else 0
    color = cm.viridis(color_val)

    # 全量数据节点可能很多，稍微调小一点默认尺寸
    size = max(500 - node['_depth'] * 60, 100)
    font_size = max(12 - node['_depth'], 7) # 最小字号7

    nodes_list.append({
        'x': node['_x'],
        'y': node['_y'],
        'label': node.get('label', ''),
        'level': node.get('level', ''),
        'color': color,
        'size': size,
        'font_size': font_size
    })

    for child in node.get('children', []):
        edges_list.append({
            'x1': node['_x'], 'y1': node['_y'],
            'x2': child['_x'], 'y2': child['_y']
        })
        collect_draw_data(child, nodes_list, edges_list, max_depth)

def visualize_file(filename):
    input_path = OUTPUT_DIR / filename
    output_path = OUTPUT_DIR / f"{input_path.stem}.png"

    print(f"处理中: {filename} ...")
    data = load_json_data(input_path)
    if not data:
        print(f"  [跳过] 文件不存在: {input_path}")
        return

    # 1. 统计与预处理
    max_depth, total_leaves = get_tree_stats(data)
    print(f"  - 树规模: 深度 {max_depth}, 叶子节点 {total_leaves}")

    # 2. 计算坐标
    leaf_y_counter = [0]
    # x_step=5.0 为了给更长的真实标题留空间
    calculate_positions(data, leaf_y_counter, x_step=5.0, y_step=1.0)

    # 3. 收集绘图对象
    nodes = []
    edges = []
    collect_draw_data(data, nodes, edges, max_depth)

    # 4. 设置画布 (全量数据通常很大，需要更大的画布)
    # 高度：每个叶子预留 0.35 英寸 (稍微紧凑一点以免图片过大)
    fig_height = max(10, total_leaves * 0.35)
    fig_width = max(15, max_depth * 4.0)

    # 限制最大尺寸防止内存溢出 (例如高度限制在 500 inch)
    if fig_height > 500:
        print(f"  [警告] 图片高度 ({fig_height}) 过大，已限制为 500。可能会有重叠。")
        fig_height = 500

    print(f"  - 画布尺寸: {fig_width:.1f} x {fig_height:.1f} inches")

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.set_facecolor('#FAFAFA')

    # 5. 绘制连线
    for e in edges:
        ax.plot([e['x1'], e['x2']], [e['y1'], e['y2']], c='#BBBBBB', lw=0.8, alpha=0.5, zorder=1)

    # 6. 绘制节点
    for n in nodes:
        ax.scatter(n['x'], n['y'], s=n['size'], c=[n['color']],
                   edgecolors='white', linewidths=1.0, zorder=2, alpha=1.0)

        display_label = n['label']
        # 生产数据标题可能很长，截断处理
        if len(display_label) > 20:
             display_label = display_label[:19] + ".."

        ax.text(n['x'] + 0.15, n['y'], display_label,
                va='center', ha='left', fontsize=n['font_size'],
                color='#333333', fontweight='normal', zorder=3)

    # 7. 调整坐标轴
    pad_x = 2.0
    pad_y = 1.0
    max_x = max((n['x'] for n in nodes), default=0)
    max_y = max((n['y'] for n in nodes), default=0)
    min_y = min((n['y'] for n in nodes), default=0)

    ax.set_xlim(-0.5, max_x + 8.0)
    ax.set_ylim(min_y - pad_y, max_y + pad_y)
    ax.axis('off')

    plt.title(f"Tree: {filename} (Depth: {max_depth}, Leaves: {total_leaves})", fontsize=18, pad=20)

    # 保存
    try:
        # dpi=100 足够浏览，太高会导致文件巨大
        plt.tight_layout()
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close()
        print(f"  -> 已生成: {output_path}")
    except Exception as e:
        print(f"  [错误] 保存图片失败 (可能是尺寸太大): {e}")

def main():
    if not OUTPUT_DIR.exists():
        print(f"[ERROR] 输出目录不存在: {OUTPUT_DIR}")
        return

    for fname in TARGET_FILES:
        visualize_file(fname)

    print("\n✅ 所有全量可视化图表已生成。")

if __name__ == "__main__":
    main()
