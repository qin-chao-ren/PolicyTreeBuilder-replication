#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Production Tree Visualization (Radial/Circular Layout)
全量生产数据可视化 - 圆形辐射版
功能：将 JSON 树渲染为圆形辐射图，解决长条图看不清的问题。
运行：python scripts/visualize_production_radial.py
"""

import json
import sys
import math
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import platform
from pathlib import Path

# --- 1. 基础配置 ---
sys.setrecursionlimit(5000)

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent  # roundC_v4/
OUTPUT_DIR = PROJECT_ROOT / "data" / "intermediate_outputs"

# 需要可视化的文件列表
TARGET_FILES = [
    "v4_tree_final.json",           # 重点看这个
    "v4_tree_coarse_global.json",
    "v4_tree_refined.json"
]

# 字体设置
system_name = platform.system()
if system_name == 'Windows':
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial']
elif system_name == 'Darwin':
    plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Arial Unicode MS']
else:
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# --- 2. 核心算法 ---

def load_json_data(filepath):
    if not filepath.exists():
        return None
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def count_leaves(node):
    """递归计算每个子树的叶子节点数量 (权重)"""
    children = node.get('children', [])
    if not children:
        node['_leaves'] = 1
        return 1

    total = 0
    for child in children:
        total += count_leaves(child)
    node['_leaves'] = total
    return total

def get_max_depth(node, d=0):
    """计算最大深度"""
    children = node.get('children', [])
    if not children:
        return d
    return max(get_max_depth(c, d+1) for c in children)

def calculate_polar_coords(node, start_angle, end_angle, current_depth, radius_step):
    """
    递归计算极坐标 (Radius, Angle) 并转为笛卡尔坐标 (x, y)
    start_angle, end_angle: 弧度制
    """
    # 1. 计算当前节点的角度 (扇区中心)
    mid_angle = (start_angle + end_angle) / 2

    # 2. 计算半径
    radius = current_depth * radius_step

    # 3. 转笛卡尔坐标
    x = radius * math.cos(mid_angle)
    y = radius * math.sin(mid_angle)

    node['_x'] = x
    node['_y'] = y
    node['_angle'] = mid_angle
    node['_radius'] = radius
    node['_depth'] = current_depth

    # 4. 递归分配子节点扇区
    children = node.get('children', [])
    if not children:
        return

    total_leaves = node['_leaves']
    # 扇区总宽度
    total_span = end_angle - start_angle

    current_start = start_angle
    for child in children:
        child_leaves = child['_leaves']
        # 按叶子权重分配扇区比例
        wedge_size = (child_leaves / total_leaves) * total_span

        calculate_polar_coords(
            child,
            current_start,
            current_start + wedge_size,
            current_depth + 1,
            radius_step
        )
        current_start += wedge_size

def collect_draw_objects(node, nodes_list, edges_list, max_depth):
    """收集绘图对象"""
    # 颜色映射
    color_val = node['_depth'] / max_depth if max_depth > 0 else 0
    color = cm.viridis(color_val)

    # 节点大小随深度递减
    size = max(600 - node['_depth'] * 80, 50)
    font_size = max(10 - node['_depth'], 6)

    nodes_list.append({
        'x': node['_x'],
        'y': node['_y'],
        'angle': node['_angle'],
        'label': node.get('label', ''),
        'depth': node['_depth'],
        'color': color,
        'size': size,
        'font_size': font_size
    })

    for child in node.get('children', []):
        edges_list.append({
            'x1': node['_x'], 'y1': node['_y'],
            'x2': child['_x'], 'y2': child['_y']
        })
        collect_draw_objects(child, nodes_list, edges_list, max_depth)

# --- 3. 绘图主逻辑 ---

def visualize_file(filename):
    input_path = OUTPUT_DIR / filename
    output_path = OUTPUT_DIR / f"{Path(filename).stem}_radial.png"

    print(f"正在处理: {filename} ...")
    data = load_json_data(input_path)
    if not data:
        print(f"  [跳过] 文件不存在: {input_path}")
        return

    # 1. 预计算
    total_leaves = count_leaves(data)
    max_depth = get_max_depth(data)
    print(f"  - 树结构: 深度={max_depth}, 叶子节点={total_leaves}")

    # 2. 布局计算 (0 到 2pi)
    # 半径步长: 决定了圆环之间的距离
    radius_step = 2.0
    calculate_polar_coords(data, 0, 2 * math.pi, 0, radius_step)

    # 3. 收集对象
    nodes = []
    edges = []
    collect_draw_objects(data, nodes, edges, max_depth)

    # 4. 设置画布
    # 尺寸由最大半径决定
    max_radius = max_depth * radius_step
    # 画布大小 (inch), 加上 padding
    fig_size = max(15, max_radius * 1.5)
    # 限制最大尺寸
    if fig_size > 50: fig_size = 50

    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    ax.set_aspect('equal') # 保证是正圆
    ax.set_facecolor('#FAFAFA')

    # 5. 绘制连线
    for e in edges:
        ax.plot([e['x1'], e['x2']], [e['y1'], e['y2']], c='#AAAAAA', lw=0.6, alpha=0.4, zorder=1)

    # 6. 绘制节点和文字
    for n in nodes:
        # 画点
        ax.scatter(n['x'], n['y'], s=n['size'], c=[n['color']],
                   edgecolors='white', linewidths=0.5, zorder=2, alpha=0.9)

        # 画文字 (关键：旋转处理)
        if n['depth'] == 0:
            # 根节点不旋转
            ax.text(n['x'], n['y'], "ROOT", ha='center', va='center', fontsize=12, fontweight='bold', zorder=3)
        else:
            # 计算文字角度：将弧度转为角度
            deg = math.degrees(n['angle'])

            # 逻辑：左半圆(90~270度)的文字要翻转180度，防止倒着读
            # 同时调整对齐方式
            rotation = deg
            ha = 'left'

            if 90 < deg <= 270:
                rotation += 180
                ha = 'right'
                # 稍微往圆心方向偏移一点点，防止压在点上
                offset_x = -0.15
            else:
                ha = 'left'
                offset_x = 0.15

            # 文本截断
            label_txt = n['label']
            if len(label_txt) > 15: label_txt = label_txt[:14] + "."

            # 极坐标下的文字位置微调 (沿径向向外推)
            # 简单的 x, y 偏移可能不够，这里直接在点旁边画
            # 为了美观，我们沿半径方向稍微延伸一点写字
            text_dist = n['size'] / 3000.0 + 0.1 # 动态距离
            text_x = n['x'] + math.cos(n['angle']) * text_dist
            text_y = n['y'] + math.sin(n['angle']) * text_dist

            # 如果是左半球，因为是右对齐，不需要额外加距离，反而可能要减

            ax.text(n['x'], n['y'], label_txt,
                    rotation=rotation, ha=ha, va='center',
                    rotation_mode='anchor', # 关键：围绕锚点旋转
                    fontsize=n['font_size'], color='#333333', zorder=3)

    # 7. 去除坐标轴
    limit = max_radius + 2.0 # 留边距
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.axis('off')

    plt.title(f"Radial Tree: {filename}", fontsize=20, y=0.98)

    try:
        plt.savefig(output_path, dpi=120, bbox_inches='tight')
        plt.close()
        print(f"  -> 已生成: {output_path}")
    except Exception as e:
        print(f"  [Error] 保存失败: {e}")

def main():
    if not OUTPUT_DIR.exists():
        print(f"[ERROR] 目录不存在: {OUTPUT_DIR}")
        return

    for fname in TARGET_FILES:
        visualize_file(fname)

    print("\n✅ 所有圆形可视化图表已生成。")

if __name__ == "__main__":
    main()
