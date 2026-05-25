#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Production Tree Visualization (Elegant Radial)
全量生产数据可视化 - 莫兰迪精美版
功能：将 v4 流程产生的关键 JSON 树渲染为学术级精美辐射图。
运行：python scripts/visualize_production_radial_elegant.py
"""

import json
import math
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from pathlib import Path
import platform

# --- 1. 基础环境配置 ---
sys.setrecursionlimit(5000) # 防止全量树递归溢出

# 路径锚点
HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent  # roundC_v4/
OUTPUT_DIR = PROJECT_ROOT / "data" / "intermediate_outputs"

# 目标文件配置 (文件名, 图表标题)
TARGET_FILES = [
    ("v4_tree_coarse_global.json", "Step 3: 粗糙树 (Coarse Global)"),
    ("v4_tree_s4_2.json",          "Step 4.2: 结构塑形 (Shaping)"),
    ("v4_tree_final.json",         "Step 4.5: 最终交付树 (Final Structure)")
]

# --- 2. 样式配置 (莫兰迪色系) ---

# 字体配置
system_name = platform.system()
if system_name == 'Windows':
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial']
elif system_name == 'Darwin':
    plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Arial Unicode MS']
else:
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# 配色方案
COLORS = {
    'background': '#F7F5F3',      # 温暖米白背景
    'ROOT': '#8B7E74',            # 莫兰迪灰棕
    'L1': '#5B8FB9',              # 雾霾蓝
    'L2': '#7FB5B5',              # 莫兰迪青
    'L3': '#D4A5A5',              # 莫兰迪粉
    'L4': '#C9B79C',              # 莫兰迪棕
    'L5': '#999999',              # 兜底灰
    'text': '#4A4543',            # 深灰文字
    'text_light': '#7A7573',      # 浅灰文字
    'line': '#C5BEB8',            # 连线色
    'line_alpha': 0.4,
}

# 节点大小配置
NODE_SIZES = {
    'ROOT': 900,
    'L1': 500,
    'L2': 250,
    'L3': 120,
    'L4': 60,
    'L5': 30
}

# 字体大小配置
FONT_SIZES = {
    'ROOT': 16,
    'L1': 12,
    'L2': 10,
    'L3': 8,
    'L4': 6,
    'L5': 5
}

# 半径步长 (决定圆环间距)
RADIUS_STEP = 3.0

# 文字截断长度
MAX_LABEL_LENGTH = 20


# ============ 3. 核心算法 ============

def load_json(filepath):
    """加载JSON文件"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def count_leaves(node):
    """递归计算叶子节点数量（用于扇区分配权重）"""
    children = node.get('children', [])
    if not children:
        node['_leaves'] = 1
        return 1
    total = sum(count_leaves(c) for c in children)
    node['_leaves'] = total
    return total


def get_max_depth(node, depth=0):
    """计算最大深度"""
    children = node.get('children', [])
    if not children:
        return depth
    return max([get_max_depth(c, depth + 1) for c in children] or [depth])


def calculate_polar_layout(node, start_angle, end_angle, depth, radius_step):
    """
    计算极坐标布局
    - start_angle, end_angle: 弧度制
    - depth: 当前深度
    """
    # 计算当前节点位置 (扇区中心)
    mid_angle = (start_angle + end_angle) / 2
    radius = depth * radius_step

    # 转笛卡尔坐标
    x = radius * math.cos(mid_angle)
    y = radius * math.sin(mid_angle)

    node['_x'] = x
    node['_y'] = y
    node['_angle'] = mid_angle
    node['_radius'] = radius
    node['_depth'] = depth

    # 递归处理子节点
    children = node.get('children', [])
    if not children:
        return

    total_leaves = node['_leaves']
    total_span = end_angle - start_angle
    current_start = start_angle

    for child in children:
        child_leaves = child['_leaves']
        # 根据子树规模分配角度宽度
        wedge_size = (child_leaves / total_leaves) * total_span
        calculate_polar_layout(child, current_start, current_start + wedge_size, depth + 1, radius_step)
        current_start += wedge_size


def collect_elements(node, nodes_list, edges_list):
    """收集所有节点和边"""
    level = node.get('level', 'L5')
    if level not in COLORS: level = 'L5' # 兜底

    nodes_list.append({
        'x': node['_x'],
        'y': node['_y'],
        'angle': node['_angle'],
        'depth': node['_depth'],
        'label': node.get('label', ''),
        'level': level,
        'color': COLORS.get(level, COLORS['L5']),
        'size': NODE_SIZES.get(level, 50),
        'font_size': FONT_SIZES.get(level, 6),
    })

    for child in node.get('children', []):
        edges_list.append({
            'x1': node['_x'], 'y1': node['_y'],
            'x2': child['_x'], 'y2': child['_y'],
        })
        collect_elements(child, nodes_list, edges_list)


def truncate_label(label, max_len=MAX_LABEL_LENGTH):
    """截断过长标签"""
    if len(label) > max_len:
        return label[:max_len-1] + '…'
    return label


def visualize_tree(data, output_path, title="Radial Tree"):
    """
    生成精美的放射状树形图
    """
    # 1. 数据预处理
    count_leaves(data)
    max_depth = get_max_depth(data)
    # 布局计算 (0 到 2pi)
    calculate_polar_layout(data, 0, 2 * math.pi, 0, RADIUS_STEP)

    # 2. 收集绘图元素
    nodes = []
    edges = []
    collect_elements(data, nodes, edges)

    print(f"  > 节点: {len(nodes)} | 边: {len(edges)} | 深度: {max_depth}")

    # 3. 创建画布 (根据深度动态调整大小)
    max_radius = max_depth * RADIUS_STEP
    fig_size = max(16, max_radius * 2.0)
    fig_size = min(fig_size, 50)  # 限制最大尺寸防止内存爆炸

    fig, ax = plt.subplots(figsize=(fig_size, fig_size), dpi=150)
    ax.set_facecolor(COLORS['background'])
    fig.patch.set_facecolor(COLORS['background'])

    # 4. 绘制装饰性同心圆（辅助线）
    for d in range(1, max_depth + 1):
        circle = plt.Circle((0, 0), d * RADIUS_STEP,
                           fill=False,
                           color=COLORS['line'],
                           alpha=0.15,
                           linewidth=0.5,
                           linestyle='--')
        ax.add_patch(circle)

    # 5. 绘制连线（底层）
    for e in edges:
        ax.plot([e['x1'], e['x2']], [e['y1'], e['y2']],
               color=COLORS['line'],
               linewidth=0.8,
               alpha=COLORS['line_alpha'],
               zorder=1)

    # 6. 绘制节点（中层）
    for n in nodes:
        ax.scatter(n['x'], n['y'],
                  s=n['size'],
                  c=n['color'],
                  edgecolors='white',
                  linewidths=1.5,
                  zorder=3,
                  alpha=0.95)

    # 7. 绘制文字（顶层，智能旋转）
    for n in nodes:
        if n['level'] == 'ROOT' or n['depth'] == 0:
            # 根节点居中显示，不旋转
            ax.text(n['x'], n['y'], 'ROOT',
                   ha='center', va='center',
                   fontsize=n['font_size'],
                   fontweight='bold',
                   color='white',
                   zorder=4)
            continue

        # 计算文字角度
        angle_deg = math.degrees(n['angle'])

        # 文字位置：沿径向稍微外移，避免压住点
        # 动态计算偏移量：点越大，偏移越多
        offset_dist = 0.2 + (n['size'] / 4000)
        text_x = n['x'] + math.cos(n['angle']) * offset_dist
        text_y = n['y'] + math.sin(n['angle']) * offset_dist

        # 智能旋转：左半圆(90~270度)的文字要翻转180度，防止倒着读
        if 90 < angle_deg <= 270:
            rotation = angle_deg + 180
            ha = 'right'
        else:
            rotation = angle_deg
            ha = 'left'

        # 截断标签
        label = truncate_label(n['label'], MAX_LABEL_LENGTH)

        # 绘制
        ax.text(text_x, text_y, label,
               rotation=rotation,
               ha=ha, va='center',
               rotation_mode='anchor', # 关键：围绕锚点旋转
               fontsize=n['font_size'],
               color=COLORS['text'] if n['depth'] <= 2 else COLORS['text_light'],
               fontweight='bold' if n['depth'] <= 1 else 'normal',
               zorder=4)

    # 8. 设置坐标轴与图例
    limit = max_radius + 3.0
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_aspect('equal')
    ax.axis('off')

    # 标题
    plt.suptitle(title, fontsize=24, fontweight='bold', color=COLORS['text'], y=0.95)

    # 图例
    legend_elements = []
    for level, color in [('L1 一级分类', COLORS['L1']),
                         ('L2 二级分类', COLORS['L2']),
                         ('L3 三级节点', COLORS['L3']),
                         ('L4 基础节点', COLORS['L4'])]:
        legend_elements.append(plt.scatter([], [], c=color, s=100,
                                          edgecolors='white', linewidths=1,
                                          label=level))

    ax.legend(handles=legend_elements,
             loc='upper left',
             fontsize=12,
             frameon=True,
             facecolor=COLORS['background'],
             edgecolor=COLORS['line'],
             framealpha=0.9,
             bbox_to_anchor=(0.02, 0.98)) # 图例位置微调

    # 9. 保存图片
    try:
        plt.savefig(output_path,
                   dpi=150,
                   bbox_inches='tight',
                   facecolor=COLORS['background'],
                   edgecolor='none',
                   pad_inches=0.5)
        plt.close()
        print(f"  ✓ 已保存: {output_path}")
    except Exception as e:
        print(f"  [ERROR] 保存失败: {e}")


def main():
    """主函数"""
    if not OUTPUT_DIR.exists():
        print(f"[ERROR] 输出目录不存在: {OUTPUT_DIR}")
        return

    print(f"开始生成精美可视化图表 (Output: {OUTPUT_DIR})...")

    for filename, title in TARGET_FILES:
        input_path = OUTPUT_DIR / filename
        output_path = OUTPUT_DIR / f"{Path(filename).stem}_elegant.png"

        if not input_path.exists():
            print(f"[跳过] 文件不存在: {input_path}")
            continue

        print(f"\n正在处理: {filename} ...")
        try:
            data = load_json(input_path)
            visualize_tree(data, output_path, title)
        except Exception as e:
            print(f"  [ERROR] 处理出错: {e}")

    print("\n✅ 所有图表生成完毕！请查看 outputs 目录。")


if __name__ == "__main__":
    main()