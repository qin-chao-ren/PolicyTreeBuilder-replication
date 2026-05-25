#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
圆形辐射树状图可视化
生成静态图像版本

python scripts/radial_tree_visualization.py -i data/final_tree/v4_tree_final.json -o figures
"""


import json
import math
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
import platform
import argparse

# --- 字体设置 ---
system_name = platform.system()
if system_name == 'Windows':
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'sans-serif']
elif system_name == 'Darwin':
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC', 'sans-serif']
else:
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
# ----------------

def load_json_data(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def preprocess_tree_depth(node, current_depth=0):
    """
    预处理：递归计算每个节点的实际拓扑深度，不依赖json里的'level'字段
    """
    node['_depth'] = current_depth
    max_d = current_depth

    if node.get('children'):
        for child in node['children']:
            child_d = preprocess_tree_depth(child, current_depth + 1)
            if child_d > max_d:
                max_d = child_d
    return max_d

def count_leaves(node):
    """计算叶子节点权重"""
    if not node.get('children') or len(node['children']) == 0:
        return 1
    return sum(count_leaves(child) for child in node['children'])

def build_adaptive_layout(root_data, max_depth):
    """构建自适应布局"""
    nodes = []
    edges = []

    # 动态参数设置
    # 半径步长：每加一层，半径增加的距离
    RADIUS_STEP = 200
    if max_depth > 6: RADIUS_STEP = 180 # 层级太深稍微紧凑一点

    def assign_pos(node, start_angle, end_angle, parent_pos=None):
        depth = node['_depth']

        # 1. 计算位置
        if depth == 0: # ROOT
            pos = (0, 0)
            angle = 0
        else:
            radius = depth * RADIUS_STEP
            angle = (start_angle + end_angle) / 2
            pos = (radius * math.cos(angle), radius * math.sin(angle))

        # 2. 动态样式
        # 字体大小随深度递减，最小不小于6
        font_size = max(14 - depth * 1.5, 5)
        # 节点大小随深度递减
        node_size = max(800 - depth * 100, 50)

        # 颜色：使用 colormap (viridis) 根据深度生成颜色
        color_val = depth / max_depth if max_depth > 0 else 0
        color = cm.viridis(color_val) # 你也可以换成 cm.coolwarm, cm.plasma 等

        nodes.append({
            'label': node.get('label', ''),
            'pos': pos,
            'angle': angle,
            'depth': depth,
            'color': color,
            'size': node_size,
            'font_size': font_size
        })

        # 添加边
        if parent_pos is not None:
            edges.append((parent_pos, pos))

        # 3. 递归处理子节点
        children = node.get('children', [])
        if children:
            total_leaves = sum(count_leaves(c) for c in children)
            if total_leaves > 0:
                angle_span = end_angle - start_angle
                angle_per_leaf = angle_span / total_leaves

                curr_angle = start_angle
                for child in children:
                    c_leaves = count_leaves(child)
                    c_span = c_leaves * angle_per_leaf
                    assign_pos(child, curr_angle, curr_angle + c_span, pos)
                    curr_angle += c_span

    # 根节点分配 0 到 2pi
    assign_pos(root_data, 0, 2 * math.pi)
    return nodes, edges, RADIUS_STEP

def plot_adaptive_tree(nodes, edges, max_depth, radius_step):
    """绘制函数"""

    # 动态计算画布大小
    # 假设每层半径200，9层就是1800半径，直径3600。
    # 为了保证清晰度，画布尺寸(inch) = 直径 / DPI * 缩放因子
    # 这是一个非常巨大的图，我们设置得大一点
    canvas_size = max(20, max_depth * 4)
    fig, ax = plt.subplots(1, 1, figsize=(canvas_size, canvas_size))
    ax.set_aspect('equal')
    fig.patch.set_facecolor('#FAFAFA')
    ax.set_facecolor('#FAFAFA') # 稍微带点灰的白，护眼

    # 1. 绘制同心圆参考线
    for d in range(1, max_depth + 1):
        r = d * radius_step
        circle = plt.Circle((0, 0), r, fill=False, edgecolor='#dddddd',
                          linestyle='--', linewidth=1, alpha=0.5)
        ax.add_patch(circle)
        # 添加层级标记
        ax.text(0, -r, f"Level {d}", ha='center', va='center',
                color='#999999', fontsize=8, alpha=0.7)

    # 2. 绘制连线 (贝塞尔曲线或直线，这里用直线保证性能)
    for p1, p2 in edges:
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], c='#bbbbbb', alpha=0.5, linewidth=0.8)

    # 3. 绘制节点和文字
    for node in nodes:
        x, y = node['pos']
        depth = node['depth']

        # 绘制点
        ax.scatter(x, y, s=node['size'], color=node['color'],
                  edgecolor='white', linewidth=1, zorder=10, alpha=0.9)

        # 绘制文字
        if depth == 0:
            # 根节点文字
            ax.text(x, y, node['label'], ha='center', va='center',
                   fontsize=16, fontweight='bold', color='white')
        else:
            # 计算文字旋转角度
            deg = math.degrees(node['angle'])

            # 逻辑：左半边的字左对齐，右半边的字右对齐，防止倒着读
            if 90 < deg <= 270:
                rotation = deg + 180
                ha = 'right'
                text_offset_x = - (node['size']**0.5 / 2 + 5)
            else:
                rotation = deg
                ha = 'left'
                text_offset_x = (node['size']**0.5 / 2 + 5)

            # 稍微偏移一点，不要压在点上
            # 这里做一个简化的偏移计算，沿着半径方向
            offset_dist = 15 + node['size']/20
            text_x = x + math.cos(node['angle']) * offset_dist
            text_y = y + math.sin(node['angle']) * offset_dist

            # 如果是最后一层，字可能会很密，把字稍微放远一点或者改颜色
            text_color = '#333333'

            ax.text(text_x, text_y, node['label'],
                   ha=ha, va='center', rotation=rotation, rotation_mode='anchor',
                   fontsize=node['font_size'], color=text_color)

    limit = (max_depth + 1) * radius_step
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.axis('off')
    plt.title(f'全层级辐射树状图 (深度: {max_depth})', fontsize=24, pad=40)
    plt.tight_layout()
    return fig

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default='v4_tree_viz_ready.json')
    parser.add_argument('-o', '--output', type=str, default='outputs')
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"找不到文件: {args.input}")
        return

    print("1. 正在加载数据...")
    data = load_json_data(args.input)

    print("2. 正在分析树结构深度...")
    max_depth = preprocess_tree_depth(data)
    print(f"   检测到最大层级深度: {max_depth}")

    print("3. 计算布局坐标...")
    nodes, edges, step = build_adaptive_layout(data, max_depth)

    print("4. 正在绘图 (由于节点较多，这可能需要几秒钟)...")
    fig = plot_adaptive_tree(nodes, edges, max_depth, step)

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    save_path = os.path.join(args.output, 'adaptive_tree_viz3.png')
    svg_path = os.path.join(args.output, 'adaptive_tree_viz3.svg')

    print(f"5. 保存图片至: {save_path}")
    # 增加 dpi 以获得高清大图
    fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='#FAFAFA')
    fig.savefig(svg_path, format='svg', bbox_inches='tight')

    print("完成！请打开图片查看细节。")
    # plt.show() # 如果图太大，弹窗可能会卡死，建议直接看文件

if __name__ == "__main__":
    main()
