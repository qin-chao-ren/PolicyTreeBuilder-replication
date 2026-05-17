#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 6 · 可视化 (Visualization)
功能：读取 Step 5 生成的 Provincial/City 树，生成高清圆形辐射图。
"""

import json
import math
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
import platform
import argparse
from pathlib import Path

# === 1. 配置路径 (自动指向你刚才生成的位置) ===
DEFAULT_DIR = Path(r"<LOCAL_SOURCE_ROOT>/data\intermediate_outputs")
TARGET_FILES = [
    "v4_tree_provincial.json",
    "v4_tree_city.json"
    # 如果你想画全局树，可以在这里加上 "v4_tree_coarse_global.json"
]

# --- 字体设置 (防止中文乱码) ---
system_name = platform.system()
if system_name == 'Windows':
    # 优先尝试常见的中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'sans-serif']
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
    """预处理：递归计算每个节点的实际拓扑深度"""
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
    
    # 半径步长
    RADIUS_STEP = 200   
    if max_depth > 6: RADIUS_STEP = 180 
    
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
        # 字体大小随深度递减
        font_size = max(16 - depth * 1.8, 6)
        # 节点大小随深度递减
        node_size = max(900 - depth * 120, 60)
        
        # 颜色：根据深度生成
        color_val = depth / max_depth if max_depth > 0 else 0
        color = cm.viridis(color_val) 
        
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

def plot_adaptive_tree(nodes, edges, max_depth, radius_step, title_text):
    """绘制函数"""
    
    # 画布大小
    canvas_size = max(24, max_depth * 4.5) 
    fig, ax = plt.subplots(1, 1, figsize=(canvas_size, canvas_size))
    ax.set_aspect('equal')
    fig.patch.set_facecolor('#FFFFFF')
    ax.set_facecolor('#FFFFFF')
    
    # 1. 绘制同心圆参考线
    for d in range(1, max_depth + 1):
        r = d * radius_step
        circle = plt.Circle((0, 0), r, fill=False, edgecolor='#eeeeee', 
                            linestyle='-', linewidth=1.5, alpha=0.6)
        ax.add_patch(circle)
        # 标记 L1, L2...
        ax.text(0, -r + 20, f"L{d}", ha='center', va='bottom', 
                color='#cccccc', fontsize=10, fontweight='bold')

    # 2. 绘制连线
    for p1, p2 in edges:
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], c='#cccccc', alpha=0.4, linewidth=0.8)

    # 3. 绘制节点和文字
    for node in nodes:
        x, y = node['pos']
        depth = node['depth']
        
        # 绘制点
        ax.scatter(x, y, s=node['size'], color=node['color'], 
                   edgecolor='white', linewidth=1.5, zorder=10, alpha=0.85)
        
        # 绘制文字
        if depth == 0:
            ax.text(x, y, node['label'], ha='center', va='center', 
                    fontsize=18, fontweight='bold', color='white')
        else:
            deg = math.degrees(node['angle'])
            # 智能旋转文字
            if 90 < deg <= 270:
                rotation = deg + 180
                ha = 'right'
                text_offset_x = - (node['size']**0.5 / 2 + 8)
            else:
                rotation = deg
                ha = 'left'
                text_offset_x = (node['size']**0.5 / 2 + 8)
            
            offset_dist = 20 + node['size']/15
            text_x = x + math.cos(node['angle']) * offset_dist
            text_y = y + math.sin(node['angle']) * offset_dist

            text_color = '#222222'
            # 最后一层稍微淡一点
            if depth == max_depth:
                text_color = '#555555'
            
            ax.text(text_x, text_y, node['label'], 
                   ha=ha, va='center', rotation=rotation, rotation_mode='anchor',
                   fontsize=node['font_size'], color=text_color, fontweight='medium')

    limit = (max_depth + 1) * radius_step
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.axis('off')
    
    # 标题
    plt.title(title_text, fontsize=30, pad=50, color='#333333', fontweight='bold')
    plt.tight_layout()
    return fig

def main():
    # 自动处理 TARGET_FILES 列表中的所有文件
    print(f"=== 开始可视化 (输入目录: {DEFAULT_DIR}) ===")
    
    for filename in TARGET_FILES:
        input_path = DEFAULT_DIR / filename
        
        if not input_path.exists():
            print(f"⚠️ 跳过 (文件不存在): {filename}")
            continue
            
        print(f"\n>>> 正在处理: {filename}")
        
        # 1. 加载
        data = load_json_data(input_path)
        
        # 2. 分析深度
        max_depth = preprocess_tree_depth(data)
        print(f"    最大深度: {max_depth}")
        if max_depth == 0:
            print("    [提示] 这是一个空树或只有根节点，跳过绘图。")
            continue
        
        # 3. 布局
        nodes, edges, step = build_adaptive_layout(data, max_depth)
        
        # 4. 绘图
        # 提取文件名主体作为标题 (去掉 .json)
        title_str = filename.replace("v4_tree_", "").replace(".json", "").capitalize() + " Tree Structure"
        print(f"    正在绘图 (节点数: {len(nodes)})...")
        fig = plot_adaptive_tree(nodes, edges, max_depth, step, title_str)
        
        # 5. 保存
        save_name_png = filename.replace('.json', '.png')
        save_name_svg = filename.replace('.json', '.svg')
        
        save_path_png = DEFAULT_DIR / save_name_png
        save_path_svg = DEFAULT_DIR / save_name_svg
        
        print(f"    保存 PNG: {save_path_png}")
        fig.savefig(save_path_png, dpi=300, bbox_inches='tight', facecolor='#FFFFFF')
        
        # 如果需要 SVG (矢量图) 取消下面注释
        # fig.savefig(save_path_svg, format='svg', bbox_inches='tight')
        
        # 关闭图形释放内存
        plt.close(fig)

    print("\n=== 全部完成，请到 outputs 目录查看图片 ===")

if __name__ == "__main__":
    main()