#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 6 (Admin) · 批量可视化行政单位树
功能：自动扫描 trees_by_admin 文件夹，为每一个行政单位生成高清辐射图。
"""

import json
import math
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import platform
import argparse
from pathlib import Path

# === 1. 配置路径 (修改点) ===
# 指向 Step 5 生成的具体行政单位树的文件夹
INPUT_DIR = Path(r"<LOCAL_SOURCE_ROOT>/data\intermediate_outputs\trees_by_admin")

# 输出图片存放的文件夹
OUTPUT_DIR = INPUT_DIR  # 默认存在同一个文件夹，也可以改为 Path(r"...\outputs\viz_admin")

# --- 字体设置 ---
system_name = platform.system()
if system_name == 'Windows':
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
    node['_depth'] = current_depth
    max_d = current_depth
    if node.get('children'):
        for child in node['children']:
            child_d = preprocess_tree_depth(child, current_depth + 1)
            if child_d > max_d:
                max_d = child_d
    return max_d

def count_leaves(node):
    if not node.get('children') or len(node['children']) == 0:
        return 1
    return sum(count_leaves(child) for child in node['children'])

def build_adaptive_layout(root_data, max_depth):
    nodes = []
    edges = []
    RADIUS_STEP = 200   
    if max_depth > 6: RADIUS_STEP = 180 
    
    def assign_pos(node, start_angle, end_angle, parent_pos=None):
        depth = node['_depth']
        if depth == 0:
            pos = (0, 0)
            angle = 0
        else:
            radius = depth * RADIUS_STEP
            angle = (start_angle + end_angle) / 2
            pos = (radius * math.cos(angle), radius * math.sin(angle))
        
        font_size = max(16 - depth * 1.8, 6)
        node_size = max(900 - depth * 120, 60)
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
        
        if parent_pos is not None:
            edges.append((parent_pos, pos))
        
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

    assign_pos(root_data, 0, 2 * math.pi)
    return nodes, edges, RADIUS_STEP

def plot_adaptive_tree(nodes, edges, max_depth, radius_step, title_text):
    canvas_size = max(24, max_depth * 4.5) 
    fig, ax = plt.subplots(1, 1, figsize=(canvas_size, canvas_size))
    ax.set_aspect('equal')
    fig.patch.set_facecolor('#FFFFFF')
    ax.set_facecolor('#FFFFFF')
    
    for d in range(1, max_depth + 1):
        r = d * radius_step
        circle = plt.Circle((0, 0), r, fill=False, edgecolor='#eeeeee', linestyle='-', linewidth=1.5, alpha=0.6)
        ax.add_patch(circle)
        ax.text(0, -r + 20, f"L{d}", ha='center', va='bottom', color='#cccccc', fontsize=10, fontweight='bold')

    for p1, p2 in edges:
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], c='#cccccc', alpha=0.4, linewidth=0.8)

    for node in nodes:
        x, y = node['pos']
        depth = node['depth']
        ax.scatter(x, y, s=node['size'], color=node['color'], edgecolor='white', linewidth=1.5, zorder=10, alpha=0.85)
        
        if depth == 0:
            ax.text(x, y, node['label'], ha='center', va='center', fontsize=18, fontweight='bold', color='white')
        else:
            deg = math.degrees(node['angle'])
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
            text_color = '#222222' if depth < max_depth else '#555555'
            
            ax.text(text_x, text_y, node['label'], ha=ha, va='center', rotation=rotation, 
                    rotation_mode='anchor', fontsize=node['font_size'], color=text_color, fontweight='medium')

    limit = (max_depth + 1) * radius_step
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.axis('off')
    plt.title(title_text, fontsize=30, pad=50, color='#333333', fontweight='bold')
    plt.tight_layout()
    return fig

def main():
    parser = argparse.ArgumentParser(description="批量可视化行政单位/分组树")
    parser.add_argument("--input-dir", default=str(INPUT_DIR), help="包含树 JSON 的目录")
    parser.add_argument("--output-dir", default=None, help="图片输出目录；默认与 input-dir 相同")
    parser.add_argument("--pattern", default="*.json", help="未指定 --files 时使用的 glob 模式")
    parser.add_argument("--files", nargs="*", default=None, help="只处理这些 JSON 文件名")
    parser.add_argument("--dpi", type=int, default=300, help="输出 PNG 的 DPI")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir

    print(f"=== 批量可视化行政单位树 ===")
    print(f"输入目录: {input_dir}")
    
    if not input_dir.exists():
        print(f"[错误] 找不到目录: {input_dir}")
        print("请先运行 Step 5 生成这些文件。")
        return

    if args.files:
        json_files = [input_dir / name for name in args.files]
    else:
        json_files = sorted(input_dir.glob(args.pattern))
    
    if not json_files:
        print("[警告] 目录下没有找到 .json 文件！")
        return

    print(f"找到 {len(json_files)} 个树文件，开始处理...\n")

    output_dir.mkdir(parents=True, exist_ok=True)

    for i, json_path in enumerate(json_files, 1):
        filename = json_path.name
        print(f"[{i}/{len(json_files)}] 处理: {filename}")
        
        # 1. 加载
        try:
            data = load_json_data(json_path)
        except Exception as e:
            print(f"    ❌ 读取失败: {e}")
            continue

        # 2. 分析深度
        max_depth = preprocess_tree_depth(data)
        if max_depth == 0:
            print("    ⚠️ 空树，跳过。")
            continue
        
        # 3. 布局
        nodes, edges, step = build_adaptive_layout(data, max_depth)
        
        # 4. 绘图标题 (去掉 tree_ 前缀和 .json 后缀)
        admin_name = filename.replace("tree_", "").replace(".json", "")
        if "provincial" in admin_name:
            title_str = "省级政策结构图"
        elif "city" in admin_name:
            title_str = "市级政策结构图"
        else:
            title_str = f"{admin_name} 政策结构图"
        
        fig = plot_adaptive_tree(nodes, edges, max_depth, step, title_str)
        
        # 5. 保存
        save_name = filename.replace('.json', '.png')
        save_path = output_dir / save_name
        
        print(f"    保存至: {save_path}")
        fig.savefig(save_path, dpi=args.dpi, bbox_inches='tight', facecolor='#FFFFFF')
        
        plt.close(fig)

    print("\n=== 全部完成 ===")

if __name__ == "__main__":
    main()
