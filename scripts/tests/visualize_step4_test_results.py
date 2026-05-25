#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualization Tool for Step 4 Mini Test
功能：将 Step 4 测试过程中的 5 个 JSON 树文件渲染为从左到右的层级图
依赖：pip install networkx matplotlib
运行：python scripts/tests/visualize_step4_test_results.py
"""

import json
import sys
import platform
import matplotlib.pyplot as plt
import networkx as nx
from pathlib import Path

# --- 1. 路径锚点 ---
HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
TEST_DIR = PROJECT_ROOT / "data" / "intermediate_outputs" / "test"

# 需要可视化的文件列表
TARGET_FILES = [
    "test_tree_input.json",
    "test_tree_s1.json",
    "test_tree_s2.json",
    "test_tree_s3.json",
    "test_tree_final.json"
]

# --- 2. 字体设置 (解决中文乱码) ---
def set_chinese_font():
    system = platform.system()
    if system == "Windows":
        # 优先尝试微软雅黑，其次黑体
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial']
    elif system == "Darwin": # Mac
        plt.rcParams['font.sans-serif'] = ['PingFang HK', 'Arial Unicode MS']
    else: # Linux
        plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

# --- 3. 自定义树布局算法 (从左到右) ---
def hierarchy_pos(G, root=None, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5):
    """
    这是一个递归布局算法，用于生成美观的树状结构坐标。
    修改版：X轴代表深度，Y轴代表节点排列。
    """
    if not nx.is_tree(G):
        # 如果不是树（比如有孤岛），回退到 shell layout
        return nx.shell_layout(G)

    pos = _hierarchy_pos(G, root, width, vert_gap, vert_loc, xcenter)
    # 旋转坐标系：将默认的 Top-Down 转为 Left-Right
    # 原逻辑: x=横向位置, y=深度(负数)
    #以此变为: x=深度(正数), y=横向位置
    return {u: (v[1], v[0]) for u, v in pos.items()}

def _hierarchy_pos(G, root, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5, pos=None, parent=None, parsed=None):
    if pos is None:
        pos = {root: (xcenter, vert_loc)}
    else:
        pos[root] = (xcenter, vert_loc)

    if parsed is None:
        parsed = []

    children = list(G.neighbors(root))
    if not isinstance(G, nx.DiGraph) and parent is not None:
        children.remove(parent)

    if len(children) != 0:
        dx = width / len(children)
        nextx = xcenter - width/2 - dx/2
        for child in children:
            nextx += dx
            pos = _hierarchy_pos(G, child, width=dx, vert_gap=vert_gap,
                                 vert_loc=vert_loc-vert_gap, xcenter=nextx,
                                 pos=pos, parent=root, parsed=parsed)
    return pos

# --- 4. 构建 NetworkX 图 ---
def build_graph_from_json(json_path):
    if not json_path.exists():
        print(f"[WARN] File not found: {json_path}")
        return None

    data = json.loads(json_path.read_text(encoding='utf-8'))
    G = nx.DiGraph()

    # 颜色映射
    color_map = []
    level_colors = {
        "ROOT": "#D3D3D3", # 灰色
        "L1": "#FF6B6B",   # 红色
        "L2": "#4ECDC4",   # 青色
        "L3": "#45B7D1",   # 蓝色
        "L4": "#FFE66D",   # 黄色
        "Other": "#EEEEEE"
    }

    def add_node_recursive(node, parent_id=None):
        node_id = node.get('node_id')
        label = node.get('label', '')
        level = str(node.get('level', 'Other')).upper()
        if not level.startswith("L") and level != "ROOT": level = "Other"

        # 节点显示的文本：截断太长的 label
        display_label = label[:10] + ".." if len(label) > 10 else label
        display_text = f"{level}\n{display_label}"

        G.add_node(node_id, label=display_text, level=level)
        if parent_id:
            G.add_edge(parent_id, node_id)

        for child in node.get('children', []):
            add_node_recursive(child, node_id)

    add_node_recursive(data)

    # 生成颜色列表
    for node in G.nodes():
        lvl = G.nodes[node]['level']
        color_map.append(level_colors.get(lvl, "#EEEEEE"))

    return G, color_map, data.get('node_id') # 返回根节点ID

# --- 5. 绘图主函数 ---
def visualize_file(filename):
    input_path = TEST_DIR / filename
    output_path = TEST_DIR / f"{input_path.stem}.png"

    print(f"Processing {filename} ...")

    res = build_graph_from_json(input_path)
    if not res: return

    G, colors, root_id = res

    # 动态调整画布大小
    # 高度取决于叶子节点数量，宽度取决于层级深度
    # 计算叶子数量
    leaves = [x for x in G.nodes() if G.out_degree(x)==0 and G.in_degree(x)==1]
    depth = nx.dag_longest_path_length(G)

    fig_height = max(8, len(leaves) * 0.6) # 每个叶子预留 0.6 unit 高度
    fig_width = max(12, depth * 3)         # 每层预留 3 unit 宽度

    plt.figure(figsize=(fig_width, fig_height))

    # 布局计算
    # 使用 spring layout 辅助调整，或者使用自定义 tree layout
    # 这里我们使用 spectral 或者自定义的
    try:
        # 左到右布局: root=(0,0), children x增加
        pos = hierarchy_pos(G, root=root_id, width=len(leaves)*1.5, vert_gap=1.0)
    except:
        # 兜底：如果树结构断裂
        pos = nx.spring_layout(G)

    # 绘制
    # 1. 边
    nx.draw_networkx_edges(G, pos, edge_color='#AAAAAA', arrows=True, arrowstyle='-|>', arrowsize=15)

    # 2. 节点
    nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=2500, alpha=0.9, node_shape='o')

    # 3. 标签
    labels = nx.get_node_attributes(G, 'label')
    nx.draw_networkx_labels(G, pos, labels, font_size=9, font_family=plt.rcParams['font.sans-serif'][0])

    plt.title(f"Tree Visualization: {filename}", fontsize=16)
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  -> Saved to {output_path}")

def main():
    if not TEST_DIR.exists():
        print(f"[ERROR] Test directory not found: {TEST_DIR}")
        print("Please run 'run_step4_mini_test.py' first.")
        return

    set_chinese_font()

    for fname in TARGET_FILES:
        visualize_file(fname)

    print("\n✅ All visualizations completed.")

if __name__ == "__main__":
    main()
