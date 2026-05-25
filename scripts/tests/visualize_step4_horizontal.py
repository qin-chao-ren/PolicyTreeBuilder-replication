#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 4 测试结果可视化 (水平展开版)
基于叶子节点权重的自适应布局，保证节点不重叠。
运行：python scripts/tests/visualize_step4_horizontal.py
"""

import json
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import platform
from pathlib import Path

# --- 1. 路径锚点 ---
HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
TEST_DIR = PROJECT_ROOT / "data" / "intermediate_outputs" / "test"

TARGET_FILES = [
    "test_tree_input.json",
    "test_tree_s1.json",
    "test_tree_s2.json",
    "test_tree_s3.json",
    "test_tree_final.json"
]

# --- 2. 字体设置 (沿用你提供的优秀配置) ---
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

# --- 3. 核心布局算法 (修改为水平笛卡尔坐标) ---

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
    """
    递归计算坐标 (Left-to-Right)
    x: 由深度决定
    y: 由叶子节点的堆叠顺序决定
    """
    # X 坐标很简单，就是层级
    node['_x'] = node['_depth'] * x_step

    children = node.get('children', [])

    if not children:
        # 叶子节点：分配下一个可用的 Y 坐标
        # 注意：为了符合直觉（上到下），我们通常用负数或者反转 Y
        # 这里我们用 current_leaf_y[0] 递增，画图时 Y 轴不需要反转，因为 matplotlib 默认左下角是原点
        # 但树通常习惯根在上或左上。为了让根在左侧中间，我们后续统一调整。
        # 这里先简单堆叠。
        node['_y'] = current_leaf_y[0] * y_step
        current_leaf_y[0] += 1
    else:
        # 非叶子节点：先处理所有孩子
        child_ys = []
        for child in children:
            calculate_positions(child, current_leaf_y, x_step, y_step)
            child_ys.append(child['_y'])

        # 父节点的 Y 坐标等于所有子节点 Y 坐标的中心
        # 这样父节点正好位于子树的垂直中间
        node['_y'] = sum(child_ys) / len(child_ys)

def collect_draw_data(node, nodes_list, edges_list, max_depth):
    """收集绘图所需的扁平化数据"""

    # 颜色根据深度渐变 (使用 viridis)
    color_val = node['_depth'] / max_depth if max_depth > 0 else 0
    color = cm.viridis(color_val) # 或者 cm.tab10, cm.Set3

    # 节点大小随深度递减
    size = max(600 - node['_depth'] * 80, 150)

    # 字体大小
    font_size = max(12 - node['_depth'], 8)

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

# --- 4. 绘图主逻辑 ---

def visualize_file(filename):
    input_path = TEST_DIR / filename
    output_path = TEST_DIR / f"{input_path.stem}.png"

    print(f"处理中: {filename} ...")
    data = load_json_data(input_path)
    if not data:
        print(f"  [跳过] 文件不存在: {input_path}")
        return

    # 1. 统计与预处理
    max_depth, total_leaves = get_tree_stats(data)

    # 2. 计算坐标
    # x_step 加大一点，留出空间写长标题
    # y_step 1.0
    # mutable list hack to pass integer by reference
    leaf_y_counter = [0]
    calculate_positions(data, leaf_y_counter, x_step=4.0, y_step=1.0)

    # 3. 收集绘图对象
    nodes = []
    edges = []
    collect_draw_data(data, nodes, edges, max_depth)

    # 4. 设置画布
    # 高度由叶子数量决定，宽度由深度决定
    # 加上边距
    fig_height = max(8, total_leaves * 0.4)
    fig_width = max(12, max_depth * 3.5)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.set_facecolor('#FAFAFA') # 护眼灰白

    # 5. 绘制连线 (先画线，压在点下面)
    for e in edges:
        # 直接直线
        ax.plot([e['x1'], e['x2']], [e['y1'], e['y2']], c='#BBBBBB', lw=1.0, alpha=0.6, zorder=1)

    # 6. 绘制节点
    for n in nodes:
        ax.scatter(n['x'], n['y'], s=n['size'], c=[n['color']],
                   edgecolors='white', linewidths=1.5, zorder=2, alpha=1.0)

        # 标签处理
        # 根节点和L1节点标签可以稍大
        display_label = n['label']
        if len(display_label) > 15: # 截断过长的
             display_label = display_label[:14] + ".."

        # 文字在点的右侧
        ax.text(n['x'] + 0.15, n['y'], display_label,
                va='center', ha='left', fontsize=n['font_size'],
                color='#333333', fontweight='normal', zorder=3)

        # 显示 Level 标记 (可选，字很小放在点上方)
        # ax.text(n['x'], n['y'] + 0.3, n['level'],
        #         va='bottom', ha='center', fontsize=6, color='#999999')

    # 7. 调整坐标轴与边距
    # 反转 Y 轴，让第一个叶子在最上面（符合阅读习惯）
    # 但由于我们的算法是 leaf_counter 递增，也就是 0, 1, 2...
    #如果不反转，0在最下。所以我们反转一下，或者在计算时用减法。
    # 这里直接用 ylim 控制

    # 留白
    pad_x = 1.0
    pad_y = 1.0
    max_x = max(n['x'] for n in nodes)
    max_y = max(n['y'] for n in nodes)
    min_y = min(n['y'] for n in nodes)

    ax.set_xlim(-0.5, max_x + 5.0) # 右边留多点给文字
    ax.set_ylim(min_y - pad_y, max_y + pad_y)

    ax.axis('off') # 隐藏坐标轴
    plt.title(f"Tree Structure: {filename} (Depth: {max_depth}, Leaves: {total_leaves})", fontsize=16, pad=20)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  -> 已生成: {output_path}")

def main():
    if not TEST_DIR.exists():
        print(f"[ERROR] 测试目录不存在: {TEST_DIR}")
        return

    for fname in TARGET_FILES:
        visualize_file(fname)

    print("\n✅ 所有可视化图表已生成。")

if __name__ == "__main__":
    main()
