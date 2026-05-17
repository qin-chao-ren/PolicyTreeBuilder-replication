'''
python scripts/222.py
'''

import json
import os

# --- 配置 ---
INPUT_FILE = 'data/intermediate_outputs/v4_tree_coarse_global.json'
OUTPUT_FILE = 'data/intermediate_outputs/v5_tree_new_structure.json'

# --- 映射逻辑：新 L1 -> 旧 L1 列表 ---
CATEGORY_MAPPING = {
    "枢纽基建与数字底座": [
        "基础设施与枢纽",
        "基础能力支撑",
        "数字基础设施"
    ],
    "市场主体与航线网络": [
        "市场主体与网络",
        "多元经营主体培育",
        "战略引资与资本合作",
        "增量市场拓展"
    ],
    "产业生态与区域协同": [
        "产业生态拓展",
        "区域协同与产业联动",
        "跨境物流服务"
    ],
    "营商环境与服务效能": [
        "通关与口岸效能",
        "营商环境优化",
        "服务管理优化",
        "资源与要素保障"
    ],
    "战略引领与创新转型": [
        "战略规划与治理",
        "智慧绿色转型"
    ]
}

def load_data(filepath):
    if not os.path.exists(filepath):
        print(f"错误: 找不到文件 {filepath}")
        return None
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def update_levels_recursive(node, current_depth):
    """
    递归更新节点的 level 属性
    ROOT -> depth 0
    L1 -> depth 1
    ...
    """
    # 更新当前节点的 level 标签
    if current_depth == 0:
        node['level'] = 'ROOT'
    else:
        node['level'] = f'L{current_depth}'
    
    # 递归更新子节点
    if 'children' in node:
        for child in node['children']:
            update_levels_recursive(child, current_depth + 1)

def reorganize_structure(original_data):
    # 1. 创建新的根节点
    new_root = {
        "node_id": "ROOT",
        "level": "ROOT",
        "label": "ROOT",
        "children": []
    }

    # 2. 建立旧节点查找表 (Label -> Node)
    original_children = original_data.get('children', [])
    node_lookup = {child['label']: child for child in original_children}
    
    # 3. 按照新映射构建结构
    for new_l1_label, old_l1_labels in CATEGORY_MAPPING.items():
        # 创建新的 L1 节点
        new_l1_node = {
            "node_id": f"NEW_L1_{abs(hash(new_l1_label))}", # 生成唯一ID
            "label": new_l1_label,
            "level": "L1",
            "children": []
        }
        
        # 将旧的 L1 节点挂载为子节点 (即变成 L2)
        for old_label in old_l1_labels:
            if old_label in node_lookup:
                old_node = node_lookup[old_label]
                new_l1_node['children'].append(old_node)
            else:
                print(f"警告: 原数据中未找到分类 '{old_label}'")
        
        new_root['children'].append(new_l1_node)

    # 4. 递归刷新所有节点的 Level 属性
    # 因为原来的 L1 变成了 L2，原来的 L2 变成了 L3，以此类推
    update_levels_recursive(new_root, 0)
    
    return new_root

def main():
    print(f"正在读取 {INPUT_FILE} ...")
    data = load_data(INPUT_FILE)
    if not data: return

    print("正在重组树结构...")
    new_tree = reorganize_structure(data)
    
    print(f"正在保存到 {OUTPUT_FILE} ...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(new_tree, f, ensure_ascii=False, indent=2)
    
    print("完成！")
    print(f"请使用 '{OUTPUT_FILE}' 运行可视化脚本。")

if __name__ == "__main__":
    main()