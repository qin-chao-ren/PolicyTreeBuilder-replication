'''
python scripts/111.py
'''


import json
import pandas as pd
import os
from collections import defaultdict

# --- 这里指向新生成的文件 ---
INPUT_FILE = 'data/intermediate_outputs/v5_tree_new_structure.json'
OUTPUT_FILE = 'data/intermediate_outputs/v5_tree_statistics.xlsx'

def load_data(filepath):
    if not os.path.exists(filepath):
        print(f"错误: 找不到文件 {filepath}")
        return None
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_all_nodes_flat(node, l1_branch_name="ROOT", parent_name="N/A", result_list=None):
    if result_list is None: result_list = []
    
    current_level = node.get('level', '')
    current_label = node.get('label', '')
    
    # 确定 L1 分支
    current_branch = l1_branch_name
    if parent_name == "ROOT": 
        current_branch = current_label
    
    node_info = {
        'ID': node.get('node_id', ''),
        '新层级': current_level,
        '名称': current_label,
        '所属新板块': current_branch,
        '父节点': parent_name
    }
    result_list.append(node_info)
    
    for child in node.get('children', []):
        get_all_nodes_flat(child, current_branch, current_label, result_list)
    return result_list

def count_levels_in_branch(node):
    counts = defaultdict(int)
    def _recurse(n):
        counts[n.get('level', 'Unknown')] += 1
        for child in n.get('children', []):
            _recurse(child)
    _recurse(node)
    return counts

def main():
    data = load_data(INPUT_FILE)
    if not data: return

    print("正在计算新结构的统计数据...")
    stats_rows = []
    
    # 现在的 children 就是那 5 个新板块
    l1_branches = data.get('children', [])
    
    for branch in l1_branches:
        branch_name = branch.get('label', 'Unnamed')
        counts = count_levels_in_branch(branch)
        
        row = {
            '新 L1 板块': branch_name,
            'L1 (自身)': counts.get('L1', 0),
            'L2 (原L1)': counts.get('L2', 0),
            'L3 (原L2)': counts.get('L3', 0),
            'L4 (原L3)': counts.get('L4', 0),
            'L5 (原L4)': counts.get('L5', 0), # 结构变深了，可能会有L5
            '板块总节点数': sum(counts.values())
        }
        stats_rows.append(row)
    
    df_stats = pd.DataFrame(stats_rows)
    
    print("正在生成明细...")
    all_nodes = get_all_nodes_flat(data, l1_branch_name="N/A", parent_name="N/A")
    df_details = pd.DataFrame(all_nodes)
    
    print(f"正在写入 Excel: {OUTPUT_FILE} ...")
    with pd.ExcelWriter(OUTPUT_FILE, engine='openpyxl') as writer:
        df_stats.to_excel(writer, sheet_name='五大板块统计', index=False)
        df_details.to_excel(writer, sheet_name='全节点明细', index=False)
            
    print("完成！")

if __name__ == "__main__":
    main()