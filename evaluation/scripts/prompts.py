"""
Prompt templates for node-level and path-level evaluation.
评分标准与方案严格一致。所有模型使用同一份 prompt。
"""

# ============ 节点级 ============
NODE_SYSTEM = """你是一名严格的政策行动框架评审员。你需要评价一棵政策行动树中的某一个节点在其局部结构中的合理性。
评分必须严格依据给定的1-5分标准，不要重写树，不要做大段解释，只输出JSON对象，不要输出```json```代码块标记。"""

NODE_USER_TEMPLATE = """请评价以下节点。

【当前节点】
{current_node_json}

【父节点】
{parent_node_json}

【兄弟节点列表】
{sibling_nodes_json}

【子节点列表】
{children_nodes_json}

【从ROOT到当前节点的完整路径】
{path_from_root_json}

【评分维度（每项1-5分）】

A1 vertical_coherence 垂直一致性:
  评价当前节点与父节点、子节点之间的语义上下位关系。
  5: 父子关系清晰，抽象层级自然递进，能概括其子节点。
  4: 父子关系基本成立，仅轻微粒度不均。
  3: 相关但不够紧密，存在语义偏移或概括不足。
  2: 勉强成立，存在概念跳跃或上下位倒置。
  1: 关系不成立或冲突。

A2 horizontal_coherence 水平一致性:
  评价当前节点与兄弟节点之间的并列关系。
  5: 兄弟基本互斥，粒度一致，覆盖父节点语义。
  4: 基本合理，存在轻微重叠或粒度不均。
  3: 一定重复或交叉，但不严重。
  2: 明显重复、粒度混杂或分类逻辑不清。
  1: 严重混乱，不应与兄弟并列。

A3 label_quality 标签质量:
  评价标签是否清晰、专业、简洁、准确。
  5: 标签专业、准确、简洁。
  4: 清晰规范，仅轻微冗余。
  3: 可理解但不够专业或过泛。
  2: 较模糊、冗长或难以判断含义。
  1: 错误、无意义或严重不适合。

【需检测的flags（可多选，无问题留空数组）】
- PARENT_CHILD_IDENTICAL: 父子标签完全相同或同义重复
- SIBLING_SEMANTIC_DUPLICATE: 与某兄弟高度重复，应合并
- LOW_ABSTRACTION_PARENT: 父节点不够抽象，无法概括兄弟集合
- SEMANTIC_DRIFT_CHILD: 当前节点与父主题明显不一致
- UNBALANCED_BRANCHING: 兄弟集规模严重失衡或父节点子节点过多
- GRANULARITY_JUMP: 与父节点粒度跨度过大
- LABEL_TOO_LONG: 标签过长(>25汉字)且含多个并列短语
- MIXED_ABSTRACTION_SIBLINGS: 兄弟节点抽象层级混杂

【输出格式（仅输出JSON，无其他内容）】
{{
  "node_id": "{node_id}",
  "node_label": "{node_label}",
  "scores": {{
    "vertical_coherence": <1-5整数>,
    "horizontal_coherence": <1-5整数>,
    "label_quality": <1-5整数>
  }},
  "flags": [<flag字符串列表>],
  "issue_summary": "<一句话概括主要问题；若无明显问题写'无明显问题'>",
  "evidence": {{
    "parent_label": "<父节点label>",
    "problematic_sibling_labels": [<有问题的兄弟label列表>],
    "problematic_child_labels": [<有问题的子label列表>]
  }},
  "suggested_fix_type": "<KEEP|MERGE_WITH_SIBLING|MOVE_TO_OTHER_PARENT|SPLIT_NODE|RENAME|COMPRESS_PARENT_CHILD|INSERT_INTERMEDIATE_NODE>",
  "suggested_fix_note": "<不超过80字>"
}}
"""

# ============ 路径级 ============
PATH_SYSTEM = """你是一名严格的政策行动框架评审员。你需要评价一棵政策行动树中的一条 ROOT-to-leaf 路径在语义递进上的合理性。
评分必须严格依据给定的1-5分标准，不要重写树，不要做大段解释，只输出JSON对象，不要输出```json```代码块标记。"""

PATH_USER_TEMPLATE = """请评价以下路径。

【完整路径（每个节点含 node_id / label / level / depth）】
{path_json}

【评分维度（每项1-5分）】

B1 path_coherence 路径语义连贯性:
  从上到下是否围绕同一政策主题逐层展开。
  5: 主题稳定，从L1到叶层层细化，逻辑清楚。
  4: 总体连贯，仅个别节点略显不自然。
  3: 基本相关，存在局部主题跳转或概括不足。
  2: 多处层级关系勉强。
  1: 主题混乱，无法构成合理分解链条。

B2 granularity_progression 粒度递进合理性:
  是否从宏观主题逐步走向具体行动，层级跨度是否合理。
  5: 粒度均匀递进，层层细化。
  4: 基本合理，仅轻微跳跃或重复。
  3: 不够均匀但仍可理解。
  2: 明显跳层或重复层。
  1: 没有清晰粒度递进，层级失效。

【需检测的flags（可多选，无问题留空数组）】
- PATH_SEMANTIC_DRIFT: 某层开始偏离 L1 主题
- PATH_GRANULARITY_JUMP: 相邻层从宏观直接跳到具体措施
- PATH_REDUNDANT_CHAIN: 相邻层语义重复
- PATH_OVER_DEEP: 层级过深但缺乏实质语义增量
- PATH_UNDER_STRUCTURED: 路径过短缺少必要中间层

【输出格式（仅输出JSON，无其他内容）】
{{
  "path_id": "{path_id}",
  "path_labels": {path_labels_json},
  "scores": {{
    "path_coherence": <1-5整数>,
    "granularity_progression": <1-5整数>
  }},
  "flags": [<flag字符串列表>],
  "issue_summary": "<一句话概括路径主要问题；若无明显问题写'无明显问题'>",
  "problematic_positions": [
    {{"from_label": "<上层label>", "to_label": "<下层label>", "problem": "<简短描述>"}}
  ],
  "suggested_fix_type": "<KEEP|COMPRESS_CHAIN|INSERT_INTERMEDIATE_NODE|MOVE_SUBTREE|RENAME_NODE>",
  "suggested_fix_note": "<不超过80字>"
}}
"""
