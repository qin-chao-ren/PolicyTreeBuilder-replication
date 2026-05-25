# Step 4 · 兄弟再安置（模板）

操作集：keep_here | move_to | create_new_parent（placement=grandparent/current_parent 等）。
占位：迁移自 v3 rehome_siblings 语义。

系统：你是结构再安置（Rehome）专家。由于刚刚发生了“父并入子（promote_child）”的纵向合并，需要对原父的其他孩子（剩余兄弟）逐一评估新的去向。

请基于候选信息（当前承接者、叔叔候选父 Top-N、全局语义）对每个剩余兄弟给出动作：
- keep_here：保留在当前承接者（新父）之下；
- move_to：移动到给定的叔叔父节点（提供 parent_id）；
- create_new_parent：为该兄弟创建新的父分组（给出 bridge_label），并裁决新父应挂在“爷爷”还是“当前新父”之下：
  - placement = "grandparent"（默认）：挂在爷爷下，层级与新父相同，成为新父的同级；
  - placement = "current_parent"：挂在当前新父之下，作为下一层中间层。

严格输出 JSON 数组，每个元素对应一个兄弟：
[
  {
    "sibling_id":"<tree_id>",
    "decision":"keep_here|move_to|create_new_parent",
    "target_parent":null|"<tree_id>",
    "bridge_label":null|string,
    "placement":null|"grandparent"|"current_parent",
    "confidence":0.0-1.0,
    "reason":"..."
  }
]

注意：
- 避免重复挂载与循环；
- 若证据不足，尽量 keep_here；
- 仅在强语义需要时才 create_new_parent；
- 当 placement 选择 grandparent 时，新建父的层级应与当前承接者相同（同级兄弟）；选择 current_parent 时，新建父层级应为承接者的下一层。

### 示例 1（自立门户，挂到爷爷）
输入：
- 新承接者：T3_智能服务（ID: L3_Naa11bb22）
- 待安置兄弟：L3_N77cc8899 标签=产业孵化与培育
- 叔叔候选：
  1) L3_N11112222 标签=公共服务 score=0.45
  2) L3_N33334444 标签=运营服务体系 score=0.50

输出：[
  {
    "sibling_id": "L3_N77cc8899",
    "decision": "create_new_parent",
    "target_parent": null,
    "bridge_label": "产业孵化与培育",
    "placement": "grandparent",
    "confidence": 0.78,
    "reason": "与新承接者/叔叔候选贴近度均不足；建议自立门户，挂到爷爷下成为同级。"
  }
]

### 示例 2（自立门户，挂到当前新父）
输入：
- 新承接者：T2_治理能力建设（ID: L2_Nbb22cc33）
- 待安置兄弟：L3_N44556677 标签=绩效评估
- 叔叔候选：
  1) L2_N99990000 标签=公共管理 score=0.40

输出：[
  {
    "sibling_id": "L3_N44556677",
    "decision": "create_new_parent",
    "target_parent": null,
    "bridge_label": "绩效评估",
    "placement": "current_parent",
    "confidence": 0.81,
    "reason": "与新承接者相关但需要一层中间聚合，挂到当前新父下一层更清晰。"
  }
]

### 示例 3（投奔叔叔 move_to）
输入：
- 新承接者：T3_数据治理（ID: L3_Nc1d2e3f4）
- 待安置兄弟：L3_Nabc01234 标签=数据目录编制
- 叔叔候选：
  1) L3_N88aa77bb 标签=主数据管理 score=0.62
  2) L3_N55ff66ee 标签=数据标准规范 score=0.83

输出：[
  {
    "sibling_id": "L3_Nabc01234",
    "decision": "move_to",
    "target_parent": "L3_N55ff66ee",
    "bridge_label": null,
    "placement": null,
    "confidence": 0.86,
    "reason": "与叔叔 '数据标准规范' 贴近度更高，建议直接改挂。"
  }
]

### 示例 4（保留 keep_here）
输入：
- 新承接者：T3_运维服务（ID: L3_Nd4e5f6a7）
- 待安置兄弟：L3_N77889900 标签=故障响应与处理
- 叔叔候选：
  1) L3_N11221122 标签=软件服务管理 score=0.35

输出：[
  {
    "sibling_id": "L3_N77889900",
    "decision": "keep_here",
    "target_parent": null,
    "bridge_label": null,
    "placement": null,
    "confidence": 0.93,
    "reason": "与新承接者 '运维服务' 语义一致，直接保留在当前新父之下。"
  }
]

### 示例 5（批量混合：move_to + create_new_parent）
输入：
- 新承接者：T3_生态培育（ID: L3_N0a1b2c3d）
- 待安置兄弟：
  - L3_Nxx001 标签=产业联盟对接；
  - L3_Nxx002 标签=生态指标与评测；
- 叔叔候选：
  1) L3_Nyy100 标签=行业协同 score=0.81
  2) L3_Nyy200 标签=评估标准体系 score=0.77

输出：[
  {
    "sibling_id": "L3_Nxx001",
    "decision": "move_to",
    "target_parent": "L3_Nyy100",
    "bridge_label": null,
    "placement": null,
    "confidence": 0.84,
    "reason": "与叔叔 '行业协同' 更贴近，直接改挂。"
  },
  {
    "sibling_id": "L3_Nxx002",
    "decision": "create_new_parent",
    "target_parent": null,
    "bridge_label": "生态评测体系",
    "placement": "grandparent",
    "confidence": 0.80,
    "reason": "与新承接者/候选叔叔均非强相关，建议自立门户，挂到爷爷下成为同级。"
  }
]

### 示例 6（证据不足，保守 keep_here）
输入：
- 新承接者：T3_安全管理（ID: L3_N1f2e3d4c）
- 待安置兄弟：L3_Nmno34567 标签=轻量化管控
- 叔叔候选：
  1) L3_Nppp111 标签=访问控制 score=0.49
  2) L3_Nppp222 标签=数据脱敏 score=0.47

输出：[
  {
    "sibling_id": "L3_Nmno34567",
    "decision": "keep_here",
    "target_parent": null,
    "bridge_label": null,
    "placement": null,
    "confidence": 0.60,
    "reason": "证据不足以支持改挂或自立门户，暂保留在当前新父，待后续轮次复核。"
  }
]
