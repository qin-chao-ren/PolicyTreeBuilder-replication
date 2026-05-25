# Step 4 · 纵向合并（模板）

占位：迁移自 v3 vertical_collapse 语义，按 L 等级表述。

系统：你是结构优化专家。任务是在“同一父节点下、相邻层级”的父子二元组之间进行纵向合并决策，优先清理显著冗余，同时保持层级清晰。

动作选项（四选一）：
- promote_child：子节点上位，替换当前父的地位（父的其他孩子并入子）；
- absorb_child：父吸收子，保留父为主位（子被移除，其子树并入父）；
- keep：保持不动；
- rename_then_keep：重命名后保留（给出 new_label）。

请仅输出严格 JSON：
{"decision":"promote_child|absorb_child|keep|rename_then_keep","new_label":null|string,"confidence":0.0-1.0,"reason":"..."}

判断要点（提示，不是硬规则）：
- 语义相近度：标签 Jaccard、质心 Cosine 高时更可能合并；
- 结构特征：父仅单子链时更容易上卷；
- 标签风格：若仅风格不统一，可 rename_then_keep；
- 优先让结构更清晰：子更具体清晰→promote_child；父更规范抽象→absorb_child。

### 示例 1（完全同名，保留父）
输入：
父 T2: 推动消费品流通提质
子 T3: 推动消费品流通提质
【证据】单子链=True, Jaccard=1.00, Cosine=0.99
输出：{"decision":"absorb_child","new_label":null,"confidence":1.0,"reason":"完全同名且同义，单子链，保留父吸收子"}

### 示例 2（语义不同步，保持）
输入：
父 T1: 建设先行示范的保障体系
子 T2: 优化资源配置
【证据】单子链=False, Jaccard=0.15, Cosine=0.32
输出：{"decision":"keep","new_label":null,"confidence":0.95,"reason":"语义边界清晰，非单子链，需保留层级"}

### 示例 3（子更具体，子上位）
输入：
父 T2: 监测分析
子 T3: 强化监测分析
【证据】单子链=True, Jaccard=0.80, Cosine=0.90
输出：{"decision":"promote_child","new_label":null,"confidence":0.85,"reason":"单子链，子标签'强化...'更有行动导向，建议上位替代虚父节点"}

### 示例 4（T3↔T4 子更具体，子上位）
输入：
父 T3: 智能服务
子 T4: 语音机器人服务
【证据】单子链=True, Jaccard=0.60, Cosine=0.85
输出：{"decision":"promote_child","new_label":null,"confidence":0.82,"reason":"T4是具体服务形态，T3过于空泛，且为单子链，建议子上位"}

### 示例 5（T3↔T4 完全同名，保留父）
输入：
父 T3: 新增安装监测
子 T4: 新增安装监测
【证据】单子链=True, Jaccard=1.00, Cosine=0.97
输出：{"decision":"absorb_child","new_label":null,"confidence":0.95,"reason":"底层完全同名冗余，保留T3父节点"}
