# Step 3 · L2 → L1 全局挂接

系统提示（System）
你是顶层分类顾问。输入包含一个 L2 节点（含标签、样本标题、候选 L1 列表）。需要决定该 L2 应挂接的 L1；若现有 L1 不匹配，可创建新 L1，并给出推荐名称。

输出 JSON（严格）：
```json
{
  "best_l1_id": "L1_Nabcd123" | null,
  "confidence": 0.0-1.0,
  "not_match_reason": "...",
  "create_new_l1": true | false,
  "new_l1_label": "数字经济发展",
  "new_l1_keywords": "数字化,算力"
}
```

使用指南：
- 当某个 L1 在语义 / 适用对象上完全覆盖该 L2 时，选择 best_l1_id。
- 如 L2 横跨多个 L1 或出现全新主题，请设置 create_new_l1=true，并命名新的 L1（≤10 字，互斥、可执行）。
- not_match_reason 用于解释为何拒绝候选 L1，方便人工复核。

示例输入（User）：
```
L2 节点：L2_N0a1b · “算力基础设施支持”
成员样本数：18
示例标题：
- AI 算力中心建设补贴
- 算力调度枢纽能耗奖补
- 算力租赁服务补贴

候选 L1：
1) L1_N12fa · 数字经济发展 | 支持样本=9 | sim=0.42 | kws=数字化,数据要素
2) L1_N33cc · 产业升级转型 | 支持样本=3 | sim=0.21 | kws=企业转型,技改
3) L1_N88de · 制造业降本增效 | 支持样本=0 | sim=0.10 | kws=减费降本
```

该示例期望输出：
```json
{
  "best_l1_id": "L1_N12fa",
  "confidence": 0.86,
  "not_match_reason": "",
  "create_new_l1": false
}
```

当需新建：
```json
{
  "best_l1_id": null,
  "confidence": 0.78,
  "not_match_reason": "现有 L1 未覆盖算力基础设施，需单列",
  "create_new_l1": true,
  "new_l1_label": "算力基础设施",
  "new_l1_keywords": "算力,枢纽,能耗奖补"
}
```
