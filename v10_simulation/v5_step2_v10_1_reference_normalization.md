# Step 2 v10.1 方案（定稿草案）

## 0. 版本定位

**v10.1 = 接在 v9.1 之后的后处理 patch**。  
它不重做 extraction，不推翻三轨，不构树，不替代 Step 3。v9.1 仍然是 Step 2 的抽取与 `pau_final` 基线，且继续保留“固定搭配优先、对象限定优先、业务区分度优先”等硬规则。

### v10.1 只新增
一个后处理阶段：

> **reference-guided local normalization**  
> （参照簇引导的局部标准化）

它要解决的，正是 v9.1 留给 Step 3 的残余问题：direction 近义写法、对象短语微差、领域默认省略、局部重复过多。现有 v10 草案也已经把这些问题明确列出来了。

---

## 1. v10.1 处理边界

### 1.1 做什么
- 局部写法统一
- 领域默认省略恢复
- guard 保留前提下的标准化
- 少量并列动作修正
- 生成更稳的 `pau_std_v10`
- 产出局部关系判断线索

### 1.2 不做什么
- 不重做 leaf extraction
- 不改 Track A / B / C
- 不重写 v9.1 `components`
- 不做全局 consolidation
- 不做 Step 3 构树
- 不要求一组节点最后长成完全一样

---

## 2. 核心思想

v10.1 不再走“先发明更多 canonical 字段”的路线。  
而是改成：

1. 先基于 **O+S 优先、A 条件加权** 构建局部近邻
2. 再让 LLM 在局部参照窗口里，对**当前节点**做标准化
3. 最后再给出局部关系判断

这里的参照窗口不是合并簇，而是：

> **reference window / owner window**

它的作用是让当前节点参考“相似表达通常怎么写”，而不是让整组节点被压成一个表达。

评估报告指出的重复问题——例如  
`引进培育货运龙头企业 / 培育物流龙头企业 / 引入培育龙头企业`，  
以及多种“全球/国际/优质/覆盖全球”的航线网络表述——正适合用这种局部参照式标准化处理。

---

## 3. 输入与输出

### 3.1 输入
直接吃 v9.1 输出：

- `doc_id`
- `track`
- `tool_nature`
- `parent_title`
- `leaf_name`
- `components`
- `pau_final`

其中 Track C 的 monetary / non_monetary 分流继续沿用 v9.1，不混并。

### 3.2 输出
新增字段：

- `reference_text_v10`
- `reference_neighbor_ids_v10`
- `components_v10`
- `pau_std_v10`
- `guard_tags_v10`
- `normalization_log_v10`
- `reference_relations_v10`
- `merge_candidate_status_v10`
- `owner_window_id_v10`
- `consistency_review_flag_v10`

### 不新增
- 不再引入 `canonical_core`
- 不再引入多层 key 体系

---

## 4. reference window 构建

### 4.1 基本原则
不是先把全体节点硬聚成若干簇，  
而是对**每个节点**构建一个以它为中心的局部参照窗口。

因此：

- window 可以重叠
- 一个节点可以出现在很多别人的 window 里
- 但每个节点只在自己的 owner window 里被正式标准化一次

---

### 4.2 召回权重设计

召回时，不直接用 `pau_final` 做唯一主键，而是采用 **分字段向量 + 加权融合**。

#### 4.2.1 权重规则
设：

- `O_vec = embedding(O)`
- `S_vec = embedding(S_scope + S_focus + S_stage + S_type)`
- `A_vec = embedding(A)`，仅当 `A_type ∈ {substantive, operational}` 时才启用

则召回向量：

#### 若 A 不是实义动作
`query_vec = 1.0 * O_vec + 0.8 * S_vec`

#### 若 A 是 substantive / operational
`query_vec = 1.0 * O_vec + 0.8 * S_vec + 0.6 * A_vec`

也就是说：

- **O 是主轴**
- **S 是修正项**
- **A 只有在实义时才轻度加权**

这和 v9.1 的结构逻辑是一致的：`pau_final` 的主标签本来就主要由 `S_* + O` 构成，而 A 并不是默认主标签核心。

#### 4.2.2 说明
这里的 1.0 / 0.8 / 0.6 是 **v10.1 初始系数**，不是理论常数。  
MVP 阶段允许调参，但第一轮先按这个版本跑。

---

### 4.3 两段式召回

#### 第一步：embedding 粗召回
对每个节点召回较大的近邻池。

#### 第二步：reranker 精排
对粗召回结果重排，输出更稳的局部近邻顺序。

#### reranker 输入建议
不是只喂 `pau_final`，而是喂一个轻量比较文本：

- `O`
- `S_summary`
- `A`（仅 substantive / operational）
- `pau_final`
- `parent_title`（轻量）

---

### 4.4 window 大小控制

不预设“每个簇固定 20 个节点”。

真正控制的是：

> **单次送入 LLM 的 owner window 大小约为 15–30 个节点**

#### 若近邻太大
截取 reranker 排名最前的一段。

#### 若近邻太小
向外轻量扩窗，补足到有参照性的规模。

#### 说明
这里不是复杂二次聚类，  
只是一个轻量的 **window shaping**。

---

## 5. owner window 机制

### 5.1 定义
每个节点都有一个自己的 `owner_window_id_v10`。

- 这个 window 只负责生成该节点的正式输出
- 其他节点在这个 window 里只是参照样本
- 参照样本不在此处被正式改写

### 5.2 结果唯一性
一个节点的正式输出：

- `components_v10`
- `pau_std_v10`
- `guard_tags_v10`

只允许在它自己的 owner window 中生成一次。

这样就避免了“同一节点在不同窗口里得到不同正式结果”。

---

## 6. snapshot 机制（关键）

v10.1 采用 **snapshot-based owner-window mechanism**。

### 6.1 Pass 1
所有 owner window **统一只看 v9.1 原始快照**：

- `components`
- `pau_final`

在这一轮中，任何窗口都**不允许引用**别的节点在本轮刚生成的新结果。

输出的是：

- `components_v10_proposed`
- `pau_std_v10_proposed`
- `guard_tags_v10`
- `normalization_log_v10`

### 6.2 Pass 1 结束后
统一写出：

- `v10.1_snapshot_pass1`

### 6.3 Pass 2（可选）
如果需要继续精修，则整轮统一吃 `v10.1_snapshot_pass1`。

仍然：

- 不动态混用
- 不边改边喂
- 整轮只看一个冻结快照

#### Pass 2 只处理
- `consistency_review_flag_v10 = true`
- 局部判断明显漂移
- guard 边界不清
- 候选关系不稳定

---

## 7. LLM 在 owner window 里做什么

分两步。

---

### 7A. 局部标准化

输入：
- 当前节点
- 当前节点的 v9.1 `components`
- 当前节点的 `pau_final`
- 当前节点的 owner window（15–30 个参照节点）

输出：
- `components_v10`
- `pau_std_v10`
- `guard_tags_v10`
- `normalization_log_v10`

#### 7A.1 可统一内容
- direction 近义写法  
  如 `推进 / 推动 / 加快推进`
- 领域默认省略  
  如 `货运 → 航空货运`，`物流 → 航空物流`
- 局部对象写法  
  如括号、轻微冗余、并列表述微差
- 少量缩略恢复  
  如 `引育`

#### 7A.2 不可统一内容
- guard  
  `试点 / 新开 / 加密 / 存量 / 示范 / 战略性`
- substantive 差异  
  `建设 ≠ 改造 ≠ 管理`
- 稳定业务词块  
  `空空中转 / 一次安检 / 稳定运行 / 口岸一体化营运费用 / 异地货站 / 转运分拨中心`
- monetary / non_monetary 边界

这些都继续继承 v9.1 的保护原则。

---

### 7B. 局部关系判断

在 `pau_std_v10` 生成后，再做一次局部关系标注。

输出关系类型仅保留四类：

- `same_expression`
- `same_base_guard_diff`
- `parent_child_related`
- `different`

这里只生成**局部判断线索**，不做全局折叠。

---

## 8. `引进 / 落户` 修正规则

这一条按最新确认意见，正式写成硬规则。

### Rule PSS-REV-01
当 `引进` / `引入` 与 `落户` **同时作用于同一对象**时：

- 只保留 `引进`
- 不把 `落户` 当成并列 substantive 动作
- 不做 `引进 + 落户` 拆分
- `落户` 若需保留，只能进入 `normalization_log_v10`

#### 例外
只有当文本中**单独出现**“落户奖励 / 落户支持 / 落户补贴”等独立工具表达时，`落户` 才可保留为独立节点或独立工具标签。

#### 说明
这条规则同时修正了 round2 中对 `引进落户` 的拆分方向。v10.1 明确改为：**同时出现时只留“引进”**。

---

## 9. 并列动作处理

### 9.1 优先处理
- `引进培育`
- `引入培育`
- `引育`
- `招引培育`

### 9.2 不机械拆分
以下默认不拆：

- `建设运营`
- `研发应用`
- `规划建设`
- `建设管理`
- `优化拓展`

除非 owner window 中证据非常强。

---

## 10. 版本管理

### 10.1 历史文件保留
以下文件保留为 **v10.0-draft 历史版本**：

- `v5_step2_v10_draft.md`
- `v10_canonicalization_full_simulation.json`
- `v10_canonicalization_full_simulation_round2.json`

不覆盖，不回写。

### 10.2 新文件建议
新增：

- `v5_step2_v10_1_reference_normalization.md`
- `v10_1_reference_window_simulation.json`
- `v10_1_reference_window_review.csv`
- `step2_v10_1_change_log.md`

### 10.3 版本标识
统一写成：

- `v9.1` = canonical extraction baseline
- `v10.0-draft` = rule-heavy canonicalization exploratory draft
- `v10.1` = reference-guided local normalization patch

---

## 11. 本轮最小可行实现（MVP 前置范围）

v10.1 先只做三件事：

### M1. 生成 owner window
先验证召回是否自然：
- 航线类是否能召到真正类似的航线类
- 供地保障类是否不会被串域
- window 规模是否能稳定落在 15–30

### M2. 跑 Pass 1 标准化
只输出：
- `components_v10`
- `pau_std_v10`
- `guard_tags_v10`
- `normalization_log_v10`

### M3. 局部关系判断
只输出：
- `same_expression`
- `same_base_guard_diff`
- `parent_child_related`
- `different`

先不做全局 consolidation。

---

## 12. 一句话总结

> **v10.1 不是把相似节点压成同一个表达，而是让每个节点在一个由 15–30 个近邻构成、且基于同一冻结快照的 owner window 中，获得一个更稳、更统一、但仍保留差异边界的标准化表达。**
