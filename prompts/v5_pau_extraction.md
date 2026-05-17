# V5 PAU 提取 Prompt 模板 (For Qwen)

> **文件用途**：嵌入 Python 代码，调用 Qwen API 时使用
> **版本**：v2.0（2026-01-15）
> **Token 优化**：精简至约 800 tokens（System Prompt）

---

## 一、System Prompt（精简版）

```
你是政策文本PAU分析专家。将输入文本分解为结构化PAU(Policy Action Unit)。

## PAU结构定义
- M: 程度词（加快、持续、进一步）→ 记录但不影响分类
- A: 动词 + A_type（direction|substantive|operational|intent）
- O: 政策对象 ⚠️保持固定搭配完整
- S_scope: 范围限定（地理/行业）→ 国际、航空、枢纽
- S_focus: 战略方向 → 绿色、智慧、数字化
- S_stage: 阶段特征 → 精品、示范、试点
- S_type: 技术子类 → 全货机、冷链、定期

## 核心规则
1. 【固定搭配】以下词组不可拆分，整体作为O：
   航空货运、航线网络、机场货站、多式联运、空空中转、跨境电商
2. 【S分类】
   - 地理词(国际/亚洲)、行业词(航空/枢纽) → S_scope
   - 绿色/智慧/数字化 → S_focus（战略方向，非废话！）
   - 精品/示范/试点 → S_stage（发展阶段）
   - 全货机/冷链/定期 → S_type
   - 高效/全面/积极 → 丢弃（噪声词）
3. 【T级预判】
   - T1=领域(体系/网络) T2=方向 T3=实体 T4=工具
   - S_scope每增1个，下沉1级；S_focus/S_type下沉0.5级
4. 【Leaf判定】O为具体实体+A为substantive/operational → is_leaf_candidate=true

## 输出格式（严格JSON）
{"pau_list":[{"pau_id":"PAU_001","M":"","A":"动词","A_type":"类型","O":"对象","S_scope":[],"S_focus":[],"S_stage":[],"S_type":[],"pau_final":"标准化表述","t_level_base":"T1-4","t_level_adjusted":"T1-4","is_leaf_candidate":false,"leaf_reason":"理由"}]}
```

---

## 二、Few-Shot 示例（仅纠错案例，3个）

### 示例1：固定搭配不拆

**输入**：
```
构建全球航空货运网络
```

**输出**：
```json
{"pau_list":[{"pau_id":"PAU_001","M":"","A":"构建","A_type":"substantive","O":"航空货运网络","S_scope":["全球"],"S_focus":[],"S_stage":[],"S_type":[],"pau_final":"全球航空货运网络","t_level_base":"T1","t_level_adjusted":"T2","is_leaf_candidate":false,"leaf_reason":"O为网络级抽象"}]}
```

**⚠️纠错**：错误做法是O=网络,S=航空货运。"航空货运"是领域固定词，不可拆。

---

### 示例2：绿色/智慧是S_focus

**输入**：
```
提升枢纽智慧绿色水平
```

**输出**：
```json
{"pau_list":[{"pau_id":"PAU_001","M":"","A":"提升","A_type":"direction","O":"水平","S_scope":["枢纽"],"S_focus":["智慧"],"S_stage":[],"S_type":[],"pau_final":"枢纽智慧水平","t_level_base":"T2","t_level_adjusted":"T3","is_leaf_candidate":false,"leaf_reason":"A为direction"},{"pau_id":"PAU_002","M":"","A":"提升","A_type":"direction","O":"水平","S_scope":["枢纽"],"S_focus":["绿色"],"S_stage":[],"S_type":[],"pau_final":"枢纽绿色水平","t_level_base":"T2","t_level_adjusted":"T3","is_leaf_candidate":false,"leaf_reason":"A为direction"}]}
```

**⚠️纠错**：智慧、绿色是战略方向(S_focus)，代表不同政策分支，必须拆分为两个PAU。绝非"废话修饰词"！

---

### 示例3：精品是S_stage

**输入**：
```
打造精品洲际货运航线
```

**输出**：
```json
{"pau_list":[{"pau_id":"PAU_001","M":"","A":"打造","A_type":"substantive","O":"货运航线","S_scope":["洲际"],"S_focus":[],"S_stage":["精品"],"S_type":[],"pau_final":"精品洲际货运航线","t_level_base":"T3","t_level_adjusted":"T4","is_leaf_candidate":true,"leaf_reason":"O为具体实体,A为substantive"}]}
```

**⚠️纠错**：精品是阶段特征(S_stage)，体现发达地区的成熟度，与欠发达地区的"新开航线"形成对比。

---

## 三、上下文策略

### 3.1 Sliding Window 模式

当处理正文时，提供前后上下文增强判断准确性：

```
## 上文（最多2条）
- [前一段落摘要...]
- [前一段落摘要...]

## 待分析文本
【层级】{h_level}
【内容】{text}

## 下文（最多2条）
- [后一段落摘要...]

请输出JSON：
```

### 3.2 Hierarchical 模式

当处理 H2/H3 标题时，提供父级标题作为语义锚点：

```
## 父级标题
【H1】{parent_h1_text}
【H2】{parent_h2_text}（如有）

## 待分析文本
【层级】{h_level}
【内容】{text}

请输出JSON：
```

### 3.3 策略选择建议

| 文本类型 | 推荐策略 | 理由 |
|---------|---------|------|
| H1 标题 | 无上下文 | H1 自成语义单元 |
| H2/H3 标题 | Hierarchical | 需要父级标题定位 |
| 正文段落 | Sliding Window | 前后文有助于理解边界 |
| 法条文本 | Sliding Window + 全文摘要 | 补贴条件常跨段落 |

---

## 四、完整 User Prompt 模板

```python
USER_PROMPT_TEMPLATE = """
{context_section}

## 待分析文本
【层级】{h_level}
【内容】{text}

## 参考示例
输入: 构建全球航空货运网络
输出: {{"pau_list":[{{"O":"航空货运网络","S_scope":["全球"],...}}]}}
要点: 航空货运是固定词，不拆。

输入: 提升枢纽智慧绿色水平  
输出: 拆为2个PAU，S_focus分别为[智慧]和[绿色]
要点: 智慧/绿色是战略方向，非废话。

请输出严格JSON格式的PAU分解结果：
"""
```

---

## 五、Token 预算

| 组件 | 预估 Tokens |
|-----|-------------|
| System Prompt | ~600 |
| Few-Shot (3个) | ~400 |
| 上下文 (4条) | ~200 |
| 输入文本 | ~100 |
| **总计** | **~1300** |

**优化建议**：
1. 首次调用包含 Few-Shot，后续可省略
2. 对于简单的 H1 标题，可跳过上下文
3. 使用 `temperature=0.1` 减少随机性

---

## 六、错误恢复策略

### 6.1 JSON 解析失败

```python
def extract_json_from_response(text: str) -> dict:
    """从响应中提取JSON，处理各种格式问题"""
    # 移除 markdown 代码块
    if '```json' in text:
        text = text.split('```json')[1].split('```')[0]
    elif '```' in text:
        text = text.split('```')[1].split('```')[0]
    
    # 尝试解析
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        # 尝试修复常见问题
        text = text.replace("'", '"')  # 单引号替换
        text = re.sub(r',\s*}', '}', text)  # 移除尾随逗号
        text = re.sub(r',\s*]', ']', text)
        return json.loads(text)
```

### 6.2 必填字段缺失

如果 O 字段为空，使用原文作为 fallback：

```python
if not pau.get('O'):
    # 提取名词短语作为 O
    pau['O'] = extract_noun_phrase(original_text)
    pau['validation_note'] = 'O_field_inferred'
```

---

## 七、版本记录

| 版本 | 日期 | 修改内容 |
|-----|------|---------|
| v1.0 | 2026-01-15 | 初版（过长，~2500 tokens） |
| v2.0 | 2026-01-15 | 精简至 ~1300 tokens，仅保留纠错案例 |
