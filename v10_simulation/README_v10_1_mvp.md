# Step 2 v10.1 MVP

## Summary

本目录实现 `v5_step2_simulation_full_v9_1.json` 之后的一个最小后处理 patch：
`reference-guided local normalization`。

v10.1 只做：

- 基于 v9.1 现有节点构建 owner window
- 在冻结快照上做 Pass 1 单节点标准化
- 产出可回溯 JSON 和 review CSV

v10.1 不做：

- 重做 Step 2 原流程
- 重做 Track A / B / C 抽取
- 重做 leaf extraction
- 重做 `pau_final`
- 进入 Step 3 构树
- 做全局 consolidation
- 做 Pass 2 refinement

## Current MVP Scope

当前输入文件真实可标准化节点数为 152：

- `Track A = 26`
- `Track B = 84`
- `Track C monetary = 40`
- `Track C non_monetary = 2`

另有 1 个 Track A 噪音标题 `DF0459_B019` 保留在原结构中，但不参与 window 与标准化。

## Implementation Plan

### 1. Flatten

将 v9.1 输入扁平化为内部节点表，并生成稳定 ID：

- `TA_001` - `TA_026`
- `TB_001` - `TB_084`
- `TC_001` - `TC_042`

每个节点保留：

- `track`
- `doc_id`
- `semantic_block_id`
- `parent_title`
- `leaf_index`
- `tool_nature`
- `source_path_v9`
- `components`
- `pau_final`

### 2. Pooling

按当前方案调整为：

- `policy_pool = Track A + Track B + Track C non_monetary`
- `tool_pool_monetary = Track C monetary`

说明：

- `Track C monetary` 继续单独封池
- `Track C non_monetary` 不单独封池
- `Track C non_monetary` 允许参考规划侧表达

### 3. Retrieval

每个节点生成内部检索字段：

- `O_text = O`
- `S_text = S_scope + S_focus + S_stage + S_type`
- `A_text = A`，仅 `A_type in {substantive, operational}` 时启用
- `reference_text_v10_internal = O/S/A/pau_final/parent_title_or_tool_nature`

召回策略：

- embedding 粗召回 `top_k=40`
- rerank 精排
- 组合权重：
  - `w_O = 1.0`
  - `w_S = 0.8`
  - `w_A = 0.6`

### 4. Window Shaping

window 参数改为：

- `target_window_size = 20`
- `min_window_size = 15`
- `max_window_size = 30`

规则：

- owner window 总量包含 owner 自身
- `reference_neighbor_ids_v10` 不包含 owner
- 先按召回排序形成候选集
- 再做轻量扩窗或截窗

### 5. Pass 1 Normalization

所有节点只读取 v9.1 冻结快照，不读取同轮新结果。

LLM 只输出当前节点：

- `components_v10`
- `pau_std_v10`
- `guard_tags_v10`
- `normalization_log_v10`
- `consistency_review_flag_v10`

### 6. Hard Rules

本 MVP 明确保留以下规则：

- `引进/引入 + 落户` 同指一对象时，只保留 `引进`，`落户` 只进 log
- `落户奖励 / 落户支持 / 落户补贴` 等独立工具表达允许保留 `落户`
- 重点处理：`引进培育`、`引入培育`、`引育`、`招引培育`
- 默认不机械拆：`建设运营`、`研发应用`、`规划建设`、`建设管理`、`优化拓展`
- guard 强保护：`试点`、`新开`、`加密`、`存量`、`示范`、`战略性`
- 稳定词块强保护：`空空中转`、`一次安检`、`稳定运行`、`口岸一体化营运费用`、`异地货站`、`转运分拨中心`
- 不混淆：`建设 ≠ 改造 ≠ 管理`
- 不跨 `monetary / non_monetary`

## Files

计划新增：

- `run_v10_1_reference_normalization.py`
- `prompt_v10_1_reference_normalization.md`
- `test_v10_1_llm_connectivity.py`
- `test_v10_1_retrieval_connectivity.py`
- `v10_1_reference_window_simulation.json`
- `v10_1_reference_window_review.csv`

## Reuse

默认复用：

- `scripts/utils/llm_client.py`
- `configs/roundC_v4.env`

embedding / rerank 请求协议参考：

- `scripts/step2_embed_and_nn.py`

## Model Configuration

不硬编码模型名。

默认从 `configs/roundC_v4.env` 读取：

- `PRIMARY_LLM_MODEL`
- `EMBED_MODEL_NAME`
- `RERANK_MODEL_NAME`

同时支持 CLI override：

- `--llm-model`
- `--embed-model`
- `--rerank-model`

## Offline-Only Validation

本次实现阶段不主动执行任何联网、调用 API、或依赖本地网络环境的命令。

如果需要验证连通性或 provider 配置，使用单独测试脚本，由用户手动运行：

- `test_v10_1_llm_connectivity.py`
- `test_v10_1_retrieval_connectivity.py`

## Run

主脚本预计使用方式：

```powershell
python v10_simulation\run_v10_1_reference_normalization.py `
  --input v10_simulation\v5_step2_simulation_full_v9_1.json `
  --env configs\roundC_v4.env
```

可选覆盖：

```powershell
python v10_simulation\run_v10_1_reference_normalization.py `
  --llm-model <model_name> `
  --embed-model <model_name> `
  --rerank-model <model_name>
```

## Current Simplifications

- 仅做 Pass 1
- 不做关系标注
- 不做全局 merge / consolidation
- 不做复杂 cluster orchestration
- `reference_text_v10` 仅作内部检索字段，不单独落盘
