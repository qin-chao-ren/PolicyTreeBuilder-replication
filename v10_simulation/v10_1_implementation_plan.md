# v10.1 Implementation Plan

## Goal

在 `v5_step2_simulation_full_v9_1.json` 之上新增一个可运行、可审查的 v10.1 MVP 后处理阶段，验证
`reference-guided local normalization` 是否可跑通。

## Confirmed Decisions

- 处理 152 个有效节点，不人为补到 153
- 保留原层级结构，并额外输出完整 `nodes_flat_v10`
- `policy_pool = Track A + Track B + Track C non_monetary`
- `tool_pool_monetary = Track C monetary`
- `target_window_size=20`
- `min_window_size=15`
- `max_window_size=30`
- 不自动执行联网/API 验证
- 模型名从 env 读取，并支持 CLI override

## Implementation Order

1. 写 prompt 文件，固定单节点 owner-window 输出 schema
2. 写主脚本，完成扁平化、pooling、embedding/rerank、window shaping、Pass 1 标准化、结果回写
3. 写两个最小连通性测试脚本，仅供手动执行
4. 本地只做不联网校验：语法检查、参数帮助、JSON/CSV 结构检查

## Acceptance Criteria

- 能从 v9.1 输入稳定扁平化出 152 个有效节点
- 能按新版 pooling/window 参数生成 owner window
- 能输出双轨结果：
  - 原层级追加 v10 字段
  - `nodes_flat_v10`
- 能生成 review CSV
- 脚本默认不依赖硬编码模型名
- 代码内不主动触发联网验证
