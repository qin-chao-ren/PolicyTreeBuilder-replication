#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Action-unit extraction · Extract T5 Action Units (With Traceability & De-duplication)
- Merges consecutive NA blocks.
- Extracts Action Units using LLM (Action + Object).
- Generates unique IDs linked to source block range.

python scripts/extract_policy_action_units.py `
  --source "data/source/policy_action_segments.csv" `
  --output "data/intermediate_outputs/policy_action_units_raw.csv" `
  --env "configs/.env"
"""
import argparse
import json
import os
import logging
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

# 假设你的 llm_client 位于 utils 目录
from utils.llm_client import chat_json, load_env_file

# === 1. System Prompt (保持不变) ===
SYSTEM_PROMPT = action_unit_prompt = {
    "role": "system",
    "content": """你是一位资深的公共政策架构师。你的任务是将具体的政策文本转化为**高度概括、标准化的“政策行动标签” (Standardized Policy Action Tags)**。

### 核心目标
从具体的实施细节中，提炼出可跨城市、跨文件复用的**业务功能模块**。
**原则：我们要的是“做什么（业务本质）”，而不是“怎么做（具体手段/指标/地点）”。**

### 提取思维链 (必须严格执行)
在生成输出前，请对文本进行以下三次“蒸馏”：

1.  **第一层：去地名与去特定化 (De-localization)**
    * **彻底删除**具体城市名（如“上海”、“成都”）、具体机场名（如“浦东机场”、“天府机场”）。
    * **泛化处理**：将具体机场统一改为“枢纽机场”或“机场”；将具体区域（如“长三角”）改为“周边区域”或直接省略（如果对业务无核心影响）。
    * *例*：“修编上海国际航空枢纽战略规划” -> “修编航空枢纽战略规划”。

2.  **第二层：列表归纳与抽象 (Enumeration Abstraction)**
    * **拒绝罗列**：当遇到顿号分隔的长串名词（如“冷链、医药、活体动物...”）时，**必须**将其归纳为上级概念（如“特种货物”、“专业物流”）。
    * *例*：“完善冷链、医药、危险品设施” -> “完善特种货物处理设施”。

3.  **第三层：语义压缩 (Semantic Compression)**
    * **缩短定语**：删除“新设立”、“一次性”、“落户”等非核心限定词，只保留核心业务对象。
    * **动宾短语标准化**：结构固定为 **[2字核心动词] + [核心名词对象]**，尽量控制在 **6-10个字**以内。
    * *例*：“给予新设立基地货运航空公司一次性落户财政奖励” -> “奖励基地航司落户” 或 “补贴货运航司引进”。

### 负面案例 vs 正面案例 (对比学习)

#### 场景1：去地名
* ❌ **错误**：修编浦东国际机场总体规划
* ✅ **正确**：修编机场总体规划
* ❌ **错误**：启动新东货运区建设
* ✅ **正确**：启动货运区建设

#### 场景2：去罗列
* ❌ **错误**：完善快递、冷链、医药、及空空联运设施 (太碎)
* ✅ **正确**：完善专业货运与中转设施 (归纳)

#### 场景3：去长定语
* ❌ **错误**：给予新开洲际航线三年期市场开拓费用补贴 (太长)
* ✅ **正确**：补贴新开洲际航线 (保留核心)
* ❌ **错误**：推动中外航空公司积极对接需求集聚资源 (太虚)
* ✅ **正确**：推动航司集聚资源 (抓住主干)

### 约束检查
1.  提取结果不能包含任何数字。
2.  提取结果不能包含具体地名（除非是“国际”、“国内”、“洲际”这种范围词）。
3.  如果提取出的内容与【上级标题】含义几乎完全重复（无信息增量），则**不要提取**。

### 输出格式
仅输出一个标准的 JSON 对象：
{
  "policy_units": [
    {
      "unit_text": "标准化后的短语(建议6-10字)",
      "source_snippet": "对应的原文片段(用于溯源)"
    }
  ]
}
"""
}

# === 2. 日志配置函数 ===
def setup_logging(out_dir: Path):
    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"extract_policy_action_units_{timestamp}.log"

    # 配置 logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            # 如果你想在屏幕上也看到刷屏的日志，可以解开下面这行，
            # 但这会打断 tqdm 进度条，建议只保留 FileHandler
            # logging.StreamHandler()
        ]
    )
    print(f"[LOG] Detailed logs will be saved to: {log_file}")
    return logging.getLogger(__name__)

# === 3. 任务构建逻辑 ===
def build_tasks(df: pd.DataFrame):
    """
    预处理：合并连续的 NA 块，并找到它们最近的父标题。
    返回：待处理的任务列表。
    """
    tasks = []

    # 状态变量
    current_h = None  # 当前最近的标题 (H1-H4)
    na_buffer = []    # 连续 NA 的缓存
    buffer_start_idx = -1 # 缓存开始的 block_idx

    # 确保按顺序处理
    # 转换辅助列用于排序
    df['block_idx_int'] = df['block_idx'].astype(int)
    df = df.sort_values(['doc_id', 'block_idx_int'])

    def flush_buffer():
        """将缓存的 NA 打包成一个任务"""
        nonlocal na_buffer, buffer_start_idx
        if na_buffer and current_h:
            merged_text = "\n".join(na_buffer)
            # 记录来源范围，例如 "16-18"
            #buffer_start_idx 已经是 int 了，可以直接计算
            end_idx = buffer_start_idx + len(na_buffer) - 1
            source_range = f"{buffer_start_idx}-{end_idx}"

            tasks.append({
                "doc_id": current_h['doc_id'],
                "parent_title_text": current_h['text'],
                # 关键：这是为了挂接树，使用 Title 的第一个分身 ID
                "anchor_parent_id": current_h['source_id'] + "_01",
                "content_text": merged_text,
                "source_block_start": buffer_start_idx,
                "source_block_range": source_range
            })
        # 清空缓存
        na_buffer = []
        buffer_start_idx = -1

    # 遍历每一行
    for _, row in df.iterrows():
        level = str(row['final_level']).upper()
        text = str(row['block_text']).strip()

        if not text: continue

        # 如果遇到标题 (H1-H4)
        if level.startswith("H"):
            # 1. 先把之前的 NA 缓存处理掉 (Flush)
            flush_buffer()

            # 2. 更新当前标题指针
            # source_id 格式: doc_id + block_idx (5位)
            sid = f"{row['doc_id']}_{str(row['block_idx']).zfill(5)}"
            current_h = {
                "text": text,
                "source_id": sid,
                "doc_id": row['doc_id']
            }

        # 如果遇到内容 (NA)
        elif level == "NA":
            if current_h:
                if not na_buffer:
                    # [修正点]：强制转换为 int
                    buffer_start_idx = int(row['block_idx'])
                na_buffer.append(text)

            # 如果 current_h 为空（比如文档开头就是 NA），则忽略或挂到 ROOT（视需求而定）

    # 循环结束后，别忘了 flush 最后一组
    flush_buffer()

    return tasks

def generate_t5_id(doc_id, start_block_idx, seq_num):
    safe_doc = str(doc_id).split('.')[0]
    safe_block = str(start_block_idx).zfill(5)
    safe_seq = str(seq_num).zfill(2)
    return f"T5_{safe_doc}_{safe_block}_{safe_seq}"

# === 4. 主程序 ===
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="Raw source source CSV")
    parser.add_argument("--output", required=True, help="Output T5 Units CSV")
    parser.add_argument("--env", default="configs/.env")
    args = parser.parse_args()

    # 初始化
    load_env_file(args.env)
    llm_model = os.getenv("PRIMARY_LLM_MODEL", "gpt-4o")

    # 设置日志
    output_path = Path(args.output)
    logger = setup_logging(output_path.parent)

    logger.info("=== Action-unit extraction Start ===")
    logger.info(f"Source: {args.source}")
    logger.info(f"Model: {llm_model}")

    print(f"[INFO] Loading source: {args.source}")
    # keep_default_na=False 告诉 pandas：不要把 "NA" 当作空值，它就是个字符串 "NA"
    df = pd.read_csv(args.source, dtype=str, keep_default_na=False)

    # --- 修改点 2: 打印一下前几行的 level 看看是什么 ---
    print(f"[DEBUG] Preview of loaded levels: {df['final_level'].unique()[:10]}")

    # 1. 构建任务
    print("[INFO] Merging consecutive NA blocks...")
    tasks = build_tasks(df)
    logger.info(f"Built {len(tasks)} extraction tasks from source CSV.")
    print(f"[INFO] Generated {len(tasks)} extraction tasks.")

    results = []

    # 使用 tqdm 显示进度条，同时在循环内部写日志
    for i, task in enumerate(tqdm(tasks, desc="Extracting")):
        doc_id = task['doc_id']
        range_str = task['source_block_range']
        parent_title = task['parent_title_text']

        # === LOG: 发送前 ===
        logger.info(f"[{i+1}/{len(tasks)}] REQ -> Doc: {doc_id} | Range: {range_str} | Title: {parent_title[:20]}...")

        user_content = f"**上级标题**：{parent_title}\n**具体内容**：\n{task['content_text']}"

        try:
            # 调用 LLM
            resp, data = chat_json(
                system=action_unit_prompt["content"], # 直接传入 Prompt 字典
                user=user_content,
                model=llm_model,
                temperature=0.1,
                max_tokens=1000
            )

            if resp.ok and isinstance(data, dict):
                units = data.get("policy_units", [])
                # === LOG: 成功收到 ===
                logger.info(f"[{i+1}/{len(tasks)}] RES <- OK. Extracted {len(units)} units.")

                for idx, u in enumerate(units, start=1):
                    t5_id = generate_t5_id(doc_id, task['source_block_start'], idx)
                    results.append({
                        "t5_node_id": t5_id,
                        "unit_text": u.get("unit_text", "").strip(),
                        "source_snippet": u.get("source_snippet", ""),
                        "anchor_parent_id": task['anchor_parent_id'],
                        "source_doc_id": doc_id,
                        "source_block_range": range_str,
                        "parent_title_text": parent_title
                    })
            else:
                # === LOG: LLM 返回错误或格式不对 ===
                logger.warning(f"[{i+1}/{len(tasks)}] RES <- FAIL or Invalid JSON. Error: {resp.error} | Raw: {resp.raw[:100]}...")

        except Exception as e:
            # === LOG: 异常 ===
            logger.error(f"[{i+1}/{len(tasks)}] CRITICAL ERROR: {str(e)}", exc_info=True)

        # 周期性保存（每50条保存一次，防止跑断了全丢）
        if (i + 1) % 50 == 0:
            temp_out = output_path.with_suffix('.tmp.csv')
            pd.DataFrame(results).to_csv(temp_out, index=False, encoding="utf-8-sig")
            logger.info(f"Checkpoint saved to {temp_out}")

    out_df = pd.DataFrame(results)
    out_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    # 删除临时文件
    temp_out = output_path.with_suffix('.tmp.csv')
    if temp_out.exists():
        temp_out.unlink()

    final_msg = f"=== Finished. Saved {len(out_df)} units to {output_path} ==="
    logger.info(final_msg)
    print("-" * 30)
    print(final_msg)
    print(f"Check logs at: {output_path.parent}/logs/")

if __name__ == "__main__":
    main()