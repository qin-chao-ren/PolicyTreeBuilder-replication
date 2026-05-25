import pandas as pd
import json
import random
import argparse
from pathlib import Path

def prepare_audit_data(t5_result_path, source_corpus_path, output_path, sample_size=50):
    # 1. 读取数据
    print(f"Loading T5 Results: {t5_result_path}")
    df_t5 = pd.read_csv(t5_result_path, dtype=str, keep_default_na=False)

    print(f"Loading Source Corpus: {source_corpus_path}")
    df_source = pd.read_csv(source_corpus_path, dtype=str, keep_default_na=False)

    # 建立源数据索引 (DocID -> BlockIdx -> Text)
    # 假设 block_idx 是数字，我们需要转成 int 以便 slicing
    df_source['block_idx_int'] = df_source['block_idx'].astype(int)
    source_map = {}

    for doc_id, group in df_source.groupby('doc_id'):
        # 创建一个 dict: block_idx -> text
        source_map[doc_id] = group.set_index('block_idx_int')['block_text'].to_dict()

    # 2. 聚合 T5 结果 (按来源 Block 聚合)
    # 也就是把同一个段落提取出的所有 T5 放在一起
    grouped = df_t5.groupby(['source_doc_id', 'source_block_range', 'parent_title_text'])

    audit_tasks = []

    print("Reconstructing context...")
    for (doc_id, block_range, title), group in grouped:
        # 解析 Range (例如 "16-18")
        try:
            if "-" in str(block_range):
                start, end = map(int, block_range.split("-"))
                indices = range(start, end + 1)
            else:
                indices = [int(block_range)]

            # 重组原文
            original_texts = []
            if doc_id in source_map:
                for idx in indices:
                    txt = source_map[doc_id].get(idx, "")
                    if txt:
                        original_texts.append(txt)

            full_text = "\n".join(original_texts)

            if not full_text:
                full_text = "[WARNING: Text not found in source corpus]"

            # 收集该段落提取出的所有 Units
            extracted_units = group['unit_text'].tolist()

            audit_tasks.append({
                "audit_id": f"AUDIT_{doc_id}_{block_range}",
                "context": {
                    "parent_title": title,
                    "original_text": full_text
                },
                "model_extracted": extracted_units
            })

        except Exception as e:
            print(f"Error processing {doc_id} range {block_range}: {e}")

    # 3. 随机抽样
    total_count = len(audit_tasks)
    real_sample_size = min(sample_size, total_count)
    sampled_tasks = random.sample(audit_tasks, real_sample_size)

    print(f"Total Blocks: {total_count}. Sampled: {real_sample_size}")

    # 4. 输出
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(sampled_tasks, f, indent=2, ensure_ascii=False)

    print(f"Audit data saved to: {output_path}")

if __name__ == "__main__":
    # 配置你的路径
    # 注意：请确保这些路径指向你真实的文件
    T5_FILE = "data/intermediate_outputs/v4_units_t5_raw_ruled.csv"
    SOURCE_FILE = "roundB_outputs/roundB_types_merged1121.csv"
    OUTPUT_FILE = "data/intermediate_outputs/v4_audit_dataset_50.json"

    prepare_audit_data(T5_FILE, SOURCE_FILE, OUTPUT_FILE)