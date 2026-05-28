# Intermediate Outputs

This folder contains included intermediate outputs and logs corresponding to the 353-node final tree.

These files are included so reviewers can inspect the pipeline state without rerunning LLM-dependent steps.

## How To Read This Directory

| Category | Examples | Reviewer use |
| --- | --- | --- |
| Cleaned corpus and calibrated inputs | `policy_corpus_cleaned.csv`, `policy_corpus_filtered.csv`, `policy_corpus_calibrated.csv` | Inspect the source records after cleaning, filtering, and granularity calibration. |
| Similarity and embedding artifacts | `policy_corpus_embeddings.parquet`, `policy_similarity_pairs_initial.csv`, `policy_similarity_rerank_edges.csv`, `cluster_similarity_edges_L*.csv` | Reproduce or audit clustering/linking inputs without rerunning embedding and reranking services. |
| Tree construction states | `policy_tree_initial.json`, `policy_tree_after_vertical_collapse.json`, `policy_tree_after_structure_balancing.json`, `policy_tree_refined.json` | Follow how the final tree evolved before finalization. |
| Node and membership tables | `tree_nodes_L*.csv`, `tree_node_membership_L*.csv`, `tree_parent_links_*.csv` | Inspect level-wise nodes, memberships, and parent links. |
| Trace and operation logs | `vertical_collapse_trace.json`, `structure_balancing_trace.json`, `tree_edit_operations.jsonl` | Audit structural changes applied during refinement. |
| LLM audit trail | `logs/llm_*.jsonl` | Optional raw model-call evidence; not the primary review entry point. |
| Human-audit support | `policy_action_units_raw.csv`, `policy_action_units_ruled.csv`, `policy_audit_sample.json` | Optional action-unit extraction audit materials. |

The primary review path remains `data/final_tree/` for final results and `evaluation/outputs/` for quality evaluation summaries.
