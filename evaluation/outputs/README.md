# Evaluation Outputs

This directory contains archived tree-quality evaluation artifacts for `data/final_tree/policy_tree_final.json`.

## Primary Review Files

| File | Purpose |
| --- | --- |
| `final_summary.json` | Machine-readable evaluation summary, including tree stats, rule checks, agreement, and final framework score. |
| `final_summary_zh.txt` | Human-readable Chinese summary of the same evaluation. |
| `agreement_node.csv`, `agreement_path.csv` | Pairwise weighted kappa and Krippendorff alpha for score dimensions. |
| `agreement_node_flags.csv`, `agreement_path_flags.csv` | Agreement statistics for binary issue flags. |
| `final_node_scores.csv`, `final_path_scores.csv` | Aggregated per-model scores after penalties. |

## Supporting Files

| File group | Purpose |
| --- | --- |
| `nodes.csv`, `edges.csv`, `paths.jsonl` | Deterministic tables extracted from the final tree. |
| `structure_report.json` | Rule-based structural checks. |
| `sampled_nodes_for_judge.jsonl`, `sampled_paths_for_judge.jsonl`, `sampling_meta.json` | Judge input samples and sampling metadata. |
| `judge_*_scores.jsonl` | Parsed model-judge outputs used by aggregation. |
| `judge_*_raw.jsonl` | Raw model text responses retained for score-level audit. |
| `divergent_nodes.json`, `divergent_paths.json` | Samples with high model disagreement or flag disagreement. |

The raw judge JSONL files are retained for transparency, but most reviewers should start with `final_summary.json` and the agreement CSVs.
