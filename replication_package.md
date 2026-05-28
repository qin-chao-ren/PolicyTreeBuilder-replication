# Replication Package Index

This repository is organized for review of the ATRS 2026 353-node PolicyTreeBuilder result.

## Reviewer Entry Points

| Path | Purpose |
| --- | --- |
| `data/final_tree/policy_tree_final.json` | Final 353-node policy tree used for ATRS 2026. |
| `data/final_tree/policy_tree_final_en_radial.jpg` | Final English radial tree figure image prepared for the paper. |
| `evaluation/outputs/final_summary.json` | Archived multi-model evaluation summary. |
| `README.md` | Quick review path and setup instructions. |
| `TECHNICAL_README.md` | Pipeline and evaluation rerun workflow. |
| `FILE_INDEX.tsv` | Public file manifest with size and SHA256. |
| `LEGACY_NAME_MAP.tsv` | Legacy-to-public path mapping for traceability. |

## Package Contents

| Path | Purpose |
| --- | --- |
| `data/source/policy_action_segments.csv` | Primary source input for the public pipeline. |
| `data/source/administrative_unit_metadata.csv` | Administrative metadata used for administrative tree splitting. |
| `data/intermediate_outputs/` | Included intermediate outputs, traces, LLM logs, embeddings, and audit samples. |
| `data/final_tree/` | Final tree, final tabular outputs, academic English variants, administrative subsets, and figure assets. |
| `scripts/` | Public main pipeline and tree-refinement scripts. |
| `visualization/` | Administrative tree splitting, visualization repair, and radial figure scripts. |
| `audit/` | Optional human-audit preparation scripts for action-unit extraction checks. |
| `evaluation/` | Public tree-quality evaluation scripts, model-judge samples, archived judge scores/raw outputs, agreement tables, and final summaries. |
| `prompts/` | LLM prompt templates used by the pipeline. |
| `configs/` | Pipeline YAML configs, unified model profile template, and safe environment template. |
| `SCRIPT_PROVENANCE.tsv` | Hash mapping from extracted source scripts to public path-normalized scripts. |

## Output Categories

- Core final results are in `data/final_tree/`.
- Deterministic and LLM-dependent intermediate states are in `data/intermediate_outputs/`.
- Pipeline LLM raw logs are retained as an audit trail, not as the primary reading path.
- Evaluation raw JSONL files are retained for score-level review; most reviewers should start with `evaluation/outputs/final_summary.json` and the agreement CSVs.

## Version Notes

- The 353-node tree is the final paper version.
- The 317-node tree is superseded and is not part of this public package.
- The legacy local `policy_tree_eval` package has been integrated as the public `evaluation/` module after removing local environments, secrets, caches, and private path assumptions.

## Generated Manifest

`FILE_INDEX.tsv` lists public files with size and SHA256 after repository preparation.
