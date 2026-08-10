# Replication Package Index

This repository is organized for review of the archived ATRS 2026 353-node PolicyTreeBuilder result and for fail-closed validation of future candidates.

## Reviewer Entry Points

| Path | Purpose |
| --- | --- |
| `data/final_tree/policy_tree_final.json` | Archived 353-node policy tree used for ATRS 2026. |
| `data/final_tree/policy_tree_final_en_radial.jpg` | Archived English radial tree figure prepared for the paper. |
| `evaluation/outputs/final_summary.json` | Archived multi-model evaluation summary. |
| `scripts/validate_tree_e0.py` | Deterministic E0 validator for historical negative controls and new candidates. |
| `schemas/semantic_tree_decision.schema.json` | Shared relation/action response schema for all four refinement/finalization stages. |
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
| `data/final_tree/` | Archived tree, tabular outputs, academic English variants, administrative subsets, and figure assets. |
| `scripts/` | Public main pipeline and tree-refinement scripts. |
| `visualization/` | Administrative tree splitting, visualization repair, and radial figure scripts. |
| `audit/` | Optional human-audit preparation scripts for action-unit extraction checks. |
| `evaluation/` | Public tree-quality evaluation scripts, model-judge samples, archived judge scores/raw outputs, agreement tables, and final summaries. |
| `prompts/` | LLM prompt templates used by the pipeline. |
| `schemas/` | Machine-readable semantic decision contract used by refinement and finalization. |
| `configs/` | Pipeline YAML configs, unified model profile template, and safe environment template. |
| `SCRIPT_PROVENANCE.tsv` | Hash mapping from extracted source scripts to public path-normalized scripts. |

## Output Categories

- Archived paper results are in `data/final_tree/`.
- Deterministic and LLM-dependent intermediate states are in `data/intermediate_outputs/`.
- Pipeline LLM raw logs are retained as an audit trail, not as the primary reading path.
- Evaluation raw JSONL files are retained for score-level review; most reviewers should start with `evaluation/outputs/final_summary.json` and the agreement CSVs.

## Version Notes

- The 353-node tree is the immutable historical paper version and an E0 negative control; this does not claim engineering compliance.
- Future corrected candidates must pass structural E0 and the independent semantic-contract publication gate, then receive a separate release decision before becoming a default artifact.
- The 317-node tree is superseded and is not part of this public package.
- The legacy local `policy_tree_eval` package has been integrated as the public `evaluation/` module after removing local environments, secrets, caches, and private path assumptions.

## Generated Manifest

`FILE_INDEX.tsv` lists every other public file with size and SHA256 after repository preparation. The manifest excludes itself because a file cannot contain a stable hash of its own complete contents.
