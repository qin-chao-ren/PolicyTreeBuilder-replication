# PolicyTreeBuilder Technical Workflow

This document describes the public ATRS 2026 replication workflow for the fixed 353-node PolicyTreeBuilder result.

## Inputs

Primary source input:

```text
data/source/policy_action_segments.csv
```

Administrative metadata input:

```text
data/source/administrative_unit_metadata.csv
```

## Configuration

YAML configs are in `configs/`.

Create a local environment file only if rerunning LLM-dependent steps:

```powershell
Copy-Item configs\.env.example configs\.env
```

The committed `.env.example` contains only variable names. Real credentials are intentionally excluded.

## Pipeline Overview

The public pipeline follows this sequence:

1. Prepare and clean the source corpus with `scripts/prepare_policy_corpus.py`.
2. Filter non-policy records with `scripts/filter_non_policy_records.py`.
3. Generate embeddings and similarity/rerank edges with `scripts/embed_policy_corpus.py`.
4. Calibrate policy-action granularity with `scripts/calibrate_policy_granularity.py`.
5. Define and assign top-level categories with `scripts/define_top_level_categories.py` and `scripts/assign_top_level_categories.py`.
6. Build action clusters, assign parent clusters, and build the initial tree.
7. Collapse redundant hierarchy, balance structure, polish labels, and finalize the tree.
8. Split administrative trees with `visualization/split_tree_by_administrative_unit.py` and render radial figures with `visualization/render_radial_tree_figure.py` when needed.
9. Run optional tree-quality evaluation with the public scripts in `evaluation/scripts/`.

Intermediate outputs are stored in `data/intermediate_outputs/`. The fixed publication tree is stored in `data/final_tree/`.

## Minimal Rerun Skeleton

The included `run_policy_tree_pipeline.ps1` is the full command template. It requires local credentials for external LLM, embedding, and reranking services. Reviewers who only need to inspect or validate the published result can use the included outputs and the deterministic evaluation commands below.

Typical direct commands use this pattern:

```powershell
python scripts/prepare_policy_corpus.py `
  --source data/source/policy_action_segments.csv `
  --env configs/.env `
  --outdir data/intermediate_outputs `
  --llm-clean yes

python scripts/build_initial_policy_tree.py `
  --config configs/tree_build_config.yaml

python scripts/finalize_policy_tree.py `
  --input data/intermediate_outputs/policy_tree_refined.json `
  --output data/final_tree/policy_tree_final.json `
  --config configs/tree_refinement_config.yaml `
  --l1-def data/intermediate_outputs/top_level_categories.json `
  --audit-out data/final_tree/policy_tree_final_audit.json `
  --flat-csv data/final_tree/policy_tree_final_flat.csv
```

Some steps call external LLM or embedding services. Reviewers without access to the same services can inspect the included intermediate and final outputs directly.

## LLM Runtime Boundary

The main tree-building pipeline and the evaluation judges intentionally keep separate LLM wrappers:

- Main pipeline scripts use `scripts/common_llm.py` and helper code in `scripts/utils/`.
- Evaluation judge scripts use `evaluation/scripts/llm_clients.py` because they preserve the archived judge-key convention used by `evaluation/outputs/`.

This package does not merge those clients, so the public code stays close to the verified replication workflow. A future maintenance branch could consolidate them into one runtime after a full rerun validation.

## Evaluation Workflow

The evaluation module defaults to the fixed final tree and writes to `evaluation/outputs/`:

```powershell
python evaluation/scripts/01_extract_tables.py
python evaluation/scripts/02_structure_check.py
python evaluation/scripts/03_sampling.py
python evaluation/scripts/06_aggregate.py
python evaluation/scripts/07_status.py
```

The deterministic steps above do not call external APIs. Model judging is optional and requires local credentials in `evaluation/.env`, created from `evaluation/.env.example`:

```powershell
python evaluation/scripts/04_run_node_judge.py --judge A_kimi --limit 3
python evaluation/scripts/05_run_path_judge.py --judge A_kimi --limit 3
```

The archived evaluation outputs include 278 sampled node judgments, 51 sampled path judgments, and the final multi-model framework score reported in `evaluation/outputs/final_summary.json`.

## Final Output Contract

The publication tree must satisfy:

- `data/final_tree/policy_tree_final.json`
- 353 nodes
- 352 edges
- 272 leaf nodes
- maximum depth 6
- SHA256 `9242d4961e417ffa1e30e728d82e73bf669e4facdedd12f4f03f10d19b157983` after repository LF normalization

The source archive extracted-file SHA256 before LF normalization is `6ee8e666dfc5bb7b8611a42c96c8ac93f766290b290115529686f9f3ca67918b`.
