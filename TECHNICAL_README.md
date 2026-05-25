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
8. Split administrative trees with `scripts/split_tree_by_administrative_unit.py` and render radial figures with `scripts/render_radial_tree_figure.py` when needed.

Intermediate outputs are stored in `data/intermediate_outputs/`. The fixed publication tree is stored in `data/final_tree/`.

## Minimal Rerun Skeleton

The included `run_policy_tree_pipeline.ps1` is the full command template. Typical direct commands use this pattern:

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

## Final Output Contract

The publication tree must satisfy:

- `data/final_tree/policy_tree_final.json`
- 353 nodes
- 352 edges
- 272 leaf nodes
- maximum depth 6
- SHA256 `9242d4961e417ffa1e30e728d82e73bf669e4facdedd12f4f03f10d19b157983` after repository LF normalization

The source archive extracted-file SHA256 before LF normalization is `6ee8e666dfc5bb7b8611a42c96c8ac93f766290b290115529686f9f3ca67918b`.
