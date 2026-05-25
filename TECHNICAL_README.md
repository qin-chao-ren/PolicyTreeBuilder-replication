# Round C v4 Technical Workflow

This document describes the technical workflow for the ATRS 2026 353-node Round C v4 replication package. It is adapted from the original Round C v4 technical README and uses repository-relative paths.

## Inputs

Primary input:

```text
data/source/roundB_types_merged1121.csv
```

Administrative mapping input:

```text
data/source/admin_mapping/roundA_final_overview_scored_selected1120.csv
```

## Configuration

YAML configs are in `configs/`.

Create a local environment file only if rerunning LLM-dependent steps:

```powershell
Copy-Item configs\.env.example configs\.env
```

The committed `.env.example` contains only variable names. Real credentials are intentionally excluded.

## Pipeline Overview

The Round C v4 pipeline follows this sequence:

1. Prepare and clean the Round B corpus with `scripts/step1_prepare_corpus.py`.
2. Filter T0 records with `scripts/step1_5_filter_t0.py`.
3. Generate embeddings and nearest-neighbor/rerank pairs with `scripts/step2_embed_and_nn.py`.
4. Calibrate granularity levels with `scripts/step2_5_calibrate_levels.py`.
5. Define and classify L1 categories with `scripts/step2_8_define_l1.py` and `scripts/step2_8_classify_levels.py`.
6. Build the coarse tree with Step 3 merge/link scripts and `scripts/step3_build_coarse_tree.py`.
7. Refine the tree with Step 4 skeleton, shaping, polishing, and final override scripts.
8. Split administrative trees with `scripts/split_final_tree_by_admin_revised_0509.py` and render radial visualizations with `scripts/visualize_radial_tree_v6_style_0510.py` when needed.

Intermediate outputs are stored in `data/intermediate_outputs/`. The fixed final publication tree is stored in `data/final_tree/`.

The paper-figure companion files are also stored in `data/final_tree/`, including the academic English label map, English final/provincial/city tree JSONs, and the final English radial JPG.

## Minimal Rerun Skeleton

The included `run_v4_pipeline.ps1` is a template for the original full pipeline. It has been rewritten to use repository-relative paths.

Typical rerun commands use this pattern:

```powershell
python scripts/step1_prepare_corpus.py `
  --source data/source/roundB_types_merged1121.csv `
  --env configs/.env `
  --outdir data/intermediate_outputs `
  --llm-clean yes

python scripts/step3_build_coarse_tree.py `
  --config configs/step3_config.yaml

python scripts/step4_5_overall_structure.py `
  --input data/intermediate_outputs/v4_tree_refined.json `
  --output data/final_tree/v4_tree_final.json `
  --config configs/step4_config.yaml `
  --l1-def data/intermediate_outputs/v4_l1_definition.json `
  --audit-out data/final_tree/v4_final_audit.json `
  --flat-csv data/final_tree/v4_tree_final_flat.csv
```

Some steps call external LLM or embedding services. Reviewers who do not have access to the same services can inspect the included intermediate and final outputs directly.

## Final Output Contract

The publication tree must satisfy:

- `data/final_tree/v4_tree_final.json`
- 353 nodes
- 352 edges
- 272 leaf nodes
- maximum depth 6
- SHA256 `9242d4961e417ffa1e30e728d82e73bf669e4facdedd12f4f03f10d19b157983` after repository LF normalization

The source archive extracted-file SHA256 before LF normalization is `6ee8e666dfc5bb7b8611a42c96c8ac93f766290b290115529686f9f3ca67918b`.
