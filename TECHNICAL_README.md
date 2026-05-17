# Technical Workflow for the Fixed Round C v4 Replication Package

This document is the cleaned and path-normalized technical workflow for the
fixed public replication package. It is adapted from the original local
technical note `ROUND-C-README-V4_1222.md`, which described the Round C v4
pipeline used during paper development.

The current repository layout is different from the original development
workspace. In this replication package, paths are normalized around the
repository root and public artifacts are grouped under `data/`, `figures/`,
`scripts/`, `prompts/`, and `configs/`.

## 1. Core Goal

Round C v4 builds a hierarchical policy-action tree from processed Chinese
air-cargo policy-action records. Its main technical goal is to resolve the
mismatch between physical document heading depth and semantic granularity, then
construct a stable tree using L1 thematic anchors.

The pipeline separates four level systems:

- **H level**: physical heading level in source documents, such as H1, H2, H3.
  It is observable but inconsistent across documents.
- **T level**: semantic granularity level calibrated by Step 2.5. T1 is broad;
  T4 is concrete.
- **L level**: final structural tree level. L1 is the top-level thematic anchor;
  L2-L4 are progressively finer structural nodes.
- **F level**: reserved final/evaluation level used by later audit or
  judge-style checks.

The final tree should remain traceable from every structural node back to source
records, cleaned titles, document IDs, and intermediate assignments.

## 2. Repository Mapping

The original local project used paths under `roundC_v4/`. The public package
uses this normalized layout:

```text
PolicyTreeBuilder-replication/
|-- data/
|   |-- source/                    # source input for the main public run
|   |-- intermediate_outputs/      # main Round C v4 intermediate artifacts
|   |-- final_tree/                # final tree, flattened tree, audit files
|   `-- historical_outputs_1120/   # historical run archive
|-- figures/                       # rendered tree visualizations
|-- scripts/                       # pipeline and utility scripts
|-- prompts/                       # LLM prompt templates
|-- configs/                       # YAML configs and .env examples only
|-- assets/                        # L1 sample assets
|-- v10_simulation/                # supplementary reference-normalization work
`-- run_v4_pipeline.ps1            # staged rerun entrypoint
```

Important path translations:

- Original source input: `roundB_outputs/roundB_types_merged1121.csv`
- Public source input: `data/source/roundB_types_merged1121.csv`
- Original main outputs: `roundC_v4/outputs/`
- Public main intermediate outputs: `data/intermediate_outputs/`
- Public final outputs: `data/final_tree/`
- Public figures: `figures/`

Raw `.env` files, raw LLM call dumps, local logs, caches, and the local frozen
archive are intentionally excluded from Git.

## 3. Identity and Traceability

The v4 workflow treats IDs as the backbone of reproducibility.

- **Sample ID** is the atomic unit. It originates in Step 1 and should remain
  stable through later stages.
- **L1 anchor ID** represents a top-level thematic category. L1 anchors are
  created and assigned in Step 2.8, then used as constraints in Step 3.
- **L2-L4 node IDs** represent structural tree nodes built from member samples
  or lower-level nodes.
- **Link files** represent parent-child relations, such as L4-to-L3,
  L3-to-L2, and L2-to-L1.
- **Operation and audit files** record later structural changes made in Step 4.

Key traceability files:

```text
data/intermediate_outputs/v4_l1_node_assignments.csv
data/intermediate_outputs/v4_membership_L2.csv
data/intermediate_outputs/v4_membership_L3.csv
data/intermediate_outputs/v4_membership_L4.csv
data/intermediate_outputs/v4_links_L2_to_L1.csv
data/intermediate_outputs/v4_links_L3_to_L2.csv
data/intermediate_outputs/v4_links_L4_to_L3.csv
data/intermediate_outputs/v4_operations_log.jsonl
data/final_tree/v4_final_audit.json
```

## 4. Environment and Model Configuration

The original implementation uses OpenAI-compatible APIs for LLM calls,
embedding, and reranking.

For public release, real credentials were removed. To rerun LLM-dependent
stages:

1. Copy `configs/roundC_v4.env.example` to `configs/roundC_v4.env`.
2. Fill in model names, base URLs, API keys, embedding model configuration, and
   reranker configuration.
3. Keep `configs/roundC_v4.env` local. It is ignored by Git.

The v4 config files are:

```text
configs/step3_config.yaml
configs/step4_config.yaml
configs/roundC_v4.env.example
```

`configs/roundC.env.example` is retained for older helper modules that still
read the legacy Round C environment format.

## 5. Pipeline Overview

The public package already includes precomputed outputs. Reviewers can inspect
the artifacts directly, or rerun selected stages after filling in local API
credentials.

### Step 1: Prepare Corpus

Script: `scripts/step1_prepare_corpus.py`

Purpose:

- read the source policy-action table;
- normalize titles and source identifiers;
- optionally use an LLM prompt to clean long or noisy titles;
- produce the cleaned v4 corpus.

Main input:

```text
data/source/roundB_types_merged1121.csv
```

Main output:

```text
data/intermediate_outputs/v4_cluster_corpus_cleaned.csv
```

### Step 1.5: Filter T0 / Non-action Structure

Script: `scripts/step1_5_filter_t0.py`

Purpose:

- remove or mark records that are structural headings rather than usable
  policy-action units;
- produce the corpus used for embedding and semantic calibration.

Main output:

```text
data/intermediate_outputs/v4_corpus_filtered.csv
```

### Step 2: Embedding, Nearest Neighbors, and Reranking

Script: `scripts/step2_embed_and_nn.py`

Purpose:

- build embedding vectors for cleaned policy-action titles;
- compute nearest-neighbor candidate edges;
- optionally rerank candidate pairs using a reranking endpoint.

Main outputs:

```text
data/intermediate_outputs/v4_embeddings.parquet
data/intermediate_outputs/v4_rerank_edges.csv
data/intermediate_outputs/v4_edges_L2.csv
data/intermediate_outputs/v4_edges_L3.csv
data/intermediate_outputs/v4_edges_L4.csv
```

### Step 2.5: Semantic Granularity Calibration

Script: `scripts/step2_5_calibrate_levels.py`

Purpose:

- map inconsistent document heading depth into calibrated T-level semantic
  granularity;
- add calibrated-level labels and confidence information.

Main outputs:

```text
data/intermediate_outputs/v4_corpus_calibrated.csv
data/intermediate_outputs/v4_calibration_report.json
```

### Step 2.8: L1 Anchor Construction

Scripts:

```text
scripts/step2_8_define_l1.py
scripts/step2_8_classify_levels.py
```

Purpose:

- define a stable set of L1 thematic anchors;
- assign calibrated records or nodes to those L1 anchors;
- keep uncertain assignments available for review.

Main outputs:

```text
data/intermediate_outputs/v4_l1_definition.json
data/intermediate_outputs/v4_l1_node_assignments.csv
data/intermediate_outputs/v4_l1_doc_assignments.csv
data/intermediate_outputs/v4_l1_classification_review.csv
```

### Step 3: L1-constrained Tree Construction

Scripts:

```text
scripts/step3_layer_merge.py
scripts/step3_vertical_link.py
scripts/step3_link_t2_l1.py
scripts/step3_build_coarse_tree.py
```

Purpose:

- merge semantically close items within the same target layer;
- vertically link lower-level nodes to higher-level parent nodes;
- attach L2 nodes to L1 anchors;
- assemble the coarse global tree before Step 4 refinement.

The practical execution sequence is:

1. initialize or merge L4 nodes;
2. initialize or merge L3 nodes;
3. initialize or merge L2 nodes;
4. link L4 to L3;
5. link L3 to L2;
6. link L2 to L1;
7. build the coarse global tree.

Main outputs:

```text
data/intermediate_outputs/v4_nodes_L1.csv
data/intermediate_outputs/v4_nodes_L2.csv
data/intermediate_outputs/v4_nodes_L3.csv
data/intermediate_outputs/v4_nodes_L4.csv
data/intermediate_outputs/v4_links_L2_to_L1.csv
data/intermediate_outputs/v4_links_L3_to_L2.csv
data/intermediate_outputs/v4_links_L4_to_L3.csv
data/intermediate_outputs/v4_tree_coarse_global.json
```

### Step 4: Structural Refinement

Scripts:

```text
scripts/step4_1_skeleton.py
scripts/step4_2_shaping.py
scripts/step4_3_polishing.py
scripts/step4_5_overall_structure.py
```

Purpose:

- improve the coarse tree while preserving traceability;
- collapse or rehome structurally weak nodes;
- balance oversized branches;
- polish labels and finalize trace information;
- run a final L1-level audit and realignment pass.

Stage roles:

- **Step 4.1 Skeleton**: removes obvious vertical redundancy and repairs coarse
  skeleton structure.
- **Step 4.2 Shaping**: balances structure, promotes or bridges nodes where
  needed, and improves branch organization.
- **Step 4.3 Polishing**: refines labels and consolidates trace information.
- **Step 4.5 Final Override**: applies final high-level audit decisions and
  writes the final tree.

Main outputs:

```text
data/intermediate_outputs/v4_tree_s4_1.json
data/intermediate_outputs/v4_tree_s4_2.json
data/intermediate_outputs/v4_tree_refined.json
data/final_tree/v4_tree_final.json
data/final_tree/v4_tree_final_flat.csv
data/final_tree/v4_tree_levels.csv
data/final_tree/v4_final_audit.json
```

### Step 5 and Later: Optional Review and Visualization

Relevant scripts include:

```text
scripts/step5_1_extract_t5.py
scripts/step5_data_preparation.py
scripts/step5_split_tree_by_admin.py
scripts/step6_visualize_trees.py
scripts/step6_visualize_admin_trees.py
```

These scripts support later audit sampling, T5-style extraction review,
administrative-unit splitting, and visualizations. The administrative split
requires an optional mapping file that is not included in the current public
package:

```text
data/source/admin_mapping/roundA_final_overview_scored_selected1120.csv
```

## 6. Main Public Outputs

For paper review, the most important files are:

```text
data/source/roundB_types_merged1121.csv
data/intermediate_outputs/v4_tree_coarse_global.json
data/intermediate_outputs/v4_operations_log.jsonl
data/final_tree/v4_tree_final.json
data/final_tree/v4_tree_final_flat.csv
data/final_tree/v4_tree_levels.csv
data/final_tree/v4_final_audit.json
figures/v4_tree_provincial.png
figures/v4_tree_city.png
```

`FILE_INDEX.tsv` gives file sizes and SHA256 hashes for the public package
contents.

## 7. Running the Pipeline

The package includes a path-normalized PowerShell entrypoint:

```powershell
.\run_v4_pipeline.ps1
```

Several early LLM-heavy preparation steps are commented out because the public
package already includes precomputed source and intermediate outputs. Reviewers
can uncomment and rerun them after configuring credentials.

## 8. Historical Output Archive

`data/historical_outputs_1120/` is retained for auditability. It corresponds to
an earlier run archived from the development workspace.

That historical run references `roundB_types_merged1113_test.csv`, which was
not found during packaging. Therefore:

- use `data/intermediate_outputs/` and `data/final_tree/` as the main public
  replication path;
- treat `data/historical_outputs_1120/` as a historical reference archive, not
  the primary reproducible run.

## 9. Public-release Safety Decisions

The following were excluded from Git:

- real `.env` files and API keys;
- local key files such as `gemini_key.txt` and `qwen_key.txt`;
- raw LLM call dumps;
- local log directories;
- Python cache files;
- the local-only frozen archive under `_local_archive/`.

The local archive and SHA256 manifest remain on the author's machine to preserve
the exact paper-version `roundC_v4` snapshot.
