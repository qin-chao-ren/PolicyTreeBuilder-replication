# PolicyTreeBuilder Replication Package

This repository contains the public replication package for the ATRS 2026 PolicyTreeBuilder experiment. The fixed publication result is a 353-node policy action tree.

## Final Version

Final tree file: `data/final_tree/policy_tree_final.json`

Expected structure:

- Nodes: 353
- Edges: 352
- Leaf nodes: 272
- Maximum depth: 6
- Repository SHA256 for `data/final_tree/policy_tree_final.json`: `9242d4961e417ffa1e30e728d82e73bf669e4facdedd12f4f03f10d19b157983`
- Source archive extracted-file SHA256 before repository LF normalization: `6ee8e666dfc5bb7b8611a42c96c8ac93f766290b290115529686f9f3ca67918b`

The older 317-node package is superseded and is not part of this public replication package.

## Reviewer Quick Check

Reviewers who do not need to rerun the full LLM pipeline can inspect the fixed outputs directly:

| Purpose | File or command |
| --- | --- |
| Final policy tree | `data/final_tree/policy_tree_final.json` |
| Final radial figure | `data/final_tree/policy_tree_final_en_radial.jpg` |
| Final tree tables and audit files | `data/final_tree/` |
| Evaluation summary | `evaluation/outputs/final_summary.json` |
| Package file manifest | `FILE_INDEX.tsv` |
| Legacy-to-public name map | `LEGACY_NAME_MAP.tsv` |

No-API validation commands:

```powershell
python evaluation/scripts/01_extract_tables.py --output-dir evaluation/outputs_scratch
python evaluation/scripts/02_structure_check.py --output-dir evaluation/outputs_scratch
python evaluation/scripts/03_sampling.py --output-dir evaluation/outputs_scratch --seed 20260430
python evaluation/scripts/06_aggregate.py --output-dir evaluation/outputs
python evaluation/scripts/07_status.py --output-dir evaluation/outputs
```

The full pipeline rerun is optional and requires local credentials for external LLM, embedding, and reranking services. Use the included outputs for review when those services are unavailable.

## Repository Layout

- `scripts/`: main pipeline and tree refinement scripts.
- `visualization/`: administrative tree splitting, visualization repair, and figure rendering scripts.
- `audit/`: optional human-audit preparation scripts for action-unit extraction checks.
- `evaluation/`: public tree-quality evaluation scripts and archived evaluation outputs.
- `prompts/`: LLM prompt templates used by the pipeline.
- `configs/`: YAML pipeline configs and a safe `.env.example` template.
- `data/source/`: source input segments and administrative-unit metadata.
- `data/intermediate_outputs/`: included intermediate outputs, logs, embeddings, and trace files.
- `data/final_tree/`: final tree, final tabular outputs, academic English tree variants, and paper figure assets.
- `LEGACY_NAME_MAP.tsv`: mapping from legacy development names to the public package names.
- `SCRIPT_PROVENANCE.tsv`: source and public hashes for path-normalized scripts.

## Setup

Use Python 3.12 or later.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

For exact package versions from the review environment, see `requirements-lock.txt`.

To rerun LLM-dependent steps, copy the template and fill in local credentials:

```powershell
Copy-Item configs\.env.example configs\.env
```

Do not commit `configs/.env`.

For optional reruns of the evaluation judges, copy the evaluation-specific template:

```powershell
Copy-Item evaluation\.env.example evaluation\.env
```

Do not commit `evaluation/.env`.

## Main Reproduction Path

The final outputs are already included. To rerun the pipeline from the source input, use `run_policy_tree_pipeline.ps1` and the workflow in `TECHNICAL_README.md`. This script is a full rerun template and is not required for basic review.

Primary input:

```text
data/source/policy_action_segments.csv
```

Administrative metadata:

```text
data/source/administrative_unit_metadata.csv
```

Main output:

```text
data/final_tree/policy_tree_final.json
```

## Notes For Reviewers

The public evaluation module is in `evaluation/`. Its archived outputs evaluate the same 353-node final tree and include deterministic structure checks, sampled node/path judge inputs, model judge outputs, agreement tables, divergent-sample reports, and final summaries.

The legacy local directory name `policy_tree_eval` is intentionally not restored. The legacy input `v4_tree_final.json` maps to `data/final_tree/policy_tree_final.json`.

See `PUBLICATION_SNAPSHOT.md` for the fixed publication snapshot and `replication_package.md` for the package index.
