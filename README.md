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

## Repository Layout

- `scripts/`: pipeline, tree refinement, administrative splitting, and figure rendering scripts.
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

## Main Reproduction Path

The final outputs are already included. To rerun the pipeline from the source input, use `run_policy_tree_pipeline.ps1` and the workflow in `TECHNICAL_README.md`.

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

`policy_tree_eval` is not included in this public package. It was kept locally only as a consistency check: its input tree matches the 353-node final tree by extracted-file SHA256, but its scores were not used in the paper's main results.

See `PUBLICATION_SNAPSHOT.md` for the fixed publication snapshot and `replication_package.md` for the package index.
