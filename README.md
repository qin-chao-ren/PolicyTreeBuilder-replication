# PolicyTreeBuilder Replication Package

This repository contains the replication package for the ATRS 2026 PolicyTreeBuilder experiment.

The public replication version is the final 353-node Round C v4 tree used for the paper. The code, prompts, YAML configs, source inputs, intermediate outputs, final tree files, and figures in this package were rebuilt from the Round C v4 project archive that produced the 353-node tree.

## Final Version

Final tree file: `data/final_tree/v4_tree_final.json`

Expected structure:

- Nodes: 353
- Edges: 352
- Leaf nodes: 272
- Maximum depth: 6
- Repository SHA256 for `data/final_tree/v4_tree_final.json`: `9242d4961e417ffa1e30e728d82e73bf669e4facdedd12f4f03f10d19b157983`
- Source archive extracted-file SHA256 before repository LF normalization: `6ee8e666dfc5bb7b8611a42c96c8ac93f766290b290115529686f9f3ca67918b`

The older 317-node package is not part of the ATRS 2026 replication version. It is retained only as a local superseded archive and is not uploaded.

## Repository Layout

- `scripts/`: Round C v4 pipeline scripts from the 353-node source version.
- `prompts/`: LLM prompt templates used by the pipeline.
- `configs/`: YAML pipeline configs and a safe `.env.example` template.
- `data/source/`: source inputs for the 353-node run.
- `data/intermediate_outputs/`: intermediate Round C v4 outputs and provenance logs.
- `data/final_tree/`: final 353-node tree and final tabular outputs.
- `figures/`: figures generated from the 353-node Round C v4 outputs.
- `v10_simulation/`: supplementary simulation materials retained from the previous package; not part of the main 353-node tree-building pipeline.
- `SCRIPT_PROVENANCE.tsv`: source and public hashes for the Round C v4 scripts after repository path normalization.

## Setup

Use Python 3.12 or later. The original local environment used Python 3.12.x.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

For exact local package versions from the previous review environment, see `requirements-lock.txt`.

To run LLM-dependent steps, copy the template and fill in local credentials:

```powershell
Copy-Item configs\.env.example configs\.env
```

Do not commit `configs/.env`.

## Main Reproduction Path

The provided final outputs are already included. To rerun the main pipeline from the source input, use the commands documented in `TECHNICAL_README.md` and the template `run_v4_pipeline.ps1`.

Primary input:

```text
data/source/roundB_types_merged1121.csv
```

Main output:

```text
data/final_tree/v4_tree_final.json
```

Historical/test input retained for traceability:

```text
data/source/roundB_types_merged1113_test.csv
```

Administrative mapping source:

```text
data/source/admin_mapping/roundA_final_overview_scored_selected1120.csv
```

## Notes For Reviewers

`policy_tree_eval` is not included in this public package. It was kept locally only as a consistency check: its input tree matches the 353-node final tree by extracted-file SHA256, but its scores were not used in the paper's main results.

See `PUBLICATION_SNAPSHOT.md` for the fixed publication snapshot and `replication_package.md` for the file index.
