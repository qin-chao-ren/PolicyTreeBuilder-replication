# PolicyTreeBuilder Replication Package

This repository contains the public replication package for the paper:

**PolicyTreeBuilder: An LLM-assisted framework for constructing hierarchical policy-action trees from Chinese air-cargo policy documents**

The package is organized for review and auditability. It includes the source input used by the main Round C v4 run, prompts, scripts, intermediate artifacts, final tree outputs, and visualizations. Private API keys, local environment files, raw log directories, and local-only archives are excluded from Git.

## Repository Structure

```text
PolicyTreeBuilder-replication/
|-- assets/                         # L1 sample assets used by the pipeline
|-- configs/                        # YAML configs and .env examples only
|-- data/
|   |-- source/                     # Public source inputs for replication
|   |-- intermediate_outputs/       # Main Round C v4 intermediate artifacts
|   |-- final_tree/                 # Final policy-action tree and audit files
|   `-- historical_outputs_1120/    # Historical output archive, not main path
|-- figures/                        # Rendered tree visualizations
|-- prompts/                        # Prompt templates used by the pipeline
|-- scripts/                        # Processing, tree-building, and visualization scripts
|-- v10_simulation/                 # Additional reference-normalization simulation materials
|-- requirements.txt                # Lightweight dependency list
|-- requirements-lock.txt           # Exact package versions from the author's Python 3.12.7 environment
|-- TECHNICAL_README.md             # Detailed Round C v4 technical workflow
|-- replication_package.md          # File index and replication notes
`-- run_v4_pipeline.ps1             # PowerShell pipeline entrypoint, path-normalized for this repo
```

## Environment

The original development environment used Python 3.12.7 on Windows.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

For closer reproduction of the author's environment:

```powershell
python -m pip install -r requirements-lock.txt
```

## API Configuration

The public repository does not include real API keys. To rerun LLM-dependent steps:

1. Copy `configs/roundC_v4.env.example` to `configs/roundC_v4.env`.
2. Fill in the OpenAI-compatible base URLs, API keys, model names, embedding model, and reranker endpoint.
3. Keep `configs/roundC_v4.env` local. It is ignored by Git.

`configs/roundC.env.example` is included for older helper code that still reads the legacy Round C environment format.

## Main Replication Path

Primary source input:

```text
data/source/roundB_types_merged1121.csv
```

Precomputed outputs:

```text
data/intermediate_outputs/
data/final_tree/
figures/
```

Key final outputs:

```text
data/final_tree/v4_tree_final.json
data/final_tree/v4_tree_final_flat.csv
data/final_tree/v4_tree_levels.csv
data/final_tree/v4_final_audit.json
```

For the detailed technical workflow, see `TECHNICAL_README.md`.

To rerun after configuring API credentials, review and run:

```powershell
.\run_v4_pipeline.ps1
```

Some commands are intentionally commented in the pipeline file so reviewers can run stages selectively.

## Known Optional or Historical Inputs

- `data/source/admin_mapping/roundA_final_overview_scored_selected1120.csv` is not included yet. Place it there when available. It is only required for administrative-unit split/visualization scripts.
- `data/historical_outputs_1120/` is preserved as a historical output archive. Its metadata references `roundB_types_merged1113_test.csv`, which was not found under `<LOCAL_SEARCH_ROOT>` during packaging, so it is not treated as the primary reproducible path.
- Raw LLM call dumps and log directories are excluded from the public package for safety. Structural operation logs and audit files needed for review are retained where appropriate.

## Local Paper Snapshot

On the author's local machine, `_local_archive/roundC_v4_paper_snapshot_2026-05-17.zip` and `_local_archive/MANIFEST.sha256` preserve the exact `roundC_v4` folder snapshot used to build this public package. `_local_archive/` is intentionally ignored by Git and is not part of the public release.
