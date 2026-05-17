# Replication Package Description

This file describes the public review package for PolicyTreeBuilder.

## 1. Source Input

- `data/source/roundB_types_merged1121.csv`: source file used by the main Round C v4 run.
- `data/source/admin_mapping/`: placeholder for the optional administrative mapping file used only by administrative split/visualization scripts.
- `data/source/missing_optional/`: notes on historical inputs referenced by archived output metadata but not found during packaging.

## 2. Prompts and Configuration

- `prompts/`: prompt templates for title cleaning, semantic calibration, L1 anchoring, tree construction, and final structure adjustment.
- `configs/*.yaml`: public YAML configuration files with repository-relative paths.
- `configs/*.env.example`: environment templates. Real `.env` files are intentionally excluded.

## 3. Scripts

- `scripts/`: Round C v4 processing scripts and utility modules.
- `run_v4_pipeline.ps1`: path-normalized PowerShell entrypoint for staged reruns.
- `v10_simulation/`: supplementary reference-normalization simulation materials.

## 4. Intermediate Outputs

- `data/intermediate_outputs/`: main intermediate artifacts from the Round C v4 pipeline.
- Raw LLM call dumps and log directories are excluded from the public package for safety.

## 5. Final Tree and Figures

- `data/final_tree/v4_tree_final.json`: final hierarchical policy-action tree.
- `data/final_tree/v4_tree_final_flat.csv`: flattened final tree.
- `data/final_tree/v4_tree_levels.csv`: final tree level table.
- `data/final_tree/v4_final_audit.json`: final audit report.
- `figures/`: rendered visualizations.

## 6. Historical Output Archive

- `data/historical_outputs_1120/`: archived historical run. This is included for auditability but is not the primary replication path because its metadata references a missing source file, `roundB_types_merged1113_test.csv`.

## 7. Reproducibility Notes

- Python version in the author's environment: 3.12.7.
- `requirements.txt` gives the concise dependency set.
- `requirements-lock.txt` gives exact installed package versions.
- Local-only frozen source snapshot and SHA256 manifest are stored under `_local_archive/` on the author's machine and ignored by Git.
