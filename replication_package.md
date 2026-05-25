# Replication Package Index

This repository is organized for review of the ATRS 2026 353-node PolicyTreeBuilder result.

## Core Files

| Path | Purpose |
| --- | --- |
| `data/source/roundB_types_merged1121.csv` | Primary Round B input for the Round C v4 pipeline. |
| `data/source/admin_mapping/roundA_final_overview_scored_selected1120.csv` | Administrative mapping source used for administrative tree splitting. |
| `data/intermediate_outputs/` | Intermediate Round C v4 outputs, logs, embeddings, and trace files. |
| `data/final_tree/v4_tree_final.json` | Final 353-node policy tree used for ATRS 2026. |
| `figures/` | Figures generated from the 353-node Round C v4 outputs. |
| `scripts/` | Round C v4 source scripts from the 353-node source version. |
| `prompts/` | LLM prompt templates used by Round C v4. |
| `configs/` | YAML configs and safe environment template. |
| `SCRIPT_PROVENANCE.tsv` | Hash mapping from extracted Round C v4 source scripts to public path-normalized scripts. |

## Version Notes

- The 353-node tree is the final paper version.
- The 317-node tree is superseded and is not part of this public package.
- `policy_tree_eval` is local-only and not part of the public replication package.
- `v10_simulation/` is supplementary and not required for rebuilding the 353-node tree.

## Generated Manifest

`FILE_INDEX.tsv` lists public files with size and SHA256 after repository preparation.
