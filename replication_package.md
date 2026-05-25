# Replication Package Index

This repository is organized for review of the ATRS 2026 353-node PolicyTreeBuilder result.

## Core Files

| Path | Purpose |
| --- | --- |
| `data/source/policy_action_segments.csv` | Primary source input for the public pipeline. |
| `data/source/administrative_unit_metadata.csv` | Administrative metadata used for administrative tree splitting. |
| `data/intermediate_outputs/` | Included intermediate outputs, logs, embeddings, and trace files. |
| `data/final_tree/policy_tree_final.json` | Final 353-node policy tree used for ATRS 2026. |
| `data/final_tree/policy_tree_label_map_en_academic.json` | Academic English label map used for paper figure preparation. |
| `data/final_tree/policy_tree_final_en_academic.json` | Academic English final tree used by the radial figure script. |
| `data/final_tree/policy_tree_provincial_en_academic.json` | Academic English provincial subset tree. |
| `data/final_tree/policy_tree_city_en_academic.json` | Academic English city-level subset tree. |
| `data/final_tree/policy_tree_final_en_radial.jpg` | Final English radial tree figure image prepared for the paper. |
| `scripts/` | Public pipeline and figure-generation scripts. |
| `prompts/` | LLM prompt templates used by the pipeline. |
| `configs/` | YAML configs and safe environment template. |
| `LEGACY_NAME_MAP.tsv` | Legacy-to-public path mapping for traceability. |
| `SCRIPT_PROVENANCE.tsv` | Hash mapping from extracted source scripts to public path-normalized scripts. |

## Version Notes

- The 353-node tree is the final paper version.
- The 317-node tree is superseded and is not part of this public package.
- `policy_tree_eval` is local-only and not part of the public replication package.

## Generated Manifest

`FILE_INDEX.tsv` lists public files with size and SHA256 after repository preparation.
