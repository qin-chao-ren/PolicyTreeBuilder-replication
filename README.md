# PolicyTreeBuilder Replication Package

This repository contains the public replication package for the ATRS 2026 PolicyTreeBuilder experiment. The archived publication result is a 353-node policy action tree.

## Archived Paper Version

Archived tree file: `data/final_tree/policy_tree_final.json`

Expected structure:

- Nodes: 353
- Edges: 352
- Leaf nodes: 272
- Maximum depth: 6
- Repository SHA256 for `data/final_tree/policy_tree_final.json`: `9242d4961e417ffa1e30e728d82e73bf669e4facdedd12f4f03f10d19b157983`
- Source archive extracted-file SHA256 before repository LF normalization: `6ee8e666dfc5bb7b8611a42c96c8ac93f766290b290115529686f9f3ca67918b`

The older 317-node package is superseded and is not part of this public replication package.

### Historical snapshot and E0 status

The committed 353-node tree is preserved byte-for-byte as the historical paper snapshot; the hardening changes do not silently replace it. The deterministic E0 validator rejects that snapshot as a negative control: it contains 8 exact sibling-duplicate groups, 10 declared-level/physical-depth mismatches, and 34 logged `applied` merges whose sources are still present. These findings affect the engineering integrity of the archived output, not the ability to inspect the exact artifact used for the paper.

New pipeline candidates are fail-closed: `finalize_policy_tree.py` publishes the formal tree only after raw/index/parent consistency, unique IDs, acyclicity, exact-label uniqueness, level/depth alignment, closed lineage, membership conservation, and applied-operation postconditions all pass. An explicitly requested membership input must exist, and a caught write failure restores the pre-publication formal outputs. The historical 353-node tree remains unchanged until a separately versioned corrected run passes that contract.

## Reviewer Quick Check

Reviewers who do not need to rerun the full LLM pipeline can inspect the archived outputs directly:

| Purpose | File or command |
| --- | --- |
| Archived policy tree | `data/final_tree/policy_tree_final.json` |
| Archived radial figure | `data/final_tree/policy_tree_final_en_radial.jpg` |
| Archived tree tables and audit files | `data/final_tree/` |
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

Run the new E0 validator independently with:

```powershell
python scripts/validate_tree_e0.py `
  --tree data/final_tree/policy_tree_final.json `
  --membership data/final_tree/policy_tree_final_membership.csv `
  --operations data/final_tree/policy_tree_final_operations.jsonl `
  --require-membership `
  --report evaluation/outputs_scratch/policy_tree_e0_report.json
```

The command intentionally exits nonzero for the frozen 353-node negative control. A newly published candidate must exit zero.

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
- `data/final_tree/`: archived tree, tabular outputs, academic English tree variants, and paper figure assets.
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

Model profile wiring is centralized in `configs/llm_profiles.yaml.example`; copy it to `configs/llm_profiles.yaml` only if you need to change profile names, providers, or env-var bindings.

Do not commit `configs/.env`.

For optional reruns of the evaluation judges, copy the evaluation-specific template:

```powershell
Copy-Item evaluation\.env.example evaluation\.env
```

Do not commit `evaluation/.env`.

## Main Reproduction Path

The archived outputs are already included. To create a new candidate from the source input, use `run_policy_tree_pipeline.ps1` and the workflow in `TECHNICAL_README.md`. The wrapper stops on the first nonzero Python exit, and finalization replaces formal outputs only after E0 passes.

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

See `PUBLICATION_SNAPSHOT.md` for the historical publication snapshot and `replication_package.md` for the package index.
