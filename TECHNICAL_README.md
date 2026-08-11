# PolicyTreeBuilder Technical Workflow

This document describes the public ATRS 2026 replication workflow and its archived 353-node PolicyTreeBuilder result.

## Inputs

Primary source input:

```text
data/source/policy_action_segments.csv
```

Administrative metadata input:

```text
data/source/administrative_unit_metadata.csv
```

## Configuration

YAML configs are in `configs/`.

Create a local environment file only if rerunning LLM-dependent steps:

```powershell
Copy-Item configs\.env.example configs\.env
```

The committed `.env.example` contains only variable names. Real credentials are intentionally excluded.

## Pipeline Overview

The public pipeline follows this sequence:

1. Prepare and clean the source corpus with `scripts/prepare_policy_corpus.py`.
2. Filter non-policy records with `scripts/filter_non_policy_records.py`.
3. Generate embeddings and similarity/rerank edges with `scripts/embed_policy_corpus.py`.
4. Calibrate policy-action granularity with `scripts/calibrate_policy_granularity.py`.
5. Define and assign top-level categories with `scripts/define_top_level_categories.py` and `scripts/assign_top_level_categories.py`.
6. Build action clusters, assign parent clusters, and build the initial tree.
7. Collapse redundant hierarchy, balance structure, polish labels, and finalize the tree.
8. Split administrative trees with `visualization/split_tree_by_administrative_unit.py` and render radial figures with `visualization/render_radial_tree_figure.py` when needed.
9. Run optional tree-quality evaluation with the public scripts in `evaluation/scripts/`.

Intermediate outputs are stored in `data/intermediate_outputs/`. The archived paper tree is stored in `data/final_tree/`; a rerun is a new candidate and must pass E0 before publication.

## Minimal Rerun Skeleton

The included `run_policy_tree_pipeline.ps1` is the full command template. It requires local credentials for external LLM, embedding, and reranking services. Reviewers who only need to inspect or validate the published result can use the included outputs and the deterministic evaluation commands below.

Typical direct commands use this pattern:

```powershell
python scripts/prepare_policy_corpus.py `
  --source data/source/policy_action_segments.csv `
  --env configs/.env `
  --outdir data/intermediate_outputs `
  --llm-clean yes

python scripts/build_initial_policy_tree.py `
  --config configs/tree_build_config.yaml

python scripts/finalize_policy_tree.py `
  --input data/intermediate_outputs/policy_tree_refined.json `
  --output data/final_tree/policy_tree_final.json `
  --config configs/tree_refinement_config.yaml `
  --l1-def data/intermediate_outputs/top_level_categories.json `
  --audit-out data/final_tree/policy_tree_final_audit.json `
  --flat-csv data/final_tree/policy_tree_final_flat.csv `
  --membership-input data/intermediate_outputs/policy_tree_final_membership.csv `
  --membership-out data/final_tree/policy_tree_final_membership.csv `
  --lineage-in data/intermediate_outputs/policy_tree_lineage.json `
  --lineage-out data/final_tree/policy_tree_final_lineage.json `
  --operations-input data/intermediate_outputs/tree_refinement_operations.jsonl `
  --operations-out data/final_tree/policy_tree_final_operations.jsonl
```

Some steps call external LLM or embedding services. Reviewers without access to the same services can inspect the included intermediate and final outputs directly.

## LLM Runtime Boundary

All external model calls use `scripts/llm_runtime.py` and named profiles from `configs/llm_profiles.yaml.example`.

- Main pipeline chat calls use `pipeline_primary` and, where needed, `pipeline_secondary`.
- Evaluation judges use `judge_A_kimi`, `judge_B_claude`, and `judge_C_gemini` profiles while preserving the archived `judge_*` output filenames.
- Embedding and reranking use `embedding_default` and `rerank_default`.

Copy `configs/llm_profiles.yaml.example` to `configs/llm_profiles.yaml` only if you need to change provider/profile wiring. Credentials and endpoints should stay in local `.env` files.

### Call-local semantic references

The four refinement/finalization model stages (14a–14d) never ask a model to reproduce persistent tree node IDs. For each logical call, the runtime creates a one-call mapping such as `PARENT`, `CHILD`, `LEFT`, `RIGHT`, or `N0`, plus a deterministic `context_token`. That token locks the localized user context, known-node-set digest, system prompt/schema contract, task, and expected decision count. The model-facing schema is `schemas/local_reference_semantic_tree_decision.schema.json`; `scripts/utils/local_reference_binding.py` verifies the token and exact refs, runs the stage's pure scope check, binds refs to persistent IDs, and only then hands the bound decision to the existing semantic contract.

Unknown, duplicate, differently cased, cross-context, or real-ID values fail before a live-tree mutation. There is no prefix, edit-distance, or similar-ID fallback. A parseable local-schema or reference-scope error may receive at most one reference-only repair, and the repair must preserve the decision count/order and every non-reference semantic field. A repair is not sent if its prompt would echo a real-ID-shaped value. Transport/JSON retries and that repair share one limit of four actual request attempts. `scripts/llm_runtime.py` records each actual attempt so the logical-call log can be reconciled with request logs. Multi-decision 14b/14d responses execute as candidate-tree transactions and commit to the live tree only when the whole batch passes.

Every model-response field, including free-text evidence, rejects known or real-ID-shaped values. Historical replay keeps original archived evidence byte-locked as source provenance, but its synthetic model-facing copy replaces ID tokens with deterministic local archival refs before binding. Candidate-tree, lineage/stat sidecars, in-memory audit rows, and each logical-call JSONL batch are committed together or restored to their pre-call state if persistence fails.

## Evaluation Workflow

The evaluation module defaults to the archived paper tree and writes to `evaluation/outputs/`:

```powershell
python evaluation/scripts/01_extract_tables.py
python evaluation/scripts/02_structure_check.py
python evaluation/scripts/03_sampling.py
python evaluation/scripts/06_aggregate.py
python evaluation/scripts/07_status.py
```

The deterministic steps above do not call external APIs. Model judging is optional and requires local credentials in `evaluation/.env`, created from `evaluation/.env.example`:

```powershell
python evaluation/scripts/04_run_node_judge.py --judge A_kimi --limit 3
python evaluation/scripts/05_run_path_judge.py --judge A_kimi --limit 3
```

The archived evaluation outputs include 278 sampled node judgments, 51 sampled path judgments, and the final multi-model framework score reported in `evaluation/outputs/final_summary.json`.

## Historical Output Contract

The committed paper snapshot is identified by:

- `data/final_tree/policy_tree_final.json`
- 353 nodes
- 352 edges
- 272 leaf nodes
- maximum depth 6
- SHA256 `9242d4961e417ffa1e30e728d82e73bf669e4facdedd12f4f03f10d19b157983` after repository LF normalization

The source archive extracted-file SHA256 before LF normalization is `6ee8e666dfc5bb7b8611a42c96c8ac93f766290b290115529686f9f3ca67918b`.

These values identify the archived artifact; they do not imply E0 compliance. The frozen tree is retained as a regression negative control and currently fails E0 with 8 exact sibling-duplicate groups and 10 level/depth mismatches. Its final operation log also has 34 `applied` merges whose source nodes remain present.

## Candidate Publication Contract

`finalize_policy_tree.py` validates candidate data before it replaces formal output files. Structural E0 requires:

- unique non-empty node IDs, one root, no cycles, and exactly one parent per non-root node;
- raw DFS IDs equal to the live manager index, with every raw child edge equal to `parent_map`;
- zero exact sibling duplicates and zero exact parent-child duplicates unless explicitly allowlisted;
- declared `level` equal to physical depth;
- acyclic, transitively closed lineage whose terminal targets are live and whose redirected sources are absent;
- no dangling membership targets and exact conservation of membership row identities;
- true postconditions for every refinement or finalization operation recorded as `applied`.

The separate semantic-contract gate requires every applied semantic operation to retain its pre-operation context and a passing shared decision report. Only `exact_duplicate` or `synonym` may authorize destructive merge; direct source membership needs explicit full-representation evidence; and merge, flatten, move, bridge, and split operations require complete compatible child plans where children are affected. The public schema is `schemas/semantic_tree_decision.schema.json`.

Step 4 writes fresh operation and membership outputs for each refinement run; an explicitly requested membership input must exist. Finalization combines the operation trace with its own records, reports structural E0 and the semantic contract separately, and publishes the combined JSONL only when both pass. The audit report is always written. On any critical violation, the command exits nonzero and leaves the formal tree, membership, flat table, lineage, and operation files unchanged. On PASS, auxiliary files are atomically replaced first and the formal tree JSON is replaced last. If a caught write fails partway through publication, the formal outputs are restored from their pre-publication byte snapshots and the audit is changed to FAIL. The PowerShell wrapper stops immediately on every nonzero exit.

The structural E0 validator can also be run independently without API access:

```powershell
python scripts/validate_tree_e0.py `
  --tree path/to/candidate.json `
  --membership path/to/candidate_membership.csv `
  --lineage path/to/candidate_lineage.json `
  --operations path/to/candidate_operations.jsonl `
  --require-membership `
  --report path/to/e0_report.json
```

Offline regression tests use the Python standard library:

```powershell
python -m unittest discover -s tests -v
```
