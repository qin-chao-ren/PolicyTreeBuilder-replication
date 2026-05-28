# Policy Tree Evaluation

This directory contains the public evaluation module for the fixed 353-node policy action tree.

The included `outputs/` files are the archived evaluation artifacts generated from `data/final_tree/policy_tree_final.json`. They include deterministic table extraction, structure checks, sampled judge inputs, model judge outputs, agreement tables, divergent-sample reports, and final summaries.

## Inputs And Outputs

- Default input tree: `data/final_tree/policy_tree_final.json`
- Default output directory: `evaluation/outputs/`
- Safe local environment template: `evaluation/.env.example`
- Optional model registry template: `evaluation/models_registry.yaml.example`
- Path helper: `evaluation/scripts/eval_paths.py` centralizes repository-relative defaults for the public package.

The legacy evaluation input name was `policy_tree_eval/data/v4_tree_final.json`. It is not copied into this repository because it corresponds to the current public final tree.

## Deterministic Steps

These steps do not call external APIs:

```powershell
python evaluation/scripts/01_extract_tables.py
python evaluation/scripts/02_structure_check.py
python evaluation/scripts/03_sampling.py
python evaluation/scripts/06_aggregate.py
python evaluation/scripts/07_status.py
```

To write to a separate scratch directory:

```powershell
python evaluation/scripts/01_extract_tables.py --output-dir evaluation/outputs_scratch
python evaluation/scripts/02_structure_check.py --output-dir evaluation/outputs_scratch
python evaluation/scripts/03_sampling.py --output-dir evaluation/outputs_scratch --seed 20260430
```

## Optional Model Judging

Model judging calls external APIs. Copy the safe template and fill in local values before running:

```powershell
Copy-Item evaluation/.env.example evaluation/.env
```

Then run one judge or all configured judges:

```powershell
python evaluation/scripts/04_run_node_judge.py --judge A_kimi --limit 3
python evaluation/scripts/05_run_path_judge.py --judge A_kimi --limit 3
python evaluation/scripts/04_run_node_judge.py --judge all
python evaluation/scripts/05_run_path_judge.py --judge all
```

The judging scripts resume from existing `judge_*_scores.jsonl` files by default. Use `--no-resume` only when intentionally appending a new run to a clean output directory.

## Archived Summary

The archived evaluation reports:

- 353 nodes, 352 edges, 272 root-to-leaf paths
- 278 sampled nodes and 51 sampled paths
- Judges: `A_kimi`, `B_claude`, `C_gemini`
- Final multi-model framework score: `2.476`
