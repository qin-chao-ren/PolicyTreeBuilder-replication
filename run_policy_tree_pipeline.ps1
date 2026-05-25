# PolicyTreeBuilder public replication pipeline template.
# Run from the repository root after configuring configs/.env.

Write-Host "[1/14] Prepare policy corpus"
python scripts/prepare_policy_corpus.py `
  --source data/source/policy_action_segments.csv `
  --env configs/.env `
  --outdir data/intermediate_outputs `
  --llm-clean yes `
  --debug-dump-llm yes `
  --title-max-len 40

Write-Host "[2/14] Filter non-policy records"
python scripts/filter_non_policy_records.py `
  --corpus data/intermediate_outputs/policy_corpus_cleaned.csv `
  --outdir data/intermediate_outputs `
  --mode delete

Write-Host "[3/14] Embed policy corpus and build similarity edges"
python scripts/embed_policy_corpus.py `
  --corpus data/intermediate_outputs/policy_corpus_filtered.csv `
  --env configs/.env `
  --outdir data/intermediate_outputs `
  --embed-text-template title_plus_path `
  --recompute no `
  --nn-topk 294 `
  --mutual-knn no `
  --local-k 20 `
  --min-sim 0.0 `
  --use-reranker yes `
  --rerank-scope all `
  --rerank-threshold 0.55 `
  --rerank-mode soft

Write-Host "[4/14] Calibrate policy granularity"
python scripts/calibrate_policy_granularity.py `
  --corpus data/intermediate_outputs/policy_corpus_filtered.csv `
  --env configs/.env `
  --outdir data/intermediate_outputs

Write-Host "[5/14] Define top-level categories"
python scripts/define_top_level_categories.py `
  --calibrated data/intermediate_outputs/policy_corpus_calibrated.csv `
  --assets-dir assets/l1_samples `
  --env configs/.env `
  --outdir data/intermediate_outputs

Write-Host "[6/14] Assign top-level categories"
python scripts/assign_top_level_categories.py `
  --l1-def data/intermediate_outputs/top_level_categories.json `
  --corpus data/intermediate_outputs/policy_corpus_calibrated.csv `
  --env configs/.env `
  --outdir data/intermediate_outputs

Write-Host "[7/14] Build L4 action clusters"
python scripts/build_action_clusters.py --level L4 --config configs/tree_build_config.yaml

Write-Host "[8/14] Build L3 action clusters"
python scripts/build_action_clusters.py --level L3 --config configs/tree_build_config.yaml

Write-Host "[9/14] Build L2 action clusters"
python scripts/build_action_clusters.py --level L2 --config configs/tree_build_config.yaml

Write-Host "[10/14] Assign L4 clusters to L3 parents"
python scripts/assign_parent_clusters.py --from-level L4 --to-level L3 --config configs/tree_build_config.yaml

Write-Host "[11/14] Assign L3 clusters to L2 parents"
python scripts/assign_parent_clusters.py --from-level L3 --to-level L2 --config configs/tree_build_config.yaml

Write-Host "[12/14] Link L2 clusters to top-level categories"
python scripts/link_second_level_to_top_level.py --config configs/tree_build_config.yaml

Write-Host "[13/14] Build initial policy tree"
python scripts/build_initial_policy_tree.py --config configs/tree_build_config.yaml --emit-samples no

Write-Host "[14/14] Refine and finalize policy tree"
python scripts/collapse_redundant_hierarchy.py `
  --input data/intermediate_outputs/policy_tree_initial.json `
  --output data/intermediate_outputs/policy_tree_after_vertical_collapse.json `
  --config configs/tree_refinement_config.yaml

python scripts/balance_tree_structure.py `
  --input data/intermediate_outputs/policy_tree_after_vertical_collapse.json `
  --output data/intermediate_outputs/policy_tree_after_structure_balancing.json `
  --config configs/tree_refinement_config.yaml

python scripts/polish_tree_labels.py `
  --input data/intermediate_outputs/policy_tree_after_structure_balancing.json `
  --output data/intermediate_outputs/policy_tree_refined.json `
  --config configs/tree_refinement_config.yaml

python scripts/finalize_policy_tree.py `
  --input data/intermediate_outputs/policy_tree_refined.json `
  --output data/final_tree/policy_tree_final.json `
  --config configs/tree_refinement_config.yaml `
  --l1-def data/intermediate_outputs/top_level_categories.json `
  --audit-out data/final_tree/policy_tree_final_audit.json `
  --flat-csv data/final_tree/policy_tree_final_flat.csv

Write-Host "PolicyTreeBuilder pipeline finished."
