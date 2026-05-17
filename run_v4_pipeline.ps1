# Round C v4 pipeline entrypoint for the public replication package.
# Run from the repository root:
#   .\run_v4_pipeline.ps1

$ErrorActionPreference = "Stop"

$EnvFile = "configs/roundC_v4.env"
$OutDir = "data/intermediate_outputs"
$SourceFile = "data/source/roundB_types_merged1121.csv"

Write-Host "Round C v4 pipeline starting..."
Write-Host "Output directory: $OutDir"

if (-not (Test-Path -LiteralPath $EnvFile)) {
    Write-Host "[WARN] $EnvFile was not found. LLM-dependent steps require API credentials."
    Write-Host "       Copy configs/roundC_v4.env.example to configs/roundC_v4.env and fill it in before rerunning those steps."
}

# Optional source-preparation steps. They are commented out because this public
# package already includes the source input and precomputed intermediate outputs.
#
# python scripts/step1_prepare_corpus.py `
#   --source $SourceFile `
#   --env $EnvFile `
#   --outdir $OutDir `
#   --llm-clean yes `
#   --debug-dump-llm yes `
#   --title-max-len 40
#
# python scripts/step1_5_filter_t0.py `
#   --corpus "$OutDir/v4_cluster_corpus_cleaned.csv" `
#   --outdir $OutDir `
#   --mode delete
#
# python scripts/step2_embed_and_nn.py `
#   --corpus "$OutDir/v4_corpus_filtered.csv" `
#   --env $EnvFile `
#   --outdir $OutDir `
#   --embed-text-template "title_plus_path" `
#   --recompute "no" `
#   --nn-topk 294 `
#   --mutual-knn "no" `
#   --local-k 20 `
#   --min-sim 0.0 `
#   --use-reranker "yes" `
#   --rerank-scope "all" `
#   --rerank-threshold 0.55 `
#   --rerank-mode "soft"
#
# python scripts/step2_5_calibrate_levels.py `
#   --corpus "$OutDir/v4_corpus_filtered.csv" `
#   --env $EnvFile `
#   --outdir $OutDir
#
# python scripts/step2_8_define_l1.py `
#   --calibrated "$OutDir/v4_corpus_calibrated.csv" `
#   --assets-dir "assets/l1_samples" `
#   --env $EnvFile `
#   --outdir $OutDir
#
# python scripts/step2_8_classify_levels.py `
#   --l1-def "$OutDir/v4_l1_definition.json" `
#   --corpus "$OutDir/v4_corpus_calibrated.csv" `
#   --env $EnvFile `
#   --outdir $OutDir

Write-Host "[Step3] Initialize L4 nodes"
python scripts/step3_layer_merge.py --level L4 --config "configs/step3_config.yaml"

Write-Host "[Step3] Initialize L3 nodes"
python scripts/step3_layer_merge.py --level L3 --config "configs/step3_config.yaml"

Write-Host "[Step3] Initialize L2 nodes"
python scripts/step3_layer_merge.py --level L2 --config "configs/step3_config.yaml"

Write-Host "[Step3] Link L4 to L3"
python scripts/step3_vertical_link.py --from-level L4 --to-level L3 --config "configs/step3_config.yaml"

Write-Host "[Step3] Link L3 to L2"
python scripts/step3_vertical_link.py --from-level L3 --to-level L2 --config "configs/step3_config.yaml"

Write-Host "[Step3] Link L2 to L1"
python scripts/step3_link_t2_l1.py --config "configs/step3_config.yaml"

Write-Host "[Step3] Build coarse tree"
python scripts/step3_build_coarse_tree.py --config "configs/step3_config.yaml"

Write-Host "[Step4.1] Skeleton"
python scripts/step4_1_skeleton.py `
  --input "$OutDir/v4_tree_coarse_global.json" `
  --output "$OutDir/v4_tree_s4_1.json" `
  --config "configs/step4_config.yaml"

Write-Host "[Step4.2] Shaping"
python scripts/step4_2_shaping.py `
  --input "$OutDir/v4_tree_s4_1.json" `
  --output "$OutDir/v4_tree_s4_2.json" `
  --config "configs/step4_config.yaml"

Write-Host "[Step4.3] Polishing"
python scripts/step4_3_polishing.py `
  --input "$OutDir/v4_tree_s4_2.json" `
  --output "$OutDir/v4_tree_refined.json" `
  --config "configs/step4_config.yaml"

Write-Host "[Step4.5] Final override"
python scripts/step4_5_overall_structure.py `
  --input "$OutDir/v4_tree_refined.json" `
  --output "$OutDir/v4_tree_final.json" `
  --config "configs/step4_config.yaml"

Write-Host "Round C v4 pipeline finished."
