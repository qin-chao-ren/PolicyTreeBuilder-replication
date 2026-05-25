# Round C v4 pipeline (PowerShell template)
# [重要!] 使用前请确保您位于项目的根目录 (.) cd .
# 然后在根目录执行: .\roundC_v4\run_v4_pipeline.ps1
# Write-Host "Round C v4 pipeline starting..."

# --- Step 1: 标题清洗 ---
#Write-Host "[Step1] Prepare Corpus"
# 请将 --source 替换为实际的 Round B 源数据
#python scripts/step1_prepare_corpus.py `
#--source roundB_outputs\roundB_types_merged1121.csv `
#--env configs/.env `
#--outdir data/intermediate_outputs `
#--llm-clean yes `
#--debug-dump-llm yes `
#--title-max-len 40

# --- Step 1.5: T0 过滤 ---
#Write-Host "[Step1.5] T0 Filter"
#python scripts/step1_5_filter_t0.py --corpus "data/intermediate_outputs/v4_cluster_corpus_cleaned.csv" --outdir "data/intermediate_outputs" --mode delete

# --- Step 2: 嵌入与近邻 ---
#Write-Host "[Step2] Embed & NN"
#python scripts/step2_embed_and_nn.py --corpus "data/intermediate_outputs/v4_corpus_filtered.csv" `
    #--env "configs/.env" `
    #--outdir "data/intermediate_outputs" `
    #--embed-text-template "title_plus_path" `
    #--recompute "no" `
    #--nn-topk 294 `
    #--mutual-knn "no" `
    #--local-k 20 `
    #--min-sim 0.0 `
    #--use-reranker "yes" `
    #--rerank-scope "all" `
    #--rerank-threshold 0.55 `
    #--rerank-mode "soft"

# --- Step 2.5: 层级粒度校准 ---
#Write-Host "[Step2.5] Calibrate Levels"
#python scripts/step2_5_calibrate_levels.py --corpus "data/intermediate_outputs/v4_corpus_filtered.csv" --env "configs/.env" --outdir "data/intermediate_outputs"

# --- Step 2.8: L1 定义 ---
#Write-Host "[Step2.8] Define L1"
#python scripts/step2_8_define_l1.py --calibrated "data/intermediate_outputs/v4_corpus_calibrated.csv" --assets-dir "assets/l1_samples" --env "configs/.env" --outdir "data/intermediate_outputs"

# --- Step 2.8: L1 分类 ---
#Write-Host "[Step2.8] Classify to L1"
#python scripts/step2_8_classify_levels.py --l1-def "data/intermediate_outputs/v4_l1_definition.json" --corpus "data/intermediate_outputs/v4_corpus_calibrated.csv" --env "configs/.env" --outdir "data/intermediate_outputs"

# --- Step 3: 构树 (安全线性版) ---
# 核心原则：自底向上准备，自下而上挂接。一旦挂接完成一层，就不再回头修改该层。

# 1. 初始化所有层级节点 (Layer Merge)
#Write-Host "[Step3] 1. Init L4 Nodes"
#python scripts/step3_layer_merge.py --level L4 --config "configs/step3_config.yaml"

#Write-Host "[Step3] 2. Init L3 Nodes"
#python scripts/step3_layer_merge.py --level L3 --config "configs/step3_config.yaml"

#Write-Host "[Step3] 3. Init L2 Nodes"
#python scripts/step3_layer_merge.py --level L2 --config "configs/step3_config.yaml"

# 2. 纵向装配 (Vertical Link)
# L4 -> L3 (如找不到父节点会自动新建)
#Write-Host "[Step3] 4. Link L4 -> L3"
#python scripts/step3_vertical_link.py --from-level L4 --to-level L3 --config "configs/step3_config.yaml"

# L3 -> L2 (如找不到父节点会自动新建)
#Write-Host "[Step3] 5. Link L3 -> L2"
#python scripts/step3_vertical_link.py --from-level L3 --to-level L2 --config "configs/step3_config.yaml"

# 3. 全局挂接
# L2 -> L1
Write-Host "[Step3] 6. Link L2 -> L1 (Global)"
python scripts/step3_link_t2_l1.py --config "configs/step3_config.yaml"

# 4. 生成粗树
Write-Host "[Step3] 7. Build Coarse Tree"
python scripts/step3_build_coarse_tree.py --config "configs/step3_config.yaml" --emit-samples no

# --- Step 4: 结构精修 (三阶段拆分) ---
Write-Host "[Step4.1] Skeleton"
python scripts/step4_1_skeleton.py --input "data/intermediate_outputs/v4_tree_coarse_global.json" --output "data/intermediate_outputs/v4_tree_s4_1.json" --config "configs/step4_config.yaml"

Write-Host "[Step4.2] Shaping"
python scripts/step4_2_shaping.py --input "data/intermediate_outputs/v4_tree_s4_1.json" --output "data/intermediate_outputs/v4_tree_s4_2.json" --config "configs/step4_config.yaml"

Write-Host "[Step4.3] Polishing"
python scripts/step4_3_polishing.py --input "data/intermediate_outputs/v4_tree_s4_2.json" --output "data/intermediate_outputs/v4_tree_refined.json" --config "configs/step4_config.yaml"

# --- Step 4.5: 全局微调与对齐 ---
Write-Host "[Step4.5] Final Override & Realignment"
python scripts/step4_5_overall_structure.py `
    --input "data/intermediate_outputs/v4_tree_refined.json" `
    --output "data/intermediate_outputs/v4_tree_final.json" `
    --config "configs/step4_config.yaml" `
    --l1-def "data/intermediate_outputs/v4_l1_definition.json" `
    --audit-out "data/intermediate_outputs/v4_final_audit.json" `
    --flat-csv "data/intermediate_outputs/v4_tree_final_flat.csv"

Write-Host "Round C v4 pipeline finished."
# --- Pipeline End ---