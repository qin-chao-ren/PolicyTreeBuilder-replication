# Pipeline Scripts

This directory contains the public pipeline scripts used to build and refine the fixed 353-node policy action tree. The scripts are kept close to the verified replication workflow; most reviewers can inspect the included outputs instead of rerunning every API-dependent step.

## Step Map

| Stage | Script(s) | Main input | Main output | External API |
| --- | --- | --- | --- | --- |
| Corpus preparation | `prepare_policy_corpus.py` | `data/source/policy_action_segments.csv` | `data/intermediate_outputs/policy_corpus_cleaned.csv` | Optional LLM cleaning |
| Non-policy filtering | `filter_non_policy_records.py` | `policy_corpus_cleaned.csv` | `policy_corpus_filtered.csv` | No |
| Embeddings and reranking | `embed_policy_corpus.py` | `policy_corpus_filtered.csv` | embeddings, similarity pairs, rerank edges | Yes |
| Granularity calibration | `calibrate_policy_granularity.py` | `policy_corpus_filtered.csv` | `policy_corpus_calibrated.csv`, calibration report | Yes by default |
| Top-level categories | `define_top_level_categories.py`, `assign_top_level_categories.py` | calibrated corpus and L1 samples | top-level definitions and assignments | Yes |
| Cluster construction | `build_action_clusters.py`, `contract_similarity_edges.py` | calibrated corpus, memberships, pair edges | L2/L3/L4 nodes and similarity edges | No for contraction; clustering uses prepared inputs |
| Parent linking | `assign_parent_clusters.py`, `link_second_level_to_top_level.py` | level nodes and similarity edges | parent-link CSVs | Yes |
| Initial tree | `build_initial_policy_tree.py` | node and parent-link tables | `policy_tree_initial.json` | No |
| Refinement and finalization | `collapse_redundant_hierarchy.py`, `balance_tree_structure.py`, `polish_tree_labels.py`, `finalize_policy_tree.py` | initial/refined tree states | final tree and audit outputs | Yes |
| Lineage tracing | `trace_node_lineage.py` | corpus, membership, operation logs | lineage report | No |

The full command template is `run_policy_tree_pipeline.ps1` in the repository root.

## LLM Runtime Boundary

Main pipeline scripts use `common_llm.py` and helpers under `scripts/utils/`. Some older step scripts still perform small local environment-loading shims so they can be rerun independently. The public evaluation module intentionally uses a separate judge client in `evaluation/scripts/llm_clients.py` to preserve archived judge keys and output names.

This package does not consolidate those clients because the priority is preserving the verified replication behavior. A future maintenance branch can unify the runtime after a full regression rerun.
