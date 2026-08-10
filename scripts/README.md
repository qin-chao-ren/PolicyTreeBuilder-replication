# Pipeline Scripts

This directory contains the public pipeline scripts used to build and refine policy action trees. The committed 353-node output is the archived paper snapshot; most reviewers can inspect it without rerunning every API-dependent step.

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
| E0 integrity gate | `validate_tree_e0.py` | candidate tree, membership, lineage, operations | machine-readable PASS/FAIL report | No |
| Semantic decision contract | `utils/semantic_contract.py`, `../schemas/semantic_tree_decision.schema.json` | shared `decisions[]` responses and operation history | validated decisions and `semantic_contract` report | No |

The full command template is `run_policy_tree_pipeline.ps1` in the repository root.

Tree writes in the refinement stages use rollback plus raw/index/parent-map postconditions and record the current run in `tree_refinement_operations.jsonl`. All four refinement/finalization stages parse the same relation/action schema; destructive merge is restricted to exact duplicates or supported synonyms, and every affected child needs an explicit compatible destination plan. Finalization validates the combined history and publishes `policy_tree_final.json` only when both structural `e0` and independent `semantic_contract` reports pass. Run the offline regression suite with `python -m unittest discover -s tests -v`.

## LLM Runtime Boundary

All external model calls now go through `scripts/llm_runtime.py`. Chat LLM, evaluation judge, embedding, and reranking services use named profiles from `configs/llm_profiles.yaml.example`; local credentials stay in `configs/.env` or `evaluation/.env`.

The evaluation scripts keep the archived judge keys in their output filenames, but they no longer maintain a separate LLM client.
