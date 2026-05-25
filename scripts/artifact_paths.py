#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Repository-relative artifact paths for the public replication package."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONFIG_DIR = PROJECT_ROOT / "configs"
PROMPT_DIR = PROJECT_ROOT / "prompts"
ASSET_DIR = PROJECT_ROOT / "assets"
SOURCE_DIR = PROJECT_ROOT / "data" / "source"
INTERMEDIATE_DIR = PROJECT_ROOT / "data" / "intermediate_outputs"
FINAL_TREE_DIR = PROJECT_ROOT / "data" / "final_tree"
LOG_DIR = INTERMEDIATE_DIR / "logs"

SOURCE_SEGMENTS = SOURCE_DIR / "policy_action_segments.csv"
ADMINISTRATIVE_METADATA = SOURCE_DIR / "administrative_unit_metadata.csv"

POLICY_CORPUS_CLEANED = INTERMEDIATE_DIR / "policy_corpus_cleaned.csv"
POLICY_CORPUS_FILTERED = INTERMEDIATE_DIR / "policy_corpus_filtered.csv"
POLICY_CORPUS_CALIBRATED = INTERMEDIATE_DIR / "policy_corpus_calibrated.csv"
POLICY_CORPUS_EMBEDDINGS = INTERMEDIATE_DIR / "policy_corpus_embeddings.parquet"
POLICY_SIMILARITY_RERANK_EDGES = INTERMEDIATE_DIR / "policy_similarity_rerank_edges.csv"
TOP_LEVEL_CATEGORIES = INTERMEDIATE_DIR / "top_level_categories.json"

POLICY_TREE_INITIAL = INTERMEDIATE_DIR / "policy_tree_initial.json"
POLICY_TREE_AFTER_VERTICAL_COLLAPSE = INTERMEDIATE_DIR / "policy_tree_after_vertical_collapse.json"
POLICY_TREE_AFTER_STRUCTURE_BALANCING = INTERMEDIATE_DIR / "policy_tree_after_structure_balancing.json"
POLICY_TREE_REFINED = INTERMEDIATE_DIR / "policy_tree_refined.json"
POLICY_TREE_FINAL = FINAL_TREE_DIR / "policy_tree_final.json"


def tree_nodes_path(level: str, outdir: Path = INTERMEDIATE_DIR) -> Path:
    return outdir / f"tree_nodes_{level}.csv"


def tree_membership_path(level: str, outdir: Path = INTERMEDIATE_DIR) -> Path:
    return outdir / f"tree_node_membership_{level}.csv"


def tree_parent_links_path(from_level: str, to_level: str, outdir: Path = INTERMEDIATE_DIR) -> Path:
    return outdir / f"tree_parent_links_{from_level}_to_{to_level}.csv"


def cluster_similarity_edges_path(level: str, outdir: Path = INTERMEDIATE_DIR) -> Path:
    return outdir / f"cluster_similarity_edges_{level}.csv"
