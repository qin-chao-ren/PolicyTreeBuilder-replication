import json
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import yaml

from common_utils import read_embeddings
from llm_runtime import load_env_file, profiles_from_config


class Step4Env:
    """Environment/config loader for tree-refinement steps."""

    def __init__(self, config_path: str, env_path: str):
        self.root_dir = Path(config_path).resolve().parents[2]
        self._enforce_env(Path(env_path))
        self.config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
        self.outdir = Path(self.config.get("outdir", "data/intermediate_outputs"))
        self.outdir.mkdir(parents=True, exist_ok=True)
        self.log_dir = self.outdir / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def _enforce_env(self, env_path: Path) -> None:
        load_env_file(env_path, required=False)

    def llm_profiles(self) -> tuple[str, str]:
        return profiles_from_config(self.config.get("llm", {}))

    def primary_llm_profile(self) -> str:
        return self.llm_profiles()[0]


class EmbeddingHelper:
    """Vector lookup and centroid helper."""

    def __init__(self, parquet_path: Path):
        if parquet_path.exists():
            df = read_embeddings(parquet_path)
            self.cache = {
                str(row["sample_id"]): np.asarray(row["_vec"], dtype=np.float32)
                for _, row in df.iterrows()
            }
        else:
            self.cache = {}
            print(f"[WARN] Embedding file not found: {parquet_path}")

    def get_centroid(self, sample_ids: Iterable[str]) -> Optional[np.ndarray]:
        vecs = [self.cache[str(sid)] for sid in sample_ids if str(sid) in self.cache]
        if not vecs:
            return None
        return np.mean(vecs, axis=0)

    @staticmethod
    def cosine_sim(v1: Optional[np.ndarray], v2: Optional[np.ndarray]) -> float:
        if v1 is None or v2 is None:
            return 0.0
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 == 0 or n2 == 0:
            return 0.0
        return float(np.dot(v1, v2) / (n1 * n2))


def load_tree(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def dump_tree(path: Path, root: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(root, ensure_ascii=False, indent=2), encoding="utf-8")


def append_jsonl(path: Path, data: Dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")


def read_membership_map(outdir: Path, level: str) -> Dict[str, List[str]]:
    p = outdir / f"tree_node_membership_{level}.csv"
    if not p.exists():
        return {}
    df = pd.read_csv(p, dtype=str)
    return {
        str(nid): grp["member_id"].astype(str).tolist()
        for nid, grp in df.groupby("node_id")
    }


def read_title_map(corpus_path: Path) -> Dict[str, str]:
    if not corpus_path.exists():
        return {}
    df = pd.read_csv(corpus_path, dtype=str)
    return {str(r["sample_id"]): str(r["cleaned_title"]) for _, r in df.iterrows()}
