import os
import json
import yaml
import time
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Iterable

# 假设这些已存在，如果路径不同请调整
from common_utils import read_embeddings
from common_llm import LLMConfig


class Step4Env:
    """环境与配置加载器"""
    def __init__(self, config_path: str, env_path: str):
        self.root_dir = Path(config_path).resolve().parents[2]  # 假设在 roundC_v4/
        self._enforce_env(Path(env_path))
        self.config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
        self.outdir = Path(self.config.get("outdir", "data/intermediate_outputs"))
        self.outdir.mkdir(parents=True, exist_ok=True)
        self.log_dir = self.outdir / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def _enforce_env(self, env_path: Path):
        if not env_path.exists():
            return
        for line in env_path.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if not s or s.startswith("#") or "=" not in s:
                continue
            k, v = s.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"'))
        # 自动注入 OpenAI 兼容变量
        base = os.getenv("PRIMARY_LLM_BASE_URL")
        key = os.getenv("PRIMARY_LLM_API_KEY")
        if base:
            os.environ.setdefault("OPENAI_BASE_URL", base)
        if key:
            os.environ.setdefault("OPENAI_API_KEY", key)

    def build_llm_config(self) -> LLMConfig:
        c = self.config.get("llm", {})
        return LLMConfig(
            primary=str(c.get("primary", os.getenv("PRIMARY_LLM_MODEL", "qwen3-max"))),
            secondary=str(c.get("secondary", os.getenv("SECONDARY_LLM_MODEL", "deepseek-v3"))),
            temperature=float(c.get("temperature", 0.2)),
            max_tokens=int(c.get("max_tokens", 1500)),
            response_format=str(c.get("response_format", "json_object")),
            workers=int(c.get("workers", 1)),
            tie_breaker=str(c.get("tie_breaker", "score_margin_or_conservative")),
        )


class EmbeddingHelper:
    """向量计算辅助类"""
    def __init__(self, parquet_path: Path):
        if parquet_path.exists():
            df = read_embeddings(parquet_path)
            # 建立缓存: sample_id -> numpy array
            self.cache = {str(row["sample_id"]): np.asarray(row["_vec"], dtype=np.float32)
                          for _, row in df.iterrows()}
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


def dump_tree(path: Path, root: Dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(root, ensure_ascii=False, indent=2), encoding="utf-8")


def append_jsonl(path: Path, data: Dict):
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")


def read_membership_map(outdir: Path, level: str) -> Dict[str, List[str]]:
    p = outdir / f"v4_membership_{level}.csv"
    if not p.exists():
        return {}
    df = pd.read_csv(p, dtype=str)
    return {str(nid): grp["member_id"].astype(str).tolist() for nid, grp in df.groupby("node_id")}


def read_title_map(corpus_path: Path) -> Dict[str, str]:
    if not corpus_path.exists():
        return {}
    df = pd.read_csv(corpus_path, dtype=str)
    return {str(r["sample_id"]): str(r["cleaned_title"]) for _, r in df.iterrows()}
