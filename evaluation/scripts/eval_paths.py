from __future__ import annotations

from pathlib import Path


def evaluation_root() -> Path:
    return Path(__file__).resolve().parents[1]


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_tree_path() -> Path:
    return repository_root() / "data" / "final_tree" / "policy_tree_final.json"


def default_output_dir() -> Path:
    return evaluation_root() / "outputs"


def resolve_repo_path(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return repository_root() / p
