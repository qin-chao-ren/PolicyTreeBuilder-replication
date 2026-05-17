#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LineageTracker（v4）
- 记录节点的谱系/操作历史，写入 JSONL，便于后续审计与溯源。
- 目前提供基础接口：record / merge / move / rename / flag。
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class LineageEvent:
    ts: int
    step: str
    action: str
    node_id: Optional[str] = None
    from_parent: Optional[str] = None
    to_parent: Optional[str] = None
    target_id: Optional[str] = None
    reason: Optional[str] = None
    meta: Optional[Dict[str, Any]] = None


class LineageTracker:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, event: LineageEvent):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(asdict(event), ensure_ascii=False) + "\n")

    def merge(self, step: str, winner: str, loser: str, reason: str = "", meta: Optional[Dict[str, Any]] = None):
        event = LineageEvent(
            ts=int(time.time()),
            step=step,
            action="merge",
            node_id=loser,
            target_id=winner,
            reason=reason,
            meta=meta,
        )
        self.record(event)

    def move(self, step: str, node_id: str, from_parent: str, to_parent: str, reason: str = "", meta: Optional[Dict[str, Any]] = None):
        event = LineageEvent(
            ts=int(time.time()),
            step=step,
            action="move",
            node_id=node_id,
            from_parent=from_parent,
            to_parent=to_parent,
            reason=reason,
            meta=meta,
        )
        self.record(event)

    def rename(self, step: str, node_id: str, new_label: str, reason: str = "", meta: Optional[Dict[str, Any]] = None):
        data = dict(meta or {})
        data["new_label"] = new_label
        event = LineageEvent(
            ts=int(time.time()),
            step=step,
            action="rename",
            node_id=node_id,
            reason=reason,
            meta=data,
        )
        self.record(event)

    def flag(self, step: str, node_id: str, reason: str, meta: Optional[Dict[str, Any]] = None):
        event = LineageEvent(
            ts=int(time.time()),
            step=step,
            action="flag",
            node_id=node_id,
            reason=reason,
            meta=meta,
        )
        self.record(event)


__all__ = ["LineageTracker", "LineageEvent"]
