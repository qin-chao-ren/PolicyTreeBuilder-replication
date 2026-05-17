#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import hashlib
from typing import Iterable


def make_node_id(level_str: str, member_ids: Iterable[str]) -> str:
    """
    根据成员集合生成稳定的 L 级节点 ID。
    level_str: "T4"/"T3"/"T2"/"T1"（或 "L4"...），输出统一为 "Lx_N{hash}"。
    member_ids: 节点包含的 sample_id 集合（调用方负责去重）。
    """
    items = sorted(str(x) for x in set(member_ids))
    s = "|".join(items)
    h = hashlib.md5(s.encode("utf-8")).hexdigest()[:8]
    # 如果 level 已经是 L 级，直接取最后一位；否则默认取最后的数字
    suffix = level_str.strip().upper()
    suffix = suffix[-1] if suffix else "4"
    return f"L{suffix}_N{h}"


__all__ = ["make_node_id"]
