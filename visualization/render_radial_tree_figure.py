#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Render policy-tree JSON files as publication-style radial visualizations.

Each input file is rendered as-is in whatever language its labels are
written in; the language is auto-detected per file from the presence of
CJK characters.  No translation step is performed inside this script:
upstream tooling is expected to deliver Chinese and English JSON files
separately (e.g. ``policy_tree_final.json`` and
``policy_tree_final_en_academic.json``).
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import unicodedata
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from artifact_paths import FINAL_TREE_DIR

DEFAULT_BASE_DIR = FINAL_TREE_DIR

PALETTE = [
    "#E63946",
    "#F4A261",
    "#D4A537",
    "#2A9D8F",
    "#264653",
    "#6A4C93",
    "#1D3557",
]

ADMIN_EN = {
    "上海市": "Shanghai",
    "天津市": "Tianjin",
    "广西壮族自治区": "Guangxi Zhuang Autonomous Region",
    "江苏省": "Jiangsu Province",
    "浙江省": "Zhejiang Province",
    "海南省": "Hainan Province",
    "南通市": "Nantong",
    "合肥市": "Hefei",
    "宁波市": "Ningbo",
    "成都市": "Chengdu",
}


def configure_fonts() -> None:
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Noto Sans CJK JP",
        "Source Han Sans SC",
        "WenQuanYi Zen Hei",
        "Arial Unicode MS",
        "DejaVu Sans",
        "sans-serif",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def walk(node: dict[str, Any]):
    yield node
    for child in node.get("children") or []:
        yield from walk(child)


def detect_language(tree: dict[str, Any]) -> str:
    """Return ``"zh"`` if any node label contains a CJK ideograph, else ``"en"``.

    The split-by-admin pipeline produces Chinese trees with Chinese labels
    and English trees with English labels (no mixing within one file), so a
    single CJK character in any label is a reliable signal.
    """
    for node in walk(tree):
        for ch in str(node.get("label", "")):
            if "\u4e00" <= ch <= "\u9fff":
                return "zh"
    return "en"


def input_tree_paths(base_dir: Path, admin_dir: Path) -> list[Path]:
    """Discover tree JSON files in *base_dir* (top-level) and *admin_dir*.

    Matches the full tree, the provincial / city aggregates, and any
    suffix-suffixed academic variants such as
    ``policy_tree_final_en_academic.json``.  Per-administrative-unit
    trees are picked up from ``trees_by_admin/``.

    Auxiliary files such as label maps and run summaries are filtered out.
    """
    paths: list[Path] = []
    for pattern in (
        "policy_tree_final*.json",
        "policy_tree_provincial*.json",
        "policy_tree_city*.json",
    ):
        paths.extend(sorted(base_dir.glob(pattern)))
    if admin_dir.is_dir():
        paths.extend(sorted(admin_dir.glob("tree_*.json")))
    skip_substrings = ("audit", "flat", "label_map", "membership", "operations", "summary")
    return [
        p for p in paths
        if p.exists() and not any(s in p.stem for s in skip_substrings)
    ]


def visual_path(path: Path, lang: str, image_format: str, tag: str = "radial") -> Path:
    """Compute the output image path for *path* in *lang*.

    The input stem is normalised by stripping known language/version
    suffixes (``_en_academic``, ``_en``) so that an English file
    such as ``policy_tree_final_en_academic.json`` does not produce the
    awkward output ``policy_tree_final_en_academic_en_radial.jpg``.
    """
    stem = path.stem
    for suffix in ("_en_academic", "_en"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return path.with_name(f"{stem}_{lang}_{tag}.{image_format}")


def display_units(text: str) -> float:
    units = 0.0
    for ch in text:
        if ch.isspace():
            units += 0.28
        elif unicodedata.east_asian_width(ch) in {"W", "F", "A"}:
            units += 1.0
        else:
            units += 0.56
    return units


def char_units(ch: str) -> float:
    if ch.isspace():
        return 0.55
    if unicodedata.east_asian_width(ch) in {"W", "F", "A"}:
        return 1.0
    return 0.58


def arc_label_items(label: str) -> list[tuple[str, float]]:
    """Tokenise a label into renderable arc items.

    For purely Latin labels with whitespace, we group consecutive characters
    into words and use an explicit space token between them. The space width is
    expressed in the same em-relative units as `display_units`.
    """
    has_cjk = any(
        unicodedata.east_asian_width(ch) in {"W", "F", "A"}
        for ch in label
        if not ch.isspace()
    )
    if " " in label and not has_cjk:
        items: list[tuple[str, float]] = []
        for index, word in enumerate(label.split()):
            if index:
                # An explicit inter-word space. Tuned to read like a real word
                # break (~0.65 em ≈ slightly wider than half-width glyph) so
                # words stay clearly separated without being pushed apart.
                items.append((" ", 0.65))
            items.append((word, display_units(word)))
        return items
    return [(ch, char_units(ch)) for ch in label]


def tangent_flipped(angle: float) -> bool:
    deg = math.degrees(angle) + 90
    deg = ((deg + 180) % 360) - 180
    return deg > 90 or deg <= -90


def draw_arc_label(
    ax,
    label: str,
    angle0: float,
    angle1: float,
    radius: float,
    color: str,
    fontsize: float,
    fontweight: str,
    zorder: int,
    lang: str = "zh",
    pt_to_data: float = 0.0107,
    min_fontsize: float = 7.5,
) -> None:
    """Draw a label one glyph at a time along an annular sector.

    Compact-first layout:
    - Each glyph is placed using its natural advance width plus a controllable
      letter-spacing.
    - When the natural width is shorter than the available arc, the label stays
      tightly grouped at the wedge centre instead of being spread to fill the
      whole arc.
    - When the natural width exceeds the available arc, letter-spacing is
      compressed first; only if that is not enough is the font size reduced.

    Language-specific tuning:
    - "zh" (Chinese): adds a moderate letter-spacing so square CJK glyphs
      breathe; works well for the typical 6-7 character L1 labels.
    - "en" (English): word-mode rendering keeps each word internally tight and
      relies on the explicit space character for inter-word separation.
    """
    label = str(label).strip()
    if not label:
        return

    items = arc_label_items(label)
    if not items:
        return

    span = angle1 - angle0
    pad = min(math.radians(0.9), span * 0.04)
    available_arc = max(span - 2 * pad, span * 0.92)
    midpoint = (angle0 + angle1) / 2

    # word_mode is True when arc_label_items returned (word, " ", word, ...)
    word_mode = any(text == " " for text, _ in items)

    # Per-language letter spacing (in em, i.e. fraction of font size)
    if lang == "zh":
        base_letter_spacing_em = 0.22  # CJK glyphs feel cramped without breath
        min_letter_spacing_em = 0.04
    else:
        base_letter_spacing_em = 0.04  # English tracks naturally inside words
        min_letter_spacing_em = 0.0

    # Count non-space items, and the gaps between consecutive non-space items.
    # In word_mode, gaps between words are realised by the explicit " " items
    # already accounted for in their own widths, so we should not double-count.
    visible_items = [(t, w) for t, w in items if not t.isspace()]
    if not visible_items:
        return

    if word_mode:
        # Inter-word spacing is the " " item; no extra kerning between items.
        n_extra_gaps = 0
    else:
        # Single-glyph rendering (CJK style): one extra gap between each pair
        # of consecutive visible glyphs.
        n_extra_gaps = max(len(visible_items) - 1, 0)

    def total_arc_at(font_size: float, letter_spacing_em: float) -> float:
        """Arc length (in radians) consumed when drawn at the given size/space."""
        per_em = font_size * pt_to_data
        chars_data = sum(w * per_em for _, w in items)
        kerning_data = n_extra_gaps * letter_spacing_em * per_em
        return (chars_data + kerning_data) / radius

    fs = fontsize
    letter_spacing_em = base_letter_spacing_em

    # Pass 1: shrink letter spacing if we are over budget.
    while (
        total_arc_at(fs, letter_spacing_em) > available_arc
        and letter_spacing_em > min_letter_spacing_em
    ):
        letter_spacing_em = max(min_letter_spacing_em, letter_spacing_em - 0.02)

    # Pass 2: shrink font size if still over budget.
    while total_arc_at(fs, letter_spacing_em) > available_arc and fs > min_fontsize:
        fs -= 0.5

    # Pass 3: as a last resort, allow letter_spacing to go to zero and
    # accept tiny overflow if the label is still longer than the arc.
    if total_arc_at(fs, 0.0) <= available_arc:
        letter_spacing_em = max(0.0, min(letter_spacing_em, base_letter_spacing_em))
    else:
        letter_spacing_em = 0.0

    used_arc = total_arc_at(fs, letter_spacing_em)
    direction = -1 if tangent_flipped(midpoint) else 1

    per_em = fs * pt_to_data
    extra_kerning_arc = (letter_spacing_em * per_em / radius) if not word_mode else 0.0

    cursor = -used_arc / 2
    last_was_visible = False
    for text, width in items:
        item_arc = width * per_em / radius
        if not text.isspace() and last_was_visible:
            cursor += extra_kerning_arc
        center_offset = cursor + item_arc / 2
        cursor += item_arc
        if text.isspace():
            last_was_visible = False
            continue
        last_was_visible = True

        angle = midpoint + direction * center_offset
        tx = radius * math.cos(angle)
        ty = radius * math.sin(angle)
        txt = ax.text(
            tx,
            ty,
            text,
            ha="center",
            va="center",
            fontsize=fs,
            fontweight=fontweight,
            color=color,
            rotation=upright(math.degrees(angle) + 90),
            rotation_mode="anchor",
            zorder=zorder,
        )
        txt.set_path_effects([pe.withStroke(linewidth=0.7, foreground=(0, 0, 0, 0.18))])


def depth_key(depth: int) -> str:
    return "ROOT" if depth == 0 else f"L{depth}"


def prepare_tree(root: dict[str, Any]) -> tuple[list[dict[str, Any]], list[tuple[dict[str, Any], dict[str, Any]]], int]:
    def weight(node: dict[str, Any]) -> int:
        children = node.get("children") or []
        if not children:
            node["_w"] = 1
            return 1
        node["_w"] = sum(weight(child) for child in children)
        return node["_w"]

    def assign(node: dict[str, Any], a0: float, a1: float) -> None:
        node["_a0"], node["_a1"], node["_a"] = a0, a1, (a0 + a1) / 2
        children = node.get("children") or []
        if not children:
            return
        total = sum(child["_w"] for child in children)
        cur = a0
        span = a1 - a0
        for child in children:
            end = cur + span * child["_w"] / total
            assign(child, cur, end)
            cur = end

    root = copy.deepcopy(root)
    weight(root)
    l1_nodes = root.get("children") or []
    root["_a"], root["_a0"], root["_a1"] = 0.0, 0.0, 2 * math.pi
    if l1_nodes:
        gap = math.radians(3.5)
        available = 2 * math.pi - gap * len(l1_nodes)
        total = sum(child["_w"] for child in l1_nodes)
        cur = -math.pi / 2
        for child in l1_nodes:
            end = cur + available * child["_w"] / total
            assign(child, cur, end)
            cur = end + gap

    def color_subtree(node: dict[str, Any], color: str) -> None:
        node["_c"] = color
        for child in node.get("children") or []:
            color_subtree(child, color)

    root["_c"] = "#222222"
    for index, child in enumerate(l1_nodes):
        color_subtree(child, PALETTE[index % len(PALETTE)])

    nodes: list[dict[str, Any]] = []
    edges: list[tuple[dict[str, Any], dict[str, Any]]] = []
    max_depth = 0

    def place(node: dict[str, Any], parent: dict[str, Any] | None = None, depth: int = 0) -> None:
        nonlocal max_depth
        node["_depth"] = depth
        node["_level_key"] = depth_key(depth)
        max_depth = max(max_depth, depth)
        nodes.append(node)
        if parent is not None:
            edges.append((parent, node))
        for child in node.get("children") or []:
            place(child, node, depth + 1)

    place(root)
    return nodes, edges, max_depth


def compute_radii(max_depth: int) -> tuple[dict[int, float], float, float, float]:
    base = [0.0, 1.8, 3.7, 5.6, 7.5, 9.3, 10.9]
    radii = {}
    for depth in range(max_depth + 1):
        if depth < len(base):
            radii[depth] = base[depth]
        else:
            radii[depth] = radii[depth - 1] + 1.45
    sector_r0 = radii[max_depth] + 1.75
    sector_r1 = sector_r0 + 0.85
    limit = sector_r1 + 1.55
    return radii, sector_r0, sector_r1, limit


def annotate_positions(nodes: list[dict[str, Any]], radii: dict[int, float]) -> None:
    for node in nodes:
        radius = radii[node["_depth"]]
        angle = node["_a"]
        node["_r"] = radius
        node["_x"] = radius * math.cos(angle)
        node["_y"] = radius * math.sin(angle)


def compute_angle_gaps(nodes: list[dict[str, Any]]) -> None:
    by_depth: dict[int, list[dict[str, Any]]] = {}
    for node in nodes:
        by_depth.setdefault(node["_depth"], []).append(node)

    for items in by_depth.values():
        items.sort(key=lambda x: x["_a"])
        for index, node in enumerate(items):
            gaps = []
            if index > 0:
                gaps.append(abs(node["_a"] - items[index - 1]["_a"]))
            if index < len(items) - 1:
                gaps.append(abs(items[index + 1]["_a"] - node["_a"]))
            node["_min_gap"] = min(gaps) if gaps else math.pi


def upright(deg: float) -> float:
    deg = ((deg + 180) % 360) - 180
    if deg > 90:
        deg -= 180
    elif deg <= -90:
        deg += 180
    return deg


def count_by_depth(nodes: list[dict[str, Any]]) -> dict[int, int]:
    counts: dict[int, int] = {}
    for node in nodes:
        if node["_depth"] > 0:
            counts[node["_depth"]] = counts.get(node["_depth"], 0) + 1
    return counts


def title_for(path: Path, lang: str) -> str:
    stem = path.stem
    # Strip known language/version suffixes added by the splitting pipeline
    # so the same lookup table works for both raw and academic-suffix names.
    for suffix in ("_en_academic", "_en"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    if stem == "policy_tree_final":
        return "航空货运政策行动树" if lang == "zh" else "Air-Cargo Policy Action Tree"
    if stem == "policy_tree_provincial":
        return "省级政策行动树" if lang == "zh" else "Provincial Policy Action Tree"
    if stem == "policy_tree_city":
        return "市级政策行动树" if lang == "zh" else "City-Level Policy Action Tree"
    if stem.startswith("tree_"):
        name = stem.removeprefix("tree_")
        if lang == "zh":
            return f"{name}政策行动树"
        return f"{ADMIN_EN.get(name, name)} Policy Action Tree"
    return stem


def subtitle_for(nodes: list[dict[str, Any]], lang: str) -> str:
    counts = count_by_depth(nodes)
    total = len(nodes) - 1
    if lang == "zh":
        return (
            f"{counts.get(1, 0)} 个一级板块 · "
            f"{counts.get(2, 0)} 个二级节点 · "
            f"{counts.get(3, 0)} 个三级节点 · 共 {total} 个节点"
        )
    return (
        f"{counts.get(1, 0)} L1 pillars · "
        f"{counts.get(2, 0)} L2 nodes · "
        f"{counts.get(3, 0)} L3 nodes · {total} total nodes"
    )


def draw_edge(ax, parent: dict[str, Any], child: dict[str, Any]) -> None:
    a0, r0 = parent["_a"], parent["_r"]
    a1, r1 = child["_a"], child["_r"]
    color = child["_c"]
    depth = child["_depth"]

    if r0 == 0:
        ax.plot([0, child["_x"]], [0, child["_y"]], color=color, linewidth=1.0, alpha=0.72, zorder=1)
        return

    arc_angles = np.linspace(a0, a1, 22)
    xa = r0 * np.cos(arc_angles)
    ya = r0 * np.sin(arc_angles)
    radial = np.linspace(r0, r1, 8)
    xr = radial * np.cos(a1)
    yr = radial * np.sin(a1)
    xs = np.concatenate([xa, xr])
    ys = np.concatenate([ya, yr])
    linewidth = 1.0 if depth == 1 else (0.7 if depth == 2 else (0.5 if depth == 3 else 0.38))
    alpha = 0.7 if depth in {1, 2} else 0.42
    ax.plot(xs, ys, color=color, linewidth=linewidth, alpha=alpha, zorder=1)


def draw_radial_tree(
    tree: dict[str, Any],
    output_path: Path,
    lang: str,
    title: str,
    image_format: str,
    dpi: int,
) -> None:
    nodes, edges, max_depth = prepare_tree(tree)
    radii, sector_r0, sector_r1, limit = compute_radii(max_depth)
    annotate_positions(nodes, radii)
    compute_angle_gaps(nodes)

    fig, ax = plt.subplots(figsize=(26, 26), dpi=dpi)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_facecolor("white")
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)

    # Conversion factor between font points and data units. With a square
    # figure of side `fig_side_in` inches that maps to a data range of
    # `2 * limit`, one point (1/72 inch) corresponds to:
    #     pt_to_data_geom = (2 * limit) / (fig_side_in * 72)
    # We then apply a per-language calibration factor that compensates for the
    # difference between em-units and the actual glyph advance widths produced
    # by matplotlib (Latin fonts include intrinsic kerning that makes a word
    # render slightly wider than the sum of `display_units`).
    fig_side_in = float(fig.get_size_inches()[0])
    pt_to_data_geom = (2 * limit) / (fig_side_in * 72.0)
    if lang == "zh":
        # CJK glyphs are very close to 1 em advance; small overshoot is fine.
        pt_to_data = pt_to_data_geom * 1.05
    else:
        # Latin fonts render visibly wider than display_units would predict;
        # use a larger factor so reserved arc space matches actual width.
        pt_to_data = pt_to_data_geom * 1.30

    for depth, radius in radii.items():
        if radius > 0:
            ax.add_patch(plt.Circle((0, 0), radius, fill=False, color="#ececec", linewidth=0.42, zorder=0))

    for parent, child in edges:
        draw_edge(ax, parent, child)

    l1_nodes = [node for node in nodes if node["_depth"] == 1]
    for node in l1_nodes:
        wedge = Wedge(
            (0, 0),
            sector_r1,
            math.degrees(node["_a0"]),
            math.degrees(node["_a1"]),
            width=sector_r1 - sector_r0,
            facecolor=node["_c"],
            edgecolor="white",
            linewidth=1.2,
            alpha=0.96,
            zorder=2,
        )
        ax.add_patch(wedge)

        label = str(node.get("label", ""))
        arc_radius = (sector_r0 + sector_r1) / 2
        # Use a slightly larger starting size for Chinese (square glyphs are
        # easier to read at smaller sizes than thin English letters).
        starting_fs = 18.5 if lang == "zh" else 17.5
        draw_arc_label(
            ax,
            label,
            node["_a0"],
            node["_a1"],
            arc_radius,
            "white",
            starting_fs,
            "bold",
            6,
            lang=lang,
            pt_to_data=pt_to_data,
            min_fontsize=8.5,
        )

    node_sizes = {1: 340, 2: 140, 3: 50, 4: 22, 5: 14, 6: 10}
    for node in nodes:
        depth = node["_depth"]
        if depth == 0:
            continue
        size = node_sizes.get(depth, 8)
        edge_color = "white" if depth <= 3 else node["_c"]
        linewidth = 1.6 if depth == 1 else (1.1 if depth == 2 else (0.6 if depth == 3 else 0.4))
        ax.scatter(
            [node["_x"]],
            [node["_y"]],
            s=size,
            c=node["_c"],
            edgecolors=edge_color,
            linewidths=linewidth,
            zorder=3,
        )

    root_circle = plt.Circle((0, 0), 0.85, facecolor="#1a1a1a", edgecolor="white", linewidth=2.5, zorder=4)
    ax.add_patch(root_circle)
    if lang == "zh":
        ax.text(0, 0.10, "航空货运", ha="center", va="center", fontsize=16, fontweight="bold", color="white", zorder=5)
        ax.text(0, -0.22, "枢纽", ha="center", va="center", fontsize=19, fontweight="bold", color="white", zorder=5)
    else:
        ax.text(0, 0.08, "AIR CARGO", ha="center", va="center", fontsize=15, fontweight="bold", color="white", zorder=5)
        ax.text(0, -0.20, "HUB", ha="center", va="center", fontsize=18, fontweight="bold", color="white", zorder=5)

    level_cfg = {
        2: dict(fs=9.0, weight="bold", stroke=2.4, color_self=True, off=0.16, z=6, ref=0.10),
        3: dict(fs=7.0, weight="bold", stroke=1.8, color_self=True, off=0.14, z=5, ref=0.06),
        4: dict(fs=5.6, weight="normal", stroke=1.3, color_self=False, off=0.12, z=4, ref=0.04),
        5: dict(fs=4.9, weight="normal", stroke=1.0, color_self=False, off=0.11, z=4, ref=0.04),
        6: dict(fs=4.6, weight="normal", stroke=0.9, color_self=False, off=0.10, z=4, ref=0.05),
    }

    for depth, cfg in level_cfg.items():
        for node in [item for item in nodes if item["_depth"] == depth]:
            angle = node["_a"]
            deg = math.degrees(angle)
            gap_ratio = min(1.0, node.get("_min_gap", cfg["ref"]) / cfg["ref"])
            fs = cfg["fs"] * (0.78 + 0.22 * gap_ratio)
            units = display_units(str(node.get("label", "")))
            if depth >= 4 and units > 18:
                fs *= 0.90
            if lang == "zh" and depth >= 4:
                fs *= 1.05
            if math.cos(angle) >= 0:
                rotation, ha = deg, "left"
            else:
                rotation, ha = deg + 180, "right"
            label_radius = node["_r"] + cfg["off"]
            tx = label_radius * math.cos(angle)
            ty = label_radius * math.sin(angle)
            color = node["_c"] if cfg["color_self"] else "#222222"
            txt = ax.text(
                tx,
                ty,
                str(node.get("label", "")),
                ha=ha,
                va="center",
                fontsize=fs,
                fontweight=cfg["weight"],
                color=color,
                rotation=rotation,
                rotation_mode="anchor",
                zorder=cfg["z"],
            )
            txt.set_path_effects([pe.withStroke(linewidth=cfg["stroke"], foreground="white")])

    ax.text(0, limit - 0.25, title, ha="center", va="top", fontsize=24, fontweight="bold", color="#1a1a1a")
    ax.text(
        0,
        limit - 0.92,
        subtitle_for(nodes, lang),
        ha="center",
        va="top",
        fontsize=13,
        color="#666666",
        style="italic",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_kwargs: dict[str, Any] = {"dpi": dpi, "bbox_inches": "tight", "facecolor": "white"}
    if image_format.lower() in {"jpg", "jpeg"}:
        save_kwargs["format"] = "jpg"
        save_kwargs["pil_kwargs"] = {"quality": 92, "optimize": True}
    fig.savefig(output_path, **save_kwargs)
    plt.close(fig)


def main() -> None:
    configure_fonts()
    parser = argparse.ArgumentParser(
        description="Render policy-tree JSON files as publication-style radial visualizations. "
                    "Each input file is rendered as-is in whatever language its labels "
                    "are written in (Chinese vs English is auto-detected per file)."
    )
    parser.add_argument("--base-dir", default=str(DEFAULT_BASE_DIR),
                        help="Directory containing top-level tree JSONs")
    parser.add_argument("--admin-dir", default=None,
                        help="Directory containing per-admin tree JSONs (default: <base-dir>/trees_by_admin)")
    parser.add_argument("--inputs", nargs="*", default=None,
                        help="Explicit list of input JSON files (overrides automatic discovery)")
    parser.add_argument("--format", default="jpg", choices=["jpg", "png"],
                        help="Output image format")
    parser.add_argument("--dpi", type=int, default=300, help="Output image DPI")
    parser.add_argument("--output-tag", default="radial",
                        help="Filename tag inserted before the language code "
                             "(default: radial; use 'radial' to drop the version suffix)")
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    admin_dir = Path(args.admin_dir) if args.admin_dir else base_dir / "trees_by_admin"

    if args.inputs:
        paths = [Path(p) for p in args.inputs]
        missing = [p for p in paths if not p.exists()]
        if missing:
            raise FileNotFoundError(
                "The following --inputs paths do not exist: "
                + ", ".join(str(p) for p in missing)
            )
    else:
        paths = input_tree_paths(base_dir, admin_dir)

    summary = {
        "base_dir": str(base_dir),
        "admin_dir": str(admin_dir),
        "output_tag": args.output_tag,
        "items": [],
    }

    print(f"[INFO] Found {len(paths)} tree JSON files")
    for path in paths:
        tree = load_json(path)
        lang = detect_language(tree)
        title = title_for(path, lang)
        out_image = visual_path(path, lang, args.format, args.output_tag)
        print(f"[{lang.upper()}] {path.name} -> {out_image.name}")
        draw_radial_tree(tree, out_image, lang, title, args.format, args.dpi)
        summary["items"].append(
            {
                "source_json": str(path),
                "language": lang,
                "title": title,
                "image": str(out_image),
            }
        )

    summary_path = base_dir / f"{args.output_tag}_summary.json"
    write_json(summary_path, summary)
    print(f"[DONE] Summary: {summary_path}")


if __name__ == "__main__":
    main()
