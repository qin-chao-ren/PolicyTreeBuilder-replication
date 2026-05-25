from __future__ import annotations

from dataclasses import dataclass
import re
from typing import List, Optional, Tuple

import pandas as pd


@dataclass
class TextBlock:
    block_id: str
    doc_id: str
    text: str
    h_level: str
    block_type: str
    context_before: List[str]
    context_after: List[str]


class Preprocessor:
    def __init__(self, config: dict) -> None:
        input_cfg = config.get("input", {})
        processing_cfg = config.get("processing", {})

        self.text_column = input_cfg.get("text_column", "block_text")
        self.level_column = input_cfg.get("level_column", "final_level")
        self.doc_id_column = input_cfg.get("doc_id_column", "doc_id")

        self.max_sentence_length = processing_cfg.get("max_sentence_length", 500)
        self.context_window_size = processing_cfg.get("context_window_size", 2)
        self.context_strategy = processing_cfg.get("context_strategy", "sliding_window")

    def load_corpus(self, filepath: str) -> pd.DataFrame:
        df = pd.read_csv(filepath, encoding="utf-8", keep_default_na=False)
        return df

    @staticmethod
    def classify_block_type(h_level: str) -> str:
        if h_level in {"H1", "H2", "H3"}:
            return "title"
        return "body"

    def split_sentences(self, text: str, max_length: Optional[int] = None) -> List[str]:
        max_len = max_length or self.max_sentence_length
        cleaned = " ".join(text.replace("\r", " ").split()).strip()
        if not cleaned:
            return []
        if len(cleaned) <= max_len:
            return [cleaned]

        parts = re.split(r"(?<=[。！？；;])", cleaned)
        parts = [p.strip() for p in parts if p.strip()]

        merged: List[str] = []
        buffer = ""
        for part in parts:
            if not buffer:
                buffer = part
                continue
            if len(buffer) + len(part) <= max_len:
                buffer = buffer + part
            else:
                merged.append(buffer)
                buffer = part
        if buffer:
            merged.append(buffer)

        final: List[str] = []
        for chunk in merged:
            if len(chunk) <= max_len:
                final.append(chunk)
                continue
            for i in range(0, len(chunk), max_len):
                final.append(chunk[i : i + max_len])

        return final

    def build_context_window(self, df: pd.DataFrame, idx: int, window_size: Optional[int] = None) -> Tuple[List[str], List[str]]:
        window = window_size or self.context_window_size
        total = len(df)
        if total == 0 or window <= 0:
            return [], []

        doc_id = self._safe_cell(df, idx, self.doc_id_column)
        before: List[str] = []
        for j in range(idx - 1, -1, -1):
            if doc_id and self._safe_cell(df, j, self.doc_id_column) != doc_id:
                break
            text = self._safe_cell(df, j, self.text_column)
            if text:
                before.append(self._clean_context(text))
            if len(before) >= window:
                break
        before.reverse()

        after: List[str] = []
        for j in range(idx + 1, total):
            if doc_id and self._safe_cell(df, j, self.doc_id_column) != doc_id:
                break
            text = self._safe_cell(df, j, self.text_column)
            if text:
                after.append(self._clean_context(text))
            if len(after) >= window:
                break

        return before, after

    def prepare_blocks(self, df: pd.DataFrame) -> List[TextBlock]:
        df = df.reset_index(drop=True)
        blocks: List[TextBlock] = []

        last_h1 = ""
        last_h2 = ""

        for idx, row in df.iterrows():
            doc_id = str(row.get(self.doc_id_column, f"DOC_{idx}"))
            text = str(row.get(self.text_column, "")).strip()
            h_level = str(row.get(self.level_column, "NA")).strip() or "NA"

            if h_level == "H1":
                last_h1 = text
                last_h2 = ""
            elif h_level == "H2":
                last_h2 = text

            if not text:
                continue

            block_type = self.classify_block_type(h_level)

            if self.context_strategy == "hierarchical" and h_level in {"H2", "H3"}:
                context_before = []
                if last_h1:
                    context_before.append(f"【H1】{self._clean_context(last_h1)}")
                if h_level == "H3" and last_h2:
                    context_before.append(f"【H2】{self._clean_context(last_h2)}")
                context_after = []
            else:
                context_before, context_after = self.build_context_window(df, idx)

            sentences = self.split_sentences(text)
            if not sentences:
                continue

            block_idx = row.get("block_idx", idx + 1)
            block_idx_str = self._format_block_idx(block_idx, idx)

            for sent_idx, sentence in enumerate(sentences):
                block_id = f"{doc_id}_{block_idx_str}"
                if len(sentences) > 1:
                    suffix = chr(97 + sent_idx)
                    block_id = f"{block_id}_{suffix}"

                blocks.append(
                    TextBlock(
                        block_id=block_id,
                        doc_id=doc_id,
                        text=sentence,
                        h_level=h_level,
                        block_type=block_type,
                        context_before=context_before,
                        context_after=context_after,
                    )
                )

        return blocks

    @staticmethod
    def _clean_context(text: str) -> str:
        return " ".join(str(text).strip().split())

    @staticmethod
    def _safe_cell(df: pd.DataFrame, idx: int, column: str) -> str:
        if column not in df.columns:
            return ""
        value = df.at[idx, column]
        if value is None:
            return ""
        return str(value).strip()

    @staticmethod
    def _format_block_idx(block_idx: object, fallback_idx: int) -> str:
        try:
            return f"{int(block_idx):03d}"
        except (TypeError, ValueError):
            return f"{fallback_idx + 1:03d}"
