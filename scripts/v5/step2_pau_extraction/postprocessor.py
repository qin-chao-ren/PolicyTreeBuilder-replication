from __future__ import annotations

import json
import logging
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class DictionaryLoader:
    def __init__(self, dict_dir: str, files: Dict[str, str]) -> None:
        self.dict_dir = Path(dict_dir)
        self.domain_compounds = self._load(files.get("domain_compounds", "domain_compounds.json"))
        self.s_keywords = self._load(files.get("s_keywords", "s_keywords.json"))
        self.a_type_mapping = self._load(files.get("a_type_mapping", "a_type_mapping.json"))

    def _load(self, filename: str) -> dict:
        path = self.dict_dir / filename
        if not path.exists():
            logger.warning("Dictionary file not found: %s", path)
            return {}
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def classify_s_keyword(self, word: str) -> Optional[str]:
        for s_type in ["S_scope", "S_focus", "S_stage", "S_type", "S_noise"]:
            keywords = self.s_keywords.get(s_type, {})
            flat_list = keywords.get("flat_list", keywords.get("keywords", []))
            if isinstance(flat_list, list) and word in flat_list:
                return s_type
        return None


class PostProcessor:
    def __init__(self, config: dict) -> None:
        self.config = config
        validation_cfg = config.get("validation", {})
        dict_cfg = config.get("dictionaries", {})

        self.require_o = validation_cfg.get("require_o_field", True)
        self.require_a = validation_cfg.get("require_a_field", True)
        self.valid_t_levels = set(validation_cfg.get("valid_t_levels", ["T1", "T2", "T3", "T4"]))
        self.valid_a_types = set(validation_cfg.get("valid_a_types", ["direction", "substantive", "operational", "intent"]))

        self.dictionaries = DictionaryLoader(
            dict_cfg.get("dir", "dictionaries"),
            {
                "domain_compounds": dict_cfg.get("domain_compounds", "domain_compounds.json"),
                "s_keywords": dict_cfg.get("s_keywords", "s_keywords.json"),
                "a_type_mapping": dict_cfg.get("a_type_mapping", "a_type_mapping.json"),
            },
        )

    def parse_llm_response(self, response: Optional[dict]) -> Optional[dict]:
        if not response:
            return None
        try:
            content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
        except (AttributeError, IndexError, KeyError):
            return None

        if not content:
            return None

        return self._extract_json_from_text(content)

    def validate_pau(self, pau: dict) -> Dict:
        issues: List[str] = []

        if self.require_o and not pau.get("O"):
            issues.append("missing_O")
        if self.require_a and not pau.get("A"):
            issues.append("missing_A")

        a_type = pau.get("A_type")
        if a_type and a_type not in self.valid_a_types:
            issues.append("invalid_A_type")

        for s_field in ["S_scope", "S_focus", "S_stage", "S_type"]:
            value = pau.get(s_field)
            if not isinstance(value, list):
                issues.append(f"invalid_{s_field}_type")
                continue
            for item in value:
                expected = self.dictionaries.classify_s_keyword(str(item))
                if expected and expected != s_field:
                    issues.append(f"misclassified_{item}_as_{s_field}_expected_{expected}")

        t_level_base = pau.get("t_level_base")
        if t_level_base and t_level_base not in self.valid_t_levels:
            issues.append("invalid_t_level_base")

        return {"is_valid": len(issues) == 0, "issues": issues}

    def process_raw_results(self, raw_path: str) -> pd.DataFrame:
        records: List[dict] = []

        path = Path(raw_path)
        if not path.exists():
            logger.warning("Raw output not found: %s", raw_path)
            return pd.DataFrame(records)

        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue

                parsed = self.parse_llm_response(item.get("llm_response"))
                if not parsed or "pau_list" not in parsed:
                    records.append(
                        {
                            "block_id": item.get("block_id"),
                            "doc_id": item.get("doc_id"),
                            "h_level": item.get("h_level"),
                            "block_type": item.get("block_type"),
                            "original_text": item.get("original_text"),
                            "status": "parse_failed",
                            "pau_id": None,
                            "O": None,
                            "A": None,
                            "A_type": None,
                            "S_scope": None,
                            "S_focus": None,
                            "S_stage": None,
                            "S_type": None,
                            "pau_final": None,
                            "t_level_base": None,
                            "t_level_adjusted": None,
                            "is_leaf_candidate": None,
                            "validation_issues": json.dumps(["parse_failed"], ensure_ascii=False),
                        }
                    )
                    continue

                for pau in parsed.get("pau_list", []):
                    validation = self.validate_pau(pau)
                    records.append(
                        {
                            "block_id": item.get("block_id"),
                            "doc_id": item.get("doc_id"),
                            "h_level": item.get("h_level"),
                            "block_type": item.get("block_type"),
                            "original_text": item.get("original_text"),
                            "status": "valid" if validation["is_valid"] else "invalid",
                            "pau_id": pau.get("pau_id"),
                            "O": pau.get("O"),
                            "A": pau.get("A"),
                            "A_type": pau.get("A_type"),
                            "S_scope": json.dumps(pau.get("S_scope", []), ensure_ascii=False),
                            "S_focus": json.dumps(pau.get("S_focus", []), ensure_ascii=False),
                            "S_stage": json.dumps(pau.get("S_stage", []), ensure_ascii=False),
                            "S_type": json.dumps(pau.get("S_type", []), ensure_ascii=False),
                            "pau_final": pau.get("pau_final"),
                            "t_level_base": pau.get("t_level_base"),
                            "t_level_adjusted": pau.get("t_level_adjusted"),
                            "is_leaf_candidate": pau.get("is_leaf_candidate"),
                            "validation_issues": json.dumps(validation["issues"], ensure_ascii=False),
                        }
                    )

        return pd.DataFrame(records)

    def generate_report(self, df: pd.DataFrame) -> Dict:
        total = len(df)
        valid = int((df["status"] == "valid").sum()) if total else 0
        invalid = int((df["status"] == "invalid").sum()) if total else 0
        failed = int(df["status"].isin(["parse_failed", "failed"]).sum()) if total else 0

        t_level_dist = df["t_level_adjusted"].value_counts(dropna=True).to_dict() if total else {}
        leaf_dist = df["is_leaf_candidate"].value_counts(dropna=True).to_dict() if total else {}

        all_issues: List[str] = []
        for issues_str in df.get("validation_issues", []).dropna().tolist():
            try:
                all_issues.extend(json.loads(issues_str))
            except (TypeError, json.JSONDecodeError):
                continue

        issue_counts = dict(Counter(all_issues))

        return {
            "summary": {
                "total_records": total,
                "valid": valid,
                "invalid": invalid,
                "failed": failed,
                "success_rate": round(valid / total, 4) if total else 0,
            },
            "t_level_distribution": t_level_dist,
            "leaf_candidate_distribution": leaf_dist,
            "common_issues": issue_counts,
        }

    @staticmethod
    def _extract_json_from_text(text: str) -> Optional[dict]:
        content = text.strip()
        if "```json" in content:
            content = content.split("```json", 1)[1].split("```", 1)[0]
        elif "```" in content:
            content = content.split("```", 1)[1].split("```", 1)[0]

        content = content.strip()
        if not content:
            return None

        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass

        try:
            start = content.find("{")
            end = content.rfind("}")
            if start >= 0 and end > start:
                return json.loads(content[start : end + 1])
        except json.JSONDecodeError:
            pass

        try:
            fixed = content.replace("'", '"')
            fixed = re.sub(r",\s*}", "}", fixed)
            fixed = re.sub(r",\s*]", "]", fixed)
            return json.loads(fixed)
        except json.JSONDecodeError:
            logger.warning("Failed to parse JSON from response")
            return None
