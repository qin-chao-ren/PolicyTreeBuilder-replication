from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import re
from pathlib import Path
from typing import Any, Optional

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from utils.llm_client import load_env_file
from v5.step2_pau_extraction.preprocessor import Preprocessor
from v5.step2_pau_extraction.extractor import PAUExtractor
from v5.step2_pau_extraction.postprocessor import PostProcessor


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_env_vars(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: resolve_env_vars(v) for k, v in value.items()}
    if isinstance(value, list):
        return [resolve_env_vars(v) for v in value]
    if isinstance(value, str):
        pattern = re.compile(r"\$\{([^}]+)\}")

        def replacer(match: re.Match) -> str:
            env_key = match.group(1)
            env_val = os.getenv(env_key)
            if env_val is None:
                raise ValueError(f"Missing environment variable: {env_key}")
            return env_val

        return pattern.sub(replacer, value)
    return value


def setup_logging(log_dir: Optional[str]) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_dir:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        log_path = Path(log_dir) / "v5_pau_extraction.log"
        handlers.append(logging.FileHandler(log_path, encoding="utf-8"))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="V5 PAU Extraction")
    parser.add_argument("--config", required=True, help="Config YAML path")
    parser.add_argument(
        "--env",
        default=str(SCRIPT_DIR.parent / "configs" / "roundC_v4.env"),
        help="Env file path (roundC_v4.env)",
    )
    args = parser.parse_args()

    load_env_file(args.env)
    config = load_yaml(args.config)
    config = resolve_env_vars(config)

    setup_logging(config.get("output", {}).get("log_dir"))
    logger = logging.getLogger(__name__)

    logger.info("=== V5 Step 2: PAU Extraction ===")

    input_path = Path(config["input"]["corpus_path"])
    if not input_path.exists():
        raise FileNotFoundError(f"Corpus not found: {input_path}")

    output_cfg = config.get("output", {})
    for key in ["raw_output", "validated_output", "report_output"]:
        if key in output_cfg:
            Path(output_cfg[key]).parent.mkdir(parents=True, exist_ok=True)

    logger.info("Step 1: Preprocessing...")
    preprocessor = Preprocessor(config)
    df = preprocessor.load_corpus(str(input_path))
    blocks = preprocessor.prepare_blocks(df)
    logger.info("Prepared %s text blocks", len(blocks))

    logger.info("Step 2: Extracting PAUs...")
    extractor = PAUExtractor(config)
    asyncio.run(extractor.run(blocks))

    logger.info("Step 3: Post-processing...")
    postprocessor = PostProcessor(config)
    result_df = postprocessor.process_raw_results(output_cfg["raw_output"])
    result_df.to_csv(output_cfg["validated_output"], index=False, encoding="utf-8-sig")

    report = postprocessor.generate_report(result_df)
    report_path = Path(output_cfg["report_output"])
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Validation Report: %s", report.get("summary"))

    logger.info("=== PAU Extraction Completed ===")


if __name__ == "__main__":
    main()
