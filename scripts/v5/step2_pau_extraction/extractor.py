from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import aiohttp

from .preprocessor import TextBlock
from .prompts import PromptManager

logger = logging.getLogger(__name__)


class PAUExtractor:
    def __init__(self, config: dict) -> None:
        self.config = config
        llm_cfg = config.get("llm", {})
        processing_cfg = config.get("processing", {})

        slot = llm_cfg.get("model_slot") or "PRIMARY_LLM"
        self.model = llm_cfg.get("model") or os.getenv(f"{slot}_MODEL") or "qwen3-max"
        self.api_key = self._resolve_api_key(
            llm_cfg.get("api_key") or os.getenv(f"{slot}_API_KEY") or os.getenv("OPENAI_API_KEY", "")
        )
        base_url = llm_cfg.get("base_url") or os.getenv(f"{slot}_BASE_URL") or os.getenv("OPENAI_BASE_URL", "")
        self.api_endpoint = llm_cfg.get("api_endpoint") or self._build_endpoint(base_url)

        self.temperature = float(llm_cfg.get("temperature", 0.1))
        self.max_tokens = int(llm_cfg.get("max_tokens", 2000))
        self.timeout = float(llm_cfg.get("timeout", 60))
        self.retries = int(llm_cfg.get("retries", 3))
        self.backoff_factor = float(llm_cfg.get("backoff_factor", 1.5))

        self.concurrent = int(processing_cfg.get("concurrent_requests", 3))
        self.batch_size = int(processing_cfg.get("batch_size", 10))

        self.output_path = Path(config["output"]["raw_output"])
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        self.prompt_manager = PromptManager()

    async def call_qwen_api(self, messages: List[dict], session: aiohttp.ClientSession) -> Dict:
        if not self.api_endpoint:
            raise RuntimeError("LLM API endpoint is missing. Set PRIMARY_LLM_BASE_URL or config llm.api_endpoint.")
        if not self.api_key:
            raise RuntimeError("LLM API key is missing. Set PRIMARY_LLM_API_KEY or config llm.api_key.")

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        last_error: Optional[str] = None
        for attempt in range(1, self.retries + 1):
            t0 = time.time()
            status = None
            raw_text = ""
            try:
                async with session.post(
                    self.api_endpoint,
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=self.timeout),
                ) as resp:
                    status = resp.status
                    raw_text = await resp.text()
                    if status == 200:
                        try:
                            data = json.loads(raw_text)
                        except json.JSONDecodeError:
                            data = {
                                "choices": [
                                    {"message": {"content": raw_text}}
                                ]
                            }
                        return {
                            "ok": True,
                            "status": status,
                            "data": data,
                            "raw_text": raw_text,
                            "attempts": attempt,
                            "latency_ms": int((time.time() - t0) * 1000),
                        }
                    if status not in {429, 500, 502, 503, 504}:
                        return {
                            "ok": False,
                            "status": status,
                            "error": f"HTTP {status}: {raw_text[:200]}",
                            "raw_text": raw_text,
                            "attempts": attempt,
                            "latency_ms": int((time.time() - t0) * 1000),
                        }
                    last_error = f"HTTP {status}: {raw_text[:200]}"
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                last_error = str(exc)

            await asyncio.sleep(self.backoff_factor ** (attempt - 1))

        return {
            "ok": False,
            "status": None,
            "error": last_error or "unknown_error",
            "raw_text": "",
            "attempts": self.retries,
            "latency_ms": None,
        }

    async def extract_single(self, block: TextBlock, session: aiohttp.ClientSession) -> Dict:
        messages = self.prompt_manager.get_messages(
            text=block.text,
            h_level=block.h_level,
            context_before=block.context_before,
            context_after=block.context_after,
        )

        response = await self.call_qwen_api(messages, session)

        return {
            "block_id": block.block_id,
            "doc_id": block.doc_id,
            "h_level": block.h_level,
            "block_type": block.block_type,
            "original_text": block.text,
            "llm_response": response.get("data") if response.get("ok") else None,
            "llm_error": None if response.get("ok") else response.get("error"),
            "status": "success" if response.get("ok") else "failed",
            "attempts": response.get("attempts"),
            "latency_ms": response.get("latency_ms"),
        }

    async def extract_batch(self, blocks: List[TextBlock], session: aiohttp.ClientSession) -> List[Dict]:
        semaphore = asyncio.Semaphore(self.concurrent)

        async def bounded(block: TextBlock) -> Dict:
            async with semaphore:
                return await self.extract_single(block, session)

        tasks = [asyncio.create_task(bounded(block)) for block in blocks]
        results: List[Dict] = []
        for task in asyncio.as_completed(tasks):
            result = await task
            self.save_result(result)
            results.append(result)
        return results

    def save_result(self, result: Dict) -> None:
        with self.output_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    async def run(self, blocks: List[TextBlock]) -> None:
        total = len(blocks)
        logger.info("Starting PAU extraction for %s blocks", total)

        async with aiohttp.ClientSession() as session:
            for start in range(0, total, self.batch_size):
                end = min(start + self.batch_size, total)
                batch = blocks[start:end]
                logger.info("Processing batch %s (%s-%s)", start // self.batch_size + 1, start + 1, end)

                await self.extract_batch(batch, session)
                await asyncio.sleep(1)

        logger.info("PAU extraction completed")

    @staticmethod
    def _resolve_api_key(value: str) -> str:
        if not value:
            return ""
        if value.startswith("${") and value.endswith("}"):
            env_key = value[2:-1]
            return os.getenv(env_key, "")
        return value

    @staticmethod
    def _build_endpoint(base_url: str) -> str:
        if not base_url:
            return ""
        return base_url.rstrip("/") + "/chat/completions"
