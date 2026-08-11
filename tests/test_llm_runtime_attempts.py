from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

import llm_runtime  # noqa: E402


class FakeResponse:
    def __init__(self, status, content="", body=""):
        self.status_code = status
        self._content = content
        self.text = body

    def json(self):
        if self.status_code != 200:
            return {}
        return {"choices": [{"message": {"content": self._content}}]}


def profile(retries=3):
    return llm_runtime.ResolvedProfile(
        name="fake",
        provider="openai_compatible",
        model="fake-model",
        api_key="fake-key",
        base_url="https://invalid.example/v1",
        retries=retries,
        backoff=1.0,
    )


class LlmRuntimeAttemptTests(unittest.TestCase):
    def call(self, side_effect, retries=3):
        with patch.object(llm_runtime, "resolve_profile", return_value=profile(retries)), \
             patch.object(llm_runtime.requests, "post", side_effect=side_effect) as post, \
             patch.object(llm_runtime.time, "sleep"):
            result = llm_runtime.call_llm_json(
                profile="fake",
                system="system",
                user="user",
                task="test",
            )
        return result, post.call_count

    def test_http_200_json_parse_failure_retries_then_records_valid_raw(self):
        result, calls = self.call([
            FakeResponse(200, "not json"),
            FakeResponse(200, '{"ok":true}'),
        ])
        self.assertTrue(result["ok"], result)
        self.assertEqual(calls, 2)
        self.assertEqual(result["attempts"], 2)
        self.assertEqual(len(result["attempt_history"]), 2)
        self.assertFalse(result["attempt_history"][0]["parse_ok"])
        self.assertTrue(result["attempt_history"][1]["parse_ok"])

    def test_non_retryable_http_reports_actual_one_attempt_not_configured_four(self):
        result, calls = self.call([FakeResponse(400, body="bad request")])
        self.assertFalse(result["ok"])
        self.assertEqual(calls, 1)
        self.assertEqual(result["attempts"], 1)
        self.assertEqual(len(result["attempt_history"]), 1)

    def test_four_parse_failures_close_at_four_with_complete_history(self):
        result, calls = self.call([FakeResponse(200, "bad") for _ in range(4)])
        self.assertFalse(result["ok"])
        self.assertEqual(calls, 4)
        self.assertEqual(result["attempts"], 4)
        self.assertEqual([item["attempt"] for item in result["attempt_history"]], [1, 2, 3, 4])

    def test_exception_then_success_keeps_both_attempts(self):
        result, calls = self.call([
            RuntimeError("temporary"),
            FakeResponse(200, '{"ok":true}'),
        ])
        self.assertTrue(result["ok"], result)
        self.assertEqual(calls, 2)
        self.assertIn("RuntimeError", result["attempt_history"][0]["error"])
        self.assertEqual(result["attempts"], len(result["attempt_history"]))


if __name__ == "__main__":
    unittest.main()
