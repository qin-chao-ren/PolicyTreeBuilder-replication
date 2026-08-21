"""C13RF27: space out transport retries, and stop swallowing the stop signal.

Two independent defects, both measured on the C13RF26 online run rather than
inferred from reading code.

**Transport retries were not spaced.**  ``call_llm_json`` used one schedule for
every retry class, ``backoff ** (attempt - 1)``, which at the shipped default of
``backoff=1.5`` is 1.0s / 1.5s / 2.25s.  For a content failure that is right --
the response arrived, it was merely unparseable, so there is nothing to wait
out.  For a transport failure it is not.  RF26's 14d run made four attempts on
its only call touching the duplicate parent/child pair; all four were read
timeouts of 120.1s, so the call spanned roughly eight minutes of which the
"backoff" was 4.75s -- about 1%.  Four attempts against one network stall is
effectively one attempt, and because that call never returned, the model never
got to propose the merge that would have cleared the E0 publication blocker.
``TransportRetryDelayTests`` and ``CallLlmJsonScheduleTests`` pin the new
schedule and, just as importantly, pin that the content path did **not** move.

**The fail-closed stop signal was swallowed (open debt D11).**  The private
harness guard wraps ``call_local_reference_json`` and
``execute_semantic_decision`` and raises ``FailClosedCallError`` to stop a run at
the first non-deferrable failure.  But 14b and 14d wrap their work in
``except Exception`` and turn any exception into a locally-handled outcome, so
the stop became a log line.  RF26 measured that too: 14d's fail-closed report was
written at 10:17:37 and the stage then issued six more API calls (10:19:16
through 10:23:44) before E0 stopped it.  14a and 14c re-raise after rolling back
and were never affected.

D11 was filed against 14d alone.  The parallel-channel scan this card ran first
-- required by the RF12/RF16/RF18 lesson that a verdict usually has more than one
copy -- found **two** channels across **four** stages, and 14b shares 14d's
shape on channel two.  Both are fixed here; ``ChannelRegistryTests`` pins the
full map so a future stage cannot quietly regrow the swallow.

The control groups carry the weight in both halves: a real transport failure
must still fail, a real stage error must still be handled locally, and a
legitimately deferred call must still be allowed to continue.  None of the scope
predicates, defer criteria or contract rules are touched by this card.
"""
from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import llm_runtime  # noqa: E402

from tests.test_semantic_stage_dispatch import load_stage_module  # noqa: E402
from utils.local_reference_binding import (  # noqa: E402
    FAIL_CLOSED_SIGNAL_TYPE_NAMES,
    is_fail_closed_stop_signal,
)


# The guard's exception type, reproduced with the same name and base class the
# private harness uses (``run_stage_fail_closed.py:16`` and its v2 successor at
# ``run_stage_fail_closed_v2.py:42``).  Production code cannot import the real
# one -- the harness is not on the public import path -- so recognition is by
# class name and this stand-in is exactly what the predicate must match.
class FailClosedCallError(RuntimeError):
    pass


class SubclassedFailClosedCallError(FailClosedCallError):
    """A future guard might subclass its own error type; the stop must survive."""


# ``build_local_reference_context`` only maps refs whose node ID is literally
# displayed in the call text, so the audited L1 has to appear in the context.
_CONTEXT_TEXT = "L1 under audit: L1_N00000001\nchildren: L2_N00000002, L3_N00000003\n"


class FakeResponse:
    def __init__(self, status, content="", body=""):
        self.status_code = status
        self._content = content
        self.text = body

    def json(self):
        if self.status_code != 200:
            return {}
        return {"choices": [{"message": {"content": self._content}}]}


def profile(retries=3, backoff=1.5):
    return llm_runtime.ResolvedProfile(
        name="fake",
        provider="openai_compatible",
        model="fake-model",
        api_key="fake-key",
        base_url="https://invalid.example/v1",
        retries=retries,
        backoff=backoff,
    )


class TransportRetryDelayTests(unittest.TestCase):
    """The schedule itself, independent of who calls it."""

    def test_nominal_schedule_grows_geometrically(self):
        # Jitter pinned to its midpoint (0.5 -> spread factor 1.0) so the
        # nominal schedule is observable exactly.
        mid = lambda: 0.5  # noqa: E731
        self.assertAlmostEqual(
            llm_runtime.transport_retry_delay_seconds(1, jitter_source=mid), 5.0
        )
        self.assertAlmostEqual(
            llm_runtime.transport_retry_delay_seconds(2, jitter_source=mid), 15.0
        )
        self.assertAlmostEqual(
            llm_runtime.transport_retry_delay_seconds(3, jitter_source=mid), 45.0
        )

    def test_schedule_is_capped(self):
        mid = lambda: 0.5  # noqa: E731
        # Without a cap, attempt 5 would be 5 * 3**4 = 405s, which would let one
        # logical call eat a meaningful slice of the wall-clock budget.
        self.assertAlmostEqual(
            llm_runtime.transport_retry_delay_seconds(5, jitter_source=mid),
            llm_runtime.TRANSPORT_BACKOFF_CAP_S,
        )
        self.assertLessEqual(
            llm_runtime.transport_retry_delay_seconds(40, jitter_source=lambda: 0.999),
            llm_runtime.TRANSPORT_BACKOFF_CAP_S * (
                1.0 + llm_runtime.TRANSPORT_BACKOFF_JITTER
            ),
        )

    def test_jitter_spreads_both_directions_and_keeps_a_floor(self):
        low = llm_runtime.transport_retry_delay_seconds(2, jitter_source=lambda: 0.0)
        high = llm_runtime.transport_retry_delay_seconds(
            2, jitter_source=lambda: 0.999999
        )
        self.assertLess(low, 15.0)
        self.assertGreater(high, 15.0)
        # Symmetric, so the expected delay is still the nominal one.
        self.assertAlmostEqual((low + high) / 2.0, 15.0, places=3)
        # A pathological source must not collapse the wait to nothing.
        self.assertGreaterEqual(
            llm_runtime.transport_retry_delay_seconds(1, jitter_source=lambda: 0.0),
            llm_runtime.TRANSPORT_BACKOFF_BASE_S / 2.0,
        )

    def test_rf26_window_would_now_be_spread(self):
        """The regression this card exists for, stated as a number.

        RF26: four attempts, 120.1s each, 4.75s of total spacing.  The same four
        attempts now straddle at least a minute of stall.
        """
        mid = lambda: 0.5  # noqa: E731
        old = sum(1.5 ** (attempt - 1) for attempt in range(1, 4))
        new = sum(
            llm_runtime.transport_retry_delay_seconds(attempt, jitter_source=mid)
            for attempt in range(1, 4)
        )
        self.assertAlmostEqual(old, 4.75)
        self.assertGreaterEqual(new, 60.0)

    def test_attempt_must_be_one_based(self):
        with self.assertRaises(ValueError):
            llm_runtime.transport_retry_delay_seconds(0)


class CallLlmJsonScheduleTests(unittest.TestCase):
    """Which schedule each retry class actually gets, and the attempt budget."""

    def call(self, side_effect, retries=3):
        with patch.object(llm_runtime, "resolve_profile", return_value=profile(retries)), \
             patch.object(llm_runtime.requests, "post", side_effect=side_effect) as post, \
             patch.object(llm_runtime.time, "sleep") as sleep:
            result = llm_runtime.call_llm_json(
                profile="fake", system="system", user="user", task="test",
            )
        return result, post.call_count, [c.args[0] for c in sleep.call_args_list]

    def test_network_exception_retries_use_the_transport_schedule(self):
        import requests as _requests

        timeout = _requests.exceptions.ReadTimeout("read timeout=120.0")
        result, calls, delays = self.call([timeout, timeout, timeout, timeout])
        self.assertFalse(result["ok"])
        self.assertEqual(calls, 4)
        self.assertEqual(result["attempts"], 4)
        self.assertEqual(len(delays), 3)
        # Every wait is a real one, and they grow.
        self.assertTrue(all(d >= llm_runtime.TRANSPORT_BACKOFF_BASE_S / 2.0 for d in delays))
        self.assertLess(delays[0], delays[-1])
        # Jitter is live here, so bound against the jittered worst case (nominal
        # 5+15+45=65s, each scaled by at least 1 - JITTER) rather than the
        # nominal sum -- asserting the nominal figure against a jittered run is
        # a flaky test, not a stricter one.
        floor = 65.0 * (1.0 - llm_runtime.TRANSPORT_BACKOFF_JITTER)
        self.assertGreaterEqual(sum(delays), floor)

    def test_retryable_http_status_uses_the_transport_schedule(self):
        result, calls, delays = self.call([
            FakeResponse(503, body="unavailable"),
            FakeResponse(503, body="unavailable"),
            FakeResponse(200, '{"ok":true}'),
        ])
        self.assertTrue(result["ok"], result)
        self.assertEqual(calls, 3)
        self.assertEqual(len(delays), 2)
        self.assertTrue(all(d >= llm_runtime.TRANSPORT_BACKOFF_BASE_S / 2.0 for d in delays))

    def test_content_failure_keeps_the_fast_schedule(self):
        """Control group: the content path must NOT have been slowed down.

        The body arrived promptly; there is no stall to wait out.  Slowing this
        down would buy nothing and would lengthen every repair round.
        """
        result, calls, delays = self.call([
            FakeResponse(200, "not json"),
            FakeResponse(200, '{"ok":true}'),
        ])
        self.assertTrue(result["ok"], result)
        self.assertEqual(calls, 2)
        self.assertEqual(len(delays), 1)
        # 1.5 ** 0 == 1.0 -- the pre-RF27 value, unchanged.
        self.assertAlmostEqual(delays[0], 1.0)
        self.assertLess(delays[0], llm_runtime.TRANSPORT_BACKOFF_BASE_S / 2.0)

    def test_non_retryable_status_still_fails_immediately(self):
        """Control group: backoff must not turn a hard rejection into a retry."""
        result, calls, delays = self.call([FakeResponse(401, body="unauthorized")])
        self.assertFalse(result["ok"])
        self.assertEqual(calls, 1)
        self.assertEqual(delays, [])

    def test_attempts_budget_is_unchanged(self):
        """Control group: this card changes *when* attempts happen, not how many.

        The attempts budget is the thing every card's HTTP ceiling is written
        against, so a schedule change must not smuggle in extra attempts.
        """
        import requests as _requests

        timeout = _requests.exceptions.ReadTimeout("read timeout=120.0")
        for retries in (0, 1, 3):
            with self.subTest(retries=retries):
                _result, calls, _delays = self.call([timeout] * (retries + 5), retries=retries)
                self.assertEqual(calls, retries + 1)

    def test_success_on_first_attempt_never_sleeps(self):
        result, calls, delays = self.call([FakeResponse(200, '{"ok":true}')])
        self.assertTrue(result["ok"], result)
        self.assertEqual(calls, 1)
        self.assertEqual(delays, [])


class FailClosedSignalPredicateTests(unittest.TestCase):
    """Recognising the stop signal, including what must NOT be recognised."""

    def test_guard_error_is_recognised(self):
        self.assertTrue(is_fail_closed_stop_signal(FailClosedCallError("stop")))

    def test_subclass_is_recognised(self):
        self.assertTrue(
            is_fail_closed_stop_signal(SubclassedFailClosedCallError("stop"))
        )

    def test_ordinary_errors_are_not_recognised(self):
        """Control group: the predicate must be narrow.

        ``FailClosedCallError`` derives from ``RuntimeError``, so matching by
        base class would swallow-then-propagate a large class of genuine stage
        errors and turn recoverable failures into aborted runs.
        """
        for exc in (
            RuntimeError("a plain runtime error"),
            ValueError("bad value"),
            KeyError("missing"),
            TimeoutError("slow"),
            Exception("generic"),
        ):
            with self.subTest(exc=type(exc).__name__):
                self.assertFalse(is_fail_closed_stop_signal(exc))

    def test_name_set_is_pinned(self):
        # If the private guard ever renames its exception, this is the single
        # place that must be updated -- and the failure will say so.
        self.assertEqual(FAIL_CLOSED_SIGNAL_TYPE_NAMES, frozenset({"FailClosedCallError"}))


class ChannelRegistryTests(unittest.TestCase):
    """The parallel-channel map, pinned so it cannot silently regrow.

    Channel one is the LLM call boundary (guard wraps
    ``call_local_reference_json``); channel two is the decision-execution
    boundary (guard wraps ``execute_semantic_decision``).  A stage is safe on a
    channel if the stop signal keeps propagating: either because nothing catches
    it, or because the handler re-raises it.
    """

    def test_stages_do_not_swallow_the_stop_signal(self):
        sources = {
            name: (SCRIPTS / f"{name}.py").read_text(encoding="utf-8")
            for name in (
                "collapse_redundant_hierarchy",
                "balance_tree_structure",
                "polish_tree_labels",
                "finalize_policy_tree",
            )
        }
        # Every stage that catches broadly around guarded work must reference the
        # predicate; 14a and 14c re-raise unconditionally and need not.
        for name in ("balance_tree_structure", "finalize_policy_tree"):
            with self.subTest(stage=name):
                self.assertIn("is_fail_closed_stop_signal", sources[name])
        for name in ("collapse_redundant_hierarchy", "polish_tree_labels"):
            with self.subTest(stage=name, expectation="re-raises unconditionally"):
                self.assertIn("raise", sources[name])

    def test_finalize_channel_one_propagates_the_stop(self):
        """14d's ``_call_llm``: the D11 defect itself."""
        module = load_stage_module("finalize_policy_tree")
        stage = _finalize_stub(module)

        with patch.object(
            module, "call_local_reference_json",
            side_effect=FailClosedCallError("guard stop"),
        ):
            with self.assertRaises(FailClosedCallError):
                stage._call_llm(_CONTEXT_TEXT, "L1_N00000001")

        # The evidence trail is still written before propagating.
        self.assertTrue(stage.llm_log.exists())
        self.assertIn("guard stop", stage.llm_log.read_text(encoding="utf-8"))

    def test_finalize_channel_one_still_handles_real_errors_locally(self):
        """Control group: a genuine stage error must not become a stop."""
        module = load_stage_module("finalize_policy_tree")
        stage = _finalize_stub(module)

        with patch.object(
            module, "call_local_reference_json",
            side_effect=RuntimeError("transport blew up"),
        ):
            result = stage._call_llm(_CONTEXT_TEXT, "L1_N00000001")

        self.assertEqual(result, [])
        self.assertEqual(stage.stats["llm_failures"], 1)
        self.assertEqual(len(stage.operation_records), 1)
        contract = stage.operation_records[0].get("semantic_contract", {})
        codes = {v.get("code") for v in contract.get("violations", [])}
        self.assertIn("LLM_CALL_FAILED", codes)


class ChannelTwoTests(unittest.TestCase):
    """The decision-execution boundary, in both stages that wrap it broadly.

    D11 was filed against 14d.  The scan found 14b shares the shape, so both are
    covered here.  In each case the abort records must still be written -- the
    fix changes only whether the run continues afterwards, never the evidence.
    """

    def _run_14d_batch(self, side_effect):
        module = load_stage_module("finalize_policy_tree")
        stage = _finalize_batch_stub(module)
        ops = [{
            "action": "keep",
            "source_id": "L2_N00000002",
            "target_id": "L2_N00000002",
            "relation": "exact_duplicate",
            "child_plan": [],
            "evidence": {"summary": "keep", "warnings": []},
        }]
        with patch.object(module, "execute_semantic_decision", side_effect=side_effect):
            return module, stage, ops, stage._apply_operations(ops, "L1_N00000001")

    def test_finalize_channel_two_propagates_the_stop(self):
        module = load_stage_module("finalize_policy_tree")
        stage = _finalize_batch_stub(module)
        ops = [{
            "action": "keep",
            "source_id": "L2_N00000002",
            "target_id": "L2_N00000002",
            "relation": "exact_duplicate",
            "child_plan": [],
            "evidence": {"summary": "keep", "warnings": []},
        }]
        with patch.object(
            module, "execute_semantic_decision",
            side_effect=FailClosedCallError("guard stop"),
        ):
            with self.assertRaises(FailClosedCallError):
                stage._apply_operations(ops, "L1_N00000001")

        # The abort trail is written before the stop propagates.
        self.assertTrue(stage.ops_log.exists())
        written = stage.ops_log.read_text(encoding="utf-8")
        self.assertIn("BATCH_SEMANTIC_ABORTED", written)

    def test_finalize_channel_two_still_aborts_locally_on_real_errors(self):
        """Control group: a real execution error stays a batch abort."""
        _module, stage, _ops, result = self._run_14d_batch(
            RuntimeError("execution blew up")
        )
        self.assertIsInstance(result, list)
        written = stage.ops_log.read_text(encoding="utf-8")
        self.assertIn("BATCH_SEMANTIC_ABORTED", written)

    def test_balance_channel_two_propagates_the_stop(self):
        module = load_stage_module("balance_tree_structure")
        stage = _balance_stub(module)
        with patch.object(
            module, "execute_semantic_decision",
            side_effect=FailClosedCallError("guard stop"),
        ):
            with self.assertRaises(FailClosedCallError):
                stage._apply("L1_N00000001", _balance_payload(), "jump")

        self.assertTrue(stage.ops_log.exists())
        self.assertIn(
            "BATCH_SEMANTIC_ABORTED", stage.ops_log.read_text(encoding="utf-8")
        )

    def test_balance_channel_two_still_aborts_locally_on_real_errors(self):
        """Control group: a real execution error stays a batch abort."""
        module = load_stage_module("balance_tree_structure")
        stage = _balance_stub(module)
        with patch.object(
            module, "execute_semantic_decision",
            side_effect=RuntimeError("execution blew up"),
        ):
            result = stage._apply("L1_N00000001", _balance_payload(), "jump")

        self.assertFalse(result)
        self.assertIn(
            "BATCH_SEMANTIC_ABORTED", stage.ops_log.read_text(encoding="utf-8")
        )


def _finalize_stub(module):
    """A 14d instance with just enough state for ``_call_llm``."""
    import tempfile

    tmp = Path(tempfile.mkdtemp(prefix="c13rf27_"))
    stage = module.OverallStructureAudit.__new__(module.OverallStructureAudit)
    stage.llm_log = tmp / "llm.jsonl"
    stage.ops_log = tmp / "ops.jsonl"
    stage.llm_profile = "fake"
    stage.stats = {"llm_failures": 0, "llm_calls": 0}
    stage.operation_records = []
    stage._pending_call_audit = None

    class _TM:
        def get_all_node_ids(self):
            # Must include the audited L1 itself: ``_call_llm`` builds a local
            # reference context with ``preferred_refs=(("L1", l1_id),)``, and the
            # binder rejects a preferred ref that is not among the candidates.
            return ["L1_N00000001", "L2_N00000002", "L3_N00000003"]

    stage.tm = _TM()
    return stage


def _tmp_dir():
    import tempfile

    return Path(tempfile.mkdtemp(prefix="c13rf27_"))


def _small_tree():
    return {
        "node_id": "ROOT",
        "label": "root",
        "level": "ROOT",
        "children": [{
            "node_id": "L1_N00000001",
            "label": "L1",
            "level": "L1",
            "children": [{
                "node_id": "L2_N00000002",
                "label": "L2",
                "level": "L2",
                "children": [],
            }],
        }],
    }


def _balance_payload():
    """A batch that clears 14b's scope preflight and reaches the guarded loop.

    ``keep`` in structure balancing requires ``source == target == candidate``
    (``balance_tree_structure.py:505``), so the decision must name the audited
    node itself -- otherwise the batch aborts at preflight and never reaches the
    channel under test.
    """
    return {"decisions": [{
        "action": "keep",
        "source_id": "L1_N00000001",
        "target_id": "L1_N00000001",
        "relation": "exact_duplicate",
        "child_plan": [],
        "evidence": {"summary": "keep", "warnings": []},
    }]}


def _finalize_batch_stub(module):
    """A 14d instance with enough state for ``_apply_operations``."""
    from utils.tree_manager import TreeManager

    tmp = _tmp_dir()
    stage = module.OverallStructureAudit.__new__(module.OverallStructureAudit)
    stage.ops_log = tmp / "ops.jsonl"
    stage.llm_log = tmp / "llm.jsonl"
    stage.tm = TreeManager(_small_tree())
    stage.redirect_map = {}
    stage.operation_records = []
    stage.membership_input = tmp / "missing_membership.csv"
    stage.membership_input_explicit = False
    stage.stats = {
        "llm_calls": 0, "llm_failures": 0,
        "ops_applied": 0, "ops_skipped": 0, "ops_deferred": 0,
    }
    stage._pending_call_audit = None
    return stage


def _balance_stub(module):
    """A 14b instance with enough state for ``_apply``."""
    from utils.tree_manager import TreeManager

    tmp = _tmp_dir()
    stage = module.ShapingProcess.__new__(module.ShapingProcess)
    stage.ops_log = tmp / "ops.jsonl"
    stage.llm_log = tmp / "llm.jsonl"
    stage.tm = TreeManager(_small_tree())
    stage.trace_map = {}
    stage.prior_lineage = {}
    stage.membership_counts = {}
    stage.deferred_candidates = set()
    return stage


if __name__ == "__main__":
    unittest.main()
