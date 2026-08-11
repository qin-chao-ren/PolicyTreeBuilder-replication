from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

from utils.local_reference_binding import (  # noqa: E402
    bind_local_reference_payload,
    build_local_reference_context,
    call_local_reference_json,
)
from utils import local_reference_binding as local_binding  # noqa: E402
from utils.tree_manager import TreeManager  # noqa: E402


def evidence(summary="The displayed relation is supported."):
    return {
        "summary": summary,
        "warnings": [],
        "target_represents_all_source_members": False,
        "membership_basis": "",
        "pure_structural_redundancy": False,
        "cross_l1_authorized": False,
    }


def child_plan(child_ref, target_ref, disposition="move"):
    return {
        "child_ref": child_ref,
        "disposition": disposition,
        "target_parent_ref": target_ref,
        "relation": "broader_narrower",
        "same_domain": True,
        "evidence": "The displayed child belongs under the displayed target.",
    }


def local_decision(
    action="keep",
    relation="broader_narrower",
    source_ref="CHILD",
    target_ref="PARENT",
    *,
    plan=None,
    **changes,
):
    value = {
        "relation": relation,
        "action": action,
        "source_ref": source_ref,
        "target_ref": target_ref,
        "new_label": None,
        "confidence": 0.95,
        "evidence": evidence(),
        "child_plan": list(plan or []),
    }
    value.update(changes)
    return value


def envelope(context, decisions):
    return {"context_token": context.context_token, "decisions": list(decisions)}


def transport_result(payload, *, attempts=1, ok=True, phase_raw="response"):
    history = []
    for index in range(1, attempts + 1):
        final = index == attempts
        history.append({
            "attempt": index,
            "status": 200 if final else 503,
            "latency_ms": index,
            "raw": phase_raw if final else "retryable",
            "parse_ok": bool(final and ok),
            "error": None if final and ok else "retryable failure",
        })
    return {
        "ok": ok,
        "json": copy.deepcopy(payload) if ok else None,
        "raw": phase_raw,
        "error": None if ok else "request failed",
        "status": 200 if ok else 503,
        "latency_ms": attempts,
        "profile": "fake",
        "provider": "fake",
        "model": "fake",
        "attempts": attempts,
        "attempt_history": history,
    }


class FakeTransport:
    def __init__(self, *results, mutate=None):
        self.results = list(results)
        self.calls = []
        self.mutate = mutate

    def __call__(self, **kwargs):
        self.calls.append(copy.deepcopy(kwargs))
        if self.mutate is not None:
            self.mutate()
        if not self.results:
            raise AssertionError("unexpected fifth/raw transport call")
        result = self.results.pop(0)
        if result["attempts"] > kwargs["retries"] + 1:
            raise AssertionError("fake transport exceeded supplied attempt budget")
        return copy.deepcopy(result)


def tree():
    return {
        "node_id": "ROOT",
        "label": "root",
        "level": "ROOT",
        "children": [{
            "node_id": "L1_N10000000",
            "label": "domain",
            "level": "L1",
            "children": [{
                "node_id": "L2_Nd39445b0",
                "label": "parent",
                "level": "L2",
                "children": [{
                    "node_id": "L3_Nd551ce89",
                    "label": "child",
                    "level": "L3",
                    "children": [{
                        "node_id": "L4_N40000000",
                        "label": "grandchild",
                        "level": "L4",
                        "children": [],
                    }],
                }],
            }],
        }],
    }


class LocalReferenceBindingTests(unittest.TestCase):
    def setUp(self):
        self.context = build_local_reference_context(
            task="14a",
            user_text=(
                "Parent L2_Nd39445b0; child L3_Nd551ce89; "
                "direct child L4_N40000000."
            ),
            candidate_node_ids=(
                "L1_N10000000",
                "L2_Nd39445b0",
                "L3_Nd551ce89",
                "L4_N40000000",
            ),
            preferred_refs=(
                ("PARENT", "L2_Nd39445b0"),
                ("CHILD", "L3_Nd551ce89"),
            ),
            contract_text="local schema",
            expected_count=1,
        )

    def test_context_is_deterministic_single_use_and_scrubs_real_ids(self):
        duplicate = build_local_reference_context(
            task="14a",
            user_text=(
                "Parent L2_Nd39445b0; child L3_Nd551ce89; "
                "direct child L4_N40000000."
            ),
            candidate_node_ids=(
                "L1_N10000000",
                "L4_N40000000",
                "L3_Nd551ce89",
                "L2_Nd39445b0",
            ),
            preferred_refs=(
                ("PARENT", "L2_Nd39445b0"),
                ("CHILD", "L3_Nd551ce89"),
            ),
            contract_text="local schema",
            expected_count=1,
        )
        self.assertEqual(self.context.context_sha256, duplicate.context_sha256)
        self.assertEqual(self.context.context_token, duplicate.context_token)
        for real_id in self.context.mapping.values():
            self.assertNotIn(real_id, self.context.model_user)
        self.assertIn("PARENT", self.context.model_user)
        self.assertIn("CHILD", self.context.model_user)

        other = build_local_reference_context(
            task="14a",
            user_text="Parent L2_N00000000; child L3_N11111111.",
            candidate_node_ids=("L2_N00000000", "L3_N11111111"),
            preferred_refs=(("PARENT", "L2_N00000000"), ("CHILD", "L3_N11111111")),
        )
        self.assertNotEqual(self.context.context_token, other.context_token)

        changed_contract = build_local_reference_context(
            task="14a",
            user_text=(
                "Parent L2_Nd39445b0; child L3_Nd551ce89; "
                "direct child L4_N40000000."
            ),
            candidate_node_ids=(
                "L1_N10000000",
                "L2_Nd39445b0",
                "L3_Nd551ce89",
                "L4_N40000000",
            ),
            preferred_refs=(
                ("PARENT", "L2_Nd39445b0"),
                ("CHILD", "L3_Nd551ce89"),
            ),
            contract_text="changed system contract",
            expected_count=1,
        )
        self.assertNotEqual(self.context.context_token, changed_contract.context_token)

        changed_task = build_local_reference_context(
            task="14a-other-task",
            user_text=(
                "Parent L2_Nd39445b0; child L3_Nd551ce89; "
                "direct child L4_N40000000."
            ),
            candidate_node_ids=self.context.known_node_ids,
            preferred_refs=(
                ("PARENT", "L2_Nd39445b0"),
                ("CHILD", "L3_Nd551ce89"),
            ),
            contract_text="local schema",
            expected_count=1,
        )
        changed_count = build_local_reference_context(
            task="14a",
            user_text=(
                "Parent L2_Nd39445b0; child L3_Nd551ce89; "
                "direct child L4_N40000000."
            ),
            candidate_node_ids=self.context.known_node_ids,
            preferred_refs=(
                ("PARENT", "L2_Nd39445b0"),
                ("CHILD", "L3_Nd551ce89"),
            ),
            contract_text="local schema",
            expected_count=None,
        )
        self.assertNotEqual(self.context.context_token, changed_task.context_token)
        self.assertNotEqual(self.context.context_token, changed_count.context_token)
        schema_bytes = local_binding.LOCAL_SCHEMA_PATH.read_bytes()
        self.assertEqual(
            self.context.schema_sha256,
            local_binding.hashlib.sha256(schema_bytes).hexdigest(),
        )
        with tempfile.TemporaryDirectory() as temporary_dir:
            changed_schema = Path(temporary_dir) / "schema.json"
            changed_schema.write_bytes(schema_bytes + b"\n")
            with mock.patch.object(local_binding, "LOCAL_SCHEMA_PATH", changed_schema):
                schema_context = build_local_reference_context(
                    task="14a",
                    user_text=(
                        "Parent L2_Nd39445b0; child L3_Nd551ce89; "
                        "direct child L4_N40000000."
                    ),
                    candidate_node_ids=self.context.known_node_ids,
                    preferred_refs=(
                        ("PARENT", "L2_Nd39445b0"),
                        ("CHILD", "L3_Nd551ce89"),
                    ),
                    contract_text="local schema",
                    expected_count=1,
                )
                mismatch_fake = FakeTransport(
                    transport_result(envelope(self.context, [local_decision()]))
                )
                with self.assertRaises(ValueError):
                    call_local_reference_json(
                        transport=mismatch_fake,
                        profile="fake",
                        system="local schema",
                        context=self.context,
                        task="14a",
                        expected_count=1,
                    )
                self.assertEqual(mismatch_fake.calls, [])
        self.assertNotEqual(self.context.context_token, schema_context.context_token)
        changed_id_pattern = local_binding.re.compile(
            r"^(?:ROOT|NODE_[A-Za-z0-9]+)$"
        )
        with mock.patch.object(
            local_binding, "REAL_NODE_ID_PATTERN", changed_id_pattern
        ):
            runtime_context = build_local_reference_context(
                task="14a",
                user_text=(
                    "Parent L2_Nd39445b0; child L3_Nd551ce89; "
                    "direct child L4_N40000000."
                ),
                candidate_node_ids=self.context.known_node_ids,
                preferred_refs=(
                    ("PARENT", "L2_Nd39445b0"),
                    ("CHILD", "L3_Nd551ce89"),
                ),
                contract_text="local schema",
                expected_count=1,
            )
        self.assertNotEqual(self.context.context_token, runtime_context.context_token)

    def test_context_rejects_duplicate_refs_targets_and_undisplayed_targets(self):
        cases = [
            (("PARENT", "L2_Nd39445b0"), ("PARENT", "L3_Nd551ce89")),
            (("PARENT", "L2_Nd39445b0"), ("CHILD", "L2_Nd39445b0")),
            (("PARENT", "L1_N10000000"),),
        ]
        for preferred in cases:
            with self.subTest(preferred=preferred), self.assertRaises(ValueError):
                build_local_reference_context(
                    task="bad",
                    user_text="L2_Nd39445b0 L3_Nd551ce89",
                    candidate_node_ids=("L2_Nd39445b0", "L3_Nd551ce89"),
                    preferred_refs=preferred,
                )

        # Complete node-token matching must not infer a displayed shorter ID
        # from a longer ID that uses the contract's .:- continuation chars.
        for text in ("Node L2_ABC.DEF.", "Node L2_X.L2_ABC."):
            with self.subTest(text=text), self.assertRaises(ValueError):
                build_local_reference_context(
                    task="prefix-forbidden",
                    user_text=text,
                    candidate_node_ids=("L2_ABC", "L2_ABC.DEF", "L2_X.L2_ABC"),
                    preferred_refs=(("SHORT", "L2_ABC"),),
                )
        exact_long = build_local_reference_context(
            task="long-token",
            user_text="Node L2_ABC.DEF.",
            candidate_node_ids=("L2_ABC", "L2_ABC.DEF"),
            preferred_refs=(("LONG", "L2_ABC.DEF"),),
        )
        self.assertEqual(exact_long.mapping, {"LONG": "L2_ABC.DEF"})
        self.assertIn("Node LONG.", exact_long.localized_user)

    def test_exact_binding_preserves_semantics_and_child_order(self):
        decision = local_decision(
            "split_reparent",
            "related",
            source_ref="CHILD",
            target_ref="CHILD",
            plan=[child_plan("N0", "PARENT", disposition="move")],
        )
        result = bind_local_reference_payload(envelope(self.context, [decision]), self.context, expected_count=1)
        self.assertTrue(result.ok, result.errors)
        bound = result.bound_payload["decisions"][0]
        self.assertEqual(bound["source_id"], "L3_Nd551ce89")
        self.assertEqual(bound["target_id"], "L3_Nd551ce89")
        self.assertEqual(bound["child_plan"][0]["child_id"], "L4_N40000000")
        self.assertEqual(bound["child_plan"][0]["target_parent_id"], "L2_Nd39445b0")
        for field in ("relation", "action", "new_label", "confidence", "evidence"):
            self.assertEqual(bound[field], decision[field])

    def test_operation_109_legacy_real_id_payload_is_rejected_without_fuzzy_fix(self):
        legacy = {
            "decisions": [{
                "relation": "broader_narrower",
                "action": "keep",
                "source_id": "L3_Nd551ce89",
                "target_id": "L2_Nd39445b05",
                "new_label": None,
                "confidence": 0.9,
                "evidence": evidence("The child is narrower than the parent."),
                "child_plan": [],
            }],
        }
        result = bind_local_reference_payload(legacy, self.context, expected_count=1)
        self.assertFalse(result.ok)
        self.assertIn("FORBIDDEN_REAL_ID_FIELD", {item["code"] for item in result.errors})
        self.assertIsNone(result.bound_payload)
        manager = TreeManager(tree())
        before = json.dumps(manager.root, sort_keys=True)
        fake = FakeTransport(transport_result(legacy))
        ingress = call_local_reference_json(
            transport=fake,
            profile="fake",
            system="local schema",
            context=self.context,
            task="14a",
            expected_count=1,
            protected_manager=manager,
        )
        executor_calls = 0
        if ingress["ok"]:
            executor_calls += 1
        self.assertFalse(ingress["ok"])
        self.assertEqual(executor_calls, 0)
        self.assertEqual(ingress["fuzzy_corrections"], 0)
        self.assertEqual(json.dumps(manager.root, sort_keys=True), before)

    def test_unknown_real_id_cross_context_case_and_whitespace_refs_fail_closed(self):
        bad_values = (
            "L2_Nd39445b05",
            "L2_Nd39445b0",
            "parent",
            "PARENT ",
            "parent",
            "ＰＡＲＥＮＴ",
        )
        for value in bad_values:
            with self.subTest(value=value):
                payload = envelope(self.context, [local_decision(target_ref=value)])
                result = bind_local_reference_payload(payload, self.context)
                self.assertFalse(result.ok)
        swapped = envelope(self.context, [local_decision()])
        swapped["context_token"] = "CTX_0000000000000000"
        result = bind_local_reference_payload(swapped, self.context)
        self.assertFalse(result.ok)
        self.assertIn("CONTEXT_TOKEN_MISMATCH", {item["code"] for item in result.errors})

    def test_duplicate_child_ref_missing_extra_and_bad_envelope_fail_closed(self):
        duplicate = local_decision(
            "split_reparent",
            "related",
            source_ref="CHILD",
            target_ref="CHILD",
            plan=[child_plan("N0", "CHILD"), child_plan("N0", "PARENT")],
        )
        cases = [
            envelope(self.context, [duplicate]),
            {"context_token": self.context.context_token, "decisions": "bad"},
            [local_decision()],
            {"context_token": self.context.context_token, "decisions": [local_decision()], "extra": True},
        ]
        missing = envelope(self.context, [local_decision()])
        del missing["decisions"][0]["source_ref"]
        cases.append(missing)
        for payload in cases:
            with self.subTest(payload=payload):
                self.assertFalse(bind_local_reference_payload(payload, self.context).ok)

    def test_virtual_bridge_ref_binds_only_in_target_slots(self):
        context = build_local_reference_context(
            task="14b",
            user_text="Candidate L2_Nd39445b0; child L3_Nd551ce89; target __NEW_BRIDGE__.",
            candidate_node_ids=("L2_Nd39445b0", "L3_Nd551ce89"),
            preferred_refs=(("CANDIDATE", "L2_Nd39445b0"),),
            virtual_refs=(("NEW_BRIDGE", "__NEW_BRIDGE__"),),
        )
        decision = local_decision(
            "create_bridge",
            source_ref="CANDIDATE",
            target_ref="NEW_BRIDGE",
            new_label="bridge",
            plan=[child_plan("N0", "NEW_BRIDGE")],
        )
        result = bind_local_reference_payload(envelope(context, [decision]), context)
        self.assertTrue(result.ok, result.errors)
        bound = result.bound_payload["decisions"][0]
        self.assertEqual(bound["target_id"], "__NEW_BRIDGE__")
        self.assertEqual(bound["child_plan"][0]["target_parent_id"], "__NEW_BRIDGE__")
        decision["source_ref"] = "NEW_BRIDGE"
        self.assertFalse(bind_local_reference_payload(envelope(context, [decision]), context).ok)

    def test_all_stage_action_shapes_bind_without_real_ids(self):
        actions = (
            ("keep", "broader_narrower", "CHILD", "PARENT", []),
            ("reject_merge", "means_goal", "CHILD", "PARENT", []),
            ("uncertain", "uncertain", "CHILD", "PARENT", []),
            ("merge", "synonym", "CHILD", "PARENT", [child_plan("N0", "PARENT")]),
            ("rename", "related", "CHILD", "CHILD", []),
            ("move", "misplaced", "CHILD", "PARENT", [child_plan("N0", "CHILD", "retain_under_source")]),
            ("move_across_l1", "misplaced", "CHILD", "PARENT", [child_plan("N0", "CHILD", "retain_under_source")]),
            ("flatten", "broader_narrower", "CHILD", "PARENT", [child_plan("N0", "PARENT")]),
            ("split_reparent", "related", "CHILD", "CHILD", [child_plan("N0", "PARENT")]),
        )
        context = build_local_reference_context(
            task="all-actions",
            user_text=(
                "Parent L2_Nd39445b0; child L3_Nd551ce89; "
                "direct child L4_N40000000."
            ),
            candidate_node_ids=self.context.known_node_ids,
            preferred_refs=(
                ("PARENT", "L2_Nd39445b0"),
                ("CHILD", "L3_Nd551ce89"),
            ),
            expected_count=None,
        )
        decisions = [
            local_decision(action, relation, source, target, plan=plan)
            for action, relation, source, target, plan in actions
        ]
        result = bind_local_reference_payload(envelope(context, decisions), context)
        self.assertTrue(result.ok, result.errors)
        self.assertEqual([item["action"] for item in result.bound_payload["decisions"]], [item[0] for item in actions])
        mismatch = bind_local_reference_payload(
            envelope(context, decisions), context, expected_count=1
        )
        self.assertFalse(mismatch.ok)
        self.assertIn(
            "EXPECTED_COUNT_CONTRACT_MISMATCH",
            {item["code"] for item in mismatch.errors},
        )

    def test_one_schema_only_repair_shares_the_four_attempt_budget(self):
        invalid = envelope(self.context, [local_decision(target_ref="UNKNOWN")])
        repaired = envelope(self.context, [local_decision(target_ref="PARENT")])
        fake = FakeTransport(
            transport_result(invalid, attempts=3, phase_raw="initial-invalid"),
            transport_result(repaired, attempts=1, phase_raw="repair-valid"),
        )
        result = call_local_reference_json(
            transport=fake,
            profile="fake",
            system="local schema",
            context=self.context,
            task="14a",
            expected_count=1,
        )
        self.assertTrue(result["ok"], result)
        self.assertEqual(result["attempts"], 4)
        self.assertEqual(result["repair_count"], 1)
        self.assertEqual(len(result["attempt_history"]), 4)
        self.assertEqual([item["global_attempt"] for item in result["attempt_history"]], [1, 2, 3, 4])
        self.assertEqual(len(fake.calls), 2)
        self.assertEqual(fake.calls[0]["retries"], 3)
        self.assertEqual(fake.calls[1]["retries"], 0)
        self.assertEqual(result["final_bound"]["decisions"][0]["target_id"], "L2_Nd39445b0")

    def test_bad_envelope_can_receive_one_repair_without_semantic_or_evidence_drift(self):
        original = local_decision(target_ref="UNKNOWN")
        malformed = {
            "context_token": self.context.context_token,
            "decision": original,
        }
        repaired = envelope(self.context, [
            {**copy.deepcopy(original), "target_ref": "PARENT"}
        ])
        fake = FakeTransport(
            transport_result(malformed),
            transport_result(repaired, attempts=3),
        )
        result = call_local_reference_json(
            transport=fake,
            profile="fake",
            system="local schema",
            context=self.context,
            task="14a",
            expected_count=1,
        )
        self.assertTrue(result["ok"], result)
        self.assertEqual(result["attempts"], 4)
        self.assertEqual(result["repair_count"], 1)
        self.assertEqual(
            result["final_bound"]["decisions"][0]["evidence"],
            original["evidence"],
        )
        self.assertEqual([item["phase"] for item in result["attempt_history"]], ["initial", "repair", "repair", "repair"])

    def test_model_response_real_ids_are_terminal_including_evidence(self):
        direct = local_decision()
        direct["evidence"]["summary"] = "Archived provenance L3_Nd551ce89."
        direct_fake = FakeTransport(
            transport_result(envelope(self.context, [direct]))
        )
        direct_result = call_local_reference_json(
            transport=direct_fake,
            profile="fake",
            system="local schema",
            context=self.context,
            task="14a",
            expected_count=1,
        )
        self.assertFalse(direct_result["ok"], direct_result)
        self.assertEqual(direct_result["error"], "LOCAL_REFERENCE_BINDING_FAILED")
        self.assertIn(
            "REAL_NODE_ID_IN_MODEL_RESPONSE",
            {item["code"] for item in direct_result["initial_binding_errors"]},
        )
        self.assertEqual(direct_result["repair_count"], 0)
        self.assertEqual(len(direct_fake.calls), 1)
        self.assertEqual(
            direct_result["opaque_evidence_id_hits"][0]["path"],
            "decisions[0].evidence",
        )
        self.assertEqual(
            direct_result["opaque_evidence_id_hits"][0]["hit_count"], 1
        )

        original = local_decision(target_ref="UNKNOWN")
        original["evidence"]["summary"] = "Do not echo L3_Nd551ce89."
        malformed = envelope(self.context, [original])
        fake = FakeTransport(transport_result(malformed))
        result = call_local_reference_json(
            transport=fake,
            profile="fake",
            system="local schema",
            context=self.context,
            task="14a",
            expected_count=1,
        )
        self.assertFalse(result["ok"])
        self.assertEqual(result["error"], "LOCAL_REFERENCE_BINDING_FAILED")
        self.assertIn(
            "REAL_NODE_ID_IN_MODEL_RESPONSE",
            {item["code"] for item in result["initial_binding_errors"]},
        )
        self.assertEqual(result["repair_count"], 0)
        self.assertEqual(len(fake.calls), 1)

        unsafe_system = "Schema example real node: L2_Nd39445b0"
        unsafe_context = build_local_reference_context(
            task="unsafe-prompt",
            user_text="Parent L2_Nd39445b0; child L3_Nd551ce89.",
            candidate_node_ids=("L2_Nd39445b0", "L3_Nd551ce89"),
            preferred_refs=(("PARENT", "L2_Nd39445b0"), ("CHILD", "L3_Nd551ce89")),
            contract_text=unsafe_system,
            expected_count=1,
        )
        unsafe_fake = FakeTransport(
            transport_result(envelope(unsafe_context, [local_decision()]))
        )
        unsafe_result = call_local_reference_json(
            transport=unsafe_fake,
            profile="fake",
            system=unsafe_system,
            context=unsafe_context,
            task="unsafe-prompt",
            expected_count=1,
        )
        self.assertFalse(unsafe_result["ok"])
        self.assertEqual(unsafe_result["error"], "MODEL_PROMPT_REAL_ID_RISK")
        self.assertEqual(unsafe_result["attempts"], 0)
        self.assertEqual(len(unsafe_fake.calls), 0)
        embedded_system = "Schema example xL2_Nd39445b0y"
        embedded_context = build_local_reference_context(
            task="unsafe-embedded-prompt",
            user_text="Parent L2_Nd39445b0; child L3_Nd551ce89.",
            candidate_node_ids=("L2_Nd39445b0", "L3_Nd551ce89"),
            preferred_refs=(("PARENT", "L2_Nd39445b0"), ("CHILD", "L3_Nd551ce89")),
            contract_text=embedded_system,
            expected_count=1,
        )
        embedded_fake = FakeTransport(
            transport_result(envelope(embedded_context, [local_decision()]))
        )
        embedded_result = call_local_reference_json(
            transport=embedded_fake,
            profile="fake",
            system=embedded_system,
            context=embedded_context,
            task="unsafe-embedded-prompt",
            expected_count=1,
        )
        self.assertFalse(embedded_result["ok"])
        self.assertEqual(embedded_result["error"], "MODEL_PROMPT_REAL_ID_RISK")
        self.assertEqual(embedded_result["attempts"], 0)
        self.assertEqual(len(embedded_fake.calls), 0)
        for punctuation in (".", ":", "-"):
            with self.subTest(prompt_punctuation=punctuation):
                shaped_system = (
                    "Schema example node L2_N00000000" + punctuation
                )
                shaped_context = build_local_reference_context(
                    task="unsafe-shaped-prompt",
                    user_text="Parent L2_Nd39445b0; child L3_Nd551ce89.",
                    candidate_node_ids=("L2_Nd39445b0", "L3_Nd551ce89"),
                    preferred_refs=(
                        ("PARENT", "L2_Nd39445b0"),
                        ("CHILD", "L3_Nd551ce89"),
                    ),
                    contract_text=shaped_system,
                    expected_count=1,
                )
                shaped_fake = FakeTransport(
                    transport_result(
                        envelope(shaped_context, [local_decision()])
                    )
                )
                shaped_result = call_local_reference_json(
                    transport=shaped_fake,
                    profile="fake",
                    system=shaped_system,
                    context=shaped_context,
                    task="unsafe-shaped-prompt",
                    expected_count=1,
                )
                self.assertFalse(shaped_result["ok"])
                self.assertEqual(
                    shaped_result["error"], "MODEL_PROMPT_REAL_ID_RISK"
                )
                self.assertEqual(shaped_fake.calls, [])

        embedded_unknown = local_decision()
        embedded_unknown["new_label"] = "xL4_OUTSIDE99y"
        embedded_unknown_fake = FakeTransport(
            transport_result(envelope(self.context, [embedded_unknown]))
        )
        embedded_unknown_result = call_local_reference_json(
            transport=embedded_unknown_fake,
            profile="fake",
            system="local schema",
            context=self.context,
            task="14a",
            expected_count=1,
        )
        self.assertFalse(embedded_unknown_result["ok"])
        self.assertIn(
            "REAL_NODE_ID_IN_MODEL_RESPONSE",
            {
                item["code"]
                for item in embedded_unknown_result["initial_binding_errors"]
            },
        )
        self.assertEqual(embedded_unknown_result["repair_count"], 0)
        self.assertEqual(len(embedded_unknown_fake.calls), 1)

    def test_raw_response_real_ids_are_terminal_on_initial_and_repair(self):
        clean = envelope(self.context, [local_decision()])
        initial_fake = FakeTransport(transport_result(
            clean,
            phase_raw="Leaked L4_OUTSIDE99 before the clean JSON envelope.",
        ))
        initial_result = call_local_reference_json(
            transport=initial_fake,
            profile="fake",
            system="local schema",
            context=self.context,
            task="14a",
            expected_count=1,
        )
        self.assertFalse(initial_result["ok"], initial_result)
        self.assertEqual(
            initial_result["error"], "REAL_NODE_ID_IN_RAW_RESPONSE"
        )
        self.assertEqual(initial_result["repair_count"], 0)
        self.assertIsNone(initial_result["final_bound"])
        self.assertEqual(len(initial_fake.calls), 1)

        initial_bad = envelope(
            self.context, [local_decision(target_ref="UNKNOWN")]
        )
        repair_fake = FakeTransport(
            transport_result(initial_bad, phase_raw="clean-initial"),
            transport_result(
                clean,
                phase_raw="Repair leaked xL4_OUTSIDE99y before clean JSON.",
            ),
        )
        repair_result = call_local_reference_json(
            transport=repair_fake,
            profile="fake",
            system="local schema",
            context=self.context,
            task="14a",
            expected_count=1,
        )
        self.assertFalse(repair_result["ok"], repair_result)
        self.assertEqual(
            repair_result["error"], "REAL_NODE_ID_IN_RAW_RESPONSE"
        )
        self.assertEqual(repair_result["repair_count"], 1)
        self.assertIsNone(repair_result["final_bound"])
        self.assertEqual(len(repair_fake.calls), 2)

    def test_scope_wrong_valid_refs_can_receive_one_reference_only_repair(self):
        invalid = envelope(self.context, [
            local_decision(source_ref="PARENT", target_ref="PARENT")
        ])
        repaired = envelope(self.context, [local_decision()])

        def pair_scope(payload):
            decision = payload["decisions"][0]
            if {decision["source_id"], decision["target_id"]} == {
                "L2_Nd39445b0", "L3_Nd551ce89"
            }:
                return []
            return [{
                "code": "DECISION_SCOPE_VIOLATION",
                "message": "decision must use the displayed pair",
                "context": {
                    "mutable_ref_paths": [
                        "decisions[0].source_ref",
                        "decisions[0].target_ref",
                    ],
                    "repairable": True,
                },
            }]

        fake = FakeTransport(transport_result(invalid), transport_result(repaired))
        result = call_local_reference_json(
            transport=fake,
            profile="fake",
            system="local schema",
            context=self.context,
            task="14a",
            expected_count=1,
            bound_validator=pair_scope,
        )
        self.assertTrue(result["ok"], result)
        self.assertEqual(result["repair_count"], 1)
        self.assertEqual(result["attempts"], 2)
        events = [item["event"] for item in result["events"]]
        self.assertLess(events.index("scope_validation_rejected"), events.index("repair_started"))
        self.assertIn("repair_scope_validation_pass", events)

    def test_four_initial_attempts_leave_no_hidden_repair_budget(self):
        invalid = envelope(self.context, [local_decision(target_ref="UNKNOWN")])
        fake = FakeTransport(transport_result(invalid, attempts=4))
        result = call_local_reference_json(
            transport=fake,
            profile="fake",
            system="local schema",
            context=self.context,
            task="14a",
            expected_count=1,
        )
        self.assertFalse(result["ok"])
        self.assertEqual(result["attempts"], 4)
        self.assertEqual(result["repair_count"], 0)
        self.assertEqual(len(fake.calls), 1)

    def test_second_schema_failure_semantic_drift_and_valid_ref_drift_stay_closed(self):
        invalid = envelope(self.context, [local_decision(target_ref="UNKNOWN")])
        still_invalid = envelope(self.context, [local_decision(target_ref="MISSING")])
        drift = envelope(self.context, [local_decision(target_ref="PARENT", action="reject_merge")])
        ref_drift_initial = envelope(self.context, [local_decision(source_ref="CHILD", target_ref="UNKNOWN")])
        ref_drift_repair = envelope(self.context, [local_decision(source_ref="PARENT", target_ref="CHILD")])
        numeric_drift_initial = envelope(self.context, [local_decision(
            target_ref="UNKNOWN", confidence=1,
        )])
        numeric_drift_repair = envelope(self.context, [local_decision(
            target_ref="PARENT", confidence=1.0,
        )])
        for initial, repaired, code in (
            (invalid, still_invalid, "REPAIR_BINDING_FAILED"),
            (invalid, drift, "REPAIR_SEMANTIC_DRIFT"),
            (ref_drift_initial, ref_drift_repair, "REPAIR_VALID_REF_DRIFT"),
            (numeric_drift_initial, numeric_drift_repair, "REPAIR_SEMANTIC_DRIFT"),
        ):
            with self.subTest(code=code):
                fake = FakeTransport(transport_result(initial), transport_result(repaired))
                result = call_local_reference_json(
                    transport=fake,
                    profile="fake",
                    system="local schema",
                    context=self.context,
                    task="14a",
                    expected_count=1,
                )
                self.assertFalse(result["ok"])
                self.assertEqual(result["error"], code)
                self.assertEqual(result["repair_count"], 1)
                self.assertEqual(len(fake.calls), 2)

    def test_real_id_and_cross_context_fail_without_repair(self):
        real_id = envelope(self.context, [local_decision(target_ref="L2_Nd39445b0")])
        hidden_real_id = envelope(self.context, [local_decision(target_ref="L1_N10000000")])
        other_context_real_id = envelope(self.context, [local_decision(target_ref="L2_N00000000")])
        known_id_new_label = envelope(self.context, [local_decision(
            action="rename",
            relation="synonym",
            source_ref="CHILD",
            target_ref="CHILD",
            new_label="L2_Nd39445b0",
        )])
        shaped_id_new_label = envelope(self.context, [local_decision(
            action="rename",
            relation="synonym",
            source_ref="CHILD",
            target_ref="CHILD",
            new_label="L2_N00000000",
        )])
        embedded_known_id_new_label = envelope(self.context, [local_decision(
            action="rename",
            relation="synonym",
            source_ref="CHILD",
            target_ref="CHILD",
            new_label="xL2_Nd39445b0y",
        )])
        punctuated_shaped_labels = [
            envelope(self.context, [local_decision(
                action="rename",
                relation="synonym",
                source_ref="CHILD",
                target_ref="CHILD",
                new_label="L2_N00000000" + punctuation,
            )])
            for punctuation in (".", ":", "-")
        ]
        wrong_context = envelope(self.context, [local_decision()])
        wrong_context["context_token"] = "CTX_FFFFFFFFFFFFFFFF"
        missing_context = envelope(self.context, [local_decision()])
        del missing_context["context_token"]
        for payload in (
            real_id,
            hidden_real_id,
            other_context_real_id,
            known_id_new_label,
            embedded_known_id_new_label,
            shaped_id_new_label,
            *punctuated_shaped_labels,
            wrong_context,
            missing_context,
        ):
            fake = FakeTransport(transport_result(payload))
            manager = TreeManager(tree())
            before = json.dumps(manager.root, sort_keys=True)
            result = call_local_reference_json(
                transport=fake,
                profile="fake",
                system="local schema",
                context=self.context,
                task="14a",
                expected_count=1,
                protected_manager=manager,
            )
            self.assertFalse(result["ok"])
            self.assertEqual(result["repair_count"], 0)
            self.assertEqual(len(fake.calls), 1)
            self.assertEqual(json.dumps(manager.root, sort_keys=True), before)

    def test_transport_audit_mismatch_fails_closed(self):
        valid = envelope(self.context, [local_decision()])
        bad = transport_result(valid)
        bad["attempts"] = 2
        fake = FakeTransport(bad)
        result = call_local_reference_json(
            transport=fake,
            profile="fake",
            system="local schema",
            context=self.context,
            task="14a",
            expected_count=1,
        )
        self.assertFalse(result["ok"])
        self.assertEqual(result["error"], "ATTEMPT_HISTORY_INVALID")


    def test_mutation_before_validation_is_detected_rolled_back_and_not_bound(self):
        valid = envelope(self.context, [local_decision()])
        for mutation_kind in ("root", "parent_map", "index"):
            with self.subTest(mutation_kind=mutation_kind):
                manager = TreeManager(tree())
                before = json.dumps(manager.root, sort_keys=True)

                def mutate():
                    if mutation_kind == "root":
                        manager.rename_node("L3_Nd551ce89", "tampered")
                    elif mutation_kind == "parent_map":
                        manager.parent_map["L3_Nd551ce89"] = "ROOT"
                    else:
                        manager.index.pop("L3_Nd551ce89")

                fake = FakeTransport(transport_result(valid), mutate=mutate)
                result = call_local_reference_json(
                    transport=fake,
                    profile="fake",
                    system="local schema",
                    context=self.context,
                    task="14a",
                    expected_count=1,
                    protected_manager=manager,
                )
                self.assertFalse(result["ok"])
                self.assertTrue(result["mutation_before_validation"])
                self.assertEqual(result["error"], "MUTATION_BEFORE_VALIDATION")
                self.assertEqual(json.dumps(manager.root, sort_keys=True), before)
                self.assertEqual(manager.validate_consistency(), [])

    def test_local_schema_and_all_four_prompts_publish_the_ref_contract(self):
        schema = json.loads(
            (REPO_ROOT / "schemas" / "local_reference_semantic_tree_decision.schema.json").read_text(encoding="utf-8")
        )
        self.assertEqual(schema["required"], ["context_token", "decisions"])
        decision = schema["$defs"]["decision"]
        self.assertIn("source_ref", decision["required"])
        self.assertIn("target_ref", decision["required"])
        self.assertNotIn("source_id", decision["properties"])
        for name in (
            "collapse_redundant_hierarchy.md",
            "balance_tree_structure.md",
            "polish_tree_labels.md",
            "finalize_tree_structure.md",
        ):
            text = (REPO_ROOT / "prompts" / name).read_text(encoding="utf-8")
            self.assertIn('"context_token"', text)
            self.assertIn('"source_ref"', text)
            self.assertIn('"target_ref"', text)
            self.assertIn('"source_id"', text)  # named only in the explicit forbidden-fields rule

        schema_invalid = []
        bad_relation = local_decision()
        bad_relation["relation"] = "BOGUS"
        schema_invalid.append((bad_relation, "DECISION_SEMANTIC_VALUE_INVALID"))
        bad_action = local_decision()
        bad_action["action"] = "BOGUS"
        schema_invalid.append((bad_action, "DECISION_SEMANTIC_VALUE_INVALID"))
        bad_confidence = local_decision()
        bad_confidence["confidence"] = 1.01
        schema_invalid.append((bad_confidence, "DECISION_SEMANTIC_VALUE_INVALID"))
        duplicate_warnings = local_decision()
        duplicate_warnings["evidence"]["warnings"] = ["duplicate", "duplicate"]
        schema_invalid.append((duplicate_warnings, "EVIDENCE_VALUE_INVALID"))
        empty_summary = local_decision()
        empty_summary["evidence"]["summary"] = ""
        schema_invalid.append((empty_summary, "EVIDENCE_VALUE_INVALID"))
        bad_child = local_decision(
            "split_reparent",
            "related",
            source_ref="CHILD",
            target_ref="CHILD",
            plan=[child_plan("N0", "PARENT", disposition="move")],
        )
        bad_child["child_plan"][0]["disposition"] = "BOGUS"
        schema_invalid.append((bad_child, "CHILD_SEMANTIC_VALUE_INVALID"))
        for invalid, expected_code in schema_invalid:
            with self.subTest(expected_code=expected_code, invalid=invalid):
                result = bind_local_reference_payload(
                    envelope(self.context, [invalid]),
                    self.context,
                    expected_count=1,
                )
                self.assertFalse(result.ok)
                self.assertIn(expected_code, {item["code"] for item in result.errors})


if __name__ == "__main__":
    unittest.main()
