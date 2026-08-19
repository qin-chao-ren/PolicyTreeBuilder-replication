"""C13RF21: narrow the leaked-node-ID grammar to the real identifier population.

The RF20 online run stopped at 14c call 97 on a false positive.  The model wrote
``different_parent_L1_domains`` in ``evidence.warnings`` -- a correct, harmless
caveat -- and the leak scanner read ``L1_domains`` as a leaked node ID.  Nothing
leaked.  The defect was that the grammar claimed far more of the string space
than real identifiers occupy: ``L[1-4]_`` followed by any ordinary characters.

The narrow grammar is derived from the generators, not from intuition:

* ``scripts/common_id.py:make_node_id``            -> ``L{1-4}_N{md5[:8]}``
* ``balance_tree_structure.generate_bridge_id``    -> ``{parent}_BR_{md5[:6]}``

plus one shape no current generator produces: a bare 4-hex group after a bridge
segment (``L1_N6be998d8_BR_8df89b_f5a4``), present in the still-live frozen 353
baseline.  A byte census of every reachable tree (446 distinct identifiers across
170 files) drove that grammar, and ``GrammarCoverageTests`` keeps every observed
shape pinned -- narrowing past a real shape would turn a false-positive fix into
a missed detection, which is strictly worse than the bug being fixed.

Three boundaries this file locks, each of which the work actually turned up:

* **Tokenization keeps the wide grammar on purpose.**  Detection and
  tokenization pull in opposite directions: narrow is correct for deciding
  whether a value leaked, wide is correct for splitting prompt text into
  complete tokens, because greedy tokenization is what stops a short candidate
  from being read as "displayed" out of the middle of a longer identifier.
  Narrowing both reopened that prefix-inference hole and broke three existing
  tests; ``TokenizationWidthTests`` pins the split.
* **A JSON-escaped identifier used to slip the raw gate.**  The gate scanned
  only the raw transport text, so ``L1_\\u004E762e6689`` carried no literal
  identifier byte-for-byte while decoding to a real one.  Measured on the RF19
  baseline: allowed.  ``EscapedIdentifierTests`` pins it closed.
* **A true hit used to archive the identifier verbatim.**  RF20's evidence
  bundle stored the full raw response; it was harmless only because that
  response held no real ID.  ``RedactionTests`` drives a response that does.

``ROOT`` stays a known residual false positive: it is a legitimate identifier and
an ordinary English word, so no grammar can separate ``"ROOT CAUSE analysis"``
from the root node.  That is registered as D1 in ``docs/OPEN_DEBT.md`` by user
decision, and ``KnownResidualTests`` pins the current behaviour rather than
pretending it is fixed.
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from utils import tree_manager  # noqa: E402
from utils.local_reference_binding import (  # noqa: E402
    REAL_NODE_ID_EMBEDDED_PATTERN,
    REAL_NODE_ID_PATTERN,
    REAL_NODE_ID_TOKENIZATION_PATTERN,
    _legacy_wide_shape_paths,
    _raw_response_id_hits,
    _real_id_shape_hits,
    _redact_real_ids,
    build_local_reference_context,
    call_local_reference_json,
)


class _Ctx:
    """Minimal stand-in exposing only what the raw gate reads."""

    def __init__(self, known=()):
        self.known_node_ids = tuple(known)


_MISSING = object()


def _gate(raw, known=(), parsed=_MISSING):
    entry = {"attempt": 1, "raw": raw}
    if parsed is not _MISSING:
        entry["json"] = parsed
    return _raw_response_id_hits([entry], _Ctx(known))


# --- the population the grammar must cover, byte-for-byte -------------------
# Every entry is a real identifier observed in a reachable tree, or a documented
# generator output.  Shapes, not samples: one per distinct shape found.
REAL_IDENTIFIERS = (
    "ROOT",                                 # root sentinel
    "L1_N05f93386",                         # base, all four levels observed
    "L2_N0109e16a",
    "L3_N0362ceea",
    "L4_N003ca61a",
    "L1_N40f698d4_BR_2c1402",               # one bridge segment
    "L2_N0109e16a_BR_448437",
    "L1_N6be998d8_BR_8df89b_BR_ff1d64",     # chained bridge segments
    "L1_N6be998d8_BR_8df89b_f5a4",          # bare 4-hex tail, frozen 353 tree
    "L1_N70e79b29_BR_6c0dca_7ecc",
    "ROOT_BR_8f69f8",                       # PITFALLS.md: 14b root fan-out
)

# Ordinary language the pre-RF21 grammar hard-rejected as leaked identifiers.
NATURAL_LANGUAGE = (
    "different_parent_L1_domains",          # the exact RF20 call-97 killer
    "L1_domains",
    "L2_policy",
    "L3_strategy",
    "L4_leaf",
    "L1_label",
    "L2_SCOPE",
    "L1_Nabc123",                           # 6 hex, not 8
    "L1_Nzzzzzzzz",                         # not hex at all
    "L5_N05f93386",                         # only L1-L4 exist
    "both nodes sit under different L1_domains entirely",
)


class GrammarCoverageTests(unittest.TestCase):
    """Narrowing must not drop a single real identifier shape."""

    def test_every_real_identifier_still_fullmatches(self):
        for value in REAL_IDENTIFIERS:
            with self.subTest(value=value):
                self.assertTrue(
                    REAL_NODE_ID_PATTERN.fullmatch(value),
                    f"{value!r} is a real identifier and must stay detected",
                )

    def test_every_real_identifier_is_found_inside_prose(self):
        # Detection may not be defeated by surrounding text or by gluing
        # ordinary characters onto an identifier.
        for value in REAL_IDENTIFIERS:
            for template in ("see {} here", "x{}y", "foo_{}_bar"):
                with self.subTest(value=value, template=template):
                    self.assertTrue(
                        REAL_NODE_ID_EMBEDDED_PATTERN.search(template.format(value)),
                        f"{value!r} must be detected inside {template!r}",
                    )

    def test_natural_language_is_no_longer_rejected(self):
        for value in NATURAL_LANGUAGE:
            with self.subTest(value=value):
                self.assertEqual(
                    [], _real_id_shape_hits([value]),
                    f"{value!r} is ordinary text, not an identifier",
                )

    def test_rf20_call_97_response_now_passes_the_raw_gate(self):
        # Reduced from the archived response that stopped the run.
        raw = json.dumps({
            "context_token": "CTX_097763563A0CA09A",
            "decisions": [{
                "relation": "related", "action": "reject_merge",
                "source_ref": "RIGHT", "target_ref": "LEFT",
                "evidence": {
                    "summary": "LEFT grows cargo volume; RIGHT extends services.",
                    "warnings": [
                        "different_parent_L1_domains",
                        "different_strategic_roles",
                    ],
                },
                "child_plan": [],
            }],
        }, ensure_ascii=False)
        self.assertEqual([], _gate(raw, known=("L3_Nc594b785", "L3_N607e22c3")))

    def test_a_real_identifier_in_that_same_field_still_stops_the_call(self):
        # Control: the fix must not make evidence a free channel for IDs.
        raw = json.dumps({
            "decisions": [{
                "evidence": {"summary": "merge into L3_Nc594b785 directly"},
            }],
        }, ensure_ascii=False)
        self.assertEqual(
            ["L3_Nc594b785"], _gate(raw, known=("L3_Nc594b785",)),
        )


class TokenizationWidthTests(unittest.TestCase):
    """Detection narrowed; tokenization deliberately did not.

    Narrowing both is the mistake this class exists to prevent.  Greedy
    tokenization is a *safety* property: it stops ``L2_ABC`` from being treated
    as displayed when the prompt only ever showed ``L2_ABC.DEF``.
    """

    def test_continuation_char_identifiers_still_tokenize_whole(self):
        for value in ("L2_ABC.DEF", "L2_X.L2_ABC", "L3_A:B-C"):
            with self.subTest(value=value):
                self.assertTrue(REAL_NODE_ID_TOKENIZATION_PATTERN.fullmatch(value))

    def test_those_same_values_are_not_treated_as_leaked_identifiers(self):
        for value in ("L2_ABC.DEF", "L2_X.L2_ABC", "L3_A:B-C"):
            with self.subTest(value=value):
                self.assertFalse(REAL_NODE_ID_PATTERN.fullmatch(value))
                self.assertEqual([], _real_id_shape_hits([value]))

    def test_short_candidate_is_not_inferred_from_a_longer_identifier(self):
        # The prefix-inference hole: only ``L2_ABC.DEF`` is displayed, so a
        # candidate ``L2_ABC`` must not bind.  Regression guard for the split.
        with self.assertRaises(ValueError):
            build_local_reference_context(
                task="prefix-forbidden",
                user_text="Node L2_ABC.DEF.",
                candidate_node_ids=("L2_ABC", "L2_ABC.DEF"),
                preferred_refs=(("SHORT", "L2_ABC"),),
            )

    def test_the_full_identifier_binds_normally(self):
        context = build_local_reference_context(
            task="long-token",
            user_text="Node L2_ABC.DEF.",
            candidate_node_ids=("L2_ABC", "L2_ABC.DEF"),
            preferred_refs=(("LONG", "L2_ABC.DEF"),),
        )
        self.assertEqual({"LONG": "L2_ABC.DEF"}, context.mapping)


class EscapedIdentifierTests(unittest.TestCase):
    """A JSON-escaped identifier must not slip the raw gate.

    Measured on the RF19 baseline: allowed through, because the gate read only
    the raw transport text where no identifier appears byte-for-byte.
    """

    def test_unicode_escaped_identifier_is_caught_via_the_decoded_payload(self):
        raw = '{"s": "L1_\\u004E762e6689"}'
        self.assertNotIn("L1_N762e6689", raw)          # not literal in raw
        self.assertIn("L1_N762e6689", json.loads(raw)["s"])   # but is after decode
        self.assertEqual(["L1_N762e6689"], _gate(raw))

    def test_escape_in_the_leading_character_is_also_caught(self):
        raw = '{"s": "\\u004C1_N762e6689"}'
        self.assertNotIn("L1_N762e6689", raw)
        self.assertEqual(["L1_N762e6689"], _gate(raw))

    def test_parsed_payload_is_scanned_even_when_raw_is_absent(self):
        hits = _raw_response_id_hits(
            [{"attempt": 1, "json": {"evidence": "L1_N762e6689"}}], _Ctx()
        )
        self.assertEqual(["L1_N762e6689"], hits)

    def test_identifiers_hidden_in_object_keys_are_caught(self):
        raw = json.dumps({"L1_N762e6689": "value"})
        self.assertEqual(["L1_N762e6689"], _gate(raw))

    def test_clean_escaped_text_is_not_a_false_positive(self):
        # Control: escapes alone must not trip the gate.
        raw = '{"s": "\\u004C1_domains and \\u0041BC"}'
        self.assertEqual([], _gate(raw))


class RedactionTests(unittest.TestCase):
    """A response rejected for leaking an ID must not archive that ID.

    RF20's bundle stored the full raw response.  Harmless there only because
    that response held no real identifier -- these tests drive one that does.
    """

    SECRET = "L3_Nc594b785"

    def _leaky_call(self):
        system = "sys contract text"
        context = build_local_reference_context(
            task="polish_tree_labels",
            user_text=f"LEFT={self.SECRET} RIGHT=L3_N607e22c3",
            candidate_node_ids=(self.SECRET, "L3_N607e22c3"),
            expected_count=1,
            contract_text=system,
        )
        leaked = json.dumps({
            "context_token": context.context_token,
            "decisions": [{
                "relation": "synonym", "action": "merge",
                "source_ref": "RIGHT", "target_ref": "LEFT",
                "new_label": None, "confidence": 0.9,
                "evidence": {"summary": f"merge into {self.SECRET} directly"},
                "child_plan": [],
            }],
        }, ensure_ascii=False)

        def transport(**kwargs):
            return {
                "ok": True, "status": 200, "raw": leaked,
                "json": json.loads(leaked), "attempts": 1, "latency_ms": 5,
                "profile": "p", "provider": "v", "model": "m",
                "attempt_history": [
                    {"attempt": 1, "raw": leaked, "status": 200, "latency_ms": 5}
                ],
            }

        return call_local_reference_json(
            transport=transport, profile="p", system=system, context=context,
            task="polish_tree_labels", expected_count=1, max_raw_attempts=1,
        )

    def test_the_call_is_still_rejected(self):
        result = self._leaky_call()
        self.assertFalse(result["ok"])
        self.assertEqual("REAL_NODE_ID_IN_RAW_RESPONSE", result["error"])

    def test_the_identifier_does_not_reach_the_archived_record(self):
        result = self._leaky_call()
        # ``context`` is our own binding table, not model output: it legitimately
        # holds the mapping.  Every other field carries untrusted text.
        archived = {k: v for k, v in result.items() if k != "context"}
        blob = json.dumps(archived, ensure_ascii=False, default=str)
        self.assertNotIn(self.SECRET, blob)

    def test_a_fingerprint_is_recorded_in_its_place(self):
        result = self._leaky_call()
        blob = json.dumps(
            {k: v for k, v in result.items() if k != "context"},
            ensure_ascii=False, default=str,
        )
        self.assertIn("<REDACTED_NODE_ID:", blob)

    def test_the_error_record_carries_only_a_count(self):
        result = self._leaky_call()
        for item in result["initial_binding_errors"]:
            self.assertNotIn(self.SECRET, json.dumps(item, ensure_ascii=False))
            self.assertIn("hit_count", item["context"])

    def test_redaction_is_stable_and_reversible_only_via_the_digest(self):
        once = _redact_real_ids(f"a {self.SECRET} b", [self.SECRET])
        twice = _redact_real_ids(f"a {self.SECRET} b", [self.SECRET])
        self.assertEqual(once, twice)
        self.assertNotIn(self.SECRET, once)

    def test_redaction_preserves_surrounding_text(self):
        redacted = _redact_real_ids(
            {"summary": f"merge into {self.SECRET} now"}, [self.SECRET]
        )
        self.assertTrue(redacted["summary"].startswith("merge into "))
        self.assertTrue(redacted["summary"].endswith(" now"))

    def test_a_clean_rejection_is_not_redacted(self):
        # Control: redaction is scoped to the ID-leak disposition only, so
        # ordinary failures keep a byte-faithful audit trail.
        text = "no identifier here at all"
        self.assertEqual(text, _redact_real_ids(text, ["L3_Nc594b785"]))


class TelemetryTests(unittest.TestCase):
    """Narrowing stays observable instead of silent.

    A value that only the pre-RF21 grammar matched is allowed through, but the
    field path is recorded so production can be reviewed rather than trusted.
    """

    def test_a_narrowed_away_value_is_reported_as_a_path(self):
        payload = {"decisions": [{"evidence": {"warnings": ["different_parent_L1_domains"]}}]}
        paths = _legacy_wide_shape_paths(payload)
        self.assertEqual(["decisions[0].evidence.warnings[0]:1"], paths)

    def test_the_report_never_contains_the_matched_text(self):
        payload = {"evidence": {"summary": "L1_domains and L2_policy"}}
        for entry in _legacy_wide_shape_paths(payload):
            self.assertNotIn("L1_domains", entry)
            self.assertNotIn("L2_policy", entry)

    def test_clean_payloads_report_nothing(self):
        payload = {"decisions": [{"evidence": {"summary": "no shapes here"}}]}
        self.assertEqual([], _legacy_wide_shape_paths(payload))

    def test_a_genuine_identifier_is_not_downgraded_to_telemetry(self):
        # Control: real IDs belong to the blocking tier, not this one.
        payload = {"evidence": {"summary": "L1_N05f93386"}}
        self.assertEqual([], _legacy_wide_shape_paths(payload))
        self.assertEqual(["L1_N05f93386"], _real_id_shape_hits(payload))


class LoadBoundaryTests(unittest.TestCase):
    """The grammar invariant lives at the entry boundary, not the generator.

    Base identifiers originate upstream in PolicyTreeBuilder, which this
    repository only consumes, so the shape contract can only be asserted where
    trees are loaded.  It is advisory by measurement: every real tree conforms,
    while the regression fixtures use synthetic identifiers.
    """

    def test_conforming_trees_report_no_violations(self):
        root = {"node_id": "ROOT", "children": [
            {"node_id": "L1_N05f93386", "children": [
                {"node_id": "L1_N40f698d4_BR_2c1402", "children": []},
            ]},
        ]}
        self.assertEqual([], tree_manager.node_id_grammar_violations(root))
        self.assertEqual([], tree_manager.TreeManager(root).node_id_grammar_violations)

    def test_the_legacy_four_hex_tail_is_accepted(self):
        root = {"node_id": "ROOT", "children": [
            {"node_id": "L1_N6be998d8_BR_8df89b_f5a4", "children": []},
        ]}
        self.assertEqual([], tree_manager.node_id_grammar_violations(root))

    def test_non_conforming_identifiers_are_reported(self):
        root = {"node_id": "A", "children": [{"node_id": "L1_A", "children": []}]}
        self.assertEqual(["A", "L1_A"], tree_manager.node_id_grammar_violations(root))

    def test_the_report_is_advisory_and_does_not_block_loading(self):
        # Synthetic fixtures must keep loading; a hard reject here would break
        # the existing suites without catching anything real.
        manager = tree_manager.TreeManager(
            {"node_id": "A", "children": [{"node_id": "L1_A", "children": []}]}
        )
        self.assertEqual(2, len(manager.index))
        self.assertEqual(["A", "L1_A"], manager.node_id_grammar_violations)


class KnownResidualTests(unittest.TestCase):
    """``ROOT`` collides with ordinary English and no grammar can fix that.

    Registered as D1 in ``docs/OPEN_DEBT.md`` by user decision on 2026-08-19:
    log it, do not migrate the ID namespace.  These tests pin the real current
    behaviour so the residual cannot be quietly forgotten or misreported.
    """

    def test_bare_root_remains_a_detected_identifier(self):
        self.assertTrue(REAL_NODE_ID_PATTERN.fullmatch("ROOT"))

    def test_english_prose_containing_root_is_still_rejected(self):
        for text in ("ROOT CAUSE analysis shows", "the ROOT node has 12 children"):
            with self.subTest(text=text):
                self.assertEqual(["ROOT"], _real_id_shape_hits([text]))

    def test_lowercase_root_is_unaffected(self):
        # The collision is confined to the uppercase spelling.
        self.assertEqual([], _real_id_shape_hits(["the root cause of the issue"]))
