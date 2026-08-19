from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from utils.semantic_contract import resolve_membership_counts  # noqa: E402
from utils.tree_integrity import (  # noqa: E402
    STAGE_LINEAGE_LEDGERS,
    LineageError,
    cross_validate_ledger_against_operations,
    load_stage_lineage_ledger,
    stages_present_in_operations,
)

VC = "vertical_collapse_trace.json"
SB = "structure_balancing_trace.json"


class StageDetectionTests(unittest.TestCase):
    """Which ledgers are required is read off the operations log, not the disk."""

    def test_maps_both_balancing_step_names_to_one_ledger(self):
        records = [
            {"step": "structure_balancing_depth"},
            {"step": "structure_balancing_fanout"},
        ]
        self.assertEqual(stages_present_in_operations(records), {"structure_balancing"})

    def test_detects_vertical_collapse(self):
        self.assertEqual(
            stages_present_in_operations([{"step": "vertical_collapse"}]),
            {"vertical_collapse"},
        )

    def test_ignores_unknown_and_missing_steps(self):
        records = [{"step": "label_polishing"}, {"step": None}, {}, "not a mapping"]
        self.assertEqual(stages_present_in_operations(records), set())

    def test_ledger_table_covers_both_upstream_stages(self):
        self.assertEqual(
            {stage for stage, _ in STAGE_LINEAGE_LEDGERS},
            {"vertical_collapse", "structure_balancing"},
        )


class LedgerLoadTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.outdir = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def write(self, name, payload):
        (self.outdir / name).write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8"
        )

    def test_missing_required_ledger_is_rejected(self):
        with self.assertRaises(LineageError) as ctx:
            load_stage_lineage_ledger(
                self.outdir, "vertical_collapse", VC, required=True
            )
        self.assertIn("vertical_collapse", str(ctx.exception))
        self.assertIn(VC, str(ctx.exception))

    def test_missing_optional_ledger_is_empty(self):
        """A stage that never ran legitimately has no ledger."""
        self.assertEqual(
            load_stage_lineage_ledger(
                self.outdir, "vertical_collapse", VC, required=False
            ),
            {},
        )

    def test_present_but_empty_ledger_is_accepted_when_required(self):
        """The control for the case above: ran, redirected nothing. RF11's 14b."""
        self.write(SB, {})
        self.assertEqual(
            load_stage_lineage_ledger(
                self.outdir, "structure_balancing", SB, required=True
            ),
            {},
        )

    def test_populated_ledger_round_trips(self):
        self.write(VC, {"L4_Naf94a663": "L3_Nc3d125b3"})
        self.assertEqual(
            load_stage_lineage_ledger(self.outdir, "vertical_collapse", VC, required=True),
            {"L4_Naf94a663": "L3_Nc3d125b3"},
        )

    def test_non_object_ledger_is_rejected(self):
        self.write(VC, ["not", "an", "object"])
        with self.assertRaises(LineageError):
            load_stage_lineage_ledger(self.outdir, "vertical_collapse", VC, required=True)

    def test_unreadable_ledger_is_rejected(self):
        (self.outdir / VC).write_text("{ this is not json", encoding="utf-8")
        with self.assertRaises(LineageError):
            load_stage_lineage_ledger(self.outdir, "vertical_collapse", VC, required=True)

    def test_empty_target_is_rejected(self):
        self.write(VC, {"L4_Naf94a663": "   "})
        with self.assertRaises(LineageError):
            load_stage_lineage_ledger(self.outdir, "vertical_collapse", VC, required=True)


class CrossValidationTests(unittest.TestCase):
    """Ledger vs operations log: disagreement is a fault, absence is not."""

    def test_agreeing_sources_report_agreement(self):
        records = [
            {
                "step": "vertical_collapse",
                "status": "applied",
                "source_id": "L4_Naf94a663",
                "lineage_target": "L3_Nc3d125b3",
            }
        ]
        verdict = cross_validate_ledger_against_operations(
            {"L4_Naf94a663": "L3_Nc3d125b3"}, records, "vertical_collapse"
        )
        self.assertTrue(verdict["comparable"])
        self.assertTrue(verdict["agrees"])

    def test_disagreement_is_reported(self):
        records = [
            {
                "step": "vertical_collapse",
                "status": "applied",
                "source_id": "L4_Naf94a663",
                "lineage_target": "L3_Nc3d125b3",
            }
        ]
        verdict = cross_validate_ledger_against_operations(
            {"L4_Naf94a663": "L3_NDIFFERENT"}, records, "vertical_collapse"
        )
        self.assertTrue(verdict["comparable"])
        self.assertFalse(verdict["agrees"])
        self.assertIn("L4_Naf94a663", verdict["disagreements"])

    def test_pre_contract_schema_is_not_comparable_not_a_mismatch(self):
        """C13F's log has applied records with no lineage_target at all.

        Treating that as a mismatch would hard-reject every historical archive,
        so it must come back not-comparable rather than disagreeing.
        """
        records = [
            {
                "step": "vertical_collapse",
                "status": "applied",
                "source_id": "L2_N4f9516d8",
                "target_id": "L3_N55664f5d",
            }
        ]
        verdict = cross_validate_ledger_against_operations(
            {"L2_N4f9516d8": "L3_N55664f5d"}, records, "vertical_collapse"
        )
        self.assertFalse(verdict["comparable"])
        self.assertEqual(verdict["applied_records"], 1)
        self.assertNotIn("agrees", verdict)

    def test_rejected_records_are_not_treated_as_redirects(self):
        records = [
            {
                "step": "vertical_collapse",
                "status": "rejected",
                "source_id": "L4_Naf94a663",
                "lineage_target": "L3_Nc3d125b3",
            }
        ]
        verdict = cross_validate_ledger_against_operations({}, records, "vertical_collapse")
        self.assertFalse(verdict["comparable"])
        self.assertEqual(verdict["applied_records"], 0)

    def test_balancing_verdict_reads_both_step_names(self):
        records = [
            {
                "step": "structure_balancing_depth",
                "status": "applied",
                "source_id": "L3_Nd290d812",
                "lineage_target": "L2_Nbec5a9c5",
            }
        ]
        verdict = cross_validate_ledger_against_operations(
            {"L3_Nd290d812": "L2_Nbec5a9c5"}, records, "structure_balancing"
        )
        self.assertTrue(verdict["comparable"])
        self.assertTrue(verdict["agrees"])


class MembershipCountTests(unittest.TestCase):
    """D7: an unresolvable count must not be dropped silently."""

    def test_resolves_through_lineage(self):
        self.assertEqual(
            resolve_membership_counts(
                {"L4_Naf94a663": 1}, {"L4_Naf94a663": "L3_Nc3d125b3"}, ["L3_Nc3d125b3"]
            ),
            {"L3_Nc3d125b3": 1},
        )

    def test_aggregates_multiple_sources_onto_one_target(self):
        counts = {"L4_Naf94a663": 2, "L4_Na8ba2f4a": 3, "L3_Nc3d125b3": 1}
        lineage = {"L4_Naf94a663": "L3_Nc3d125b3", "L4_Na8ba2f4a": "L3_Nc3d125b3"}
        self.assertEqual(
            resolve_membership_counts(counts, lineage, ["L3_Nc3d125b3"]),
            {"L3_Nc3d125b3": 6},
        )

    def test_unresolvable_count_raises_instead_of_dropping(self):
        with self.assertRaises(LineageError) as ctx:
            resolve_membership_counts({"L4_Naf94a663": 1}, {}, ["L3_Nc3d125b3"])
        message = str(ctx.exception)
        self.assertIn("not live", message)
        self.assertIn("L4_Naf94a663", message)

    def test_error_names_the_resolved_target_when_lineage_points_somewhere_dead(self):
        with self.assertRaises(LineageError) as ctx:
            resolve_membership_counts(
                {"L4_Naf94a663": 1}, {"L4_Naf94a663": "L3_NDEADBEEF"}, ["L3_Nc3d125b3"]
            )
        self.assertIn("L3_NDEADBEEF", str(ctx.exception))

    def test_live_node_with_no_lineage_entry_is_fine(self):
        self.assertEqual(
            resolve_membership_counts({"L3_Nc3d125b3": 4}, {}, ["L3_Nc3d125b3"]),
            {"L3_Nc3d125b3": 4},
        )

    def test_empty_counts_stay_empty(self):
        self.assertEqual(resolve_membership_counts({}, {}, ["L3_Nc3d125b3"]), {})


class GateLooseningRegressionTests(unittest.TestCase):
    """Why D7 mattered: a dropped count made two gates *looser*, not stricter.

    These pin the arithmetic the gates depend on.  If a future change reverts to
    dropping, the count seen by the gate goes to 0 and both of these flip.
    """

    def test_dropped_count_would_have_bypassed_merge_representation_gate(self):
        # SOURCE_MEMBERSHIP_NOT_REPRESENTED only fires when count > 0.
        with self.assertRaises(LineageError):
            resolve_membership_counts({"L4_Naf94a663": 1}, {}, ["L3_Nc3d125b3"])
        # With complete lineage the count survives and the gate can still fire.
        resolved = resolve_membership_counts(
            {"L4_Naf94a663": 1}, {"L4_Naf94a663": "L3_Nc3d125b3"}, ["L3_Nc3d125b3"]
        )
        self.assertGreater(resolved["L3_Nc3d125b3"], 0)

    def test_dropped_count_would_have_faked_flatten_precondition(self):
        # FLATTEN_SOURCE_HAS_MEMBERSHIP requires count == 0; a drop fakes it.
        with self.assertRaises(LineageError):
            resolve_membership_counts({"L4_Naf94a663": 5}, {}, ["L3_Nc3d125b3"])


if __name__ == "__main__":
    unittest.main()


def _load_balance_stage():
    """Import 14b with numpy/pandas stubbed out.

    The stage pulls in the scientific stack at module scope but the lineage
    loader touches none of it; stubbing keeps this test runnable in the same
    stdlib-only environment as the rest of the suite.
    """
    import types

    created = []
    for name in ("numpy", "pandas"):
        if name not in sys.modules:
            module = types.ModuleType(name)
            module.ndarray = object
            sys.modules[name] = module
            created.append(name)
    try:
        import balance_tree_structure  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - environment guard
        raise unittest.SkipTest(f"14b not importable: {exc}") from exc
    return balance_tree_structure, created


class BalanceStagePriorLineageTests(unittest.TestCase):
    """14b had the same fail-open as 14c; the card only named 14c.

    Without this, reverting the 14b fix left the whole suite green -- the
    strictness probe caught that blind spot, so it gets pinned here.
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.outdir = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        self.module, _ = _load_balance_stage()

    def _call(self):
        """Invoke the loader with only the two attributes it reads."""
        stage = self.module.ShapingProcess.__new__(self.module.ShapingProcess)
        stage.env = type("Env", (), {"outdir": self.outdir})()
        stage.ops_log = self.outdir / "tree_refinement_operations.jsonl"
        return self.module.ShapingProcess._load_prior_lineage(stage)

    def write_ops(self, records):
        (self.outdir / "tree_refinement_operations.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records),
            encoding="utf-8",
        )

    def test_rejects_missing_ledger_when_14a_ran(self):
        self.write_ops([{"step": "vertical_collapse", "status": "applied"}])
        with self.assertRaises(LineageError) as ctx:
            self._call()
        self.assertIn("vertical_collapse", str(ctx.exception))

    def test_accepts_missing_ledger_when_14a_never_ran(self):
        self.write_ops([{"step": "label_polishing", "status": "applied"}])
        self.assertEqual(self._call(), {})

    def test_accepts_missing_ops_log_entirely(self):
        """A fresh run starts with no operations log at all."""
        self.assertEqual(self._call(), {})

    def test_loads_ledger_when_present(self):
        self.write_ops([{"step": "vertical_collapse", "status": "applied"}])
        (self.outdir / VC).write_text(
            json.dumps({"L4_Naf94a663": "L3_Nc3d125b3"}), encoding="utf-8"
        )
        self.assertEqual(self._call(), {"L4_Naf94a663": "L3_Nc3d125b3"})
