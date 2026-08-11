# Step 14c · Pairwise semantic polishing

You are reviewing exactly two displayed nodes, their displayed parents, direct children, direct membership counts, and record-title examples. Similarity thresholds only generate the pair; they are not merge evidence.

## Output contract

Return exactly one decision in JSON:

```json
{
  "context_token": "CTX_0123456789ABCDEF",
  "decisions": [
    {
      "relation": "exact_duplicate|synonym|broader_narrower|related|complementary|means_goal|carrier_outcome|misplaced|uncertain",
      "action": "merge|move|rename|split_reparent|keep|reject_merge|uncertain",
      "source_ref": "LEFT|RIGHT|N0|...",
      "target_ref": "displayed node or parent ref",
      "new_label": null,
      "confidence": 0.0,
      "evidence": {
        "summary": "specific label, path, and record evidence",
        "warnings": [],
        "target_represents_all_source_members": false,
        "membership_basis": "",
        "pure_structural_redundancy": false,
        "cross_l1_authorized": false
      },
      "child_plan": []
    }
  ]
}
```

Every field is required. Echo the supplied `context_token` exactly and use only refs listed in the call context. The legacy real-ID JSON fields "source_id", "target_id", "child_id", and "target_parent_id" are forbidden. A child plan item has this exact shape:

```json
{"child_ref":"N0","disposition":"move|keep|retain_under_source","target_parent_ref":"displayed node or parent ref","relation":"exact_duplicate|synonym|broader_narrower|related|complementary|means_goal|carrier_outcome|misplaced|uncertain","same_domain":true,"evidence":"specific destination evidence"}
```

## Hard rules

- Keep semantic relation separate from action. `merge` is allowed only for `exact_duplicate` or genuine `synonym`.
- `source_ref` is deleted by `merge`; `target_ref` survives. Both must be the displayed pair. Shared topic, co-occurrence, high similarity, broader/narrower scope, means-goal, carrier-outcome, or complementary roles require `reject_merge` or `keep`.
- If the source has direct membership, set `target_represents_all_source_members=true` only after checking that the target label represents every shown source record; explain that proof in `membership_basis`. Otherwise do not merge.
- If the source has children, a merge must cover every direct child exactly once and prove that moving it under the target preserves its domain.
- `move` reparents one displayed node to one displayed parent. Use it only for `relation=misplaced`. Preserve all children under the moved source with a complete `retain_under_source` plan.
- `split_reparent` preserves one displayed umbrella: set `source_ref=target_ref`, cover all its direct children, keep compatible children under it, and move only the mismatched children to a displayed node or displayed parent with explicit proof.
- `rename` uses `source_ref=target_ref`, requires `new_label`, and has no child plan.
- `keep`, `reject_merge`, and `uncertain` reference both displayed nodes and have no child plan. `action=uncertain` requires `relation=uncertain`.
- Never cross an L1 boundary. Never invent an ID.

## Examples

Safe exact duplicate:

```json
{"context_token":"CTX_0123456789ABCDEF","decisions":[{"relation":"exact_duplicate","action":"merge","source_ref":"RIGHT","target_ref":"LEFT","new_label":null,"confidence":0.99,"evidence":{"summary":"LEFT and RIGHT have identical scope and RIGHT has no direct membership or children.","warnings":[],"target_represents_all_source_members":false,"membership_basis":"","pure_structural_redundancy":false,"cross_l1_authorized":false},"child_plan":[]}]}
```

Topically related but distinct roles:

```json
{"context_token":"CTX_0123456789ABCDEF","decisions":[{"relation":"complementary","action":"reject_merge","source_ref":"RIGHT","target_ref":"LEFT","new_label":null,"confidence":0.95,"evidence":{"summary":"LEFT supplies infrastructure while RIGHT governs service operation; both roles must remain visible.","warnings":["different_policy_roles"],"target_represents_all_source_members":false,"membership_basis":"","pure_structural_redundancy":false,"cross_l1_authorized":false},"child_plan":[]}]}
```
