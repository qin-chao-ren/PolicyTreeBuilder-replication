# Step 14c · Pairwise semantic polishing

You are reviewing exactly two displayed nodes, their displayed parents, direct children, direct membership counts, and record-title examples. Similarity thresholds only generate the pair; they are not merge evidence.

## Output contract

Return exactly one decision in JSON:

```json
{
  "decisions": [
    {
      "relation": "exact_duplicate|synonym|broader_narrower|related|complementary|means_goal|carrier_outcome|misplaced|uncertain",
      "action": "merge|move|rename|split_reparent|keep|reject_merge|uncertain",
      "source_id": "displayed node id",
      "target_id": "displayed node or parent id",
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

Every field is required. A child plan item has this exact shape:

```json
{"child_id":"direct child id","disposition":"move|keep|retain_under_source","target_parent_id":"displayed node or parent id","relation":"exact_duplicate|synonym|broader_narrower|related|complementary|means_goal|carrier_outcome|misplaced|uncertain","same_domain":true,"evidence":"specific destination evidence"}
```

## Hard rules

- Keep semantic relation separate from action. `merge` is allowed only for `exact_duplicate` or genuine `synonym`.
- `source_id` is deleted by `merge`; `target_id` survives. Both must be the displayed pair. Shared topic, co-occurrence, high similarity, broader/narrower scope, means-goal, carrier-outcome, or complementary roles require `reject_merge` or `keep`.
- If the source has direct membership, set `target_represents_all_source_members=true` only after checking that the target label represents every shown source record; explain that proof in `membership_basis`. Otherwise do not merge.
- If the source has children, a merge must cover every direct child exactly once and prove that moving it under the target preserves its domain.
- `move` reparents one displayed node to one displayed parent. Use it only for `relation=misplaced`. Preserve all children under the moved source with a complete `retain_under_source` plan.
- `split_reparent` preserves one displayed umbrella: set `source_id=target_id`, cover all its direct children, keep compatible children under it, and move only the mismatched children to a displayed node or displayed parent with explicit proof.
- `rename` uses `source_id=target_id`, requires `new_label`, and has no child plan.
- `keep`, `reject_merge`, and `uncertain` reference both displayed nodes and have no child plan. `action=uncertain` requires `relation=uncertain`.
- Never cross an L1 boundary. Never invent an ID.

## Examples

Safe exact duplicate:

```json
{"decisions":[{"relation":"exact_duplicate","action":"merge","source_id":"B","target_id":"A","new_label":null,"confidence":0.99,"evidence":{"summary":"A and B have identical scope and B has no direct membership or children.","warnings":[],"target_represents_all_source_members":false,"membership_basis":"","pure_structural_redundancy":false,"cross_l1_authorized":false},"child_plan":[]}]}
```

Topically related but distinct roles:

```json
{"decisions":[{"relation":"complementary","action":"reject_merge","source_id":"B","target_id":"A","new_label":null,"confidence":0.95,"evidence":{"summary":"A supplies infrastructure while B governs service operation; both roles must remain visible.","warnings":["different_policy_roles"],"target_represents_all_source_members":false,"membership_basis":"","pure_structural_redundancy":false,"cross_l1_authorized":false},"child_plan":[]}]}
```
