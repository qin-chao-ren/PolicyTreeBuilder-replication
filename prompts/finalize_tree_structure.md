# Step 14d · L1-bounded semantic finalization

You are auditing one displayed L1 subtree batch. The finalizer has two independent publication gates: structural E0 and the semantic contract below. A structurally valid operation is still rejected when its semantic proof is incomplete.

## Output contract

Return JSON only. Return `{"decisions":[]}` when no safe operation is justified.

```json
{
  "decisions": [
    {
      "relation": "exact_duplicate|synonym|broader_narrower|related|complementary|means_goal|carrier_outcome|misplaced|uncertain",
      "action": "merge|move|rename|flatten|split_reparent|keep|reject_merge|uncertain",
      "source_id": "displayed non-L1 node id",
      "target_id": "displayed node id",
      "new_label": null,
      "confidence": 0.0,
      "evidence": {
        "summary": "specific path and membership evidence",
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

Every field is required. Each child plan item must contain:

```json
{"child_id":"direct child id","disposition":"move|keep|retain_under_source","target_parent_id":"displayed node id","relation":"exact_duplicate|synonym|broader_narrower|related|complementary|means_goal|carrier_outcome|misplaced|uncertain","same_domain":true,"evidence":"specific destination evidence"}
```

## Publication-safe rules

- Do not modify an L1 node and do not cross the displayed L1 boundary.
- Relation and action are separate. Only `exact_duplicate` and genuine `synonym` may authorize destructive `merge`.
- A repeated word, a repetitive path marker, a single-child marker, topical relatedness, broader/narrower scope, means-goal, carrier-outcome, or complementary function does not authorize merge.
- `source_id` is removed by merge and `target_id` survives. If the displayed source has direct membership above zero, merge only when every source record is represented by the target label; set `target_represents_all_source_members=true` and give a concrete `membership_basis`. The batch does not show record titles, so do not claim this proof merely from labels or counts.
- A merge with children needs a complete per-child move plan to the target. A parent-into-child merge additionally requires an exact single-child chain and a complete plan to the source parent.
- `flatten` is distinct from merge. It requires `target_id` equal to the source parent, direct membership exactly zero, `pure_structural_redundancy=true`, and a complete per-child move plan proving the destination domain.
- `move` uses `relation=misplaced`; preserve every source child with a complete `retain_under_source` plan.
- `split_reparent` keeps the umbrella with `source_id=target_id`, covers every direct child once, and moves only individually proven mismatches.
- `rename` uses `source_id=target_id`, requires `new_label`, and has no child plan.
- `keep`, `reject_merge`, and `uncertain` never mutate and have an empty child plan. `action=uncertain` requires `relation=uncertain`.
- Do not invent IDs, silently omit a child, or use free-text reasoning as operation authority. Unknown or insufficient evidence must fail closed.

## Safe examples

Non-equivalent single-child chain:

```json
{"decisions":[{"relation":"broader_narrower","action":"reject_merge","source_id":"N3","target_id":"N2","new_label":null,"confidence":0.97,"evidence":{"summary":"N3 adds a narrower policy domain, so the single-child shape is not semantic redundancy.","warnings":["scope_differs"],"target_represents_all_source_members":false,"membership_basis":"","pure_structural_redundancy":false,"cross_l1_authorized":false},"child_plan":[]}]}
```

Safe empty-wrapper flatten with all children proven:

```json
{"decisions":[{"relation":"broader_narrower","action":"flatten","source_id":"W","target_id":"P","new_label":null,"confidence":0.96,"evidence":{"summary":"W has zero direct membership and adds no domain beyond P; both children remain in P's domain.","warnings":[],"target_represents_all_source_members":false,"membership_basis":"","pure_structural_redundancy":true,"cross_l1_authorized":false},"child_plan":[{"child_id":"C1","disposition":"move","target_parent_id":"P","relation":"broader_narrower","same_domain":true,"evidence":"C1 is a direct subtype of P."},{"child_id":"C2","disposition":"move","target_parent_id":"P","relation":"broader_narrower","same_domain":true,"evidence":"C2 is a direct subtype of P."}]}]}
```
