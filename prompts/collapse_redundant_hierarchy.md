# Step 14a · Vertical semantic decision

You are reviewing exactly one displayed parent-child pair. Similarity scores only select a candidate; they never authorize a destructive operation.

## Semantic contract

Return exactly one JSON object with exactly one item in `decisions`:

```json
{
  "decisions": [
    {
      "relation": "exact_duplicate|synonym|broader_narrower|related|complementary|means_goal|carrier_outcome|misplaced|uncertain",
      "action": "merge|rename|split_reparent|keep|reject_merge|uncertain",
      "source_id": "node id",
      "target_id": "node id",
      "new_label": null,
      "confidence": 0.0,
      "evidence": {
        "summary": "specific semantic evidence",
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

Every shown field is required. Use only the listed enum values. Do not invent IDs or omit fields.

Each `child_plan` item must contain all fields below:

```json
{
  "child_id": "direct child id",
  "disposition": "move|keep|retain_under_source",
  "target_parent_id": "existing displayed node id",
  "relation": "exact_duplicate|synonym|broader_narrower|related|complementary|means_goal|carrier_outcome|misplaced|uncertain",
  "same_domain": true,
  "evidence": "why this child belongs under that target"
}
```

## Hard rules

- `relation` describes meaning; `action` describes the requested operation. Never use free-text reasoning as an implicit merge approval.
- `merge` is allowed only for `exact_duplicate` or genuine `synonym`. Shared words, high Jaccard/Cosine, topical relation, broader/narrower scope, means-goal, carrier-outcome, or a more action-oriented label are not synonyms.
- For `merge`, `source_id` is removed and `target_id` survives. They must be the displayed pair.
- If the source has direct membership, set `target_represents_all_source_members=true` only when the target label represents every source record, and give a concrete `membership_basis`. Otherwise choose `reject_merge`, `keep`, or `uncertain`.
- If a merged source has children, `child_plan` must cover every direct child exactly once and prove each destination is in-domain.
- Merging a parent into its child is eligible only on an exact single-child chain and still requires a complete child plan to the displayed grandparent implied by the pair context. If that proof is absent, reject it.
- `split_reparent` preserves its source umbrella: set `source_id=target_id` to one displayed node, cover every direct child exactly once, keep compatible children under the source, and move only proven out-of-place children to the other displayed node.
- `rename` also uses `source_id=target_id`, has an empty child plan, and supplies `new_label`.
- For `keep`, `reject_merge`, or `uncertain`, reference both displayed nodes and use an empty child plan. `action=uncertain` requires `relation=uncertain`.
- Cross-L1 changes are forbidden; `cross_l1_authorized` must be false.

## Safe examples

Exact duplicate with no direct membership or children:

```json
{"decisions":[{"relation":"exact_duplicate","action":"merge","source_id":"C","target_id":"P","new_label":null,"confidence":0.99,"evidence":{"summary":"The labels denote the same policy action and C has no direct membership or children.","warnings":[],"target_represents_all_source_members":false,"membership_basis":"","pure_structural_redundancy":false,"cross_l1_authorized":false},"child_plan":[]}]}
```

Related but non-equivalent pair:

```json
{"decisions":[{"relation":"means_goal","action":"reject_merge","source_id":"C","target_id":"P","new_label":null,"confidence":0.96,"evidence":{"summary":"C is a means and P is the goal; deleting either would erase a distinct role.","warnings":["not_synonymous"],"target_represents_all_source_members":false,"membership_basis":"","pure_structural_redundancy":false,"cross_l1_authorized":false},"child_plan":[]}]}
```
