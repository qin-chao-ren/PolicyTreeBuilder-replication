# Step 14b · Structure balancing semantic decision

You are reviewing one current candidate node, its parent, and its direct children. The scene may be a level gap, excessive fanout, or a deep branch. Structural pressure never authorizes deleting a semantic category.

## Output contract

Return JSON only:

```json
{
  "context_token": "CTX_0123456789ABCDEF",
  "decisions": [
    {
      "relation": "exact_duplicate|synonym|broader_narrower|related|complementary|means_goal|carrier_outcome|misplaced|uncertain",
      "action": "create_bridge|move|flatten|split_reparent|keep|reject_merge|uncertain",
      "source_ref": "CANDIDATE|PARENT|N0|...",
      "target_ref": "CANDIDATE|PARENT|N0|...|NEW_BRIDGE",
      "new_label": null,
      "confidence": 0.0,
      "evidence": {
        "summary": "specific semantic and membership evidence",
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

Every field is required. Echo the supplied `context_token` exactly and use only refs listed in the call context. The legacy real-ID JSON fields "source_id", "target_id", "child_id", and "target_parent_id" are forbidden. Each child plan item must contain:

```json
{"child_ref":"N0","disposition":"move|keep|retain_under_source","target_parent_ref":"CANDIDATE|PARENT|N0|...|NEW_BRIDGE","relation":"exact_duplicate|synonym|broader_narrower|related|complementary|means_goal|carrier_outcome|misplaced|uncertain","same_domain":true,"evidence":"specific destination evidence"}
```

Several `create_bridge` decisions may be returned for a fanout scene. Otherwise return only operations justified by the displayed context. If no mutation is safe, return one non-mutating decision for the current candidate.

## Stage rules

- `merge` is not available in this stage. Use `flatten` only for a genuinely empty structural wrapper; use `split_reparent` when only some children are misplaced.
- `create_bridge`: `source_ref` is `CANDIDATE`; `target_ref` and every selected child's `target_parent_ref` must literally be `NEW_BRIDGE`; `new_label` is required. List only direct children that share a coherent broader domain. The program deterministically materializes the bridge ID after validation.
- `flatten`: `source_ref` is `CANDIDATE` and `target_ref` is `PARENT`. It is allowed only when direct membership is exactly zero, the source is pure structural redundancy, and `child_plan` covers every direct child with `disposition=move`, the displayed parent as target, and explicit same-domain evidence.
- `split_reparent`: preserve the candidate with `source_ref=target_ref=CANDIDATE`. Cover every direct child exactly once. A kept child targets the source; a moved child may target only a node visible in the current context and needs same-domain evidence.
- `move`: the source must be a displayed direct child and the target must be the displayed parent of the current candidate. If the moved source has children, list every child with `retain_under_source` and target the source itself.
- `keep`, `reject_merge`, and `uncertain` use `source_ref=target_ref=CANDIDATE` and an empty child plan.
- Cross-L1 changes are forbidden. Use `relation=uncertain, action=uncertain` when destination evidence is insufficient.
- Similar labels, high vector similarity, a depth limit, or fanout greater than seven are candidate signals only. Never regroup heterogeneous policy domains merely to improve shape.

## Examples

Safe bridge over two coherent direct children:

```json
{"context_token":"CTX_0123456789ABCDEF","decisions":[{"relation":"broader_narrower","action":"create_bridge","source_ref":"CANDIDATE","target_ref":"NEW_BRIDGE","new_label":"Financing support","confidence":0.94,"evidence":{"summary":"N0 and N1 are distinct financing instruments under one coherent financing-support domain.","warnings":[],"target_represents_all_source_members":false,"membership_basis":"","pure_structural_redundancy":false,"cross_l1_authorized":false},"child_plan":[{"child_ref":"N0","disposition":"move","target_parent_ref":"NEW_BRIDGE","relation":"broader_narrower","same_domain":true,"evidence":"N0 is a financing-support instrument."},{"child_ref":"N1","disposition":"move","target_parent_ref":"NEW_BRIDGE","relation":"broader_narrower","same_domain":true,"evidence":"N1 is a financing-support instrument."}]}]}
```

Unsafe flatten because the wrapper carries records:

```json
{"context_token":"CTX_0123456789ABCDEF","decisions":[{"relation":"broader_narrower","action":"keep","source_ref":"CANDIDATE","target_ref":"CANDIDATE","new_label":null,"confidence":0.98,"evidence":{"summary":"CANDIDATE has direct membership, so it is not an empty structural wrapper.","warnings":["direct_membership_present"],"target_represents_all_source_members":false,"membership_basis":"","pure_structural_redundancy":false,"cross_l1_authorized":false},"child_plan":[]}]}
```
