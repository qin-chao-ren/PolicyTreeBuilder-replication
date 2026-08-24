# Known Limitations

This document records the known limitations of this replication package as of 2026-08-24.
It is written for reviewers and for anyone porting this code into a successor system.

All statements below are backed by offline measurements on committed artifacts. No claim here
is an estimate. Where a limitation was investigated and a hypothesis was ruled out, the
refutation is stated as plainly as the limitation itself.

## Scope of the engineering status

The refinement pipeline (stages 14a–14d plus two independent E0 validations) has been executed
end-to-end once, on 2026-08-22, producing a 238-node tree that passed every gate that exists.

**That means "the pipeline runs", not "the tree is semantically clean".** The distinction is
load-bearing and is the subject of the first two limitations below.

## 1. Exact-duplicate zero does not imply semantic-duplicate zero

The E0 validator reports `sibling_duplicate_groups: 0` for the finished tree. The precise
meaning of that field is **"no two sibling labels are byte-identical after NFKC normalization
and whitespace folding"** (`scripts/utils/tree_integrity.py`, `canonical_exact_label()`).

It does not mean the tree contains no semantic duplicates. Two families of near-synonymous
nodes are known to remain in the finished 238-node tree.

**Family A — never reached the model** (three nodes, all siblings under one parent):

- 构建覆盖全球的货运航线网络
- 构建全球航空货运网络
- 构建优质航线网络

**Family B — reached the model, which proposed merging, but the merge was not executed**
(three nodes, two of them siblings):

- 提升口岸通关便利化水平
- 提升口岸通关便利化
- 提高口岸通关便利化水平

These two families have different causes, described in limitations 2 and 4. They are listed
together here because their effect on the published tree is identical: a reader auditing the
tree for semantic redundancy will find them.

**Therefore this package does not claim that all semantic duplicates have been removed.**

## 2. The 14c sibling-merge prefilter can withhold true duplicates from the model

Before asking the model whether two sibling nodes should merge, stage 14c applies a similarity
prefilter (`scripts/polish_tree_labels.py`, `_quick_check_similarity`). A pair that clears
neither the lexical nor the vector channel is skipped without ever being shown to the model.

Current thresholds:

| Comparison | Jaccard | Cosine |
|---|---|---|
| Sibling merge | 0.75 | 0.85 |
| Cross-parent unify | 0.80 | 0.90 |

The known escaping pair — 构建全球国际航线网络 vs 构建覆盖全球的货运航线网络, same parent —
scores Jaccard **0.5714** and bare cosine **0.832628**. Both channels reject it, so the model
never evaluates it.

### A plausible cause was tested and ruled out

Node vectors are computed as the centroid of the member policy-text embeddings
(`EmbeddingHelper.get_centroid`); the node label itself never enters the vector. This invites
the hypothesis that the prefilter underscores true duplicates because it compares long policy
prose rather than the labels being deduplicated.

**For this pair, that hypothesis is false.** The embedded text for both members is
byte-identical to the node label:

| Member | Node label | Embedded text | Identical |
|---|---|---|---|
| `DF0542_00016_01` | 构建全球国际航线网络 | 构建全球国际航线网络 | yes |
| `DF0403_00012_02` | 构建覆盖全球的货运航线网络 | 构建覆盖全球的货运航线网络 | yes |

The centroid already *is* the label vector for these nodes, so switching the representation to
"label vector" would leave 0.832628 unchanged. The corpus stores pre-segmented action phrases,
not whole documents.

Across the 246-node input tree: 61 nodes have member text identical to the label, 142 have
`label [SEP] section-heading`, 25 have multiple members, 18 have none. The label is always
present in the vector; the only variation is a trailing section heading on some nodes.

### Threshold sensitivity

Measured on the actual 14c input tree (output of 14b, 246 nodes, 366 sibling pairs, of which
318 have computable vectors on both sides):

| `thr_cos` | Pairs reaching the model | Change |
|---|---|---|
| 0.85 (current) | 77 | — |
| 0.84 | 86 | +9 |
| 0.83 | 95 | +18 |
| 0.82 | 103 | +26 |
| 0.80 | 116 | +39 |

Of the 18 pairs admitted by lowering to 0.83, roughly 3 look like genuine near-synonyms
(including the known pair) and roughly 15 are topically related but describe different actions
(e.g. 打造高效便利的营商环境 ‖ 提升口岸通关效率) or merely share a section heading
(强化航空货运枢纽建设项目用地支持 ‖ 加强航空货运人才队伍建设, Jaccard 0.33).

**The threshold is deliberately left unchanged.** 14c is intended as a conservative backstop,
not as a substitute for upstream semantic consolidation, and the measured signal-to-noise of
lowering it is about 1 useful pair per 5 additional model calls.

### An observation that is not a causal claim

Among single-member sibling pairs, those whose members share the same section heading average
cosine **0.8755** (54 pairs), while those with different headings average **0.7354** (98 pairs).

This suggests that concatenating section headings into the embedded text may enlarge the
same-section candidate pool. However, this offline comparison cannot separate the effect of
section context from the effect of action semantics themselves, so **no causal claim is made
and nothing is changed in this package on the strength of it.**

## 3. E0 is an exact-label deterministic gate by design

E0 does not perform semantic equivalence detection. It has no embedding, fuzzy-matching, or
model-based comparison, and none is planned for this package. Its duplicate checks
(`SIBLING_EXACT_DUPLICATE`, `PARENT_CHILD_EXACT_DUPLICATE`) are string equality.

This is a design property, not a defect: E0 is the deterministic publication gate, and keeping
it deterministic is what makes its verdicts reproducible. Semantic consolidation belongs
upstream, in the stages that call a model.

A consequence worth stating explicitly: **E0 having rejected a duplicate in the past does not
generalize to near-duplicates.** One historical rejection involved a byte-identical
parent/child pair, which is precisely the case E0 does detect.

### All 37 violation codes are `critical`, with one exemption

Every violation code carries `severity: "critical"`, and `critical` is not merely a report
field — it enters control flow. `publish_tree_if_valid()` computes `passed = not violations`
and raises `E0ValidationError` when any violation is present, so the candidate tree is not
written at all.

E0 never deletes, filters, or modifies nodes; it only decides whether publication proceeds.

The only caller-side exemption is `parent_child_allowlist`, which suppresses
`PARENT_CHILD_EXACT_DUPLICATE` for explicitly listed pairs.

The operational consequence: there is no "warn but publish" tier. A purely cosmetic finding
(for example a declared-level/physical-depth mismatch) blocks publication exactly as hard as a
genuine structural fault. The authoritative 2026-08-22 run is unaffected — it passed with
`critical_count: 0` — but future runs carry this risk.

## 4. `split_reparent` has never been produced by a model

The decision contract permits eight actions. In the authoritative run the model emitted only
four: `merge`, `keep`, `reject_merge`, `create_bridge`.

`split_reparent`, `move`, `rename`, and `flatten` were produced **zero** times — across the
authoritative run's 310 operation records, an earlier full run's 237 records, and a 325-decision
offline corpus.

**This zero occurs at the generation end.** It is not the case that the action was proposed and
then dropped by the executor: the 22 `applied` operations in the authoritative run are 14
`merge` and 8 `create_bridge`, and no `split_reparent` was ever proposed to be executed.

**Consequently this package does not claim that structural split-and-reparent has been
validated, verified, or completed.** The capability exists in the contract and is covered by
tests, but it has no empirical evidence of use.

Three candidate causes remain unseparated: the prompt may not explain or exemplify the action;
presenting one node pair at a time may make "split this open and reattach elsewhere" hard to
conceive; and the 14a scope gate's conditions for the action are intricate. Zero occurrences in
the corpus indicate the action was never even attempted, which weighs toward the first cause.

Nothing about the prompt, candidate presentation, action set, binder, or executor is changed in
this package.

## 5. `deferred` is a recorded controlled abandonment, with no consumer stage

When a stage judges a decision semantically reasonable but inexpressible under the current
contract, it records the decision as `status: "deferred"` and continues, instead of halting the
whole tree.

**No downstream stage ever retries a deferred decision.** The present meaning of `deferred` is
therefore "recorded, and the run continued" — *not* "will be done later". A deferred
restructuring will never be executed.

The authoritative run produced 4 deferred records, across three stages:

| Stage | Count | Cause |
|---|---|---|
| `vertical_collapse` (14a) | 1 | proposed merge target outside the presented pair |
| `structure_balancing_fanout` (14b) | 1 | balancing decision inexpressible |
| `label_polishing` (14c) | 2 | `DECISION_SCOPE_INEXPRESSIBLE` — child plan referenced a non-source child |

Regarding what this cost the tree, three things must be distinguished:

1. **Nodes still exist.** Every node involved in a deferred decision is still in the finished
   tree, with one exception noted below.
2. **The proposed operation was not executed.** That is the whole content of a defer.
3. **Nothing was silently lost.** All 4 are recorded in
   `intermediate_outputs/tree_refinement_operations.jsonl` with their contract violations. No
   node or policy action disappeared without a trace.

The one apparent exception is instructive: `L2_Nd0b47c6e` is absent from the finished tree, but
not because of the defer — it was merged into `L2_N08a346fe` by a later *applied* 14c operation
(`relation: synonym`), and its member remains fully traceable in the membership export
(`final_node_id: L2_N08a346fe`, `original_node_id: L2_Nd0b47c6e`).

The two 14c defers are the direct cause of Family B in limitation 1: the model examined that
pair (cosine 0.9511, well clear of the prefilter), judged the labels an
`exact_duplicate` differing only by a stylistic suffix, and proposed the merge — which was then
deferred because the accompanying child plan referenced a child that did not belong to the
merge source. **So near-duplicates survive by two independent routes: never being shown to the
model (Family A), and being shown and correctly judged but not executable under the contract
(Family B).** Lowering the prefilter threshold would not have helped Family B.

There is no human-review record for the deferred decisions.

## 6. Cross-run reuse of a stale membership roster is not supported

The membership roster (`tree_node_membership_L*.csv`) is a snapshot taken upstream. Stages
14a–14c delete and merge nodes, and no mechanism refreshes the roster across runs.

A fresh full run is unaffected: the authoritative input's 244 active nodes against its 278-row
roster yield **0 references to inactive nodes**, verified by recomputation.

Reuse and resume scenarios are affected: a stale roster referencing deleted nodes is rejected
fail-closed on entry to 14a. **That rejection is the gate working correctly, not a bug.** The
practical limitation is that you cannot cheaply resume from a previous run's roster; a fresh
run is the supported path.

## 7. The spend-reconciliation check assumes one execution per stage per scratch

The stage verifier compares the model's self-reported attempt count against the number of
requests the HTTP interceptor recorded, and fails the stage when they disagree. This is the
only mechanism that can detect requests being issued without being accounted for, and it is
deliberately retained.

The two logs have different lifecycles: the model log `intermediate_outputs/logs/llm_<stage>.jsonl`
is **appended**, while `runlogs/<stage>.http.jsonl` is **truncated and rebuilt** on each
invocation. Running the same stage twice in the same scratch directory therefore guarantees a
mismatch.

Network retries do **not** trigger this falsely: a retry increments both sides together.

**Operational requirement:** before re-running a stage in a scratch directory that has already
run it, move that stage's `llm_<stage>.jsonl` aside. This is a manual step; the assumption in
the verifier is unchanged.

## 8. `ROOT` is both a node ID and an English word

The root node's ID is the literal string `ROOT` (assigned in
`scripts/build_initial_policy_tree.py`). The leak-prevention scanner that keeps real node IDs
out of model-facing content must treat `ROOT` as a real ID, because it is one.

Consequently a model writing ordinary English prose containing `ROOT` in a free-text field —
`"ROOT CAUSE analysis"` — trips the scanner and the call is rejected. No regular expression can
resolve this, because the bytes are genuinely ambiguous: any algorithm must choose between
false positives and missed detections.

This is an ID namespace design issue, not a scanner defect, and it is not addressed here. It has
not caused a failure in the authoritative run. Resolving it means renaming the root identity,
which is a coordinated change best made during a port rather than patched in place.

## Status of these limitations

These limitations are recorded rather than fixed. The functional code, prompts, thresholds, and
run configuration of this package are frozen as of the authoritative 2026-08-22 run; the
remaining work is carried on a successor-system migration list, where each item can be resolved
against a different architecture rather than patched into a package whose value is that it
currently runs end-to-end.

None of the above blocks reproduction of the published artifacts. All of it constrains what may
be claimed about them.
