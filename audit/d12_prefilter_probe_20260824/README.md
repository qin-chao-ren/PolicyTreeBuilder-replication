# D12 prefilter probe — 2026-08-24

Offline evidence for the 14c sibling-merge prefilter limitation recorded in
`KNOWN_LIMITATIONS.md` section 2.

**Zero API calls.** Every number is computed from committed artifacts plus a precomputed
embedding parquet, using the package's own tokenizer, centroid helper and cosine function
(imported, not reimplemented).

## What question this answers

A pair of same-parent nodes that mean the same thing — 构建全球国际航线网络 and
构建覆盖全球的货运航线网络 — is never shown to the model by stage 14c, so it survives into the
finished tree. Two candidate causes were on the table:

1. **Representation.** Node vectors are the centroid of member *policy-text* embeddings and never
   include the label, so perhaps the prefilter compares the wrong text.
2. **Threshold.** The pair may simply sit below a correctly-shaped threshold.

The probe tests both. **Result: cause 1 is ruled out for this pair, and the limitation is a
threshold boundary.** No code was changed on the strength of this finding — see
`KNOWN_LIMITATIONS.md` section 2 for why the threshold is deliberately left alone.

## Key measurements

| Measurement | Value |
|---|---|
| Known pair, Jaccard | 0.5714 (threshold 0.75 → fail) |
| Known pair, bare cosine | 0.832628 (threshold 0.85 → fail) |
| Known pair, `[0,1]`-mapped cosine | 0.916314 — **not** the production channel |
| Member embedded text vs node label, known pair | byte-identical on both sides |
| Input tree | 246 nodes, 366 sibling pairs, 318 with vectors on both sides |
| Pairs reaching the model at 0.85 | 77 |
| Pairs reaching the model at 0.83 | 95 (+18, of which ~3 look genuinely near-synonymous) |

The `[0,1]`-mapped variant (`common_utils.cosine_sim`) rescales to 0.916314 and would appear to
clear the 0.85 threshold, supporting the opposite conclusion. Production compares the bare
cosine. Any reuse of these figures must state which is which.

### Why "switch to label vectors" would change nothing here

For both nodes of the known pair the embedded text is byte-identical to the node label, so the
centroid already *is* the label vector. The corpus stores pre-segmented action phrases rather
than whole documents. Across the input tree: 61 nodes have member text identical to the label,
142 carry `label [SEP] section-heading`, 25 have multiple members, 18 have none — the label is
always present in the vector.

This refutation is specific to the pair and tree measured. It is **not** a claim that no
embedding representation problem exists anywhere in the pipeline.

## Files

| File | Contents |
|---|---|
| `probe_d12_prefilter.py` | The probe. Takes all paths as arguments; no machine-specific paths embedded. |
| `sibling_pairs.csv` | Per-pair scores for all 366 sibling pairs: parent, both node ids, both labels, Jaccard, bare cosine, and whether vectors were computable. |

## Reproducing

```bash
python probe_d12_prefilter.py \
    --scripts      <repo>/scripts \
    --embeddings   <run>/input_assets/policy_corpus_embeddings.parquet \
    --intermediate <run>/intermediate_outputs \
    --out-csv      sibling_pairs.csv
```

`--intermediate` must hold the 14c input tree
(`policy_tree_after_structure_balancing.json`, i.e. the output of 14b) and the
`tree_node_membership_L*.csv` rosters. Requires pandas, numpy, pyarrow.

The figures above were produced against the 2026-08-22 run whose 14b output tree is the 246-node
`policy_tree_after_structure_balancing.json`. Running against a different tree will give
different counts; the threshold table in particular is tree-specific and must always be cited
together with the tree it was measured on.

## What is deliberately not archived here

The embedding parquet itself is not copied into this directory (it is a large binary artifact
that belongs with its run), and no run scratch content, credentials, or logs are included.
