# Publication Snapshot

This package is the ATRS 2026 final replication snapshot for PolicyTreeBuilder.

## Historical Public Version

- Final tree: `data/final_tree/policy_tree_final.json`
- Nodes: 353
- Edges: 352
- Leaf nodes: 272
- Maximum depth: 6
- Repository SHA256: `9242d4961e417ffa1e30e728d82e73bf669e4facdedd12f4f03f10d19b157983`
- Source archive extracted-file SHA256 before repository LF normalization: `6ee8e666dfc5bb7b8611a42c96c8ac93f766290b290115529686f9f3ca67918b`

These identifiers freeze the artifact used for the paper; they are not an E0 integrity claim. The tree is retained byte-for-byte as a negative control. The deterministic validator reports 8 exact sibling-duplicate groups and 10 level/depth mismatches, while the archived final operation log contains 34 `applied` merges whose source IDs remain in the tree. The historical files are not silently overwritten by this hardening change.

## Source Selection

The public code and data are based on the source archive that produced the 353-node tree. The previous 317-node public package is superseded and retained only in a local archive.

## Evaluation Package

The former local `policy_tree_eval` materials have been integrated as the public `evaluation/` module after removing local environments, credentials, caches, and private path assumptions. The archived evaluation outputs evaluate the same 353-node final tree and provide supporting quality checks for reviewers.

## Corrected Candidate Policy

Any future corrected tree is a separately versioned candidate. Before publication, it must pass structural E0, including structure, exact-label uniqueness, level/depth, lineage, membership-conservation, and applied-operation checks, plus the independent semantic-contract gate over every applied refinement decision. Whether such a candidate replaces the historical default is a separate release decision.

## Non-public Material

The public repository excludes local archives, real `.env` files, API keys, virtual environments, caches, scratch outputs, and the superseded 317-node package.

## FILE_INDEX.tsv / SCRIPT_PROVENANCE.tsv Are Frozen Baseline Snapshots (C13RF25)

`FILE_INDEX.tsv` and `SCRIPT_PROVENANCE.tsv` were generated once, at the `4066e6a` (2026-08-06) publication baseline, and are **not** regenerated on every commit. As of C13RF25 (2026-08-21), 9 of `FILE_INDEX.tsv`'s 212 rows no longer match the corresponding file's current SHA256 (hash is the 3rd column, headed `sha256` — the 2nd column is `bytes`, not a hash):

```
scripts/balance_tree_structure.py
scripts/collapse_redundant_hierarchy.py
scripts/finalize_policy_tree.py
scripts/polish_tree_labels.py
scripts/utils/local_reference_binding.py
scripts/utils/semantic_contract.py
scripts/utils/tree_integrity.py
scripts/utils/tree_manager.py
tests/test_local_reference_binding.py
```

This is expected: it is exactly the set of production files C13RF16 through C13RF22 (2026-08-18 through 2026-08-19) modified after the baseline snapshot was taken, and no others. Do not regenerate the file to make it match current `HEAD` — that would destroy its value as a point-in-time publication baseline. Instead, treat a `FILE_INDEX.tsv` mismatch as a **change-audit signal**: any row whose 3rd-column hash disagrees with the file's current SHA256 has been touched since 2026-08-06, and the list above is the complete such set as of 2026-08-21. `SCRIPT_PROVENANCE.tsv`'s hash column is named `public_sha256` (4th column); read it by that column name, not by position, since an earlier draft of this note compared the wrong column and reported the baseline as fully stale.
