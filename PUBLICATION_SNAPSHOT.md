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
