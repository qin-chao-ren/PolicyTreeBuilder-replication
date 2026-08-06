# Final Tree

This folder contains the archived ATRS 2026 final 353-node tree outputs.

Primary file: `policy_tree_final.json`.

Expected structure: 353 nodes, 352 edges, 272 leaf nodes, maximum depth 6.

The files are immutable historical artifacts, not E0-compliant corrected outputs. The C13 validator intentionally rejects `policy_tree_final.json` as a frozen negative control (8 exact sibling-duplicate groups and 10 level/depth mismatches); its operation log also contains 34 `applied` merges whose sources remain in the raw tree. A corrected tree must be published as a separately versioned candidate after E0 passes.
