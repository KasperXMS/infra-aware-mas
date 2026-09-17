# astropy__astropy-14309 open-ended semantic-switch report

Runs discovered: **12**.

## Materialized worlds

| World | Repository site | Repository bytes | Bandwidth (Mbps) | RTT (ms) |
| --- | --- | ---: | ---: | ---: |
| H1_edge | edge | 85732111 | 1.0 | 50.0 |
| H2_cloud | cloud | 85732111 | 1.0 | 50.0 |

## Cell aggregates

| World | Visibility | Runs | Resolved | Median E2E (ms) | Planner tokens | Planner latency (ms) | Reasoning calls | Cross-site bytes | Context bytes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| H1_edge | snapshot | 3 | 100.0% | 559750.0 | 323049 | 542434.1 | 19 | 59328 | 52651 |
| H1_edge | static | 3 | 100.0% | 839853.3 | 302101 | 821016.3 | 19 | 60213 | 50938 |
| H2_cloud | snapshot | 3 | 66.7% | 1181139.5 | 405834 | 1178965.3 | 26 | 0 | 49983 |
| H2_cloud | static | 3 | 33.3% | 1515052.6 | 581225 | 1513141.8 | 30 | 0 | 49584 |

### Workflow and execution medians

| World | Visibility | Search | Read | Edit | Apply patch | Targeted tests | Full tests | Retries | Tool service (ms) | Transfer (ms) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| H1_edge | snapshot | 7 | 12 | 3 | 1 | 4 | 1 | 6 | 18419.1 | 2577.4 |
| H1_edge | static | 9 | 8 | 3 | 2 | 5 | 0 | 4 | 21956.3 | 3368.1 |
| H2_cloud | snapshot | 9 | 10 | 9 | 1 | 2 | 1 | 7 | 1350.7 | 0.0 |
| H2_cloud | static | 9 | 10 | 11 | 4 | 3 | 1 | 13 | 1734.4 | 0.0 |

## Main questions

- `G_snapshot(H1) != G_snapshot(H2)`: **True**
- Clean shift in the calibrated direction: **False**
  - H1 snapshot median search+read: **19**; H2 snapshot: **19**.
  - The snapshot workflows differ structurally, but the expected clean shift from more edge-local preprocessing in H1 to less preprocessing in H2 is not supported: median search+read counts are equal. H2 transmits slightly less code context but uses more reasoning and edit calls.
- `H1_edge: C_snapshot(H) < C_static(H)` without a correctness drop: **True** (median E2E change: -33.4%)
- `H2_cloud: C_snapshot(H) < C_static(H)` without a correctness drop: **True** (median E2E change: -22.0%)

## Per-run realized workflows

| Run | World | Visibility | Resolved | E2E (ms) | Workflow |
| --- | --- | --- | ---: | ---: | --- |
| formal-h1-snapshot-r1 | H1_edge | snapshot | True | 559750.0 | read_file -> read_file -> read_file -> search_code -> search_code -> search_code -> read_file -> search_code -> read_file -> read_file -> read_file -> search_code -> read_file -> read_file -> read_file -> read_file -> read_file -> edit_file -> search_code -> edit_file -> run_targeted_test -> run_targeted_test -> run_targeted_test -> edit_file -> run_targeted_test -> run_targeted_test -> apply_patch -> submit_patch |
| formal-h1-snapshot-r2 | H1_edge | snapshot | True | 361699.7 | read_file -> search_code -> search_code -> search_code -> read_file -> search_code -> read_file -> search_code -> search_code -> search_code -> edit_file -> read_file -> read_file -> read_file -> search_code -> edit_file -> search_code -> run_targeted_test -> read_file -> read_file -> apply_patch -> run_full_test -> run_targeted_test -> run_targeted_test -> run_targeted_test -> submit_patch |
| formal-h1-snapshot-r3 | H1_edge | snapshot | True | 1194837.8 | read_file -> search_code -> read_file -> read_file -> read_file -> search_code -> read_file -> read_file -> search_code -> search_code -> read_file -> read_file -> read_file -> search_code -> read_file -> search_code -> read_file -> search_code -> apply_patch -> edit_file -> edit_file -> edit_file -> run_targeted_test -> run_targeted_test -> apply_patch -> edit_file -> edit_file -> apply_patch -> edit_file -> run_targeted_test -> run_full_test -> edit_file -> edit_file -> edit_file -> read_file -> submit_patch |
| formal-h1-static-r1 | H1_edge | static | True | 839853.3 | read_file -> search_code -> search_code -> search_code -> read_file -> read_file -> search_code -> search_code -> read_file -> search_code -> search_code -> read_file -> read_file -> read_file -> search_code -> search_code -> read_file -> edit_file -> apply_patch -> apply_patch -> apply_patch -> run_targeted_test -> run_targeted_test -> run_targeted_test -> run_targeted_test -> run_targeted_test -> submit_patch |
| formal-h1-static-r2 | H1_edge | static | True | 1063625.3 | read_file -> search_code -> read_file -> read_file -> read_file -> read_file -> search_code -> search_code -> read_file -> search_code -> read_file -> search_code -> read_file -> read_file -> search_code -> search_code -> search_code -> search_code -> edit_file -> edit_file -> edit_file -> edit_file -> run_targeted_test -> run_targeted_test -> run_targeted_test -> read_file -> apply_patch -> run_targeted_test -> edit_file -> edit_file -> edit_file -> run_targeted_test -> run_targeted_test -> run_targeted_test -> run_targeted_test -> run_targeted_test -> run_targeted_test -> run_targeted_test -> run_targeted_test -> run_targeted_test -> apply_patch -> run_targeted_test -> submit_patch |
| formal-h1-static-r3 | H1_edge | static | True | 462516.9 | read_file -> search_code -> read_file -> search_code -> search_code -> search_code -> search_code -> read_file -> read_file -> read_file -> read_file -> search_code -> search_code -> read_file -> read_file -> search_code -> search_code -> edit_file -> edit_file -> apply_patch -> edit_file -> run_targeted_test -> run_targeted_test -> submit_patch |
| formal-h2-snapshot-r1 | H2_cloud | snapshot | True | 519417.5 | read_file -> search_code -> read_file -> search_code -> read_file -> read_file -> search_code -> read_file -> search_code -> read_file -> read_file -> search_code -> read_file -> search_code -> search_code -> read_file -> search_code -> search_code -> edit_file -> run_targeted_test -> run_full_test -> submit_patch |
| formal-h2-snapshot-r2 | H2_cloud | snapshot | False | 1181139.5 | read_file -> search_code -> read_file -> search_code -> search_code -> read_file -> read_file -> search_code -> search_code -> search_code -> search_code -> read_file -> read_file -> read_file -> search_code -> read_file -> search_code -> search_code -> edit_file -> edit_file -> read_file -> edit_file -> edit_file -> run_targeted_test -> run_targeted_test -> edit_file -> apply_patch -> run_full_test -> edit_file -> edit_file -> edit_file -> run_targeted_test -> edit_file -> edit_file -> edit_file -> edit_file -> read_file |
| formal-h2-snapshot-r3 | H2_cloud | snapshot | True | 1278730.5 | read_file -> search_code -> search_code -> search_code -> read_file -> read_file -> search_code -> search_code -> search_code -> read_file -> read_file -> search_code -> edit_file -> read_file -> search_code -> read_file -> edit_file -> search_code -> read_file -> edit_file -> edit_file -> apply_patch -> run_targeted_test -> run_full_test -> apply_patch -> edit_file -> run_targeted_test -> edit_file -> apply_patch -> read_file -> edit_file -> edit_file -> read_file -> edit_file -> submit_patch |
| formal-h2-static-r1 | H2_cloud | static | True | 988778.3 | read_file -> search_code -> search_code -> search_code -> read_file -> read_file -> read_file -> search_code -> read_file -> read_file -> search_code -> search_code -> read_file -> read_file -> search_code -> search_code -> search_code -> search_code -> apply_patch -> edit_file -> run_targeted_test -> run_targeted_test -> run_targeted_test -> read_file -> apply_patch -> apply_patch -> run_targeted_test -> run_full_test -> run_targeted_test -> edit_file -> apply_patch -> run_targeted_test -> run_targeted_test -> run_targeted_test -> run_targeted_test -> run_targeted_test -> submit_patch |
| formal-h2-static-r2 | H2_cloud | static | False | 1515052.6 | read_file -> search_code -> read_file -> search_code -> read_file -> search_code -> read_file -> search_code -> read_file -> search_code -> read_file -> search_code -> search_code -> read_file -> read_file -> search_code -> read_file -> read_file -> edit_file -> run_targeted_test -> run_full_test -> run_targeted_test -> apply_patch -> apply_patch -> edit_file -> edit_file -> edit_file -> edit_file -> edit_file -> edit_file -> read_file -> edit_file -> edit_file -> edit_file -> edit_file -> apply_patch -> edit_file -> edit_file -> apply_patch |
| formal-h2-static-r3 | H2_cloud | static | False | 1552832.2 | read_file -> search_code -> read_file -> search_code -> read_file -> read_file -> read_file -> search_code -> search_code -> search_code -> read_file -> read_file -> search_code -> read_file -> search_code -> search_code -> search_code -> edit_file -> edit_file -> edit_file -> run_targeted_test -> run_full_test -> read_file -> edit_file -> edit_file -> edit_file -> edit_file -> apply_patch -> apply_patch -> edit_file -> edit_file -> apply_patch -> run_targeted_test -> edit_file -> run_targeted_test -> edit_file -> read_file |

## Leakage audit

Reference/evaluator leakage count: **0**.

## Experimental controls

- `formal_run_ids_only`: **True**
- `static_context_identical_across_worlds`: **True**
- `snapshot_context_present_only_in_snapshot_runs`: **True**
- `same_scheduler`: **True**
- `same_planner_and_budgets`: **True**
