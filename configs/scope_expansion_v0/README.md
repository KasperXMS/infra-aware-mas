# Scope Expansion v0 reference workflows

This directory configures the Planner-free, Jetson-only MultiHop-RAG and LongBench-v2
reference experiments. The MultiHop sweep manifest must contain exactly the fixed
Top-30 candidate documents for each selected M2/M3/M4 task (or Top-50 globally, never
per-task), with every document as an independent Worker-local artifact.

The runner validates placement as:

```text
int.from_bytes(sha256(document_id)[:8], "big") % 3
0 -> a4 / A4, 1 -> a5 / A5, 2 -> a28 / A28
```

`distributed_retrieval` always uses `local_top_k_per_shard: 3`; the field is a literal
and cannot be tuned per task or world. `centralized_raw` passes all Top-N independent
documents to the same `edge-text-reasoner` executor on A28. The distributed branch
runs deterministic BM25 concurrently on A4/A5/A28 and passes only its three evidence
artifacts to that exact same final executor.

The declared final-model context window is the deployed `qwen3-vl-edge-8k` value
(`8192`), not a hypothetical larger window. Actual input/output token usage is recorded
for every final invocation.

Run a materialized sweep with:

```text
infra-mas-bench prepare-scope-expansion-v0 \
  --task-bank ../infra-bench/runs/scope_expansion_v0/task_bank.jsonl \
  --output configs/scope_expansion_v0/multihop_rag_sweep.yaml

infra-mas-bench scope-expansion-v0 --sweep <generated-sweep.yaml> \
  --output runs/scope_expansion_v0/multihop_rag
```

Each cell is resumable and fixed to one warm-up plus three measured repeats. Worker-local
binding occurs before the measured E2E boundary. Gold answers and supporting-document
metadata are not accepted by the MAS sweep schema.

After MultiHop-RAG is stable, prepare and run the two LongBench-v2 references with:

```text
infra-mas-bench prepare-longbench-v2 \
  --task-bank ../infra-bench/runs/scope_expansion_v0/task_bank.jsonl \
  --output configs/scope_expansion_v0/longbench_v2_sweep.yaml

infra-mas-bench longbench-v2 --sweep <generated-sweep.yaml> \
  --output ../infra-bench/runs/scope_expansion_v0
```

The LongBench sweep preserves natural document/record boundaries. Multi-document QA
uses `centralized_raw` versus `distributed_retrieval`; structured analysis uses
`centralized_raw` versus a plan-driven chain of generic record operators. Both keep
the final model and A28 executor fixed, and both use one warm-up plus three measured
runs in H1 and H2.
