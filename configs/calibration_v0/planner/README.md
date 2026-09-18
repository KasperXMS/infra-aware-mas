# Task 795 open-ended Planner configuration

This directory contains the controlled `795 × H1/H2 × blind/aware` configuration grid:

- `795-h1-blind.yaml`
- `795-h1-aware.yaml`
- `795-h2-blind.yaml`
- `795-h2-aware.yaml`

All four cells share `task-795.yaml`, `models.yaml`, and `executors.yaml`. The task keeps the three
original MP4 chunks, placed on A4, A5, and A28. It does not contain evaluator answers. The four
executable experiment YAMLs keep the same logical models, Planner model, efficient harness, tools,
locality scheduler, executor set, ten-turn limit, and generic sampler settings. Blind and aware
differ only in whether the Planner receives a dynamic infrastructure snapshot. H1 and H2 differ
only in the selected `resources-h1.yaml`/`resources-h2.yaml` network condition.
The resource snapshots use effective RTT (33 ms observed baseline plus the configured added delay):
83 ms in H1 and 33 ms in H2.

Validate the four files without contacting any Worker:

```bash
uv run infra-mas-bench check-planner-v0 \
  --manifest configs/calibration_v0/planner/795-h1-blind.yaml \
  --manifest configs/calibration_v0/planner/795-h1-aware.yaml \
  --manifest configs/calibration_v0/planner/795-h2-blind.yaml \
  --manifest configs/calibration_v0/planner/795-h2-aware.yaml
```

Add `--jetson-preflight` to check only A4/A5/A28. That preflight deliberately does not construct a
client for the 4090 Worker and records it as `expected_unavailable`.

## Current execution blocker

These are prepared experiment configurations, not authorization to run the Planner. The unchanged
open-ended Planner exposes `spawn_agent` and `inspect_artifact`, but not the registered
`sample_frames` operator. The deployed Ollama VLM path also does not accept raw MP4 input. Therefore
the current benchmark bridge cannot consume these pre-existing Worker-local paths as if they were
controller-local uploads. Validation reports `execution_blocked`; it does not replace the MP4s with
preprocessed images, change the task, or silently choose a data-processing path. No Planner run
should be started until those capability gaps are addressed in a separately authorized task.
