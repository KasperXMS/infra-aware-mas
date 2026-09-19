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

## Execution contract

The Planner receives logical artifact IDs and the generic `sample_frames`, `make_contact_sheet`,
`extract_clip`, `process_local_artifact`, and `aggregate_artifacts` actions. It never receives a
Worker-selection argument. The runtime chooses a capable Worker by artifact locality and performs
any required Worker-to-Worker transfer through the normal data plane. `sample_frames` can probe the
video duration on the owning Worker, so the Planner may call it with only an artifact ID.

The experiment YAMLs use `input_source: worker_local`. At run setup, each Orin imports its existing
MP4 from the allowlisted `/home/edge/xiaoming/calibration_v0/artifacts` tree into its ArtifactStore;
the controller sends only path/metadata in this setup phase, and video bytes do not pass through the
controller. Binding is traced before workflow E2E timing begins.

Planner-visible input, output, and action IDs use a fresh opaque namespace unrelated to the trace
directory/run ID. Meaningful experiment IDs remain in controller-side config and trace metadata but
cannot reveal the world or blind/aware arm through ArtifactRef IDs.

After deploying the matching Worker code/config and completing a full four-Worker preflight, one
cell can be started explicitly with:

```bash
uv run infra-mas-bench run-planner-v0 \
  --manifest configs/calibration_v0/planner/795-h1-blind.yaml \
  --run-id planner-task-run-r1
```

Configuration preparation and Jetson-only preflight do not run the Planner or contact the 4090.
