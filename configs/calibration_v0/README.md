# calibration_v0 deployment

This is a Planner-free calibration. `H1` and `H2` use the
same tasks, devices, models, and placements; only the application-layer transfer
profile changes (3 Mbps + 50 ms added RTT versus 100 Mbps + no added RTT).
Each chunk uses 12 fixed uniform 320px-wide frames in both workflows; 320px keeps
the identical temporal policy inside the Orin model's deployed 8K context window.

Two Worker-local reference manifests are ready:

- `reference_747_sweep.yaml`: Task 747 with `centralized_raw` and `local_reduction`.
- `reference_795_all_workflows_sweep.yaml`: Task 795 with `centralized_raw`,
  `visual_reduction`, and `local_reduction`.

Both use `input_source: worker_local`. Before each warm-up or measured workflow, the controller
sends only an allowlisted path and artifact metadata to A4/A5/A28. Each Worker imports its existing
chunk into its ArtifactStore locally. This binding is traced but occurs before the E2E timer, so it
does not create a controller-to-Orin video upload or contaminate workflow transfer measurements.
The formal manifests include the measured byte size of every source chunk; a mismatch deletes the
partial binding and fails before any workflow or model execution. SHA256 is intentionally deferred
to avoid re-reading every long video during each setup.
E2E ends when the final answer ArtifactRef has been produced on its Worker. The subsequent
controller download is evaluator materialization and is explicitly excluded; rows record this as
`metadata.e2e_boundary: workflow_start_to_final_artifact_created`.
Rows identify their independent series in `metadata.measurement_series_id` and use
`metadata.measurement_protocol: steady_state_1_warmup_3_measured_v1`. Resume checks are scoped to
that series, so legacy Task 795 rows cannot suppress the new formal measurements.

The historical `sweep.yaml` retains the default `controller_upload` behavior for backward
compatibility. Run a Worker-local manifest from the repository root only after deploying matching
Worker configs:

```bash
uv run infra-mas-bench calibrate-v0 \
  --sweep configs/calibration_v0/reference_747_sweep.yaml \
  --output /home/super/xiaoming/calibration_v0/runs/calibration_v0
```

All transfers after initial placement are real streamed bytes with wall-clock throttling. The Orin
Worker allowlist contains only `/home/edge/xiaoming/calibration_v0/artifacts` and
`/home/edge/xiaoming/calibration_v0/candidates`; arbitrary local paths are rejected.

Bindings use run-scoped ArtifactStore IDs. Re-running a series leaves setup copies under each
Worker's artifact root, as the historical upload path does. Remove obsolete run directories only as
an explicit post-experiment storage-maintenance step after retaining their traces and results.
