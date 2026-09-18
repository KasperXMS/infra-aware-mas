# calibration_v0 deployment

This is a Planner-free, fixed 2-task calibration. The controller runs on the
dual-4090 host so source chunks under `/home/super/xiaoming/calibration_v0/chunks`
can be staged to A4/A5/A28 before the measured interval. `H1` and `H2` use the
same tasks, devices, models, and placements; only the application-layer transfer
profile changes (3 Mbps + 50 ms added RTT versus 100 Mbps + no added RTT).
Each chunk uses 12 fixed uniform 320px-wide frames in both workflows; 320px keeps
the identical temporal policy inside the Orin model's deployed 8K context window.

Run from the repository root on the 4090 host:

```bash
uv run infra-mas-bench calibrate-v0 \
  --sweep configs/calibration_v0/sweep.yaml \
  --output /home/super/xiaoming/calibration_v0/runs/calibration_v0
```

The controller-to-Orin staging upload is initial placement and deliberately
excluded from workflow E2E and cross-agent transfer metrics. All subsequent
artifact transfers are real streamed bytes with wall-clock throttling.
