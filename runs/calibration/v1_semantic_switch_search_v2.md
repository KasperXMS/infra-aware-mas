# V1 semantic-switch calibration search V2

## Existing distributed_3x2 implementation audit

Does distributed_3x2 perform a real model-based semantic synthesis after the three VLM outputs? **YES**.

The audited pre-V2 implementation invoked `edge-vlm` for synthesis on Worker A28. The synthesis invocation was inside the measured E2E interval; its `service_ms` was included in `service_ms_sum`. Normal transfer semantics moved the A4 and A5 evidence artifacts to A28 (188 bytes in the inspected distributed trace).

The V2 rerun leaves `distributed_3x2` unchanged: its synthesis still uses `edge-vlm` and normal locality-aware placement. Only the new `distributed_6x1` invokes `remote-llm` on Worker `coordinator-remote`. Its six evidence artifacts are transferred there through the normal Worker-to-Worker path. Synthesis service time remains part of `service_ms_sum` and the invocation remains inside measured E2E.

## Calibration measurements

| World | Workflow | Median E2E | Service | Transfer bytes | Transfer ms | Accuracy |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| one-replica-a28-colocated | centralized | 14004.735499969684 | 13377.341678831726 | 0 | 0.0 | 1.0 |
| one-replica-a28-colocated | distributed_3x2 | 14101.381800021045 | 23679.053080966696 | 0 | 0.0 | 1.0 |
| one-replica-a28-colocated | distributed_6x1 | 108847.46119996998 | 295221.2860690197 | 3574 | 3402.3181000957265 | 1.0 |
| one-replica-a4-colocated | centralized | 12195.914800046012 | 11844.845565996366 | 0 | 0.0 | 1.0 |
| one-replica-a4-colocated | distributed_3x2 | 12569.44949994795 | 21351.974374003476 | 0 | 0.0 | 1.0 |
| one-replica-a5-colocated | centralized | 14095.75350000523 | 12815.472627058625 | 0 | 0.0 | 1.0 |
| one-replica-a5-colocated | distributed_3x2 | 15329.155900049955 | 23888.80789023824 | 0 | 0.0 | 1.0 |
| three-replicas-distributed | centralized | 14611.414100043476 | 13036.202046088874 | 405192 | 205.5425839498639 | 1.0 |
| three-replicas-distributed | distributed_3x2 | 10982.896200031973 | 13385.395859040727 | 188 | 441.20638398453593 | 1.0 |
| three-replicas-distributed | distributed_6x1 | 66924.89130003378 | 135405.31794518756 | 3432 | 3559.034299920313 | 1.0 |

Semantic-switch pair found: **False**.

## Pairwise winner margins

| World | Comparison | Winner | Relative margin | Correct |
| --- | --- | --- | ---: | --- |
| one-replica-a28-colocated | centralized vs distributed_3x2 | centralized | 0.006900972892459878 | True |
| one-replica-a28-colocated | centralized vs distributed_6x1 | centralized | 6.772189713986788 | True |
| one-replica-a4-colocated | centralized vs distributed_3x2 | centralized | 0.030627854164783867 | True |
| one-replica-a5-colocated | centralized vs distributed_3x2 | centralized | 0.08750170042660484 | True |
| three-replicas-distributed | centralized vs distributed_3x2 | distributed_3x2 | 0.33037896688861895 | True |
| three-replicas-distributed | centralized vs distributed_6x1 | centralized | 3.580315829926047 | True |

Reason: V1 does not provide a sufficiently robust real-system crossover

## Near-boundary reversal pairs

- [distributed_3x2] one-replica-a28-colocated (centralized, margin=0.006900972892459878) vs three-replicas-distributed (distributed_3x2, margin=0.33037896688861895)
- [distributed_3x2] one-replica-a4-colocated (centralized, margin=0.030627854164783867) vs three-replicas-distributed (distributed_3x2, margin=0.33037896688861895)
- [distributed_3x2] one-replica-a5-colocated (centralized, margin=0.08750170042660484) vs three-replicas-distributed (distributed_3x2, margin=0.33037896688861895)
