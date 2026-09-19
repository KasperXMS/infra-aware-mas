"""Worker-to-worker network microbenchmark for the three Jetson calibration sites."""

from __future__ import annotations

import asyncio
import json
import re
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

from infra_mas.core.artifact import ArtifactRef
from infra_mas.core.execution import ArtifactPullRequest
from infra_mas.execution.executor_registry import ExecutorRegistry
from infra_mas.execution.worker_client import WorkerClient

_MIB = 1024 * 1024
_WORKERS = ("a4", "a5", "a28")
_RING = (("a4", "a5"), ("a5", "a28"), ("a28", "a4"))


async def _ping(source_host: str, target_host: str) -> dict[str, float] | None:
    process = await asyncio.create_subprocess_exec(
        "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8",
        f"edge@{source_host}", "ping", "-q", "-c", "10", target_host,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    stdout, _ = await process.communicate()
    match = re.search(
        rb"(?:rtt|round-trip) min/avg/max/(?:mdev|stddev) = "
        rb"([0-9.]+)/([0-9.]+)/([0-9.]+)/([0-9.]+) ms",
        stdout,
    )
    if process.returncode != 0 or match is None:
        return None
    minimum, average, maximum, jitter = (float(value) for value in match.groups())
    return {"min_ms": minimum, "avg_ms": average, "max_ms": maximum, "jitter_ms": jitter}


def _append(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


async def run_network_microbench(
    executors_path: Path, output_directory: Path
) -> dict[str, object]:
    """Run single-flow and concurrent-flow measurements without a 4090 endpoint."""
    registry = ExecutorRegistry.from_yaml(executors_path.resolve())
    endpoints = {worker: registry.worker_endpoint(worker) for worker in _WORKERS}
    hosts = {
        worker: str(endpoints[worker]).split("//", 1)[1].split(":", 1)[0]
        for worker in _WORKERS
    }
    clients = {
        worker: WorkerClient(endpoints[worker], timeout=1200.0) for worker in _WORKERS
    }
    output_directory = output_directory.resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    raw_path = output_directory / "raw_results.jsonl"
    if raw_path.exists():
        raise FileExistsError(f"refusing to overwrite existing measurements: {raw_path}")

    rtts: dict[str, dict[str, float] | None] = {}
    for source in _WORKERS:
        for target in _WORKERS:
            if source != target:
                rtts[f"{source}->{target}"] = await _ping(hosts[source], hosts[target])

    # H1 at 3 Mbps takes ~839 s for 300 MiB, beyond the deployed Worker's
    # 300 s source-read timeout. Measure H1 with 1/10 MiB and explicitly report
    # the larger cells unavailable instead of inventing values.
    plans = [
        ("physical_baseline", "single", (1, 10, 100, 300), None, 0.0),
        ("physical_baseline", "concurrent", (1, 10, 100, 300), None, 0.0),
        ("H2_distributed_favorable", "single", (1, 10, 100, 300), 100.0, 0.0),
        ("H2_distributed_favorable", "concurrent", (1, 10, 100, 300), 100.0, 0.0),
        ("H1_distributed_constrained", "single", (1, 10), 3.0, 50.0),
        ("H1_distributed_constrained", "concurrent", (1, 10), 3.0, 50.0),
    ]

    async def transfer(
        scenario: str, mode: str, size_mib: int, source: str, target: str,
        bandwidth_mbps: float | None, rtt_ms: float, artifact: ArtifactRef, batch_id: str,
    ) -> dict[str, Any]:
        started = datetime.now(UTC)
        wall_start = perf_counter()
        try:
            result = await clients[target].pull_artifact(
                ArtifactPullRequest(
                    artifact=artifact, source_worker_id=source,
                    source_endpoint=clients[source].base_url,
                    bandwidth_mbps=bandwidth_mbps, rtt_ms=rtt_ms,
                )
            )
            wall_ms = (perf_counter() - wall_start) * 1000
            row: dict[str, Any] = {
                "schema_version": "calibration-v0-network-microbench-v1",
                "timestamp": started.isoformat(), "scenario": scenario, "mode": mode,
                "batch_id": batch_id, "source_worker_id": source,
                "target_worker_id": target, "direction": f"{source}->{target}",
                "payload_bytes": size_mib * _MIB,
                "configured_bandwidth_mbps": bandwidth_mbps,
                "configured_added_rtt_ms": rtt_ms,
                "icmp_rtt": rtts[f"{source}->{target}"], "status": "completed",
                "bytes_transferred": result.bytes_transferred,
                "transfer_latency_ms": result.transfer_ms, "controller_wall_ms": wall_ms,
                "effective_throughput_mbps": (
                    result.bytes_transferred * 8 / result.transfer_ms / 1000
                ),
            }
        except Exception as error:
            row = {
                "schema_version": "calibration-v0-network-microbench-v1",
                "timestamp": started.isoformat(), "scenario": scenario, "mode": mode,
                "batch_id": batch_id, "source_worker_id": source,
                "target_worker_id": target, "direction": f"{source}->{target}",
                "payload_bytes": size_mib * _MIB,
                "configured_bandwidth_mbps": bandwidth_mbps,
                "configured_added_rtt_ms": rtt_ms,
                "icmp_rtt": rtts[f"{source}->{target}"], "status": "failed",
                "error": f"{type(error).__name__}: {error}",
            }
        _append(raw_path, row)
        return row

    rows: list[dict[str, Any]] = []
    try:
        for client in clients.values():
            await client.health()
        with tempfile.TemporaryDirectory(prefix="infra-mas-netbench-") as directory:
            temporary_root = Path(directory)
            for scenario, mode, sizes, bandwidth, added_rtt in plans:
                directions = (
                    tuple((source, target) for source in _WORKERS for target in _WORKERS
                          if source != target)
                    if mode == "single" else _RING
                )
                for size_mib in sizes:
                    payload = temporary_root / f"payload-{size_mib}mib.bin"
                    if not payload.exists():
                        with payload.open("wb") as handle:
                            handle.truncate(size_mib * _MIB)
                    refs: dict[str, ArtifactRef] = {}
                    for source in sorted({item[0] for item in directions}):
                        artifact_id = (
                            f"network-microbench/{scenario}/{mode}/{size_mib}mib/{source}.bin"
                        )
                        refs[source] = await clients[source].upload_artifact(
                            ArtifactRef(
                                id=artifact_id, artifact_type="application/octet-stream",
                                size_bytes=size_mib * _MIB, locations=["controller"],
                            ), payload,
                        )
                    batch_id = f"{scenario}-{mode}-{size_mib}mib"
                    if mode == "single":
                        for source, target in directions:
                            rows.append(await transfer(
                                scenario, mode, size_mib, source, target, bandwidth,
                                added_rtt, refs[source], batch_id,
                            ))
                    else:
                        rows.extend(await asyncio.gather(*(
                            transfer(
                                scenario, mode, size_mib, source, target, bandwidth,
                                added_rtt, refs[source], batch_id,
                            ) for source, target in directions
                        )))
    finally:
        await asyncio.gather(*(client.aclose() for client in clients.values()))

    completed = [row for row in rows if row["status"] == "completed"]
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in completed:
        key = f"{row['scenario']}|{row['mode']}|{row['payload_bytes']}"
        groups.setdefault(key, []).append(row)
    aggregates: list[dict[str, Any]] = []
    for key, items in groups.items():
        scenario, mode, payload_text = key.split("|")
        throughputs = [float(item["effective_throughput_mbps"]) for item in items]
        latencies = [float(item["transfer_latency_ms"]) for item in items]
        baseline_rtts = [
            float(item["icmp_rtt"]["avg_ms"])
            for item in items
            if isinstance(item.get("icmp_rtt"), dict)
            and isinstance(item["icmp_rtt"].get("avg_ms"), (int, float))
        ]
        aggregates.append({
            "scenario": scenario,
            "mode": mode,
            "payload_bytes": int(payload_text),
            "flow_count": len(items),
            "configured_bandwidth_mbps": items[0]["configured_bandwidth_mbps"],
            "configured_added_rtt_ms": items[0]["configured_added_rtt_ms"],
            "measured_baseline_rtt_mean_ms": (
                sum(baseline_rtts) / len(baseline_rtts) if baseline_rtts else None
            ),
            "mean_effective_throughput_mbps": sum(throughputs) / len(throughputs),
            "aggregate_effective_throughput_mbps": sum(throughputs),
            "max_transfer_latency_ms": max(latencies),
        })
    summary: dict[str, Any] = {
        "schema_version": "calibration-v0-network-microbench-summary-v1",
        "workers": list(_WORKERS), "rtt_by_direction": rtts,
        "measurement_count": len(rows), "completed_count": len(completed),
        "failed_count": len(rows) - len(completed), "aggregates": aggregates,
        "unavailable_measurements": [{
            "scenario": "H1_distributed_constrained", "payload_mib": size,
            "reason": (
                "3 Mbps expected duration exceeds or approaches the deployed Worker's "
                "300 s source-read timeout"
            ),
        } for size in (100, 300)],
    }
    (output_directory / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    markdown = [
        "# A4/A5/A28 network microbenchmark", "",
        "All payloads moved directly between the three Jetson Workers; no 4090 endpoint was "
        "contacted. `single` runs one flow at a time across all six directions. `concurrent` "
        "runs the three-flow ring A4->A5->A28->A4 simultaneously.", "",
        "| Scenario | Mode | Payload | Configured cap | Configured added RTT | "
        "Measured baseline RTT | Flows | Measured mean/flow Mbps | "
        "Measured aggregate Mbps | Measured max latency |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in aggregates:
        configured_bandwidth = item["configured_bandwidth_mbps"]
        bandwidth_label = (
            "none"
            if configured_bandwidth is None
            else f"{float(configured_bandwidth):.1f} Mbps"
        )
        baseline_rtt = item["measured_baseline_rtt_mean_ms"]
        baseline_rtt_label = (
            "unavailable" if baseline_rtt is None else f"{float(baseline_rtt):.2f} ms"
        )
        markdown.append(
            f"| {item['scenario']} | {item['mode']} | {item['payload_bytes'] / _MIB:.0f} MiB | "
            f"{bandwidth_label} | {item['configured_added_rtt_ms']:.1f} ms | "
            f"{baseline_rtt_label} | {item['flow_count']} | "
            f"{item['mean_effective_throughput_mbps']:.2f} | "
            f"{item['aggregate_effective_throughput_mbps']:.2f} | "
            f"{item['max_transfer_latency_ms'] / 1000:.3f}s |"
        )
    markdown.extend(["",
        "H1 100/300 MiB cells are explicitly unavailable: at 3 Mbps their ideal transfer "
        "times are about 280/839 seconds before physical-network overhead; the deployed "
        "Worker's source read timeout is 300 seconds. No latency was fabricated for them.", "",
        "`configured_bandwidth_mbps` and `configured_added_rtt_ms` are application-layer "
        "pacing inputs, not measured physical-link properties. `icmp_rtt` is the observed "
        "unshaped baseline. `effective_throughput_mbps` and `transfer_latency_ms` are derived "
        "from actual transferred bytes and measured elapsed time; deviations from the simple "
        "`bytes / configured_bandwidth + added RTT` model include HTTP setup, pacing "
        "granularity, Wi-Fi jitter, and contention. Concurrent aggregate effective throughput "
        "exposes whether the three flows share a physical WLAN bottleneck.",
    ])
    (output_directory / "summary.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    return summary
