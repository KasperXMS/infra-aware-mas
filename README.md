# Infra-Aware MAS

A minimal, decoupled, and traceable distributed multi-agent system prototype. The current runtime
supports an end-to-end resource-blind baseline across independently deployed physical Workers.

## Development

```bash
uv sync
uv run pytest
uv run ruff check .
uv run pyright
```

## Real-machine resource-blind run

The example topology assumes an AGX Orin Worker and an RTX 4090 Worker. Each machine must already
have an OpenAI-compatible Chat Completions inference server exposing the models named in its Worker
configuration. Model deployment itself is intentionally outside this prototype.

### 1. Configure the Workers

Edit:

- `configs/workers/orin-1.yaml` on the Orin
- `configs/workers/gpu-1.yaml` on the RTX 4090 machine

Set each backend `base_url` and `model` to the locally reachable inference endpoint. A server that
does not require authentication can keep `api_key_env: null`. Otherwise set it to an environment
variable name and export that variable before startup.

Validate the configuration and model endpoint, then start each process:

```bash
uv sync
uv run infra-mas-worker --config configs/workers/orin-1.yaml --check
uv run infra-mas-worker --config configs/workers/orin-1.yaml
```

```bash
uv sync
uv run infra-mas-worker --config configs/workers/gpu-1.yaml --check
uv run infra-mas-worker --config configs/workers/gpu-1.yaml
```

Both Workers bind to `0.0.0.0` by default. Restrict network access with the host firewall; this
prototype deliberately has no authentication layer.

### 2. Configure the Controller

On the Controller, edit `configs/executors.yaml` and replace `orin-host` and `gpu-host` with
resolvable hostnames or LAN IP addresses. The Executor IDs must exactly match those exposed by the
two Worker YAML files. `configs/models.yaml` is the logical model directory exposed to the Planner;
it contains descriptions, modalities, and context windows but no host, device, endpoint, or replica
information.

Then edit `configs/blind.yaml`:

- choose `planner_mode`: `static_agents`, `dynamic_models`, or `hybrid`;
- choose `planner_harness`: `minimal`, `stateful`, or `efficient`;
- map each logical `model_id` to a physical executor for the fixed resource-blind scheduler;
- set the Coordinator model;
- use `api: responses` for the official OpenAI API, or `api: chat_completions` plus `base_url` for a
  compatible server;
- set `api_key_env` to the credential environment variable, or `null` for an unauthenticated local
  endpoint.

`static_agents` exposes only the presets in `agents.yaml`. `dynamic_models` makes `agents_config`
optional and lets the Planner create a role, system instructions, task, and artifact inputs for each
logical model invocation. `hybrid` exposes both mechanisms. Presets are compatibility conveniences:
they are converted to the same `InvocationSpec` used by dynamic roles before scheduling and
execution.

`minimal` preserves the basic tool results. `stateful` also returns a deterministic planning ledger
after each completed delegation or dynamic invocation, including initial-input use counts, unused
inputs, and invocation history. `efficient` uses the same execution semantics and adds only generic
guidance to consider existing evidence, avoid substantially redundant work, and stop when the task
can be adequately answered. The ledger does not parse task requirements, infer completion, reject
calls, or prescribe a workflow.

For the official API, set the configured key in the environment. For example in PowerShell:

```powershell
$env:OPENAI_API_KEY = "your-key"
```

The custom Coordinator endpoint wiring follows the [official OpenAI Agents SDK model-provider
configuration](https://openai.github.io/openai-agents-python/models/).

### 3. Preflight and run

The preflight checks configuration, HTTP health, Worker identities, and exact Executor membership.
It also checks that each OpenAI-compatible endpoint exposes the configured model through
`/v1/models`, but it does not invoke inference:

```bash
uv run infra-mas-run --config configs/blind.yaml --check
```

To verify the multi-process HTTP wiring on one development machine without model servers, start
the two mock Workers in separate terminals and run the smoke preflight:

```bash
uv run infra-mas-worker --config configs/smoke/worker-a.yaml
uv run infra-mas-worker --config configs/smoke/worker-b.yaml
uv run infra-mas-run --config configs/smoke/blind.yaml --check
```

Run the full resource-blind MAS workflow with one or more initial artifacts:

```bash
uv run infra-mas-run \
  --config configs/blind.yaml \
  --task "Analyze the supplied image and answer: what is happening?" \
  --input /path/to/image.jpg
```

Use repeated `--input` arguments for multiple files. Inputs are streamed to the configured ingress
Worker. For subsequent movement, the Controller sends only a pull instruction: the target Worker
streams bytes directly from the source Worker. Source, target, byte count, and transfer latency are
recorded by the transfer layer.

Each run prints the final answer and writes `config.yaml`, `trace.jsonl`, and `result.json` beneath
`runs/<run-id>/`. Use `--run-id name` when a stable run name is useful. A non-empty directory for
the selected run ID is always rejected; runs are never appended or silently reused.

The effective config records the selected Planner harness and the initial ledger state. Trace events
record each deterministic ledger update, so completed invocation history can be reconstructed
offline; stateful and efficient results also include the final ledger snapshot.

Execution results report `service_ms`, meaning elapsed model-service time including model-server
queueing, inference, and Worker-to-model RPC latency. `queue_ms` remains zero when queue time cannot
be observed independently; it is not an inferred compute or queue measurement.

## Supported model inputs

The OpenAI-compatible Worker backend accepts UTF-8 text/JSON/XML/YAML and common image formats.
Images are encoded as Chat Completions `image_url` data URLs by the Worker. Unsupported binary
formats fail explicitly instead of being silently inserted into the control plane.

The Planner is not given a predefined DAG or execution topology. It can adaptively invoke logical
models based on the task and current artifacts; the scheduler independently maps each logical model
to a physical executor replica. Artifact references remain the only data-movement inputs, and
OpenAI Agents SDK integration stays confined to the planner package.

## Real-system infrastructure-awareness pilot

The pilot in `configs/pilot/` uses the same open-ended `dynamic_models` + `efficient`
Coordinator, logical model, tools, and deterministic locality scheduler in both arms.
`infrastructure_visibility` supports three levels: `none` exposes no physical facts; `static`
exposes only generic, world-invariant execution semantics and service profiles; `snapshot` exposes
that identical static context plus raw artifact placement, replica sites, per-replica service
estimates, and network facts. None of the levels exposes candidate workflows, evaluator answers, or
optimizer-computed workflow costs.

The same six V1 images are placed in two worlds: `colocated` puts all six at A28, while
`distributed` puts two each at A4, A5, and A28.

Run the hand-authored reference workflows without constructing a Planner context:

```bash
uv run infra-mas-pilot --config configs/pilot/pilot.yaml --mode calibration
```

Run the paired Qwen3.8-Max Planner arms (requires `DASHSCOPE_API_KEY`):

```bash
uv run infra-mas-pilot --config configs/pilot/pilot.yaml --mode planner
```

Each run records invocation structure and bindings, transfer bytes/latency, service time, E2E,
Planner token usage, and final answer. Summaries are written under `runs/pilot/`. Reference
workflow definitions exist only in the calibration path and are never included in Planner input.

## infra-bench JSON bridge

`infra-mas-bench` executes one oracle-free JSON world exported by the sibling `infra-bench`
repository. It materializes every artifact at the specified physical site, then uses the normal
open-ended Planner, tools, deterministic locality scheduler, Worker transfer path, and trace
recorder. It writes `mas_result.json` beside `result.json` and `trace.jsonl`, including an
artifact-dependency-based realized workflow.

```bash
uv run infra-mas-bench \
  --case ../infra-bench/data/mas_exports/v1-smoke/world-a.json \
  --config configs/pilot/pilot.yaml \
  --visibility static \
  --run-id example-static-a

uv run infra-mas-bench \
  --case ../infra-bench/data/mas_exports/v1-smoke/world-a.json \
  --config configs/pilot/pilot.yaml \
  --visibility snapshot \
  --run-id example-snapshot-a
```

Use the same two commands for `world-b.json`. The benchmark case never contains hidden ground
truth or the calibration workflow bank; those remain on the `infra-bench` side.

### Repeated-run locality report and calibration sweep

Every `mas_result.json` includes cross-Worker transfer count/bytes, multimodal invocation count,
site-aligned invocation count/rate, and the initial-image grouping used by each VLM call. Aggregate
repeated V1 runs with:

```bash
uv run infra-mas-bench summarize-v1 \
  --run runs/example-static-a \
  --run runs/example-snapshot-a \
  --output runs/pilot
```

The hidden-reference calibration runner supports configuration-level replica eligibility and
per-artifact placement without changing Worker processes:

```bash
uv run infra-mas-bench calibrate \
  --sweep configs/pilot/sweeps/replica_locality.yaml \
  --output runs/calibration
```

Calibration creates no Coordinator or Planner context. Its `centralized` and `distributed_3x2`
reference IDs exist only in calibration trace metadata and reports. Admission requires correct
explicit answers, opposite median-E2E winners, and at least 20% winner margin in both worlds.

The V2 search also includes the hidden `distributed_6x1` reference. Its six one-image VLM calls
feed one real `remote-llm` synthesis invocation through normal artifact-transfer and tracing paths.
Start the calibration-only remote Worker, then run the V2 sweep:

```bash
uv run infra-mas-worker \
  --config configs/pilot/workers/coordinator-remote.yaml

uv run infra-mas-bench calibrate \
  --sweep configs/pilot/sweeps/replica_locality_v2.yaml \
  --output runs/calibration
```

The remote Worker reads `DASHSCOPE_API_KEY`. It is excluded from the normal pilot Planner model
catalog and is used only by hidden calibration references.

## Open-ended SWE-bench semantic-switch experiment

The `code_tasks` runtime connects an admitted SWE-bench case to the same Planner model interface
without exposing a candidate workflow or verified patch. It offers only generic repository tools:
code search/read, exact edit or patch application, targeted/full tests, and final patch submission.
The deterministic repository-locality scheduler binds every tool call to the Worker holding the
materialized repository. `static` supplies world-invariant execution semantics; `snapshot` appends
only the current repository location, measured artifact size, and network facts.

Start one code Worker at each materialized repository, then run one arm:

```bash
uv run infra-mas-code worker configs/semantic_switch/edge-worker.yaml
uv run infra-mas-code worker configs/semantic_switch/cloud-worker.yaml

uv run infra-mas-code run \
  --config configs/semantic_switch/astropy_14309.yaml \
  --world H1_edge \
  --visibility snapshot \
  --run-id formal-h1-snapshot-r1
```

Evaluate a submitted patch with the official SWE-bench environment and aggregate only run IDs
beginning with `formal-`:

```bash
uv run infra-mas-code evaluate \
  --run-directory runs/semantic_switch/astropy__astropy-14309/formal-h1-snapshot-r1 \
  --python /path/to/swebench-env/bin/python \
  --workdir /path/to/infra-bench

uv run infra-mas-code report \
  --runs-root runs/semantic_switch/astropy__astropy-14309 \
  --output runs/semantic_switch/astropy__astropy-14309
```
