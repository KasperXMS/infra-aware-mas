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
