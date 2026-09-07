# Infra-Aware MAS

A minimal, decoupled, and traceable distributed multi-agent system prototype.

The implementation follows the phases in the project specification. Phase 0 establishes the
package layout and development tooling; semantic models and runtime behavior are added only in
subsequent phases.

## Development

```bash
uv sync
uv run pytest
uv run ruff check .
uv run pyright
```

The architecture keeps semantic agents separate from physical executors, uses artifact references
for data movement, and confines OpenAI Agents SDK integration to the planner package.

