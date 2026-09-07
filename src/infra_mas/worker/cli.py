"""Command-line entry point for a physical Worker process."""

import argparse
import asyncio
from pathlib import Path

import uvicorn

from infra_mas.worker.config import WorkerConfig, build_worker_service
from infra_mas.worker.server import create_app


def build_parser() -> argparse.ArgumentParser:
    """Create the Worker command-line parser."""
    parser = argparse.ArgumentParser(description="Run an Infra-Aware MAS Worker")
    parser.add_argument("--config", type=Path, required=True, help="Worker YAML configuration")
    parser.add_argument("--host", help="Override the configured bind host")
    parser.add_argument("--port", type=int, help="Override the configured bind port")
    parser.add_argument("--check", action="store_true", help="Validate configuration and exit")
    parser.add_argument("--log-level", default="info", help="Uvicorn log level")
    return parser


def main() -> None:
    """Validate configuration and launch Uvicorn."""
    args = build_parser().parse_args()
    config_path = args.config.resolve()
    config = WorkerConfig.from_yaml(config_path)
    service = build_worker_service(config, config_path.parent)
    if args.check:

        async def check() -> None:
            try:
                await service.check_backends()
                status = service.status()
                print(f"worker {status.worker_id}: {', '.join(status.executors)}")
            finally:
                await service.aclose()

        asyncio.run(check())
        return

    uvicorn.run(
        create_app(service),
        host=args.host or config.host,
        port=args.port or config.port,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main()
