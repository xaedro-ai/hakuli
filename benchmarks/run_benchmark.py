"""CLI entry point for Hakuli benchmarks.

Usage:
    python benchmarks/run_benchmark.py --config benchmarks/configs/<name>.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from benchmarks.harness import run  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a single Hakuli benchmark")
    parser.add_argument("--config", required=True, help="Path to a JSON config")
    parser.add_argument("--output", default=None, help="Override output path")
    args = parser.parse_args()

    config_path = Path(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))

    result = run(config, REPO_ROOT)
    output = Path(args.output or config.get("output") or f"benchmarks/results/{config_path.stem}.json")
    if not output.is_absolute():
        output = REPO_ROOT / output
    result.save(output)
    print(json.dumps(result.__dict__, indent=2))
    print(f"\nSaved to {output}")


if __name__ == "__main__":
    main()
