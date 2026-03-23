from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Emit fixture payloads for deterministic dry runs.")
    parser.add_argument("--fixture", required=True, help="Path to the fixture JSON file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = json.loads(Path(args.fixture).read_text())
    print(json.dumps(payload))


if __name__ == "__main__":
    main()
