#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from causalalign.data.manifest import prepare


def main() -> None:
    parser = argparse.ArgumentParser(description="Build deterministic DL3DV+RE10K paper splits")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.config), indent=2))


if __name__ == "__main__":
    main()

