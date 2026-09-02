from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.misbot_io import DEFAULT_MISBOT_ROOT, MisBotDataset, PROJECT_ROOT, summarize_users


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and profile the local MisBot dataset")
    parser.add_argument("--root", type=Path, default=DEFAULT_MISBOT_ROOT)
    parser.add_argument("--full", action="store_true", help="Profile train_data.jsonl instead of the active sampled set")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs" / "misbot_profile.json")
    args = parser.parse_args()

    dataset = MisBotDataset(args.root)
    dataset.require_complete()
    report = {
        "dataset_root": str(dataset.root.resolve()),
        "files": dataset.validation_report(),
        "user_profile": summarize_users(dataset.iter_users(sampled=not args.full, limit=args.limit)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["user_profile"], ensure_ascii=False, indent=2))
    print(f"Profile written to {args.output}")


if __name__ == "__main__":
    main()

