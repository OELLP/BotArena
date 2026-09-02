from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.misbot_information import load_inference_labels, summarize_information
from scripts.misbot_io import DEFAULT_MISBOT_ROOT, MisBotDataset, PROJECT_ROOT


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile MisBot information and propagation records")
    parser.add_argument("--root", type=Path, default=DEFAULT_MISBOT_ROOT)
    parser.add_argument("--limit-per-category", type=int)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs" / "information_profile.json")
    args = parser.parse_args()

    dataset = MisBotDataset(args.root)
    dataset.require_complete()
    labels = load_inference_labels(args.root)
    report = summarize_information(dataset, labels, limit_per_category=args.limit_per_category)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Profile written to {args.output}")


if __name__ == "__main__":
    main()

