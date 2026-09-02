from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.misbot_io import (
    MisBotDataset,
    extract_user_features,
    extract_user_text,
    iter_jsonl,
    parse_user,
    summarize_users,
)


class MisBotIoTests(unittest.TestCase):
    def sample_record(self, label: int = 1) -> dict[str, object]:
        return {
            "uid": "u001",
            "tweet": ["转发微博", "转发微博", "正常内容"],
            "description": "测试用户",
            "numerical": [10, 90, 1000],
            "categorical": [0, 1] + [0] * 18,
            "label": label,
        }

    def test_parse_user_and_features(self) -> None:
        user = parse_user(self.sample_record())
        features = extract_user_features(user)
        self.assertEqual(user.uid, "u001")
        self.assertEqual(user.label, 1)
        self.assertEqual(len(features), 31)
        self.assertTrue(all(isinstance(value, float) for value in features))
        self.assertIn("测试用户", extract_user_text(user))
        self.assertIn("转发微博", extract_user_text(user))

    def test_jsonl_stream_respects_limit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.jsonl"
            path.write_text("\n".join(json.dumps(self.sample_record(index % 2)) for index in range(5)), encoding="utf-8")
            self.assertEqual(len(list(iter_jsonl(path, limit=2))), 2)

    def test_summary_counts_labels(self) -> None:
        users = iter([parse_user(self.sample_record(1)), parse_user(self.sample_record(0))])
        summary = summarize_users(users)
        self.assertEqual(summary["records"], 2)
        self.assertEqual(summary["labels"], {"0": 1, "1": 1})

    def test_validation_reports_missing_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset = MisBotDataset(directory)
            self.assertTrue(all(not item["exists"] for item in dataset.validation_report().values()))


if __name__ == "__main__":
    unittest.main()
