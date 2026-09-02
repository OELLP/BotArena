from __future__ import annotations

import json
import math
import re
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MISBOT_ROOT = PROJECT_ROOT / "data" / "raw" / "MisBot"
USER_FEATURE_NAMES = [
    "粉丝数",
    "关注数",
    "发文数",
    "关注占比",
    "采样发文量",
    "平均文本长度",
    "文本唯一率",
    "纯转发占比",
    "链接占比",
    "提及占比",
    "简介长度",
    "已认证",
    "未认证",
    "SVIP",
    "非SVIP",
] + [f"会员等级={value}" for value in range(10)] + [
    f"会员类型={value}" for value in (0, 2, 11, 12, 13, 14)
]


@dataclass(slots=True)
class UserRecord:
    uid: str
    tweets: list[str]
    description: str
    numerical: list[float]
    categorical: list[float]
    label: int | None


def iter_jsonl(path: Path, limit: int | None = None) -> Iterator[dict[str, Any]]:
    """Yield JSON objects one line at a time to keep memory use bounded."""
    emitted = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
            yield record
            emitted += 1
            if limit is not None and emitted >= limit:
                break


class MisBotDataset:
    REQUIRED_FILES = {
        "train_data": Path("User_Instances/train_data.jsonl"),
        "train_data_sampled": Path("User_Instances/train_data_sampled.jsonl"),
        "inference_data": Path("User_Instances/inference_data.jsonl"),
        "inference_labels": Path("User_Instances/inference_labels.json"),
        "misinformation": Path("Information_Instances/misinformation.jsonl"),
        "verified_information": Path("Information_Instances/verified_information.jsonl"),
        "trend_information": Path("Information_Instances/trend_information.jsonl"),
    }

    def __init__(self, root: str | Path = DEFAULT_MISBOT_ROOT) -> None:
        self.root = Path(root)

    def validation_report(self) -> dict[str, dict[str, Any]]:
        report: dict[str, dict[str, Any]] = {}
        for name, relative_path in self.REQUIRED_FILES.items():
            path = self.root / relative_path
            report[name] = {
                "exists": path.is_file(),
                "path": str(path),
                "size_bytes": path.stat().st_size if path.is_file() else 0,
            }
        return report

    def require_complete(self) -> None:
        missing = [name for name, item in self.validation_report().items() if not item["exists"]]
        if missing:
            raise FileNotFoundError(f"MisBot core files are missing: {', '.join(missing)}")

    def iter_users(self, sampled: bool = True, limit: int | None = None) -> Iterator[UserRecord]:
        filename = "train_data_sampled.jsonl" if sampled else "train_data.jsonl"
        path = self.root / "User_Instances" / filename
        for raw in iter_jsonl(path, limit=limit):
            yield parse_user(raw)


def parse_user(raw: dict[str, Any]) -> UserRecord:
    tweets = [_as_text(value) for value in (raw.get("tweet") or [])]
    numerical = [_as_number(value) for value in (raw.get("numerical") or [])]
    categorical = [_as_number(value) for value in (raw.get("categorical") or [])]
    label = raw.get("label")
    return UserRecord(
        uid=str(raw.get("uid") or ""),
        tweets=[value for value in tweets if value],
        description=_as_text(raw.get("description")),
        numerical=numerical,
        categorical=categorical,
        label=int(label) if label is not None else None,
    )


def extract_user_features(user: UserRecord) -> list[float]:
    """Build a compact, explainable feature vector for the first baseline."""
    followers, following, statuses = _pad(user.numerical, 3)
    normalized_tweets = [_normalize_text(text) for text in user.tweets if text.strip()]
    tweet_count = len(normalized_tweets)
    unique_ratio = len(set(normalized_tweets)) / max(tweet_count, 1)
    lengths = [len(text) for text in normalized_tweets]
    avg_length = sum(lengths) / max(len(lengths), 1)
    repost_ratio = sum(text in {"转发微博", "轉發微博"} for text in normalized_tweets) / max(tweet_count, 1)
    url_ratio = sum("http" in text or "网页链接" in text for text in normalized_tweets) / max(tweet_count, 1)
    mention_ratio = sum("@" in text for text in user.tweets) / max(tweet_count, 1)

    base = [
        math.log1p(max(followers, 0.0)),
        math.log1p(max(following, 0.0)),
        math.log1p(max(statuses, 0.0)),
        following / max(followers + following, 1.0),
        float(tweet_count),
        float(avg_length),
        float(unique_ratio),
        float(repost_ratio),
        float(url_ratio),
        float(mention_ratio),
        float(len(user.description.strip())),
    ]
    return base + user.categorical


def extract_user_text(user: UserRecord) -> str:
    """Combine profile and timeline text for the text-analysis baseline."""
    parts = [user.description.strip()]
    parts.extend(tweet.strip() for tweet in user.tweets if tweet.strip())
    return " [SEP] ".join(part for part in parts if part)


def summarize_users(users: Iterator[UserRecord]) -> dict[str, Any]:
    labels: Counter[int | None] = Counter()
    feature_lengths: Counter[int] = Counter()
    tweet_total = 0
    count = 0
    empty_tweets = 0
    for user in users:
        count += 1
        labels[user.label] += 1
        feature_lengths[len(extract_user_features(user))] += 1
        tweet_total += len(user.tweets)
        empty_tweets += int(not user.tweets)
    return {
        "records": count,
        "labels": {str(key): value for key, value in sorted(labels.items(), key=lambda item: str(item[0]))},
        "average_tweets": round(tweet_total / max(count, 1), 4),
        "empty_tweet_records": empty_tweets,
        "feature_vector_lengths": {str(key): value for key, value in sorted(feature_lengths.items())},
    }


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", "", text.strip().lower())


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ("text", "content", "tweet"):
            if key in value:
                return str(value[key])
    return str(value)


def _as_number(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _pad(values: list[float], size: int) -> list[float]:
    return (values + [0.0] * size)[:size]
