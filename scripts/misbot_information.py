from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from scripts.misbot_io import DEFAULT_MISBOT_ROOT, MisBotDataset, iter_jsonl


@dataclass(slots=True)
class InformationRecord:
    category: str
    content: str
    publish_time: float | None
    repost_count: int
    comment_count: int
    attitude_count: int
    comment_users: list[str]
    repost_users: list[str]
    attitude_users: list[str]
    comment_nodes: int
    comment_edges: int
    repost_nodes: int
    repost_edges: int
    activity_times: list[float]
    interaction_edges: list[tuple[str, str, str]] = field(default_factory=list)
    activity_events: list[tuple[float, str]] = field(default_factory=list)

    @property
    def participants(self) -> set[str]:
        return set(self.comment_users) | set(self.repost_users) | set(self.attitude_users)


def iter_information(
    dataset: MisBotDataset,
    category: str,
    limit: int | None = None,
) -> Iterator[InformationRecord]:
    if category not in {"misinformation", "verified_information", "trend_information"}:
        raise ValueError(f"Unsupported information category: {category}")
    path = Path(dataset.validation_report()[category]["path"])
    for raw in iter_jsonl(path, limit=limit):
        article = raw.get("article") or {}
        comment_graphs = raw.get("comment_graphs") or []
        repost_graph = raw.get("repost_graph") or {}
        comment_nodes = [node for graph in comment_graphs for node in (graph.get("nodes") or [])]
        repost_nodes = repost_graph.get("nodes") or []
        interaction_edges = _comment_edges(comment_nodes)
        for graph in comment_graphs:
            interaction_edges.extend(
                _indexed_edges(graph.get("nodes") or [], graph.get("edges") or [], "comment")
            )
        interaction_edges.extend(_indexed_edges(repost_nodes, repost_graph.get("edges") or [], "repost"))
        interaction_edges = list(dict.fromkeys(interaction_edges))
        activity_events = [
            (timestamp, action)
            for nodes, action in ((comment_nodes, "comment"), (repost_nodes, "repost"))
            for node in nodes
            if (timestamp := _optional_float(node.get("publish_time"))) is not None
        ]
        yield InformationRecord(
            category=category,
            content=str(article.get("article_content") or ""),
            publish_time=_optional_float(article.get("publish_time")),
            repost_count=_int(article.get("repost_count")),
            comment_count=_int(article.get("comment_count")),
            attitude_count=_int(article.get("attitude_count")),
            comment_users=[str(uid) for uid in (raw.get("comment_users") or [])],
            repost_users=[str(uid) for uid in (raw.get("repost_users") or [])],
            attitude_users=[str(uid) for uid in (raw.get("attitude_users") or [])],
            comment_nodes=len(comment_nodes),
            comment_edges=sum(len(graph.get("edges") or []) for graph in comment_graphs),
            repost_nodes=len(repost_nodes),
            repost_edges=len(repost_graph.get("edges") or []),
            activity_times=[timestamp for timestamp, _ in activity_events],
            interaction_edges=interaction_edges,
            activity_events=activity_events,
        )


def extract_relation_features(item: InformationRecord) -> list[float]:
    action_counts = Counter(item.comment_users + item.repost_users + item.attitude_users)
    participants = item.participants
    actions = sum(action_counts.values())
    multi_action_ratio = sum(count > 1 for count in action_counts.values()) / max(len(participants), 1)
    comment_connectivity = item.comment_edges / max(item.comment_nodes, 1)
    repost_connectivity = item.repost_edges / max(item.repost_nodes, 1)
    return [
        math.log1p(len(participants)),
        math.log1p(actions),
        multi_action_ratio,
        math.log1p(item.comment_nodes),
        math.log1p(item.comment_edges),
        math.log1p(item.repost_nodes),
        math.log1p(item.repost_edges),
        comment_connectivity,
        repost_connectivity,
    ]


def extract_propagation_features(item: InformationRecord) -> list[float]:
    actions = len(item.comment_users) + len(item.repost_users) + len(item.attitude_users)
    repost_share = len(item.repost_users) / max(actions, 1)
    comment_share = len(item.comment_users) / max(actions, 1)
    attitude_share = len(item.attitude_users) / max(actions, 1)
    return [
        math.log1p(item.repost_count),
        math.log1p(item.comment_count),
        math.log1p(item.attitude_count),
        math.log1p(actions),
        repost_share,
        comment_share,
        attitude_share,
        _burst_ratio(item.activity_times),
        math.log1p(item.comment_nodes + item.repost_nodes),
        math.log1p(item.comment_edges + item.repost_edges),
    ]


def load_inference_labels(root: str | Path = DEFAULT_MISBOT_ROOT) -> dict[str, tuple[int, float]]:
    path = Path(root) / "User_Instances" / "inference_labels.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    labels: dict[str, tuple[int, float]] = {}
    for uid, value in raw.items():
        if not isinstance(value, list) or len(value) < 2:
            continue
        labels[str(uid)] = (int(value[0]), float(value[1]))
    return labels


def summarize_information(
    dataset: MisBotDataset,
    labels: dict[str, tuple[int, float]],
    limit_per_category: int | None = None,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    all_participants: set[str] = set()
    for category in ("misinformation", "verified_information", "trend_information"):
        records = 0
        participants: set[str] = set()
        total_actions = 0
        graph_nodes = 0
        graph_edges = 0
        for item in iter_information(dataset, category, limit=limit_per_category):
            records += 1
            item_participants = item.participants
            participants.update(item_participants)
            total_actions += len(item.comment_users) + len(item.repost_users) + len(item.attitude_users)
            graph_nodes += item.comment_nodes + item.repost_nodes
            graph_edges += item.comment_edges + item.repost_edges
        all_participants.update(participants)
        labelled = {uid for uid in participants if uid in labels}
        bots = {uid for uid in labelled if labels[uid][0] == 1}
        output[category] = {
            "records": records,
            "unique_participants": len(participants),
            "participant_actions": total_actions,
            "graph_nodes": graph_nodes,
            "graph_edges": graph_edges,
            "labelled_participants": len(labelled),
            "weakly_labelled_bots": len(bots),
            "bot_ratio_among_labelled": round(len(bots) / max(len(labelled), 1), 6),
        }
    labelled_all = {uid for uid in all_participants if uid in labels}
    output["overall"] = {
        "unique_participants": len(all_participants),
        "labelled_participants": len(labelled_all),
        "weakly_labelled_bots": sum(labels[uid][0] == 1 for uid in labelled_all),
        "label_note": "Inference labels are weakly supervised and are used for exploratory analysis, not strict ground truth.",
    }
    return output


def _int(value: Any) -> int:
    try:
        return max(0, int(float(value)))
    except (TypeError, ValueError):
        return 0


def _optional_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _burst_ratio(times: list[float], window_seconds: float = 120.0) -> float:
    if len(times) < 2:
        return 0.0
    ordered = sorted(times)
    close = sum(right - left <= window_seconds for left, right in zip(ordered, ordered[1:]))
    return close / (len(ordered) - 1)


def _comment_edges(nodes: list[dict[str, Any]]) -> list[tuple[str, str, str]]:
    return [
        (str(node["user_from"]), str(node["user_to"]), "comment")
        for node in nodes
        if node.get("user_from") is not None and node.get("user_to") is not None
    ]


def _indexed_edges(
    nodes: list[dict[str, Any]],
    edges: list[Any],
    action: str,
) -> list[tuple[str, str, str]]:
    names = [str(node.get("name") or node.get("uid") or index) for index, node in enumerate(nodes)]
    output: list[tuple[str, str, str]] = []
    for edge in edges:
        if not isinstance(edge, (list, tuple)) or len(edge) < 2:
            continue
        try:
            source, target = int(edge[0]), int(edge[1])
        except (TypeError, ValueError):
            continue
        if 0 <= source < len(names) and 0 <= target < len(names):
            output.append((names[source], names[target], action))
    return output
