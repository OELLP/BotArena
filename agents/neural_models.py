from __future__ import annotations

import math
from collections import Counter
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

if TYPE_CHECKING:
    from scripts.misbot_information import InformationRecord


GRAPH_FEATURES = 6
SEQUENCE_FEATURES = 6


class GraphSAGEClassifier(nn.Module):
    def __init__(self, input_dim: int = GRAPH_FEATURES, hidden_dim: int = 48) -> None:
        super().__init__()
        self.layer1 = nn.Linear(input_dim * 2, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, 2)

    def forward(
        self,
        nodes: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        hidden = torch.relu(self.layer1(torch.cat((nodes, _neighbor_mean(nodes, edge_index)), dim=1)))
        hidden = torch.relu(self.layer2(torch.cat((hidden, _neighbor_mean(hidden, edge_index)), dim=1)))
        graph_count = int(batch.max().item()) + 1 if batch.numel() else 1
        pooled = torch.zeros(graph_count, hidden.shape[1], device=hidden.device)
        pooled.index_add_(0, batch, hidden)
        counts = torch.bincount(batch, minlength=graph_count).clamp_min(1).unsqueeze(1)
        return self.classifier(pooled / counts)


class TemporalGRUClassifier(nn.Module):
    def __init__(
        self,
        sequence_dim: int = SEQUENCE_FEATURES,
        static_dim: int = 10,
        hidden_dim: int = 48,
    ) -> None:
        super().__init__()
        self.gru = nn.GRU(sequence_dim, hidden_dim, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim + static_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim, 2),
        )

    def forward(
        self,
        sequence: torch.Tensor,
        lengths: torch.Tensor,
        static: torch.Tensor,
    ) -> torch.Tensor:
        packed = pack_padded_sequence(
            sequence,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, hidden = self.gru(packed)
        return self.classifier(torch.cat((hidden[-1], static), dim=1))


def graph_arrays(item: InformationRecord, max_nodes: int = 256) -> tuple[np.ndarray, np.ndarray]:
    source = "__source__"
    action_users = {
        "comment": item.comment_users,
        "repost": item.repost_users,
        "attitude": item.attitude_users,
    }
    candidates = set(item.participants)
    for left, right, _ in item.interaction_edges:
        candidates.update((left, right))
    selected = [source, *sorted(candidates)[: max(0, max_nodes - 1)]]
    index = {name: position for position, name in enumerate(selected)}

    edges: list[tuple[int, int]] = []
    for left, right, _ in item.interaction_edges:
        if left in index and right in index:
            edges.extend(((index[left], index[right]), (index[right], index[left])))
    if not edges:
        for users in action_users.values():
            for user in dict.fromkeys(users):
                if user in index:
                    edges.extend(((0, index[user]), (index[user], 0)))

    incoming: Counter[int] = Counter(target for _, target in edges)
    outgoing: Counter[int] = Counter(origin for origin, _ in edges)
    action_counts = {
        action: Counter(user for user in users if user in index)
        for action, users in action_users.items()
    }
    nodes = np.asarray(
        [
            [
                math.log1p(action_counts["comment"][name]),
                math.log1p(action_counts["repost"][name]),
                math.log1p(action_counts["attitude"][name]),
                math.log1p(incoming[position]),
                math.log1p(outgoing[position]),
                float(name == source),
            ]
            for position, name in enumerate(selected)
        ],
        dtype=np.float32,
    )
    edge_index = np.asarray(edges, dtype=np.int64).T if edges else np.empty((2, 0), dtype=np.int64)
    return nodes, edge_index


def temporal_array(item: InformationRecord, max_events: int = 256) -> np.ndarray:
    events = item.activity_events or [(value, "unknown") for value in item.activity_times]
    ordered = sorted(events, key=lambda event: event[0])
    if len(ordered) > max_events:
        indices = np.linspace(0, len(ordered) - 1, max_events, dtype=np.int64)
        ordered = [ordered[index] for index in indices]
    if not ordered:
        return np.zeros((1, SEQUENCE_FEATURES), dtype=np.float32)

    first = ordered[0][0]
    previous = first
    duration = max(ordered[-1][0] - first, 1.0)
    rows: list[list[float]] = []
    for position, (timestamp, action) in enumerate(ordered, start=1):
        rows.append(
            [
                math.log1p(max(timestamp - previous, 0.0)),
                (timestamp - first) / duration,
                float(action == "comment"),
                float(action == "repost"),
                float(action == "attitude"),
                position / len(ordered),
            ]
        )
        previous = timestamp
    return np.asarray(rows, dtype=np.float32)


def _neighbor_mean(nodes: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    if edge_index.numel() == 0:
        return torch.zeros_like(nodes)
    source, target = edge_index
    aggregated = torch.zeros_like(nodes)
    aggregated.index_add_(0, target, nodes[source])
    degree = torch.bincount(target, minlength=nodes.shape[0]).clamp_min(1).unsqueeze(1)
    return aggregated / degree
